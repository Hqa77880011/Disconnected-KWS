import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import nn
from torch.nn import functional as F


def audio(path):
    value, rate = sf.read(path, always_2d=True)
    value = value.mean(1)
    if rate != 16000:
        divisor = np.gcd(rate, 16000)
        value = resample_poly(value, 16000 // divisor, rate // divisor)
    return value.astype(np.float32)


def mel(value):
    x = torch.tensor(value, dtype=torch.float32)
    x = F.pad(x[:16000], (0, max(0, 16000 - len(x))))
    spectrum = torch.stft(x, n_fft=512, hop_length=160, win_length=400, window=torch.hann_window(400), return_complex=True).abs().square()
    hz = np.linspace(0, 8000, 257)
    points = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 42) / 2595) - 1)
    bank = np.maximum(0, np.minimum((hz[None] - points[:-2, None]) / (points[1:-1, None] - points[:-2, None]),
                                    (points[2:, None] - hz[None]) / (points[2:, None] - points[1:-1, None])))
    return (torch.tensor(bank, dtype=torch.float32) @ spectrum).clamp_min(1e-8).log()[None]


class Detector(nn.Module):
    def __init__(self, c):
        super().__init__()
        layers = [nn.Conv2d(1, 64, 3, padding=1), nn.ReLU()]
        for _ in range(3):
            layers.extend([nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.Conv2d(64, 64, 1), nn.ReLU(), nn.AvgPool2d(2)])
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten()])
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(64, c['classes'])

    def forward(self, x):
        feature = self.encoder(x)
        return self.head(feature), feature


class Adapter(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.v = nn.Parameter(torch.randn(64, c['rank']) * .01)
        self.u = nn.Parameter(torch.zeros(c['classes'], c['rank']))
        self.b = nn.Parameter(torch.zeros(c['classes']))

    def forward(self, feature, base):
        correction = feature @ self.v @ self.u.T + self.b
        if self.c['variant'] != 'unbounded':
            correction = self.c['amplitude'] * correction.tanh()
        return base + correction


def prepare(c, out, source):
    source = Path(source)
    frame = pd.read_csv(source)
    if not {'audio', 'label', 'speaker', 'split'} <= set(frame):
        raise ValueError('Manifest requires audio,label,speaker,split')
    frame.audio = frame.audio.map(lambda p: str((source.parent / p).resolve()))
    if frame.audio.duplicated().any():
        raise ValueError('Every original utterance must have a unique role')
    if set(frame[frame.split == 'train'].speaker) & set(frame[frame.split == 'dev'].speaker):
        raise ValueError('Training and development speakers must be disjoint')
    source_speakers = set(frame[frame.split.isin(['train', 'dev', 'anchor'])].speaker)
    target_speakers = set(frame[frame.split.isin(['support', 'accept', 'query'])].speaker)
    if source_speakers & target_speakers:
        raise ValueError('Enrollment speakers cannot appear in source training or development')
    frame.to_csv(out / 'manifest.csv', index=False)


def train(c, out):
    torch.manual_seed(c['seed'])
    frame = pd.read_csv(out / 'manifest.csv')
    model = Detector(c)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'])
    train_data = frame[frame.split == 'train']
    dev_data = frame[frame.split == 'dev']
    best = np.inf
    history = []
    for epoch in range(c['epochs']):
        model.train()
        for ids in np.array_split(np.random.default_rng(c['seed'] + epoch).permutation(len(train_data)), max(1, int(np.ceil(len(train_data) / c['batch_size'])))):
            batch = train_data.iloc[ids]
            x = torch.stack([mel(audio(path)) for path in batch.audio])
            labels = torch.tensor(batch.label.to_numpy(), dtype=torch.long)
            loss = F.cross_entropy(model(x)[0], labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        values = []
        with torch.no_grad():
            for row in dev_data.itertuples():
                values.append(float(F.cross_entropy(model(mel(audio(row.audio))[None])[0], torch.tensor([row.label]))))
        score = float(np.mean(values))
        history.append(dict(epoch=epoch, validation_cross_entropy=score))
        if score < best:
            best = score
            torch.save(model.state_dict(), out / 'base.pt')
    pd.DataFrame(history).to_csv(out / 'learning.csv', index=False)


def cached(model, rows):
    features, logits = [], []
    with torch.no_grad():
        for path in rows.audio:
            base, feature = model(mel(audio(path))[None])
            features.append(feature)
            logits.append(base)
    return torch.cat(features).half(), torch.cat(logits).half(), torch.tensor(rows.label.to_numpy(), dtype=torch.long)


def events(probabilities, c):
    current, count, last, emitted = None, 0, -np.inf, []
    for i, probability in enumerate(probabilities):
        label = int(np.argmax(probability))
        valid = label in c['wake_classes'] and probability[label] >= c['wake_threshold']
        count = count + 1 if valid and current == label else int(valid)
        current = label if valid else None
        timestamp = i * c['hop_seconds']
        if count >= c['wake_ticks'] and timestamp - last >= c['refractory_seconds']:
            emitted.append((timestamp, label))
            last, count = timestamp, 0
    return emitted


def run(c, out):
    torch.manual_seed(c['seed'])
    frame = pd.read_csv(out / 'manifest.csv')
    model = Detector(c)
    model.load_state_dict(torch.load(out / 'base.pt', weights_only=True, map_location='cpu'))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    anchors = cached(model, frame[frame.split == 'anchor'])
    workload = pd.read_csv(c['workload'])
    rows, traces = [], []
    for speaker, group in frame[frame.split.isin(['support', 'accept', 'query'])].groupby('speaker'):
        support = group[group.split == 'support']
        acceptance = group[group.split == 'accept']
        query = group[group.split == 'query']
        if len(support) < c['support_examples'] or len(acceptance) < c['acceptance_examples']:
            raise ValueError('Insufficient distinct enrollment utterances for speaker ' + str(speaker))
        support = support.iloc[:c['support_examples']]
        acceptance = acceptance.iloc[:c['acceptance_examples']]
        start = time.perf_counter()
        s, v, q = cached(model, support), cached(model, acceptance), cached(model, query)
        active = Adapter(c)
        candidate = copy.deepcopy(active)
        rollback = copy.deepcopy(active)
        parameters = sum(p.numel() for p in candidate.parameters())
        cache_bytes = sum(value.numel() * value.element_size() for values in [s, v, anchors] for value in values)
        workspace = cache_bytes + 4 * parameters * 4 + c['update_workspace_bytes']
        optimizer = torch.optim.SGD(candidate.parameters(), lr=c['adapter_learning_rate'])
        energy = c['energy_j'] - c['feature_energy_j'] * (len(support) + len(acceptance) + len(frame[frame.split == 'anchor']))
        steps, completed_at = 0, np.nan
        for tick in workload.itertuples():
            timing_ok = tick.slack_ms >= c['quantum_ms'] + c['guard_ms']
            memory_ok = tick.workspace_bytes >= workspace
            energy_ok = energy >= c['quantum_energy_j'] + c['check_energy_j']
            admitted = timing_ok and memory_ok and energy_ok
            if c['variant'] == 'immediate':
                admitted = memory_ok and energy_ok
            if c['variant'] == 'frozen':
                admitted = False
            if admitted and steps < c['steps']:
                feature, base, label = s
                logits = candidate(feature.float(), base.float())
                loss = F.cross_entropy(logits, label)
                if c['variant'] != 'no-anchor':
                    anchor_feature, anchor_base, _ = anchors
                    anchor_logits = candidate(anchor_feature.float(), anchor_base.float())
                    loss += c['anchor_weight'] * F.kl_div(anchor_logits.log_softmax(-1), anchor_base.float().softmax(-1), reduction='batchmean')
                loss += c['regularization'] * sum(p.square().sum() for p in candidate.parameters())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                steps += 1
                energy -= c['quantum_energy_j']
                if steps == c['steps']:
                    completed_at = tick.time_seconds
            traces.append(dict(speaker=speaker, time=tick.time_seconds, admitted=admitted, completed_steps=steps,
                               remaining_energy_j=energy, deadline_miss=bool(admitted and not timing_ok)))
            if steps == c['steps']:
                break
        accepted = False
        if steps == c['steps'] and energy >= c['check_energy_j']:
            with torch.no_grad():
                before = active(v[0].float(), v[1].float())
                after = candidate(v[0].float(), v[1].float())
                gain = float(F.cross_entropy(before, v[2]) - F.cross_entropy(after, v[2]))
                baseline = anchors[1].argmax(1)
                old_disagreement = (active(anchors[0].float(), anchors[1].float()).argmax(1) != baseline).float().mean()
                new_disagreement = (candidate(anchors[0].float(), anchors[1].float()).argmax(1) != baseline).float().mean()
                accepted = gain >= c['acceptance_gain'] and new_disagreement - old_disagreement <= c['anchor_tolerance']
                if c['variant'] == 'no-acceptance':
                    accepted = True
            energy -= c['check_energy_j']
            if accepted:
                active = candidate
                torch.save(active.state_dict(), out / f'adapter_{speaker}.pt')
        with torch.no_grad():
            probability = active(q[0].float(), q[1].float()).softmax(-1)
            predicted = probability.argmax(1)
            target = torch.isin(q[2], torch.tensor(c['personalized_classes']))
            recall = float((predicted[target] == q[2][target]).float().mean())
            margin = q[1].float().topk(2, dim=1).values.diff(dim=1).abs().squeeze(1)
            protected = margin > 2 * c['amplitude']
            preserved = float((predicted[protected] == q[1].argmax(1)[protected]).float().mean()) if protected.any() else np.nan
        false_wakes, exposure = 0, 0.
        background = frame[(frame.split == 'background') & (frame.speaker == speaker)]
        for sample in background.itertuples():
            wave = audio(sample.audio)
            probabilities = []
            with torch.no_grad():
                for offset in range(0, len(wave) - 16000 + 1, round(16000 * c['hop_seconds'])):
                    base, feature = model(mel(wave[offset:offset + 16000])[None])
                    probabilities.append(active(feature, base).softmax(-1)[0].numpy())
            false_wakes += len(events(probabilities, c))
            exposure += len(wave) / 16000 / 3600
        rows.append(dict(speaker=speaker, recall=recall, accepted=accepted, postponed=steps < c['steps'], steps=steps,
                         enrollment_seconds=completed_at, charged_energy_j=c['energy_j'] - energy,
                         adapter_parameters=parameters, enrollment_workspace_bytes=workspace,
                         protected_class_preservation=preserved, false_wakes=false_wakes, background_hours=exposure,
                         host_seconds=time.perf_counter() - start, rollback_bytes=sum(p.numel() * p.element_size() for p in rollback.parameters())))
    pd.DataFrame(rows).to_csv(out / 'speakers.csv', index=False)
    pd.DataFrame(traces).to_csv(out / 'schedule.csv', index=False)


def evaluate(c, out):
    d = pd.read_csv(out / 'speakers.csv')
    schedule = pd.read_csv(out / 'schedule.csv')
    result = dict(method=c['variant'], recall=d.recall.mean(), lower_decile_recall=d.recall.quantile(.1),
                  acceptance_rate=d.accepted.mean(), postponed_fraction=d.postponed.mean(),
                  enrollment_seconds=d.enrollment_seconds.mean(), charged_energy_j=d.charged_energy_j.mean(),
                  peak_enrollment_workspace_bytes=d.enrollment_workspace_bytes.max(),
                  adapter_parameters=d.adapter_parameters.max(), deadline_miss_fraction=schedule.deadline_miss.mean(),
                  protected_class_preservation=d.protected_class_preservation.mean())
    if d.background_hours.sum() > 0:
        result['false_wakes_per_hour'] = d.false_wakes.sum() / d.background_hours.sum()
    pd.DataFrame([result]).to_csv(out / 'metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'schedule.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    for speaker, g in d.groupby('speaker'):
        ax.step(g.time, g.completed_steps, where='post', label=str(speaker))
    ax.set(xlabel='Foreground trace time (s)', ylabel='Completed adaptation steps')
    fig.tight_layout()
    fig.savefig(out / 'schedule.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'train', 'run', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    p.add_argument('--variant')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.variant:
        c['variant'] = args.variant
    if c['variant'] not in ['full', 'frozen', 'unbounded', 'no-anchor', 'no-acceptance', 'immediate']:
        p.error('Unknown variant')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    if args.variant is None and settings.exists():
        c['variant'] = json.loads(settings.read_text(encoding='utf-8')).get('variant', c.get('variant', 'full'))
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if not args.input:
            p.error('--input required')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
