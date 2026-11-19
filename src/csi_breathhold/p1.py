"""Offline, interval-supervised detection of voluntary breath holds from P0 PC2."""

import csv
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


FS = 40
WINDOW = 8 * FS
STEP = FS
THRESHOLD = 0.5
EVENT_IOU = 0.3
VERSION = "pc2-offline-hold-v1"
CLASSES = {"natural_breathing": 0, "voluntary_breath_hold": 1}


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        yield from reader


@dataclass
class Entry:
    recording_id: str
    participant_id: str
    group: str
    split: str
    waveform: Path
    labels: Path
    data_origin: str
    digest: str


def _manifest(path):
    path = Path(path).resolve()
    entries = []
    registered = set()
    participants, groups = {}, {}
    digests = set()
    required = ("recording_id", "participant_id", "split", "waveform", "labels", "data_origin")
    for row in _rows(path, required):
        if any(not (row[key] or "").strip() for key in required):
            raise ValueError(f"{path}: blank manifest field")
        rid, person, split = (row[k].strip() for k in required[:3])
        group = (row.get("recording_group_id") or rid).strip()
        origin = row["data_origin"].strip()
        if split not in {"train", "val", "test"} or origin not in {"real", "synthetic"}:
            raise ValueError("split must be train/val/test and data_origin real/synthetic")
        if rid in registered:
            raise ValueError(f"Duplicate recording_id: {rid}")
        if participants.setdefault(person, split) != split or groups.setdefault(group, split) != split:
            raise ValueError(f"Participant or recording group crosses splits: {rid}")
        waveform = (path.parent / row["waveform"]).resolve()
        with waveform.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in digests:
            raise ValueError(f"Duplicate waveform contents: {rid}")
        entry = Entry(rid, person, group, split, waveform, (path.parent / row["labels"]).resolve(),
                      origin, digest)
        entries.append(entry)
        registered.add(rid)
        digests.add(digest)
    if len({e.data_origin for e in entries}) != 1:
        raise ValueError("Manifest must contain recordings of exactly one data_origin")
    return entries


def _intervals(path, rid):
    intervals = []
    for row in _rows(path, ("recording_id", "start_s", "end_s", "label")):
        # One file may contain annotations for all recordings.
        if row["recording_id"].strip() != rid:
            continue
        label = row["label"].strip()
        if label not in CLASSES:
            raise ValueError(f"{rid}: unknown label {label!r}")
        a, b = float(row["start_s"]), float(row["end_s"])
        if not (0 <= a < b and math.isfinite(b)):
            raise ValueError(f"{rid}: invalid label interval")
        intervals.append((a, b, CLASSES[label]))
    intervals.sort()
    if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
        raise ValueError(f"{rid}: overlapping label intervals")
    return intervals


@dataclass
class Windows:
    features: np.ndarray
    seconds: np.ndarray  # Start time of each scoreable 1-second decision.
    duration: int         # Number of complete seconds in the recording.
    end_time: float


def _windows(path):
    tick, values, segment = [], [], []
    for row in _rows(path, ("time_s", "pc2", "valid", "segment_id")):
        time = float(row["time_s"])
        if not math.isfinite(time) or time < 0:
            raise ValueError("P0 time must be finite and recording-relative")
        sample = round(time * FS)
        if abs(time * FS - sample) > 1e-4 or (tick and sample <= tick[-1]):
            raise ValueError("P0 must use increasing 40 Hz ticks")
        if row["valid"] not in {"0", "1"}:
            raise ValueError("P0 valid must be 0 or 1")
        valid = row["valid"] == "1"
        value = float(row["pc2"]) if valid else np.nan
        ident = int(row["segment_id"]) if valid else -1
        if valid and (not math.isfinite(value) or ident < 0):
            raise ValueError("A valid P0 row needs finite PC2 and nonnegative segment_id")
        tick.append(sample)
        values.append(value)
        segment.append(ident)
    tick = np.asarray(tick, dtype=np.int64)
    segment = np.asarray(segment, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    end_time = (tick[-1] + 1) / FS if len(tick) else 0.
    sec, features = [], []
    cuts = np.flatnonzero((np.diff(tick) != 1) | (np.diff(segment) != 0)) + 1
    for start, stop in zip(np.r_[0, cuts], np.r_[cuts, len(tick)]):
        if stop - start < WINDOW or segment[start] < 0:
            continue
        piece = values[start:stop]
        centered = piece - np.median(piece)
        scale = float(np.quantile(np.abs(centered), .75))
        if scale <= 0:
            continue
        signal = np.clip(centered / scale, -20, 20)
        # Each decision covers [s,s+1). Its context extends 3.5s on both sides.
        margin = (WINDOW - STEP) // 2
        first = math.ceil((int(tick[start]) + margin) / FS)
        last = (int(tick[stop - 1]) + 1 - margin - STEP) // FS
        for s in range(first, last + 1):
            offset = s * FS - margin - tick[start]
            window = signal[offset:offset + WINDOW]
            # Sign-invariant 100 ms features; keep within-record amplitude changes.
            blocks = window.reshape(-1, 4)
            amplitude = np.sqrt(np.mean(blocks ** 2, axis=1))
            slope = np.abs(np.diff(window, prepend=window[0])).reshape(-1, 4).mean(axis=1)
            features.append(np.log1p(np.stack([amplitude, FS * slope])))
            sec.append(s)
    return Windows(np.asarray(features, dtype=np.float32).reshape(-1, 2, WINDOW // 4),
                   np.asarray(sec, dtype=int), int(end_time), float(end_time))


def _targets(windows, intervals):
    targets = np.full(windows.duration, -1, dtype=int)
    for a, b, value in intervals:
        # Event annotations often come from a higher-rate clock. The last P0 tick
        # and acquisition endpoint can differ by less than one P0 sample.
        if b > windows.end_time + 1 / FS + 1e-6:
            raise ValueError("Label interval extends beyond this recording")
        targets[math.ceil(a):math.floor(b)] = value
    return targets


class HoldCNN(nn.Module):
    """Temporal convolutions on a PC2 amplitude envelope and absolute slope."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(2, 12, 5, padding=2), nn.ReLU(),
                                     nn.Conv1d(12, 12, 5, padding=4, dilation=2), nn.ReLU(),
                                     nn.Conv1d(12, 12, 5, padding=8, dilation=4), nn.ReLU())
        self.head = nn.Linear(24, 1)

    def forward(self, x):
        z = self.encoder(x)
        midpoint = z.shape[-1] // 2
        local = z[:, :, midpoint - 5:midpoint + 5].mean(dim=-1)
        return self.head(torch.cat([local, z.mean(dim=-1)], dim=1)).squeeze(1)


def _dataset(entries, split):
    features, labels = [], []
    for e in entries:
        if e.split != split:
            continue
        w = _windows(e.waveform)
        y = _targets(w, _intervals(e.labels, e.recording_id))[w.seconds]
        use = y >= 0
        if use.any():
            features.append(w.features[use])
            labels.append(y[use])
    if not labels or len(np.unique(np.concatenate(labels))) < 2:
        raise ValueError(f"{split} needs scoreable, explicitly labelled seconds of both classes")
    return (torch.from_numpy(np.concatenate(features)),
            torch.from_numpy(np.concatenate(labels).astype(np.float32)))


@contextmanager
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def train_model(manifest, output_dir, *, epochs=30, seed=42):
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be positive")
    entries = _manifest(manifest)
    x, y = _dataset(entries, "train")
    vx, vy = _dataset(entries, "val")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {k: [int((v == i).sum()) for i in range(2)]
              for k, v in (("train", y), ("val", vy))}
    used = [dict(recording_id=e.recording_id, participant_id=e.participant_id,
                 recording_group_id=e.group, sha256=e.digest)
            for e in entries if e.split in {"train", "val"}]
    report = dict(version=VERSION, data_origin=entries[0].data_origin, seed=seed,
                  class_order=list(CLASSES), seconds_per_class=counts, threshold=THRESHOLD,
                  learning_rate=.003, batch_size=64, epochs=epochs, history=[])
    with _cpu_threads(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = HoldCNN()
        optimizer = torch.optim.AdamW(model.parameters(), lr=report["learning_rate"], weight_decay=.001)
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(counts["train"][0] / counts["train"][1]))
        best = math.inf
        for epoch in range(1, epochs + 1):
            model.train()
            for indices in torch.randperm(len(y)).split(report["batch_size"]):
                optimizer.zero_grad()
                value = loss(model(x[indices]), y[indices])
                value.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                logits = torch.cat([model(batch) for batch in vx.split(256)])
                validation = nn.functional.binary_cross_entropy_with_logits(logits, vy).item()
            if not math.isfinite(validation):
                raise ValueError("Training produced a non-finite validation loss")
            report["history"].append({"epoch": epoch, "val_bce": validation})
            if validation < best:
                best, report["best_epoch"] = validation, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    checkpoint = dict(version=VERSION, weights=best_state, data_origin=report["data_origin"],
                      threshold=THRESHOLD, fit_records=used)
    torch.save(checkpoint, out / "model.pt")
    _json(out / "training.json", report)
    return report


def _load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if (ckpt.get("version") != VERSION or ckpt.get("threshold") != THRESHOLD
            or ckpt.get("data_origin") not in {"real", "synthetic"}):
        raise ValueError("Incompatible PC2 event checkpoint")
    model = HoldCNN()
    model.load_state_dict(ckpt["weights"])
    return model.eval(), ckpt


def _scores(model, windows):
    scores = np.full(windows.duration, np.nan)
    if len(windows.seconds):
        with torch.no_grad(), _cpu_threads():
            batches = torch.from_numpy(windows.features).split(256)
            p = torch.cat([model(batch).sigmoid() for batch in batches]).numpy()
        if not np.isfinite(p).all():
            raise ValueError("Non-finite model output")
        scores[windows.seconds] = p
    return scores


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def predict_file(model_path, waveform_path, output_dir):
    """Predict without annotations. Unscoreable seconds have a blank score."""
    model, ckpt = _load(model_path)
    w = _windows(waveform_path)
    scores = _scores(model, w)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = (dict(start_s=s, end_s=s + 1, score=float(v) if np.isfinite(v) else "",
                 decision=int(v >= THRESHOLD) if np.isfinite(v) else "", valid=int(np.isfinite(v)))
            for s, v in enumerate(scores))
    _csv(out / "scores.csv", ("start_s", "end_s", "valid", "score", "decision"), rows)
    events = _runs(scores >= THRESHOLD)
    _csv(out / "events.csv", ("start_s", "end_s"), [dict(start_s=a, end_s=b) for a, b in events])
    report = dict(version=VERSION, model_data_origin=ckpt["data_origin"], threshold=THRESHOLD,
                  status="ok" if len(w.seconds) else "insufficient_data",
                  complete_seconds=w.duration, scored_seconds=len(w.seconds), events=len(events),
                  task="offline voluntary breath hold", score_unit="one second")
    _json(out / "prediction.json", report)
    return report


def _match(pred, truth):
    """Greedy one-to-one matching ordered by temporal IoU."""
    pairs = []
    for i, (a, b) in enumerate(pred):
        for j, (c, d) in enumerate(truth):
            iou = max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
            if iou >= EVENT_IOU:
                pairs.append((iou, i, j))
    p_used, t_used = set(), set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in p_used and j not in t_used:
            p_used.add(i)
            t_used.add(j)
    return len(p_used)


def _ratio(a, b):
    return a / b if b else None


def evaluate_model(model_path, manifest, output_dir):
    model, ckpt = _load(model_path)
    entries = [e for e in _manifest(manifest) if e.split == "test"]
    if not entries:
        raise ValueError("Evaluation requires a test split")
    fit = ckpt["fit_records"]
    for e in entries:
        if any((e.recording_id == r["recording_id"] or e.participant_id == r["participant_id"]
                or e.group == r["recording_group_id"] or e.digest == r["sha256"]) for r in fit):
            raise ValueError("Held-out data overlaps checkpoint training or validation records")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, tp, fp, tn, fn = [], 0, 0, 0, 0
    matched = predicted = reference = annotated = seconds = 0
    annotated_positive = evaluated_positive = 0
    for e in entries:
        w = _windows(e.waveform)
        scores = _scores(model, w)
        labels = _targets(w, _intervals(e.labels, e.recording_id))
        use = np.isfinite(scores) & (labels >= 0)
        pos, neg = (scores >= THRESHOLD) & use, (scores < THRESHOLD) & use
        tp += int(np.sum(pos & (labels == 1)))
        fp += int(np.sum(pos & (labels == 0)))
        tn += int(np.sum(neg & (labels == 0)))
        fn += int(np.sum(neg & (labels == 1)))
        pred, ref = _runs(pos), _runs((labels == 1) & use)
        matched += _match(pred, ref)
        predicted += len(pred)
        reference += len(ref)
        annotated += int(np.sum(labels >= 0))
        annotated_positive += int(np.sum(labels == 1))
        evaluated_positive += int(np.sum((labels == 1) & use))
        seconds += int(use.sum())
        rows.extend(dict(recording_id=e.recording_id, start_s=s, end_s=s + 1,
                         label=int(labels[s]) if labels[s] >= 0 else "",
                         score=float(v) if np.isfinite(v) else "", evaluated=int(use[s]))
                    for s, v in enumerate(scores))
    _csv(out / "test_seconds.csv", ("recording_id", "start_s", "end_s", "label", "score", "evaluated"), rows)
    report = dict(version=VERSION, status="ok" if seconds else "insufficient_labels_or_coverage",
                  threshold=THRESHOLD, model_data_origin=ckpt["data_origin"], data_origin=entries[0].data_origin,
                  test_recordings=len(entries), annotated_seconds=annotated, evaluated_seconds=seconds,
                  coverage=_ratio(seconds, annotated), annotated_positive_seconds=annotated_positive,
                  evaluated_positive_seconds=evaluated_positive,
                  per_second=dict(tp=tp, fp=fp, tn=tn, fn=fn, sensitivity=_ratio(tp, tp + fn),
                                  specificity=_ratio(tn, tn + fp), precision=_ratio(tp, tp + fp),
                                  f1=_ratio(2 * tp, 2 * tp + fp + fn)),
                  events=dict(iou_threshold=EVENT_IOU, matched=matched, predicted=predicted, reference=reference,
                              precision=_ratio(matched, predicted), recall=_ratio(matched, reference),
                              scope="contiguous event fragments on annotated and scoreable whole seconds only"))
    _json(out / "evaluation.json", report)
    return report
