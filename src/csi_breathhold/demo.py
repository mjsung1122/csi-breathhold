"""Deterministic synthetic software fixture spanning the real CSV entry point."""

import csv

import numpy as np

from .cli import PROFILE, convert


def _fixture(path, index, holds):
    rng = np.random.default_rng(729 + index)
    t = np.arange(120 * 50) / 50.0
    active = np.ones(len(t))
    for start, end in holds:
        active[(t >= start) & (t < end)] = 0
    slots = np.arange(128)
    phase = rng.uniform(0, 2 * np.pi, 128)
    common = 12 * np.sin(2 * np.pi * (0.42 + 0.009 * index) * t + rng.uniform(0, 6))
    periodic = active * np.sin(2 * np.pi * (0.23 + 0.014 * index) * t + rng.uniform(0, 6))
    motion = 6 * np.cos(2 * np.pi * (slots % 64) / 64)
    amplitude = 40 + common[:, None] + periodic[:, None] * motion + rng.normal(0, 0.6, (len(t), 128))
    iq = np.empty((len(t), 256), dtype=np.int8)
    iq[:, 0::2] = np.rint(amplitude * np.sin(phase))
    iq[:, 1::2] = np.rint(amplitude * np.cos(phase))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq", "lost", "ts_us", "rssi", "first_word_invalid", "len", "csi"])
        for i, pair in enumerate(iq):
            writer.writerow((i, 0, 100_000 + i * 20_000, -55, 1, 256, " ".join(map(str, pair.tolist()))))


def make_demo(out):
    manifest_rows = []
    for i, split in enumerate(("train", "train", "train", "val", "test", "test")):
        rid = f"synthetic_{i + 1:02d}"
        folder = out / rid
        folder.mkdir()
        holds = [(26 + 2 * i, 41 + 2 * i), (72 - i, 88 - i)]
        _fixture(folder / "capture.csv", i, holds)
        result = convert(folder / "capture.csv", folder / "p0", rid, PROFILE, "lltf")
        if result["status"] != "ok":
            raise ValueError(f"Synthetic fixture produced no valid PC2 for {rid}")
        edges = [0, *holds[0], *holds[1], 120]
        with (folder / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("recording_id", "start_s", "end_s", "label"))
            for n, (start, end) in enumerate(zip(edges, edges[1:])):
                label = "voluntary_breath_hold" if n % 2 else "natural_breathing"
                writer.writerow((rid, start, end, label))
        manifest_rows.append((rid, f"fixture_person_{i + 1}", split, f"{rid}/p0/waveform.csv",
                              f"{rid}/labels.csv", "synthetic"))
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("recording_id", "participant_id", "split", "waveform", "labels", "data_origin"))
        writer.writerows(manifest_rows)
