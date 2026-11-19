import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
from csi_breathhold import p1


def write_rows(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(columns)
        writer.writerows(rows)


def make_wave(path, phase=0., gap=None, sign=1):
    t = np.arange(40 * 52) / 40
    # Fixture beats are arbitrary software inputs, with a reduced-motion interval.
    signal = sign * np.sin(2 * np.pi * .25 * t + phase) * np.where((t >= 20) & (t < 36), .01, 1.)
    valid = np.ones(len(t), dtype=bool) if gap is None else (t < gap[0]) | (t >= gap[1])
    rows = ((time, x if ok else "", int(ok), 0 if ok else -1)
            for time, x, ok in zip(t, signal, valid))
    write_rows(path, ["time_s", "pc2", "valid", "segment_id"], rows)


def make_manifest(directory):
    manifest = directory / "manifest.csv"
    rows = []
    for i, split in enumerate(["train", "val", "test"]):
        rid, pid = f"record{i}", f"person{i}"
        wave, labels = directory / f"pc{i}.csv", directory / f"labels{i}.csv"
        make_wave(wave, phase=i/10)
        write_rows(labels, ["recording_id", "start_s", "end_s", "label"],
                   [(rid, 0, 19.6, "natural_breathing"), (rid, 20.2, 35.8, "voluntary_breath_hold"),
                    (rid, 36, 52, "natural_breathing")])
        rows.append([rid, pid, split, wave.name, labels.name, "synthetic"])
    header = ["recording_id", "participant_id", "split", "waveform", "labels", "data_origin"]
    write_rows(manifest, header, rows)
    return manifest, header, rows


def test_windows_mask_gaps_edges_and_are_pc2_sign_invariant(tmp_path):
    wave = tmp_path / "p.csv"
    make_wave(wave, gap=(15., 16.))
    data = p1._windows(wave)
    assert data.seconds[0] == 4 and data.seconds[-1] == 47
    assert not np.isin(data.seconds, np.arange(11, 20)).any()
    make_wave(wave, gap=(15., 16.), sign=-1)
    other = p1._windows(wave)
    np.testing.assert_array_equal(data.features, other.features)
    targets = p1._targets(data, [(15.2, 18.9, 1), (19., 20., 0)])
    assert targets[15:21].tolist() == [-1, 1, 1, -1, 0, -1]
    assert p1._runs(np.array([False, True, False, True, True])) == [(1, 2), (3, 5)]


@pytest.mark.parametrize("change,message", [("person", "crosses splits"), ("duplicate", "Duplicate waveform"),
                                          ("group", "crosses splits"), ("origin", "data_origin")])
def test_manifest_guards(tmp_path, change, message):
    path, fields, rows = make_manifest(tmp_path)
    if change == "person":
        rows[1][1] = rows[0][1]
    if change == "duplicate":
        (tmp_path / rows[1][3]).write_bytes((tmp_path / rows[0][3]).read_bytes())
    if change == "group":
        fields.append("recording_group_id")
        rows = [r + ["same-source"] for r in rows]
    if change == "origin":
        rows[1][-1] = "real"
    write_rows(path, fields, rows)
    with pytest.raises(ValueError, match=message):
        p1._manifest(path)


def test_unlabeled_training_refused(tmp_path):
    path, _, rows = make_manifest(tmp_path)
    write_rows(tmp_path / rows[0][4], ["recording_id", "start_s", "end_s", "label"],
               [(rows[0][0], 5, 18, "natural_breathing")])
    with pytest.raises(ValueError, match="both classes"):
        p1.train_model(path, tmp_path / "blocked")
    assert not (tmp_path / "blocked/model.pt").exists()


def test_fit_predict_evaluate_without_test_label_training(tmp_path):
    path, fields, rows = make_manifest(tmp_path)
    test_labels = tmp_path / rows[-1][4]
    backup = test_labels.read_bytes()
    test_labels.unlink()
    out = tmp_path / "fit"
    training = p1.train_model(path, out, epochs=2)
    assert training["data_origin"] == "synthetic" and 1 <= training["best_epoch"] <= 2
    model = out / "model.pt"
    pred = p1.predict_file(model, tmp_path / rows[-1][3], tmp_path / "pred")
    assert pred["scored_seconds"] == 44 and pred["model_data_origin"] == "synthetic"
    with (tmp_path / "pred/scores.csv").open() as fp:
        first = next(csv.DictReader(fp))
        assert first["valid"] == "0" and first["score"] == ""
    test_labels.write_bytes(backup)
    summary = p1.evaluate_model(model, path, tmp_path / "eval")
    assert 0 < summary["evaluated_seconds"] < summary["annotated_seconds"]
    assert summary["evaluated_positive_seconds"] == 14
    confusion = summary["per_second"]
    assert sum(confusion[k] for k in ("tp", "tn", "fp", "fn")) == summary["evaluated_seconds"]
    # A disjoint-looking new manifest cannot relabel a training participant as test.
    rows[-1][1] = rows[0][1]
    write_rows(path, fields, [rows[-1]])
    with pytest.raises(ValueError, match="overlaps checkpoint"):
        p1.evaluate_model(model, path, tmp_path / "leak")


def test_overlapping_intervals_fail_and_event_matching_is_one_to_one(tmp_path):
    path = tmp_path / "labels.csv"
    write_rows(path, ["recording_id", "start_s", "end_s", "label"],
               [("r", 0, 5, "natural_breathing"), ("r", 4, 8, "voluntary_breath_hold")])
    with pytest.raises(ValueError, match="overlapping"):
        p1._intervals(path, "r")
    assert p1._match([(0, 4), (4, 8)], [(0, 8)]) == 1
