import csv

import numpy as np
import pytest

from csi_breathhold.esp32 import PROFILE, read_capture


def packet(stamp, flag="0"):
    pairs = np.zeros((128, 2), dtype=int)
    pairs[1] = [3, 4]
    pairs[2] = [-128, 127]
    pairs[65] = [-12, 5]
    return {
        "ts_us": str(stamp),
        "len": "256",
        "csi": " ".join(map(str, pairs.ravel())),
        "first_word_invalid": flag,
    }


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_single_ltf_raw_slot_clock_and_int8(tmp_path):
    path = write_rows(tmp_path / "capture.csv", [packet(900000), packet(920000)])
    t, ll, meta = read_capture(path, profile=PROFILE, view="lltf")
    np.testing.assert_allclose(t, [0.0, 0.02])
    assert ll.shape == (2, 52)
    assert meta["raw_complex_slots"] == list(range(1, 27)) + list(range(38, 64))
    np.testing.assert_allclose(ll[0, :2], [5, np.hypot(-128.0, 127.0)])
    _, ht, meta = read_capture(path, profile=PROFILE, view="htltf")
    assert ht.shape == (2, 56)
    assert ht[0, 0] == 13
    assert meta["raw_complex_slots"] == list(range(65, 93)) + list(range(100, 128))
    assert meta["raw_complex_slots"][-1] == 127


@pytest.mark.parametrize("unknown", [True, False])
def test_first_word_excluded_without_moving_other_slots(tmp_path, unknown):
    rows = [packet(1), packet(20001, "1")]
    if unknown:
        for row in rows:
            del row["first_word_invalid"]
    path = write_rows(tmp_path / "flag.csv", rows)
    _, ll, report = read_capture(path, profile=PROFILE, view="lltf")
    np.testing.assert_allclose(ll[:, 0], np.hypot(-128.0, 127.0))
    assert ll.shape == (2, 51)
    assert report["lltf_slot_1_dropped_for_common_basis"] is True
    assert report["first_word_flags_accepted"] == ({"unknown": 2} if unknown else {"0": 1, "1": 1})
    _, ht, report = read_capture(path, profile=PROFILE, view="htltf")
    assert ht.shape == (2, 56)
    assert not report["lltf_slot_1_dropped_for_common_basis"]


def test_payload_damage_keeps_file_origin_and_clock_gaps(tmp_path):
    rows = [packet(5000 + i * 20000) for i in range(5)]
    rows[0]["csi"] = rows[0]["csi"].replace("-128", "-", 1)
    rows[2]["csi"] = rows[2]["csi"].replace("127", "300", 1)
    rows[3]["csi"] = "1 2"
    path = write_rows(tmp_path / "gap.csv", rows)
    t, ll, report = read_capture(path, profile=PROFILE, view="lltf")
    np.testing.assert_allclose(t, [0.02, 0.08])
    assert ll.shape == (2, 52)
    assert report["time_origin_device_us"] == 5000
    assert report["rejection_counts"] == {"malformed_csi": 1, "csi_outside_int8": 1, "csi_token_count": 1}
    assert [r["line"] for r in report["rejection_examples_first_20"]] == [2, 4, 5]


@pytest.mark.parametrize(("field", "value", "message"), [
    ("len", "128", "layout"), ("ts_us", "2", "backwards"), ("ts_us", "10", "repeated"),
    ("ts_us", "0.2", "timestamp"), ("ts_us", str(2**32), "uint32"),
])
def test_incompatible_layout_or_clock_requires_new_input(tmp_path, field, value, message):
    rows = [packet(10), packet(20)]
    rows[-1][field] = value
    path = write_rows(tmp_path / "bad.csv", rows)
    with pytest.raises(ValueError, match=message):
        read_capture(path, profile=PROFILE, view="lltf")


def test_unknown_format_and_headers(tmp_path):
    path = tmp_path / "not_csi.csv"
    path.write_text("time_s,length,values\n0,256,blank\n")
    with pytest.raises(ValueError, match="Unsupported profile"):
        read_capture(path, profile="generic", view="lltf")
    with pytest.raises(ValueError, match="view"):
        read_capture(path, profile=PROFILE, view="both")
    with pytest.raises(ValueError, match="Missing required"):
        read_capture(path, profile=PROFILE, view="lltf")
