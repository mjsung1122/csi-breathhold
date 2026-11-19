import csv
import json

import numpy as np
import pytest

from csi_breathhold.p0 import FS, MIN_SAMPLES, _details_only, _resample_amplitudes, run_p0


def two_components(t):
    """Two orthogonal loadings: the weaker signal is known to occupy PC2."""
    common = 3 * np.sin(2 * np.pi * 0.31 * t)
    weak = np.cos(2 * np.pi * 0.43 * t)
    return 25 + np.column_stack((common + weak, common - weak, -common + weak, -common - weak))


def read_waveform(path):
    with (path / "waveform.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fixed_pc2_tracks_weaker_component_on_irregular_timestamps(tmp_path):
    ticks = np.arange(0., 90., 1 / 120)
    t = ticks + .0008 * np.sin(2 * np.pi * ticks)
    report = run_p0(t, two_components(t), tmp_path, recording_id="two-components")
    rows = [r for r in read_waveform(tmp_path) if r["valid"] == "1"]
    seconds = np.array([float(r["time_s"]) for r in rows])
    recovered = np.array([float(r["pc2"]) for r in rows])
    expected = np.cos(2 * np.pi * .43 * seconds)
    assert abs(np.corrcoef(expected, recovered)[0, 1]) > .85
    assert report == json.loads((tmp_path / "run.json").read_text())
    assert report["valid_samples"] == len(rows)
    assert report["retained_detail_levels"] == [6, 7]
    assert report["segments"][0]["psd_peak_hz"] == pytest.approx(.43, abs=.02)
    for name in ("waveform.png", "psd.png", "psd.csv"):
        assert (tmp_path / name).stat().st_size > 1000
    with pytest.raises(FileExistsError):
        run_p0(t, two_components(t), tmp_path, recording_id="other")


def test_fourier_resampling_rejects_signal_that_would_alias_into_respiration():
    t = np.arange(0, 60 + 1 / 240, 1 / 120)
    grid = np.arange(0, 60 + 1 / 80, 1 / FS)
    oscillations = np.cos(2 * np.pi * 39.7 * t)  # At 40 Hz without anti-aliasing: 0.3 Hz
    resampled = _resample_amplitudes(t, oscillations[:, None], grid)
    assert np.max(np.abs(resampled)) < 1e-7


def test_wavelet_details_preserve_inband_and_suppress_out_of_band():
    t = np.arange(0, 100, 1 / FS)
    signals = np.sin(2 * np.pi * t[:, None] * np.array([.04, .31, 2.0]))
    interior = slice(10 * FS, -10 * FS)
    filtered = _details_only(signals)[interior]
    gain = np.sqrt(np.mean(filtered**2, axis=0) / np.mean(signals[interior]**2, axis=0))
    assert gain[0] < .12 and gain[1] > .85 and gain[2] < .12


def test_gap_splits_and_keeps_unavailable_ticks(tmp_path):
    t = np.arange(0, 82, .025)
    present = (t < 40) | (t >= 42)
    x = two_components(t)
    x[~present] = np.nan
    run_p0(t, x, tmp_path, recording_id="gap")
    rows = read_waveform(tmp_path)
    assert len(rows) == len(t)
    assert {r["segment_id"] for r in rows if r["valid"] == "1"} == {"0", "1"}
    assert all(r["pc2"] == "" for r in rows if 35 <= float(r["time_s"]) < 47)
    assert all(r["segment_id"] == "-1" for r in rows if 40 <= float(r["time_s"]) < 42)


@pytest.mark.parametrize("bad", ["short", "rank-one", "constant", "absent"])
def test_unsupported_data_only_produces_diagnostics(tmp_path, bad):
    t = np.arange(0, MIN_SAMPLES + 100) / FS
    x = two_components(t)
    if bad == "short":
        t, x = t[:MIN_SAMPLES - 1], x[:MIN_SAMPLES - 1]
    elif bad == "rank-one":
        x = np.repeat(x[:, :1], 4, axis=1)
    elif bad == "constant":
        x[:] = 50
    else:
        x[:] = np.nan
    report = run_p0(t, x, tmp_path, recording_id=bad)
    assert report["status"] == "insufficient_data" and report["valid_samples"] == 0
    assert all(r["pc2"] == "" and r["valid"] == "0" for r in read_waveform(tmp_path))
    assert {p.name for p in tmp_path.iterdir()} == {"waveform.csv", "run.json"}


def test_duplicate_timestamp_refused(tmp_path):
    with pytest.raises(ValueError, match="increasing"):
        run_p0([0, 1, 1], np.ones((3, 4)), tmp_path, recording_id="repeated")
