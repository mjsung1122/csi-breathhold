"""Offline Alzaabi-style CSI amplitude processing (IEEE Access, 2024).

Only time and CSI amplitudes enter this module. PC2 has arbitrary polarity
and relative amplitude. A valid value indicates signal support, not that a
human is present or that respiratory motion has been independently verified.
"""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pywt
from scipy.signal import periodogram, resample


FS = 40
WAVELET = pywt.Wavelet("db4")
LEVEL = 7
DETAILS = (6, 7)
GAP_S = 0.25
EDGE_S = 5
MIN_SAMPLES = (WAVELET.dec_len - 1) * 2**LEVEL
BAND_HZ = (FS / 2**8, FS / 2**6)
ARTIFACTS = ("waveform.csv", "psd.csv", "waveform.png", "psd.png", "run.json")


def _resample_amplitudes(t, x, grid):
    """Interpolate irregular samples, then Fourier-resample to exact grid ticks.

    Both uniform grids share their first and last instants. Even reflection
    makes the Fourier period twice the interval and avoids joining unequal ends.
    Truncating the Fourier spectrum suppresses aliasing when downsampling.
    """
    packet_hz = (len(t) - 1) / (t[-1] - t[0])
    intervals = max(1, round((grid[-1] - grid[0]) * packet_hz))
    source_grid = np.linspace(grid[0], grid[-1], intervals + 1)
    uniform = np.column_stack([np.interp(source_grid, t, col) for col in x.T])
    if len(uniform) == len(grid):
        return uniform
    reflected = np.concatenate((uniform, uniform[-2:0:-1]), axis=0)
    return resample(reflected, 2 * (len(grid) - 1), axis=0)[:len(grid)]


def _details_only(x):
    coefficients = pywt.wavedec(x, WAVELET, mode="symmetric", level=LEVEL, axis=0)
    # wavedec returns [A7, D7, D6, D5, ..., D1].
    keep = {LEVEL - level + 1 for level in DETAILS}
    chosen = [c if i in keep else np.zeros_like(c) for i, c in enumerate(coefficients)]
    return pywt.waverec(chosen, WAVELET, mode="symmetric", axis=0)[:len(x)]


def _process_segment(t, x, grid, segment_id):
    detail = {
        "segment_id": segment_id,
        "start_s": float(t[0]),
        "end_s": float(t[-1]),
        "packets": len(t),
        "grid_samples": len(grid),
        "valid_samples": 0,
    }
    if len(grid) < MIN_SAMPLES or len(t) < 2:
        detail["reason"] = "too_short_for_db4_level_7"
        return detail, None

    complete = np.flatnonzero(np.isfinite(x).all(axis=0))
    slots = complete[np.ptp(x[:, complete], axis=0) > 0]
    detail["active_input_columns"] = slots.tolist()
    if len(slots) < 2:
        detail["reason"] = "fewer_than_two_complete_varying_channels"
        return detail, None

    amplitude = _resample_amplitudes(t, x[:, slots], grid)
    # Removing the constant offset up front avoids round-off in DWT of DC.
    filtered = _details_only(amplitude - amplitude.mean(axis=0))
    centered = filtered - filtered.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    tolerance = np.finfo(float).eps * max(centered.shape) * singular[0]
    if singular[1] <= tolerance:
        detail["reason"] = "rank_below_two"
        return detail, None

    loading = axes[1].copy()
    if loading[np.argmax(np.abs(loading))] < 0:
        loading *= -1
    score = centered @ loading
    edge = EDGE_S * FS
    score[:edge] = np.nan
    score[-edge:] = np.nan
    detail.update(
        valid_samples=int(np.isfinite(score).sum()),
        pc2_variance_fraction=float(singular[1]**2 / np.sum(singular**2)),
        pc2_loadings=loading.tolist(),
    )
    return detail, score


def _write_outputs(path, grid, values, segments, report):
    with (path / "waveform.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "pc2", "valid", "segment_id"))
        for t, y, sid in zip(grid, values, segments):
            valid = np.isfinite(y)
            writer.writerow((f"{t:.3f}", f"{y:.12g}" if valid else "", int(valid), int(sid)))

    if report["status"] == "ok":
        from matplotlib.figure import Figure

        waveform_fig = Figure(figsize=(9, 3.4), layout="constrained")
        ax = waveform_fig.subplots()
        ax.plot(grid, values, linewidth=0.85)
        ax.set(xlabel="Recording time (s)", ylabel="PC2 (relative CSI amplitude)",
               title="Respiratory-band CSI candidate")
        waveform_fig.savefig(path / "waveform.png", dpi=140)

        psd_fig = Figure(figsize=(7, 3.4), layout="constrained")
        spectrum_ax = psd_fig.subplots()
        with (path / "psd.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("segment_id", "frequency_hz", "pc2_power_per_hz"))
            for detail in report["segments"]:
                if not detail["valid_samples"]:
                    continue
                sid = detail["segment_id"]
                y = values[(segments == sid) & np.isfinite(values)]
                freq, power = periodogram(y, fs=FS, window="hann", scaling="density")
                peak = int(np.argmax(power[1:])) + 1
                detail["psd_peak_hz"] = float(freq[peak])
                detail["psd_peak_bpm"] = float(60 * freq[peak])
                writer.writerows((sid, hz, p) for hz, p in zip(freq, power))
                spectrum_ax.plot(freq, power, linewidth=1, label=f"Segment {sid}")
        spectrum_ax.set(xlim=(0, 1.25), xlabel="Frequency (Hz)", ylabel="PC2 power / Hz",
                        title="Spectrum of the valid PC2 samples")
        if len(report["segments"]) <= 8:
            spectrum_ax.legend()
        psd_fig.savefig(path / "psd.png", dpi=140)

    (path / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                   encoding="utf-8")


def run_p0(times, amplitudes, output_dir, *, recording_id):
    """Produce an offline waveform and diagnostics on a recording-relative 40 Hz grid.

    Each uninterrupted span fits its own centered PCA. Complete, varying
    amplitude columns only participate. Timestamps must already be in seconds.
    """
    t = np.asarray(times, dtype=float)
    if np.iscomplexobj(amplitudes):
        raise ValueError("P0 expects real CSI amplitudes; decode complex CSI first")
    x = np.asarray(amplitudes, dtype=float)
    if t.ndim != 1 or x.ndim != 2 or len(x) != len(t):
        raise ValueError("Expected seconds [packets] and amplitudes [packets, channels]")
    if not np.isfinite(t).all() or np.any(t < 0) or np.any(np.diff(t) <= 0):
        raise ValueError("Recording-relative seconds must be finite, nonnegative and increasing")
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise ValueError("A recording_id is required")
    if (x[np.isfinite(x)] < 0).any():
        raise ValueError("CSI amplitude must be nonnegative")

    # Use the recording clock directly: a gap never shifts later human labels.
    if len(t):
        first = int(np.ceil(t[0] * FS - 1e-8))
        last = int(np.floor(t[-1] * FS + 1e-8))
        grid = np.arange(first, last + 1, dtype=np.float64) / FS
    else:
        grid = np.empty(0)
    values = np.full(len(grid), np.nan)
    segments = np.full(len(grid), -1, dtype=int)
    finite_rows = np.flatnonzero(np.isfinite(x).any(axis=1))
    observed = t[finite_rows]
    breaks = np.flatnonzero(np.diff(observed) > GAP_S + 1e-10) + 1
    details = []
    for sid, rows in enumerate(np.split(finite_rows, breaks)):
        if not len(rows):
            continue
        lo, hi = t[rows[0]], t[rows[-1]]
        window = slice(grid.searchsorted(lo - 1e-9), grid.searchsorted(hi + 1e-9, side="right"))
        segments[window] = sid
        detail, score = _process_segment(t[rows], x[rows], grid[window], sid)
        details.append(detail)
        if score is not None:
            values[window] = score

    valid_count = int(np.isfinite(values).sum())
    report = {
        "recording_id": recording_id,
        "status": "ok" if valid_count else "insufficient_data",
        "reason": None if valid_count else "no_supported_pc2_segment",
        "method": "Alzaabi et al. (2024), offline CSI amplitude to second principal component",
        "source_doi": "10.1109/ACCESS.2024.3349700",
        "sampling_hz": FS,
        "wavelet": WAVELET.name,
        "levels": LEVEL,
        "retained_detail_levels": list(DETAILS),
        "nominal_band_hz": list(BAND_HZ),
        "gap_limit_s": GAP_S,
        "edge_trim_s": EDGE_S,
        "min_wavelet_samples": MIN_SAMPLES,
        "resampling": "timestamp interpolation followed by even-reflected Fourier resampling",
        "pc2_sign": "largest absolute loading positive per segment; arbitrary respiratory polarity",
        "packets": len(t),
        "input_channels": x.shape[1],
        "valid_samples": valid_count,
        "grid_samples": len(grid),
        "segments": details,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".p0-running"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        if any((output / name).exists() for name in ARTIFACTS):
            raise FileExistsError(f"P0 results already exist; choose a new output directory: {output}")
        with TemporaryDirectory(dir=output) as tmp:
            staged = Path(tmp)
            _write_outputs(staged, grid, values, segments, report)
            for name in ARTIFACTS:  # run.json marks completion and is published last.
                if (staged / name).exists():
                    (staged / name).rename(output / name)
    finally:
        lock.unlink()
    return report
