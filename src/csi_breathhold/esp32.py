"""Read the team's classic ESP32 HT20 dual-LTF logger, with one LTF per output.

The team's receiver enables both training fields without merge. Its CSV guide
specifies signed int8 [imaginary, real] and FFT order 0..31, -32..-1 for each
64-slot field. These assumptions apply only to the named capture profile.
"""

from array import array
from collections import Counter
import csv
from hashlib import file_digest
from pathlib import Path
import re

import numpy as np


PROFILE = "team_esp32_ht20_v1"
RECEIVER_SOURCE = (
    "https://github.com/clapppp/csi-sensing/blob/"
    "66716ed03dc7147e9c175805bbc01d5178ca5cc0/receiver/main/main.c"
)
# Offsets in the original 128-complex-slot packet, DC and guard removed.
_SLOTS = {
    "lltf": np.r_[1:27, 38:64],
    "htltf": np.r_[65:93, 100:128],
}
_UNEXPECTED_BYTE_CHAR = re.compile(r"[^0-9+\-\s]")


def _amplitudes(cell, slots):
    """Decode exactly 256 signed bytes, including the unselected field."""
    if not cell or _UNEXPECTED_BYTE_CHAR.search(cell):
        raise ValueError("malformed_csi")
    tokens = cell.split()
    if len(tokens) != 256:
        raise ValueError("csi_token_count")
    try:
        values = np.fromiter(map(int, tokens), np.int32, count=256)
    except (ValueError, OverflowError):
        raise ValueError("malformed_csi") from None
    if values.min() < -128 or values.max() > 127:
        raise ValueError("csi_outside_int8")
    pairs = values.reshape(128, 2)[slots].astype(np.float32)
    imaginary, real = pairs.T
    return np.hypot(real, imaginary)


def read_capture(csv_path, *, profile=PROFILE, view="lltf"):
    """Return relative seconds, packet × channel amplitudes and an audit dict.

    Keep the original packet clock, including gaps from rejected packets.
    Wrong layouts and nonmonotonic/reset clocks require a separate recording.
    Corrupt CSI payloads are skipped with bounded per-line diagnostics.
    """
    if profile != PROFILE:
        raise ValueError(f"Unsupported profile {profile!r}. Expected {PROFILE!r}.")
    if view not in _SLOTS:
        raise ValueError("view must be 'lltf' or 'htltf'")
    path = Path(csv_path)
    slots = _SLOTS[view]
    times_us = array("Q")
    amplitudes = array("f")
    rejected = Counter()
    examples = []
    flags = Counter()
    first_stamp = previous_stamp = None
    packet_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        headers = reader.fieldnames or []
        if len(set(headers)) != len(headers):
            raise ValueError("Duplicate CSV header. Provide one named column per field.")
        missing = {"ts_us", "len", "csi"}.difference(headers)
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")
        try:
            for row in reader:
                packet_rows += 1
                line = reader.line_num
                stamp_text = row["ts_us"] or ""
                if not stamp_text.isascii() or not stamp_text.isdecimal():
                    raise ValueError(f"Line {line}: ts_us must be a uint32 device timestamp")
                stamp = int(stamp_text)
                if stamp >= 2**32:
                    raise ValueError(f"Line {line}: ts_us exceeds the uint32 device clock")
                if previous_stamp is not None and stamp <= previous_stamp:
                    raise ValueError(f"Line {line}: ts_us repeated or moved backwards. Split this recording.")
                if first_stamp is None:
                    first_stamp = stamp
                previous_stamp = stamp
                if row["len"] != "256":
                    raise ValueError(f"Line {line}: this profile requires len=256. Check the CSI layout.")
                flag = row.get("first_word_invalid")
                try:
                    if None in row:
                        raise ValueError("unexpected_csv_field")
                    if flag not in (None, "", "0", "1"):
                        raise ValueError("invalid_first_word_flag")
                    amplitude = _amplitudes(row["csi"], slots)
                except ValueError as exc:
                    reason = str(exc)
                    rejected[reason] += 1
                    if len(examples) < 20:
                        examples.append({"line": line, "reason": reason})
                    continue
                flags[flag or "unknown"] += 1
                times_us.append(stamp)
                amplitudes.frombytes(amplitude.tobytes())
        except csv.Error as exc:
            raise ValueError(f"Line {reader.line_num}: malformed CSV: {exc}") from exc
    if not times_us:
        raise ValueError(f"No valid CSI packets. Rejections: {dict(rejected)}")
    t = (np.asarray(times_us, dtype=np.float64) - first_stamp) / 1_000_000
    matrix = np.asarray(amplitudes).reshape(len(t), len(slots))
    # PCA needs a fixed basis. Preserve every packet's usable common channels
    # without fabricating the one L-LTF amplitude lost when slot 1 is invalid.
    drop_slot_one = view == "lltf" and (flags["1"] > 0 or flags["unknown"] > 0)
    if drop_slot_one:
        matrix = matrix[:, 1:]
        slots = slots[1:]
    with path.open("rb") as raw:
        digest = file_digest(raw, "sha256").hexdigest()
    report = {
        "profile": profile,
        "view": view,
        "source_name": path.name,
        "source_sha256": digest,
        "profile_evidence": RECEIVER_SOURCE,
        "payload_order": "signed_int8_imag_real",
        "timestamp_unit": "microseconds",
        "time_origin_device_us": first_stamp,
        "packet_rows": packet_rows,
        "accepted_packets": len(t),
        "rejected_packets": sum(rejected.values()),
        "rejection_counts": dict(rejected),
        "rejection_examples_first_20": examples,
        "first_word_flags_accepted": dict(flags),
        "lltf_slot_1_dropped_for_common_basis": drop_slot_one,
        "raw_complex_slots": slots.tolist(),
        "channels": len(slots),
        "observed_packets_per_s": (len(t) - 1) / float(t[-1] - t[0]) if len(t) > 1 else None,
        "max_packet_gap_s": float(np.diff(t).max()) if len(t) > 1 else None,
    }
    return t, matrix, report
