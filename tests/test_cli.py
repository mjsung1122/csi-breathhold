import json
from pathlib import Path

import pytest

from csi_breathhold.cli import PROFILE, main, new_result


def test_malformed_capture_writes_diagnostic_without_curve(tmp_path):
    raw = tmp_path / "capture.csv"
    raw.write_text("unknown\n0\n", encoding="utf-8")
    out = tmp_path / "result"
    args = ["p0", str(raw), "--out", str(out), "--profile", PROFILE, "--recording-id", "session"]
    assert main(args) == 1
    report = (out / "run.json").read_bytes()
    assert json.loads(report)["status"] == "input_error"
    assert not (out / "waveform.csv").exists()
    assert main(args) == 1
    assert (out / "run.json").read_bytes() == report


def test_failed_run_can_be_retried_without_partial_artifacts(tmp_path):
    out = tmp_path / "nested" / "result"
    with pytest.raises(ValueError):
        with new_result(out) as stage:
            (stage / "partial").write_text("partial", encoding="utf-8")
            raise ValueError("interrupted")
    assert not out.exists()
    with new_result(out) as stage:
        (stage / "complete").write_text("ok", encoding="utf-8")
    assert sorted(path.name for path in Path(out).iterdir()) == ["complete"]
