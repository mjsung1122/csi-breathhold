"""Command-line entry points for the offline two-stage pipeline."""

import argparse
import csv
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from .esp32 import PROFILE, read_capture
from .p0 import run_p0


@contextmanager
def new_result(destination):
    """Publish a complete run together and preserve prior results on retries."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Output already exists. Choose a new directory: {target}")
    with TemporaryDirectory(dir=target.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        yield stage
        stage.rename(target)


def convert(source, out, recording_id, profile, view):
    with new_result(out) as stage:
        try:
            time, amplitudes, capture = read_capture(source, profile=profile, view=view)
        except (ValueError, csv.Error) as exc:
            report = dict(status="input_error", recording_id=recording_id, reason=str(exc),
                          profile=profile, view=view, valid_samples=0)
            (stage / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        else:
            report = run_p0(time, amplitudes, stage, recording_id=recording_id)
            (stage / "capture.json").write_text(json.dumps(capture, indent=2), encoding="utf-8")
    return report


def _parser():
    parser = argparse.ArgumentParser(prog="csi-breathhold", description=__doc__)
    cmds = parser.add_subparsers(dest="command", required=True)
    p0 = cmds.add_parser("p0", help="Extract an offline wavelet/PC2 candidate from one ESP32 CSV")
    p0.add_argument("capture", type=Path)
    p0.add_argument("--profile", required=True, choices=[PROFILE])
    p0.add_argument("--view", choices=["lltf", "htltf"], default="lltf")
    p0.add_argument("--recording-id", required=True)
    demo = cmds.add_parser("demo", help="Create six synthetic CSI sessions, P0 outputs and interval labels")
    train = cmds.add_parser("train", help="Train a voluntary breath-hold model")
    train.add_argument("manifest", type=Path)
    train.add_argument("--epochs", type=int, default=30)
    evaluate = cmds.add_parser("evaluate", help="Evaluate only held-out annotated test intervals")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("manifest", type=Path)
    predict = cmds.add_parser("predict", help="Score an offline P0 waveform without labels")
    predict.add_argument("checkpoint", type=Path)
    predict.add_argument("waveform", type=Path)
    for cmd in [p0, demo, train, evaluate, predict]:
        cmd.add_argument("--out", required=True, type=Path, help="New result directory")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "p0":
            report = convert(args.capture, args.out, args.recording_id, args.profile, args.view)
            if report["status"] == "input_error":
                print(f"Input error: {report['reason']} ({args.out / 'run.json'})", file=sys.stderr)
                return 1
            print(f"{report['status']}: {report['valid_samples']} valid samples; {args.out}")
            return 0 if report["status"] == "ok" else 2
        with new_result(args.out) as stage:
            if args.command == "demo":
                from .demo import make_demo
                make_demo(stage)
            else:
                from . import p1
                if args.command == "train":
                    p1.train_model(args.manifest, stage, epochs=args.epochs)
                elif args.command == "evaluate":
                    p1.evaluate_model(args.checkpoint, args.manifest, stage)
                else:
                    p1.predict_file(args.checkpoint, args.waveform, stage)
        print(f"{args.command}: {args.out}")
        return 0
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print('P1 requires the ai extra. Install with: python -m pip install ".[ai]"', file=sys.stderr)
        return 1
    except (ValueError, OSError, csv.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
