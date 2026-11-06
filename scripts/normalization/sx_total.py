"""Run the normalization steps on one dataset, in order.

    .venv/bin/python scripts/normalization/sx_total.py data/train/polish/YODAS2-sidon --steps 1,5,6,7,8 [--yes] [--dry]

Steps (scripts/normalization/sN_*.py); which ones fit depends on the dataset - pick them with --steps:
    1 characters    every dataset
    2 leftovers     only datasets verbalized upstream (Emilia): it drops every transcript with a digit
    3 ctc_rescoring only English Emilia (de/fr Emilia-YODAS were filtered with the current bounds)
    4 readings      only Emilia (en, de, fr)
    5 verbalize     only datasets that keep digits (YODAS2-sidon)
    6 edges         every dataset
    7 loudness      every dataset
    8 codecs        every dataset, on a CUDA box
Each step runs as its own script, in place, so the next one sees its changes. It shows its report and asks Y/N
before writing, as when run alone; an N stops the chain. --yes answers Y everywhere. --dry writes nothing: steps
6-8 get --dry, steps 1-5 an automatic N - so each step reports on the data as it is, not as the earlier steps would
have left it. --limit, --lang, --device and --vad-only are passed to the steps that take them.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.__style__ import Colors, print_test_title  # noqa: E402

HERE = Path(__file__).resolve().parent
# The options each step takes (besides the dataset root).
OPTIONS = {1: {"in_place"}, 2: {"in_place"}, 3: {"in_place", "limit", "device"}, 4: {"in_place", "limit", "device"},
           5: {"in_place", "limit", "device", "lang"}, 6: {"dry", "limit", "device", "lang", "vad_only"},
           7: {"dry", "limit"}, 8: {"dry", "limit", "device"}}
DECLINED = "Nothing written."


def command(step: int, args: argparse.Namespace) -> list[str]:
    script = next(HERE.glob(f"s{step}_*.py"))
    cmd = [sys.executable, str(script), str(args.root)]
    takes = OPTIONS[step]
    if "in_place" in takes and not args.dry:
        cmd.append("--in-place")
    if "dry" in takes and args.dry:
        cmd.append("--dry")
    for name in ("limit", "device", "lang"):
        if name in takes and getattr(args, name):
            cmd += [f"--{name}", str(getattr(args, name))]
    if "vad_only" in takes and args.vad_only:
        cmd.append("--vad-only")
    return cmd


def run(cmd: list[str], answer: str | None) -> tuple[int, str]:
    """Run a step, relaying its output as it comes; `answer` (Y or N) is typed in for its prompt, otherwise the user
    answers. Returns the exit code and the output."""
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE if answer else None, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if answer:
        process.stdin.write(f"{answer}\n".encode())
        process.stdin.close()
    output = []
    while chunk := os.read(process.stdout.fileno(), 4096):
        sys.stdout.write(chunk.decode(errors="replace"))
        sys.stdout.flush()
        output.append(chunk.decode(errors="replace"))
    return process.wait(), "".join(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dataset root, e.g. data/train/polish/YODAS2-sidon")
    parser.add_argument("--steps", required=True, help="comma-separated step numbers, e.g. 1,5,6,7,8")
    parser.add_argument("--yes", action="store_true", help="accept every step's changes without asking")
    parser.add_argument("--dry", action="store_true", help="only report what each step would do; write nothing")
    parser.add_argument("--limit", type=int, help="passed to steps 3-8: process a random sample of this many clips")
    parser.add_argument("--lang", help="passed to steps 5 and 6")
    parser.add_argument("--device", help="passed to steps 3, 4, 5, 6 and 8")
    parser.add_argument("--vad-only", action="store_true", help="passed to step 6")
    args = parser.parse_args()
    steps = [int(s) for s in args.steps.split(",")]
    if unknown := [s for s in steps if s not in OPTIONS]:
        sys.exit(f"unknown steps {unknown}; expected some of {sorted(OPTIONS)}")
    if steps != sorted(steps):
        sys.exit(f"steps must be in order, got {args.steps}")

    print_test_title(f"Normalization of {args.root}: steps {', '.join(map(str, steps))}" + (" (dry run)" if args.dry else ""))
    for step in steps:
        cmd = command(step, args)
        print(f"\n{Colors.HEADER}{Colors.BOLD}>>> step {step}: {' '.join(Path(c).name if c.endswith('.py') else c for c in cmd[1:])}{Colors.ENDC}")
        answer = "N" if args.dry and "dry" not in OPTIONS[step] else "Y" if args.yes and not args.dry else None
        code, output = run(cmd, answer)
        if code != 0:
            sys.exit(f"{Colors.FAIL}step {step} failed (exit code {code}); stopping{Colors.ENDC}")
        if not args.dry and DECLINED in output:
            print(f"{Colors.WARNING}step {step} declined; stopping before step "
                  f"{steps[steps.index(step) + 1] if step != steps[-1] else '-'}{Colors.ENDC}")
            return
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}done: steps {', '.join(map(str, steps))}{' (dry run)' if args.dry else ''}{Colors.ENDC}")


if __name__ == "__main__":
    main()
