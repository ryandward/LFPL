"""Which declared limits the generated programs and handlers stop at, counted over a range of seeds.

    python3 tools/limits_hit.py [--programs N] [--handlers N] [--seed S] [--jobs J]

tests/test_generated.py lets a generated program or handler stop at a limit the language declares, and does not say
which. This builds each of them as that test does, without running anything, and counts the refusals by the limit
they name: the text up to its first colon, or the declared prefix the test accepts. Anything else is printed in full,
since that is a bug and not a limit.
"""
from __future__ import annotations

import argparse
import collections
import multiprocessing
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.lfplo import Refused, build, compile_program  # noqa: E402
from src.ring import Reject  # noqa: E402
from src.rtlower import Limit  # noqa: E402
from src.runtime import BindingError, Options  # noqa: E402
from tests.reference import gen, hgen  # noqa: E402

DECLARED = ("nesting too deep", "section 8.2", "name longer than")


def reason(text: str) -> str:
    """The limit a refusal names, without what varies from one program to the next."""
    text = re.sub(r"^.*?(near line \d+: )", "", text.strip().splitlines()[0])
    text = re.sub(r"^.*?: refused: ", "", text)
    text = re.sub(r"near line \d+: ", "", text)
    for declared in DECLARED[::2]:
        if text.startswith(declared):
            return declared
    if text.startswith("section 8.2: "):
        words = re.sub(r"\b(fn|handler) \w+", r"\1 _", text[len("section 8.2: "):])
        return "section 8.2: " + re.sub(r"\d+", "N", words)[:100]
    return "NOT DECLARED: " + text[:200]


def program(seed: int) -> str:
    try:
        source, _expected = gen.program(seed)
    except gen.Skip:
        return "skipped by the generator"
    try:
        compile_program(source)
    except (Reject, Limit) as failure:
        return reason(str(failure))
    return "ran"


def handler(seed: int) -> str:
    try:
        made = hgen.H(seed)
        made.build()
    except hgen.Skip:
        return "skipped by the generator"
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "program.lfpl")
        with open(path, "w") as handle:
            handle.write(made.text())
        initial = {}
        for name, data in made.state_bytes().items():
            initial[name] = os.path.join(workdir, f"{name}.bin")
            with open(initial[name], "wb") as handle:
                handle.write(data)
        try:
            build(path, os.path.join(workdir, "forward"), Options(40000, 40100, 40200), initial)
        except (Refused, Reject, BindingError) as failure:
            return reason(str(failure))
    return "ran"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--programs", type=int, default=300)
    parser.add_argument("--handlers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="as LFPL_FUZZ_SEED: seeds start at 1000 times this")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    base = 1000 * args.seed
    with multiprocessing.Pool(args.jobs) as workers:
        for name, work, count in (("programs", program, args.programs), ("handlers", handler, args.handlers)):
            results = collections.Counter(workers.map(work, range(base, base + count)))
            print(f"{name}: {count} generated, seeds {base} to {base + count - 1}")
            for what, times in sorted(results.items(), key=lambda item: (-item[1], item[0])):
                print(f"  {times:5d}  {what}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
