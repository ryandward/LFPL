"""Programs generated from a syntax tree, judged by an interpreter that shares nothing with the
compiler. See tests/reference.

test_fuzz.py works out each answer alongside the text it writes, which limits it to shapes its
author thought of. These generators came from an independent review, aimed at what those miss:
tuples through joins, nested matches, spills across calls, handlers with helpers. Every program that
compiles must pass the symbolic check and print what the interpreter says. A handler must also
answer every datagram correctly on a live socket, give every cell back, and stay inside the W(n)
it was certified for, counted by a build that adds a counter at each step the checker charges for.

A program may stop at a limit the language declares: anything else is a bug.
"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lfplo import build_and_run, compile_program  # noqa: E402
from src.ring import Reject  # noqa: E402
from src.rtlower import Limit  # noqa: E402
from src.symcheck import check_translation  # noqa: E402
from tests.reference import gen, hgen  # noqa: E402
from tests.reference.lang import without_annotations  # noqa: E402
from tests.test_forward import Forwarder  # noqa: E402

COUNT = int(os.environ.get("LFPL_FUZZ", "60"))
SEED = int(os.environ.get("LFPL_FUZZ_SEED", "0"))
DATAGRAMS = 30
DECLARED = ("nesting too deep", "section 8.2", "name longer than")


class GeneratedPrograms(unittest.TestCase):
    def test_every_program_prints_what_the_interpreter_says(self):
        ran = 0
        with tempfile.TemporaryDirectory() as workdir:
            for seed in range(1000 * SEED, 1000 * SEED + COUNT):
                try:
                    source, expected = gen.program(seed)
                except gen.Skip:
                    continue
                try:
                    asm = compile_program(source)
                except (Reject, Limit) as failure:
                    self.assertTrue(any(reason in str(failure) for reason in DECLARED),
                                    f"seed {seed}: {failure}\n{source.decode()}")
                    continue
                self.assertEqual([str(p) for p in check_translation(source, asm)], [],
                                 f"seed {seed}\n{source.decode()}")
                self.assertEqual(build_and_run(asm, workdir), expected,
                                 f"seed {seed}\n{source.decode()}")
                # The right-to-left pass has to find every `@` that was written.
                self.assertEqual(compile_program(without_annotations(source)), asm,
                                 f"seed {seed}\n{source.decode()}")
                ran += 1
        print(f"generated programs: {ran} of {COUNT} ran", file=sys.stderr)
        self.assertGreater(ran, COUNT // 3)


class GeneratedHandlers(unittest.TestCase):
    def _serve(self, handler: "hgen.H", workdir: str, seed: int) -> bool:
        """Build and drive one handler. False if it stopped at a limit the language declares."""
        path = os.path.join(workdir, "program.lfpl")
        with open(path, "w") as handle:
            handle.write(handler.text())
        options = []
        for name, data in handler.state_bytes().items():
            table = os.path.join(workdir, f"{name}.bin")
            with open(table, "wb") as handle:
                handle.write(data)
            options += ["--state", f"{name}={table}"]
        try:
            router = Forwarder(bytes(1024), workdir, build_options=("--count-steps", *options),
                               source=path)
        except subprocess.CalledProcessError as failure:
            self.assertTrue(any(reason in failure.stderr for reason in DECLARED),
                            f"seed {seed}: {failure.stderr}\n{handler.text()}")
            # These do not recurse, so a bound always exists, whatever section 8.2 allows elsewhere.
            self.assertNotIn("no linear bound", failure.stderr, f"seed {seed}\n{handler.text()}")
            return False
        try:
            interpreter = handler.interp()
            before = router.stats()
            for number in range(1, DATAGRAMS + 1):
                datagram, values, body = handler.datagram()
                port, want = handler.expect(interpreter, values, body)
                router.raw(datagram)
                if port is not None and port < len(router.outputs):
                    self.assertEqual(router.outputs[port].recv(4096), want,
                                     f"seed {seed}, datagram {datagram!r}\n{handler.text()}")
                now = router.settle(number)
                self.assertLessEqual(now["steps"] - before["steps"],
                                     now["budget_steps"] - before["budget_steps"],
                                     f"seed {seed}, datagram {datagram!r}\n{handler.text()}")
                before = now
            self.assertEqual(now["pool"], now["capacity"], f"seed {seed}\n{handler.text()}")
            self.assertIsNone(router.process.poll())
        finally:
            router.close()
        return True

    def test_every_handler_answers_what_the_interpreter_says(self):
        ran = 0
        for seed in range(1000 * SEED, 1000 * SEED + COUNT // 3):
            try:
                handler = hgen.H(seed)
                handler.build()
            except hgen.Skip:
                continue
            with tempfile.TemporaryDirectory() as workdir:
                ran += self._serve(handler, workdir, seed)
        print(f"generated handlers: {ran} of {COUNT // 3} ran", file=sys.stderr)
        self.assertGreater(ran, COUNT // 9)


if __name__ == "__main__":
    unittest.main()
