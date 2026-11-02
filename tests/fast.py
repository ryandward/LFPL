"""The fast tier: what can be checked in seconds, with no live sockets and no simulation.

    python3 -m unittest tests.fast

It is what to run while working, and the full suite is what to run before pushing. `unittest discover`
does not find this file, so nothing runs twice. While it runs, opening a socket or starting a simulator
is an error, so a test that needs either cannot join this list by accident.
"""
import socket
import subprocess
import unittest
from unittest import mock

TIER = ("tests.test_simplex",
        "tests.test_rtcheck",
        "tests.test_rtlower.ValidPrograms",
        "tests.test_rtlower.TheDecreasingSet",
        "tests.test_rtlower.RefusedWithAReason",
        "tests.test_rtlower.TheRightToLeftPass",
        "tests.test_spans",
        "tests.test_surface",
        "tests.test_machine.TheBound",
        "tests.test_machine.ShapesItDoesNotServe",
        "tests.test_ipv4_model",
        "tests.test_rtverilog.TargetIndependence",
        "tests.test_rtverilog.TheEmittersOwnArithmetic",
        "tests.test_walks.ReportedBounds",
        "tests.test_walks.TheModels")
SIMULATORS = frozenset(("iverilog", "vvp", "yosys", "verilator"))


def _no_socket(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("the fast tier opens no sockets: this test belongs in the full suite")


class _Guarded(subprocess.Popen):
    def __init__(self, args: object, *rest: object, **options: object) -> None:
        name = args[0] if isinstance(args, (list, tuple)) else str(args).split()[0]
        if str(name).rsplit("/", 1)[-1] in SIMULATORS:
            raise AssertionError(f"the fast tier starts no {name}: this test belongs in the full suite")
        super().__init__(args, *rest, **options)


class NoLiveSuite(unittest.TestSuite):
    def run(self, result: unittest.TestResult, debug: bool = False) -> unittest.TestResult:
        with mock.patch.object(socket, "socket", _no_socket), mock.patch.object(subprocess, "Popen", _Guarded):
            return super().run(result, debug)


def load_tests(loader: unittest.TestLoader, _standard: unittest.TestSuite, _pattern: object) -> unittest.TestSuite:
    return NoLiveSuite(loader.loadTestsFromNames(TIER))
