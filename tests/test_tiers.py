"""The fast tier keeps its promise: what opens a socket or starts a simulator fails there."""
import socket
import subprocess
import unittest

from tests.fast import TIER, NoLiveSuite


class Probe(unittest.TestCase):
    """Not collected by name: each case is run under the guard by the tests below."""

    def opens_a_socket(self) -> None:
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).close()

    def starts_a_simulator(self) -> None:
        subprocess.run(["vvp", "-V"], capture_output=True)

    def starts_something_else(self) -> None:
        subprocess.run(["true"], check=True)


def guarded(name: str) -> unittest.TestResult:
    result = unittest.TestResult()
    NoLiveSuite([Probe(name)]).run(result)
    return result


class TheFastTier(unittest.TestCase):
    def test_a_socket_is_an_error_there(self):
        result = guarded("opens_a_socket")
        self.assertEqual(len(result.failures) + len(result.errors), 1)

    def test_a_simulator_is_an_error_there(self):
        result = guarded("starts_a_simulator")
        self.assertEqual(len(result.failures) + len(result.errors), 1)

    def test_other_programs_are_not(self):
        self.assertTrue(guarded("starts_something_else").wasSuccessful())

    def test_the_guard_goes_away_afterwards(self):
        guarded("starts_something_else")
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).close()

    def test_nothing_that_serves_a_socket_is_on_the_list(self):
        for name in TIER:
            for live in ("test_forward", "test_checksum", "test_generated", "TheChecksum", "RandomWalks",
                         "OtherHandlers", "Synthesis"):
                self.assertNotIn(live, name)
