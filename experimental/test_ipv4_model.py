"""The model of the workload, against what the RFCs say, and against itself.

The model judges the C baseline and experimental/ipv4.lfpl in experimental/test_ipv4.py, so it has to stand on
something else. These are the numbers the RFCs print, a recomputation that uses none of the update
arithmetic, and a table lookup done the slow way against what it should choose.
"""
import random
import unittest

from experimental import ipv4_model as model


class TheChecksum(unittest.TestCase):
    def test_rfc_1071s_own_example(self):
        self.assertEqual(model.one_complement_sum(bytes.fromhex("0001f203f4f5f6f7")), 0xDDF2)

    def test_rfc_1624s_own_example(self):
        # The case RFC 1141's update got wrong: it gave 0xFFFF.
        self.assertEqual(model.incremental_checksum(0xDD2F, 0x5555, 0x3285), 0x0000)

    def test_the_update_agrees_with_working_it_out_again(self):
        rng = random.Random(1624)
        for _ in range(5000):
            frame = model.ipv4_frame(rng.randbytes(4), rng.randbytes(4), ttl=rng.randrange(2, 256),
                                     protocol=rng.randrange(256), identification=rng.randrange(1 << 16))
            verdict, sent = model.forward(frame, model.by_top_bits)
            self.assertEqual(verdict, model.FORWARD)
            header = sent[model.ETHERNET_HEADER:model.HEADER_BYTES]
            self.assertEqual(model.one_complement_sum(header), 0xFFFF, frame.hex())

    def test_every_checksum_value_can_be_had_and_stays_valid_through_an_update(self):
        for target in (*range(0xFF00, 0x10000), *range(0, 0x100), 0x8000, 0x7FFF):
            for ttl in (2, 255):
                frame = model.with_checksum(target, ttl=ttl)
                self.assertEqual(int.from_bytes(frame[model.CHECKSUM_AT:model.CHECKSUM_AT + 2], "big"), target)
                verdict, sent = model.forward(frame, model.by_top_bits)
                self.assertEqual(verdict, model.FORWARD)
                self.assertEqual(model.one_complement_sum(sent[model.ETHERNET_HEADER:model.HEADER_BYTES]), 0xFFFF)


class TheVerdicts(unittest.TestCase):
    def test_each_case_in_the_corpus_has_the_verdict_it_says(self):
        for name, verdict, frame in model.corpus():
            with self.subTest(name):
                self.assertEqual(model.forward(frame, model.by_top_bits)[0], verdict)

    def test_the_corpus_has_all_three_verdicts_in_some_number(self):
        verdicts = [verdict for _name, verdict, _frame in model.corpus()]
        for verdict in (model.FORWARD, model.MALFORMED, model.EXPIRED):
            self.assertGreater(verdicts.count(verdict), 20 if verdict != model.EXPIRED else 2)

    def test_what_is_forwarded_changes_only_what_the_workload_says(self):
        frame = model.ipv4_frame(bytes((200, 1, 1, 1)), payload=b"payload", ttl=64)
        _verdict, sent = model.forward(frame, model.by_top_bits)
        self.assertEqual(len(sent), len(frame))
        self.assertEqual(sent[:6], bytes.fromhex("020000000013"))
        self.assertEqual(sent[6:12], bytes.fromhex("020000000113"))
        self.assertEqual(sent[model.TTL_AT], 63)
        changed = {at for at, (a, b) in enumerate(zip(frame, sent)) if a != b}
        self.assertLessEqual(changed, {*range(12), model.TTL_AT, model.CHECKSUM_AT, model.CHECKSUM_AT + 1})

    def test_the_order_of_the_checks_can_be_seen_from_outside(self):
        self.assertEqual(model.forward(model.ipv4_frame(ttl=1, checksum=1), model.by_top_bits)[0], model.MALFORMED)
        self.assertEqual(model.forward(model.ipv4_frame(ttl=1), model.by_top_bits)[0], model.EXPIRED)
        self.assertEqual(model.forward(model.ipv4_frame(ttl=1), lambda _destination: None)[0], model.EXPIRED)


class TheTable(unittest.TestCase):
    ROUTES = [(bytes((0, 0, 0, 0)), 0, 1), (bytes((10, 0, 0, 0)), 8, 2), (bytes((10, 1, 0, 0)), 16, 3),
              (bytes((10, 1, 2, 0)), 24, 4), (bytes((10, 1, 2, 128)), 25, 5), (bytes((10, 1, 2, 200)), 32, 6)]

    def test_the_longest_prefix_wins(self):
        look_up = model.longest_prefix(self.ROUTES)
        for destination, hop in (((11, 0, 0, 0), 1), ((10, 0, 0, 1), 2), ((10, 1, 0, 1), 3), ((10, 1, 2, 1), 4),
                                 ((10, 1, 2, 127), 4), ((10, 1, 2, 128), 5), ((10, 1, 2, 199), 5),
                                 ((10, 1, 2, 200), 6), ((10, 1, 2, 201), 5)):
            self.assertEqual(look_up(bytes(destination)), hop, destination)

    def test_with_no_route_it_is_refused(self):
        look_up = model.longest_prefix([(bytes((10, 0, 0, 0)), 8, 2)])
        self.assertIsNone(look_up(bytes((11, 0, 0, 0))))
        self.assertEqual(model.forward(model.ipv4_frame(bytes((11, 0, 0, 0))), look_up)[0], model.MALFORMED)


if __name__ == "__main__":
    unittest.main()
