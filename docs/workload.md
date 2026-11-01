# The workload

This is the fixed workload of every measurement from now on. It is what a router does to a packet, cut
down to where nothing in it is a matter of taste, so that a number measured today can be compared with
a number measured a year from now, and one forwarder's with another's.

`baseline/ipv4_forward.c` and `examples/ipv4.lfpl` both do this, and `tests/ipv4_model.py` is a third
statement of it, in Python, that shares nothing with either. `tests/test_ipv4.py` holds all three to each
other, frame by frame.

## What happens to a frame

A frame is Ethernet and IPv4, with a header of 5 words and no VLAN tag. Anything else is refused as
malformed. The checks come in this order, and the order can be seen from outside, because a frame that
fails two of them is counted for the first:

1. It has at least the 34 bytes of those two headers.
2. Its ethertype is 0x0800.
3. Its first IP byte is 0x45: version 4, and 5 words of header.
4. Its total length is at least 20, and fits in the frame. A frame can be longer than its total length,
   since Ethernet pads, and whatever follows is carried along unchanged.
5. Its header checksum is right: the one's complement sum of the header's 10 words is 0xFFFF.
6. Its TTL is above 1. A frame that fails this is refused as expired, and not as malformed.

Then the frame is forwarded, without moving it:

7. The TTL goes down by 1.
8. The header checksum is patched by RFC 1624, equation 3: HC' = ~(~HC + ~m + m'), where m is the 16-bit
   word holding the TTL. RFC 1141's older update is wrong in one case, which is the reason for that RFC,
   and the corpus has it: a checksum near 0xFFFF.
9. The next hop comes from the destination, by the longest prefix in a DIR-24-8 table. A destination with
   no route is refused as malformed, because that is what the workload does with anything else.
10. The MAC addresses are rewritten to the next hop's: destination 02:00:00:00:00:(0x10 + hop), and source
    02:00:00:00:01:(0x10 + hop). That formula stands in for an ARP table, which this workload does not have.
11. The frame is transmitted.

## The table, and matching in its place

The table is what the C baseline does with `--routes`. `examples/ipv4.lfpl` does not have one yet, and
chooses by the top two bits of the destination's first octet: below 64 is hop 0, below 128 hop 1, below
192 hop 2, and the rest hop 3. The baseline's `--match` swaps its table for exactly that, which is the
only way the two can be compared on the same frames. Read-only arrays with `get` already work in LFPL-O,
so a real table does not have to wait for the language. Until it exists, every comparison between
LFPL-O and C is without the table, and the baseline's rate with the table says what it will cost.

## The shape of the comparison

Both forwarders run on one core, with 4096 frames of 2048 bytes, rings of 4096 entries, a transmit ring
of 2048, batches of 64, and the same XDP program, loaded by `tools/xdp_load.py`.

Sizes are of the whole frame on the wire, with its 4-byte frame check sequence, so a 64-byte frame is the
60 bytes a driver handles and pktgen is told about.

## Speed

`tools/speed.py` is the only thing in this repository that reports a rate, and it refuses to on a virtual
pair. No such rate exists yet, because the card has not arrived. When it does, with two ports cabled to
each other and a driver with native XDP:

```
sudo python3 tools/speed.py --generator PORT_A --forwarder PORT_B
```

This moves the first port into a namespace and runs pktgen from it, and runs each forwarder on the second
port, pinned to a different physical core, with the frame going back out the port it came in on. It reports
the rate at 64 and 1500 bytes for the C baseline with `--match`, LFPL-O, and the C baseline with its table,
and how they compare where the comparison means something: if the baseline keeps up with everything it is
offered, its rate is the generator's or the link's, and it says so rather than give a ratio.

pktgen sends one destination unless told otherwise, so a table lookup hits the same entry every time and
costs almost nothing. `--random-destinations` is what shows what the table costs, and it slows the generator,
so look at whether the offered rate still saturates the forwarder.

Each run says whether the kernel gave it zero-copy, since a driver can have native XDP without it, and that
decides most of the number. Everything here is what has been written for a card without one to run on, so
expect the first run to find something.
