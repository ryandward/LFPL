# Experimental: an IPv4 forwarder on AF_XDP

None of this is part of main's compiler or its test suite. It uses the compiler as it is and adds a runtime
beside it, which is why it has its own build step, `experimental/build.py`, in place of a `--xdp` option on the
driver. `python3 -m unittest discover -s tests -t .` does not run its tests, and nothing in `src/` imports it.

A loopback socket is a way to test a runtime, and nothing more. `experimental/ipv4.lfpl` does what a router
does to a packet, as `experimental/workload.md` fixes it: it checks the Ethernet and IPv4 headers, refuses a TTL
of 1 or less, lowers the TTL, patches the header checksum by RFC 1624, chooses a next hop, rewrites the
MAC addresses and sends the frame. It has no routing table yet, so it chooses by the top bits of the
destination. A frame comes from the network card through AF_XDP, which lets the program and the kernel
share a region of memory and pass frames around it on four rings, so the runtime makes no system call for
a frame while the card keeps up. That region is part of the image, locked and touched at startup like
everything else.

A frame never moves, and a program never sees one. Admission copies the header fields into cells, and the
frame itself goes into the item as a value of an abstract type, which is what `Bytes` was in the first
handler:

```
data Frame;
record Tail { dst_lo: Int16, length: Int16, frame: Frame };
record Ip2 { sum: Int16, src_hi: Int16, src_lo: Int16, dst_hi: Int16, rest: Tail };
record Ip1 { total: Int16, id: Int16, frag: Int16, ttl: Int8, proto: Int8, rest: Ip2 };
record Eth2 { smac_lo: Int32, ethertype: Int16, version: Int8, tos: Int8, rest: Ip1 };
record Eth1 { dmac_hi: Int32, dmac_lo: Int16, smac_hi: Int16, rest: Eth2 };
```

Those are the frame's first 34 bytes, read in wire order. A cell holds 8 bytes of scalars beside its link,
so the header takes five cells, each owning the next. A program can do nothing with a `Frame` but hand it
on, and because it is linear it has to: it cannot read one, forge one or lose one, and the output type is
checked to hold exactly one in each constructor. The word that stands for a frame holds its address and
its length together, so there is nothing a program could change about what gets sent except the fields
it writes back, and those go, by name, to the place in the frame the item's field of that name came from:

```
record Tx2 { sum: Int16, frame: Frame };
record Tx1 { dmac_lo: Int16, smac_hi: Int16, smac_lo: Int32, rest: Tx2 };
data Out = Malformed(frame: Frame)
         | Expired(frame: Frame)
         | Send(port: Int8, ttl: Int8, dmac_hi: Int32, rest: Tx1);
```

A constructor with a `port` is sent, and the other two give their frame back to the ring that feeds
ingress, as any refusal at admission does. This all goes through the same compiler, with the language
unchanged, and it derives what it always does:

```
python3 experimental/xdp_load.py attach eth1 --pin /sys/fs/bpf/lfplo          # as root
python3 -m experimental.build experimental/ipv4.lfpl -o ipv4 --xdp eth1 --map /sys/fs/bpf/lfplo/xsks --ctl-port 9001

handler ipv4: R(n) = 0 cells, W(n) = 17 n steps, stack 80 bytes, 5 cells, image 8392784 bytes
```

An item here is always 5 cells, so that bound is 85 steps, and the dearest frame in the test corpus takes
81. `experimental/xdp_load.py` is a separate step because loading a program into the kernel has no place in a
serving loop. It writes five BPF instructions as bytes, so no C compiler is involved there either, and it
pins the map the runtime registers with. If that map is not there, the runtime refuses to start, and each
step of startup that can fail exits with its own number.

Frames and cells are counted apart, because they run out for different reasons. Cells go back to the pool
as soon as egress has written them to the frame, so one transaction's worth is all there are: 5, and with
a correct program Short cannot happen. A frame is held for longer, so a full transmit ring is a real
refusal, Full, which gives the frame back and takes no cell. Run out of frames altogether and the
kernel has nowhere to put what arrives. That is the one refusal the program does not get to make, and the
socket's own count of it comes back as `kernel_dropped`.

`experimental/ipv4_forward.c` does the same workload in plain C on the same rings, with a real DIR-24-8 table,
and it has a `--match` switch that swaps the table for what `experimental/ipv4.lfpl` does. It is two things:
the oracle for what the forwarder sends, and the ceiling for how fast it could go. `experimental/test_ipv4.py`
puts a virtual Ethernet pair in a network namespace and sends the same frames through both, on valid
frames, every malformed case, TTLs 1 and 2, and header checksums near 0xFFFF, and holds both to each other
and to a third statement of the workload in `experimental/ipv4_model.py`, which shares nothing with either. Its
tests need root, and skip without it:

```
sudo env PATH="$PATH" python3 -m unittest experimental.test_ipv4
```

Speed is another matter, because a virtual pair's rate says how fast the kernel copies memory. So
`experimental/speed.py`, which runs pktgen on one port of a card and the forwarder on the other, cabled together,
refuses to give a rate on a virtual pair, and until a card is here there is no such number to report.
What it can show here is that it goes through every step, puts everything back, and calls a run that went
wrong a failure: uncabled ports, a port with no carrier, or being told to stop.

## What is not here

The IPv4 forwarder has no table, because that has not been written, and not because the language lacks
it: read-only arrays are enough. It has only ever run on a virtual pair, where the kernel copies every
frame. Zero-copy and every rate are unmeasured, and so is what `experimental/speed.py` does on a real card, since
only its plumbing has been run. Its `port` is checked against the one port there is, and any other is
counted and refused. A frame's fate after it reaches a ring is the kernel's and the card's, so the
guarantee stops there. On a card with zero-copy, running out of frames shows in the card's own counters
and not in `kernel_dropped`.
