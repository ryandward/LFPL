# LFPL on bare silicon

LFPL is a small functional language with no way to ask for memory. A program
starts with a fixed set of 16-byte cells, and every list node or record it builds
has to name the cell it is reusing. This project compiles that language straight
to x86-64 Linux machine code, with no allocator, no garbage collector and no C
runtime underneath, and it holds its own compiler to the same discipline.

The language comes from Martin Hofmann's work on bounded space and in-place update.
His insight was that if every cell is a linear value, one that must be used exactly
once, then a functional program can overwrite its data and still be purely
functional: nobody else can be looking at a cell you own, so reusing it is
indistinguishable from building a new one. The memory a program can ever use is then
fixed before it starts, and running out stops being a crash at some unlucky
allocation. It becomes an arm of a `match` that the compiler makes you write.

What this repository adds is following that idea all the way down. Registers, stack
slots and the CPU's flags are treated the way cells are: generated code may use one
only while it holds it, and must hand it back. The rules are written out in
[the formal model](Pure%20LFPL%20on%20Bare%20Silicon%20A%20Formal%20Execution%20Model.md),
and this code is its executable reference. It is written in Python on purpose. This
first phase exists to make the rules runnable, so they can be checked and argued with
before anything is built on them; Python is not meant to be the end state.

A second specification, LFPL-O, asks what those rules buy a long-running service: a program
that, once started, never crashes, hangs or leaks, and that refuses work at its boundary
when demand exceeds what it can guarantee. Two programs of that kind now run here on real
sockets, a packet forwarder and a checksum whose work depends on the packet, and they come later
in this page.

## What a program looks like

Here is a function that adds one to every element of a list, in place:

```
data List = Nil | Cons(h: Int64, t: List);

fn inc(l: List) : List @l =
  match l with
  | Cons(d, h, t) => Cons@d(+(h, 1), inc(t))
  | Nil => Nil
  end;
```

Matching on a `Cons` takes the cell apart and gives you what was in it, `h` and `t`,
along with `d`, the emptied cell itself. `Cons@d(...)` is the only way to build a
`Cons`, and it has to spend a cell like `d` to do it, so a cell can never be conjured
out of nothing. The `@l` in the signature promises that the result is `l`'s own cell,
and the compiler checks that promise at every exit from the function. Operators are
written prefix, as in `+(h, 1)`, for a reason that will come up shortly.

That source compiles to this:

```
fn_inc_body:
    mov rax, rdi
L1:
    test rdi, rdi
    je L2
L3:
    add qword [rdi], 1
    mov rdi, qword [rdi+8]
    jmp L1
L2:
    ret
```

The recursion has become a loop, and the increment has become a single
read-modify-write on the cell. What matters most is what is missing. The source says
to rebuild each cell with a new head and a new tail, yet nothing here stores a tail,
because the tail pointer never changed. No optimization pass tidied that up afterwards.
The compiler tracks what each location last held, and it has no rule that would emit
an instruction for writing a value back to the place it came from.

That same bookkeeping turns mistakes into compile errors. A function that forgets its
list is rejected with `linear variable l is never used`, because a cell that could
quietly vanish is exactly what the language exists to rule out:

```
fn forget(l: List) : Int64 = 1;
```

## Running it

You need Python 3, `nasm` and `ld`, and no C compiler is involved in anything the compiler builds. The
hardware target also uses `iverilog`, `yosys` and `verilator`, and its tests skip themselves without them.
The one piece of C in the repository is the baseline the IPv4 forwarder is measured against.

```
python3 -m src.lower examples/inc.lfpl      # print the assembly
python3 -m src.lower examples/sum.lfpl -r   # assemble, link and run it: prints 9
```

The compiler streams its output as it goes, so a program it rejects may already have
printed some assembly. The exit status is what tells you whether it succeeded.

`examples/sum.lfpl` is a complete program and a good one to read next. A program that
needs cells declares `fn main(free: Free)`, and `_start` hands it every cell in the
binary as one free list. Because nothing can be dropped, summing a list means giving
each cell back as you read it, and `main` returns the leftovers along with its answer.
That feels strict at first, but it is what makes the guarantee worth having: when the
program finishes, every cell is accounted for.

## A program that serves

An LFPL-O program has no `main`. It has a handler, which the runtime applies to one item at a
time, threading a fixed-size state from each item to the next. This is the forwarding handler
from section 4.7 of that specification, compiled exactly as it is written there, from
`examples/forward.lfpl`:

```
data Bytes;
record Hdr { ttl: Int8, dst: Int32, body: Bytes };
data Verdict = Drop(body: Bytes) | Fwd(port: Int8, ttl: Int8, dst: Int32, body: Bytes);
state routes : Array[1024] Int8;

handler forward(routes: Array[1024] Int8, h: Hdr, free: Free)
    : (Array[1024] Int8, Verdict, Free) =
  match h with Hdr(d, ttl, dst, body) =>
    if <=(ttl, 1) then (routes, Drop@d(body), free)
    else let (routes2, port) = get(routes, idx(dst, 1024)) in
         (routes2, Fwd@d(port, -(ttl, 1), dst, body), free)
  end;
```

A packet's header cell becomes its verdict, in place, so the handler draws nothing from its
reserve. `idx` reduces the address into range, and its type says so, which is why the table
lookup needs no bounds check and has none. `data Bytes;` is declared and never defined: the
runtime builds that chain of cells, and all a program can do with it is pass it on.

```
head -c 1024 /dev/zero > routes.bin
python3 -m src.lfplo examples/forward.lfpl -o forward \
    --in-port 9000 --out-port 9100 --ctl-port 9001 --state routes=routes.bin
./forward
```

The compiler reports what it derived, and it refuses a handler for which it cannot derive it:

```
handler forward: R(n) = 0 cells, W(n) = 6 steps, stack 8 bytes, 14256 cells, image 233216 bytes
```

What comes out is a static binary with no libc in it. A datagram is `[ttl][dst, little endian][body]`,
and a forwarded one leaves for port `9100 + port` with its `ttl` one lower. That layout is not
parsed anywhere: `recvmsg` scatters each datagram straight into cells through an iovec naming
the header cell's fields and each body cell's payload, so the wire format follows from the
item's declared type, and `sendmsg` gathers the same way. The control port answers `S` with
its counters, counting the pool by walking the free list, and `Q` shuts it down.

The point of all this is where it can refuse, because each refusal happens before any work has
been done on what it refuses. The compiler refuses a program, and leaves no binary. The process
refuses to start, exiting with the number of the step that failed, if it cannot lock its image
into memory. And admission refuses one item, counting why, when it is too large, the pool is
short or the output queue is full; the item's cells never left ingress, so nothing else moves.
Once a handler starts it finishes. All the memory is laid out at link time and touched once at
startup, and the stack is inside that image with a guard page below it, so a flood changes the
refusal count and nothing else. A flood too fast to read is dropped by the kernel, before any of
this, and the socket says how many on the next datagram it delivers, so that `kernel_dropped`, `served` and
the refusals add up to what was sent. A build with `--measure-age` also asks when each datagram
arrived, at the cost of reading the clock for every item, and `max_age_ns` keeps the longest any
admitted item waited, which nothing refuses on yet. `tests/test_forward.py` checks each of these against a
running binary, from outside it, including pausing egress so that a queue really does fill.

## A program that walks its item

The forwarder never looks at a packet's body, so it does the same work for every packet, and its
bound is a constant. A checksum has to read every byte, so its work depends on the packet, and
now the promise is harder: the runtime has to know how much a handler will need before starting
it, for a packet it has never seen. `examples/checksum.lfpl` computes the Internet checksum of a
payload and writes it in place into the header, and this is the part that does the walking:

```
data Bytes = Last | Chunk(used: Int8, b0: Int8, b1: Int8, b2: Int8, b3: Int8, b4: Int8, b5: Int8,
                          b6: Int8, next: Bytes);
record Packet { port: Int8, sum: Int16, body: Bytes };

fn add(s: Int64, odd: Int64, l: Bytes, acc: Bytes) : (Int64, Bytes) = match l with
  | Last => (s, acc)
  | Chunk(d, used, b0, b1, b2, b3, b4, b5, b6, next) =>
      add(+(s, +(<<(+(pack<8>(b0), +(pack<8>(b2), +(pack<8>(b4), pack<8>(b6)))), *(odd, 8)),
                 <<(+(pack<8>(b1), +(pack<8>(b3), pack<8>(b5))), *(^(odd, 1), 8)))),
          ^(odd, 1), next, Chunk@d(used, b0, b1, b2, b3, b4, b5, b6, acc))
  end;

handler checksum(p: Packet, free: Free) : (Packet, Free) =
  match p with Packet(d, port, old, body) =>
    let (s, reversed) = add(0, 0, body, Last) in
    (Packet@d(port, finish(s), reverse(reversed, Last)), free)
  end;
```

Declaring `Bytes` with the layout the runtime builds, where the forwarder left it abstract, is what
lets the program read it. A linear program cannot read a chain and also keep it, so `add` takes
the chain apart as it goes, moving each cell onto an accumulator, and `reverse` moves them all back.
Nothing is copied and nothing is drawn from the reserve: only the link in each cell changes, and
the sum replaces the old one in the same header cell. A datagram is `[port][any two bytes][payload]`,
and it leaves for port `9100 + port` as `[checksum][payload]`. Now the report has the item's size in
it, where `n` is the number of cells the item takes:

```
$ python3 -m src.lfplo examples/checksum.lfpl -o checksum --in-port 9000 --out-port 9100 --ctl-port 9001
handler checksum: R(n) = 0 cells, W(n) = 28 n + 4 steps, stack 24 bytes, 14256 cells, image 232192 bytes
```

That bound comes from the potential method of Hofmann and Jost. Imagine every cell of a value carrying some
amount of credit, the same unknown amount for each cell. Opening a cell releases its credit. Each
instruction spends some, and building a value has to set aside the new cell's share out of what is
in hand. So a walk has to pay for each iteration out of the one cell that iteration opened, and
it has to pay for the credit of the cell it put on the accumulator, because the next walk will
need it. Every one of those sentences is a linear inequality over the unknowns, so one handler's
are one linear program, and solving it gives the smallest amounts that work: 28 steps per cell here.
A program with no such numbers is refused, which is what happens to a walk that also reverses its
accumulator at each step: it does quadratic work, and the inequalities have no solution.

The solver is not trusted. `src/rtcheck.py` writes the inequalities as it types the code, and
`src/simplex.py`, which shares nothing with it, proposes numbers in exact fractions. Then the
checker puts those numbers into every inequality it wrote, and believes none of them until all hold.
A wrong solver can therefore only cost a refusal, and that is tested by making the solver lie.

At run time the promise is kept by arithmetic on the datagram's length. Admission works out `n` from
that length alone, looks up the four numbers the compiler put in the image, and reserves `R(n)` cells
before the handler starts, or refuses the item if the pool does not have them. A handler that draws
from its reserve gets exactly what its bound says, and so never reaches the arm for running out.

A bound is only a claim until the code it describes has been timed against it. A `--count-steps`
build adds a counter at each step the checker charged for, and `tests/test_checksum.py` compares
that against `W(n)` on a live socket, along with the checksum itself against one written
independently in Python, for every short length and the longest item there is. `tests/test_walks.py`
then does the same to handlers nobody chose: random pipelines of reversals, copies, filters and
merges, each with a model in Python, each datagram judged on its bytes and its steps. Those tests
have to be able to fail, so `tools/mutate_bounds.py` breaks the rules that derive a bound and the
code that uses one, a line at a time, and reports any break that no test notices. Its first run found
that the image was too small for a handler that grows its item, and that nothing was timing a walk
with a join in it.

## The same stream in hardware

The layer between the compiler and x86 claims not to be about x86. The way to find out is to lower it
to something with no registers to run out of, no stack and no instructions, and see what breaks.
`src/rtverilog.py` turns the same checked stream into one synchronous state machine in Verilog. It
imports nothing from the x86 backend or runtime, nothing before it changed to allow it, and it writes
nothing until `src/rtcheck.py` has accepted every line.

```
$ python3 -m src.rtverilog examples/checksum.lfpl -o checksum.v
...
walk add: 24 cycles per iteration, 2 to leave
walk reverse: 4 cycles per iteration, 2 to leave
k = 1: the most cycles any one instruction takes in a walk
c = 1: the most states RT does not charge for, on a path outside the walks
cycles(n) <= 1*W(n) + 1 = 28 n + 5, from the edge that samples start to the edge that raises done
```

Every RT register becomes flip-flops, and every cell a word of block RAM. A state is one instruction,
and here is the end of `add`'s loop, as the compiler wrote it:

```
// take r25, r2[8:8]
add_s22: begin
    add_r25 <= a_dout[77:64];
    state <= add_s23;
end
// build r2, c6, none, =, =, =, =, =, =, =, =, r3
add_s23: state <= add_s24;
// tail f0, r23, r24, r25, r2
add_s24: begin
    add_r0 <= add_r23;
    add_r1 <= add_r24;
    add_r2 <= add_r25;
    add_r3 <= add_r2;
    state <= add_s0;
end
```

The `tail` is the part x86 finds hard. Four registers change at once, and `add_r3` gets the value `add_r2`
held before the clock edge, because that is what `<=` means, so the parallel move that takes
the x86 backend a scheduler and a scratch register does not exist. The `build` is a state with no
assignment, because its work is on the block RAM's write port, where `b_we = 16'hff00` enables the
eight bytes of the link and no others. Eight of its nine fields say `=`, and a field with no enable
cannot be written. Nothing checks for that: it is what the instruction means. And `add_s0`, the null
test, is also the read of the next cell, whose word is at the port in the next state, so a walk
spends no extra clock per cell.

So the emitter can say what the machine will take before anything runs, as the report above does.
`tests/test_rtverilog.py` then holds it to that, on the datagrams of the live UDP run, 240 random ones
and those that take each path all the way along, with each output compared with the x86 binary's
on the same bytes. These are the five the UDP run showed, with the steps counted by an x86 build:

| Payload bytes | n | W(n) | x86 steps | Cycles | Cycles / steps |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 32 | 32 | 33 | 1.0312 |
| 7 | 2 | 60 | 60 | 61 | 1.0167 |
| 8 | 3 | 88 | 88 | 89 | 1.0114 |
| 1000 | 144 | 4036 | 4036 | 4037 | 1.0002 |
| 1505 | 216 | 6052 | 6052 | 6053 | 1.0002 |

The one extra cycle is the read of the header cell, the first time a field of it is named. x86 reads a
field by naming it, so RT's `open` charges nothing, and a block RAM has to be asked first. That is
tested parity on those inputs, and not a proof. The random pipelines of `tests/test_walks.py`, the
forwarding handler and the generated handlers run on both targets as well, which is what covers
joins, tagged types, arrays and tail recursion modulo cons. `yosys` maps the cell store to block RAM,
`verilator -Wall` has nothing to say, and `tools/place_and_route.py` routes the checksum on an iCE40 HX8K,
with an image of 432 cells, at no worse than 81.82 MHz over five seeds, by `nextpnr`'s timing analysis.
So the largest item has a bound in time: 6053 cycles, or 73.98 microseconds. There are no caches or
interrupts in the machine for that to depend on, which is what x86 cannot say.

What is not built on this target is everything around the handler. Ingress, egress and admission are
test code driving the block RAM between items, so a datagram does not reach this machine by itself yet.

## A forwarder on a network card

A loopback socket is a way to test a runtime, and nothing more. `examples/ipv4.lfpl` does what a router
does to a packet, as `docs/workload.md` fixes it: it checks the Ethernet and IPv4 headers, refuses a TTL
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
python3 tools/xdp_load.py attach eth1 --pin /sys/fs/bpf/lfplo          # as root
python3 -m src.lfplo examples/ipv4.lfpl -o ipv4 --xdp eth1 --map /sys/fs/bpf/lfplo/xsks --ctl-port 9001

handler ipv4: R(n) = 0 cells, W(n) = 18 n steps, stack 80 bytes, 5 cells, image 8392784 bytes
```

An item here is always 5 cells, so that bound is 90 steps, and the dearest frame in the test corpus takes
86. `tools/xdp_load.py` is a separate step because loading a program into the kernel has no place in a
serving loop. It writes six BPF instructions as bytes, so no C compiler is involved there either, and it
pins the map the runtime registers with. If that map is not there, the runtime refuses to start, and each
step of startup that can fail exits with its own number.

Frames and cells are counted apart, because they run out for different reasons. Cells go back to the pool
as soon as egress has written them to the frame, so one transaction's worth is all there are: 5, and with
a correct program Short cannot happen. A frame is held for longer, so a full transmit ring is a real
refusal, Full, which gives the frame back and takes no cell. Run out of frames altogether and the
kernel has nowhere to put what arrives. That is the one refusal the program does not get to make, and the
socket's own count of it comes back as `kernel_dropped`.

`baseline/ipv4_forward.c` does the same workload in plain C on the same rings, with a real DIR-24-8 table,
and it has a `--match` switch that swaps the table for what `examples/ipv4.lfpl` does. It is two things:
the oracle for what the forwarder sends, and the ceiling for how fast it could go. `tests/test_ipv4.py`
puts a virtual Ethernet pair in a network namespace and sends the same frames through both, on valid
frames, every malformed case, TTLs 1 and 2, and header checksums near 0xFFFF, and holds both to each other
and to a third statement of the workload in `tests/ipv4_model.py`, which shares nothing with either. Its
tests need root, and skip without it:

```
sudo env PATH="$PATH" python3 -m unittest tests.test_ipv4
```

Speed is another matter, because a virtual pair's rate says how fast the kernel copies memory. So
`tools/speed.py`, which runs pktgen on one port of a card and the forwarder on the other, cabled together,
refuses to give a rate on a virtual pair, and until a card is here there is no such number to report.

## How the compiler holds itself to the same rules

Most compilers read a whole file into a syntax tree, which costs memory in proportion
to the file. This one never builds that tree. Its stages sit in a closed ring and pass
16-byte cells around, and every cell a stage finishes with goes back to the start:

```
    +------------------- empty cells ------------------+
    v                                                  |
  Ingest --> Lex --> Parse --> Check --> Lower --> Egress
   bytes    tokens   events    types     RT        written
                                         lines     out
```

In place of a tree, `src/dyck.py` emits a stream of bracketed events: open a construct,
then its children, then close it, like balanced parentheses. This is the reason for the
prefix operators. The later stages need to know what a construct is before they see its
children, and an infix `a + b` would have to announce the `+` before reading an `a` of
any size. Each stage that reads the stream keeps one small frame per open construct, so
its memory follows how deeply the source nests rather than how long it is. Only the
table of global names grows with the program, and nesting past a declared limit is a
syntax error, which is what lets the ring be sized in advance.

`src/check.py` does the linear typing as a fold over those events, and Lower turns them into
instructions. Lower is built on one idea: every expression is compiled knowing where its value
is wanted, whether that is a particular register, a field of a cell, or a pair of jump targets.
Values therefore appear where they will be used, and much of what other compilers need an
optimizer for never happens at all. A comparison under an `if` becomes a compare and a branch
with no boolean ever materialized, and a constructor taken apart immediately by a `match`
produces no code. `src/ring.py` is the machine that moves cells between stages.

There are two Lowers at the moment, and the difference between them is the most important
thing that has happened to this code. The first, `src/lower.py`, wrote x86 directly, so a single
pass over the program had to decide ownership, register allocation and instruction selection
all at once, and it kept miscompiling in ways that were hard to see. The second,
`src/rtlower.py`, knows nothing about x86. It writes [a small typed language of register
transfers](docs/rt-layer.md), with as many registers as it likes and moves that consume their
source, and you can read what it wrote:

```
$ python3 -m src.lfplo examples/forward.lfpl --rt
...
handler f0 forward(r0 : t32768, r1 : t21, r2 : t16) : t57344
    open r1, c6
    br be, r1[0:1], #1, L2, L3
L2:
    build r1, c11, write, =
    ret r0, r1, r2
L3:
    idx r4, 1024, r1[4:4]
    aget r5, r0, r4, 1
    drop r4
    op r6, sub/8, r1[0:1], #1
    build r1, c13, write, r5, r6, =, =
    drop r6
    ret r0, r1, r2
end
```

`build` is one atomic instruction, and each `=` is a field that stays where it is: `dst` and
`body` here are not moved, and that is a fact the instruction states, not an optimization.
`src/rtcheck.py` types every line of this before anything else happens, sharing no code with
the writer, using token rules like the ones for cells above. Saying a field stays when it was
overwritten is a type error, and so is storing a value back where it came from, reading a cell
after it has been rebuilt, or losing a pointer the new constructor does not name. Most of the bugs
that hand-written tests and fuzzing found in the first Lower are local type errors at this
level. Then `src/x86.py`, which is the only code that knows about fifteen registers, spills
and frames, turns the stream into assembly.

Python allocates, of course, so what this reference enforces is the model. Tokens,
events and frames live in a real array of 16-byte cells, and a cell refuses a 16th byte
of payload, so a stage cannot quietly keep more state than the model allows. A few short
Python lists name those cells. So that nothing can hide in ordinary objects instead, a
test also watches the Python heap and confirms that it barely moves between a short
source and a far longer one of the same depth.

## Why you might believe it

Nothing here is proved. These are tests, and they run on real hardware: programs are
assembled, linked and executed. `tests/test_invariants.py` checks the model's named
claims directly, such as cell conservation at every step of the pipeline and the absence
of any copy of an unchanged field. A test that cannot fail proves nothing, so these come
with controls that must trip the same detector on code that really is wrong. One of them
deliberately breaks the lowerer so that it writes the forbidden copies, and the test has
to notice.

Running a program only tells you about the answer it printed, and a bug can damage
something the answer never looks at. So `src/symcheck.py` compares meaning without running
anything. It has its own parser for the source and its own reading of x86, and a test
confirms that it imports nothing from the compiler. It works out from the source what each
function should produce, executes the emitted assembly symbolically, and requires the two
to agree wherever a value leaves the function: at every `ret`, call and tail jump, each
field of every constructor that was built must hold what the source says it holds. Where
paths join, a location that differs between them holds "this if that condition, otherwise
that", kept in a canonical form, so a source argument like `if c then 7 else z` compares
directly with a register that two branches filled. To show the check can fail, the test
suite breaks the lowerer one line at a time, mostly in ways it really has been broken, and
the check has to find each break without running the result. It has also found a bug
nothing else could see: a tag byte written over a free-list pointer that had not yet been
moved out of the cell, in programs whose answers came out right anyway.

`tests/test_fuzz.py` attacks from the other side. It generates random programs, works out
each expected answer in Python from what the language means, runs the binary and compares,
and puts every program through the symbolic check as well. It found miscompiles that every
hand-written test had missed. In one, a value joined after an `if` kept the provenance of
its last arm alone, so a store the other arm needed was silently dropped. In another, a
nested constructor overwrote a location an outer constructor was still waiting to read. A
review of the code then found two more that the generators had never reached, which is why
they now pass a cell's capability to callees and build tagged constructors in both arms of
an `if`. Each of these bugs has a named regression test, and a campaign fails if a program
is rejected for any reason other than a limit the language declares.

`src/validate.py` asks a narrower question. It reads the assembly as text and rejects the
redundant moves the model says cannot occur. That is a check for waste rather than for
correctness, since a lost store leaves nothing redundant behind and passes it.

Every generator runs through both Lowers, and running them through the new one found six bugs
in it the moment it was tried on anything but the handler. Then an independent review, told only
to report failures it could reproduce, found much worse, because it did not share the author's idea
of what a program looks like. A single datagram could crash a running binary through a program's
own definition of `Free`. An `Int8` literal of 300 gave a wrong answer past every check, because it
was 44 to `+` and not 44 to `==`. The type checker for the RT layer accepted streams that lost
pointers, forged array indices or skipped stores. And the symbolic check refused most correct
programs of certain shapes. All of that is fixed and tested. The generators it wrote are now in
`tests/reference`, each with an interpreter that shares nothing with the compiler, and
`tests/test_generated.py` runs them on every test run, handlers included, on live sockets.

For a handler, the checks are not something you run afterwards. `src/lfplo.py` runs the type check
on every line, verifies the bound's certificate, runs the symbolic check on the result and an
independent recomputation of the stack on the assembly, and any failure
removes the binary, including one left by an earlier build. The test suite damages the compiled
handler one instruction at a time, in every way that still parses, and requires the check to
catch all of it, apart from a short list of changes that do not alter what it means, each with
the reason written beside it.

```
python3 -m unittest tests.fast                                        # seconds: no sockets, no simulation
python3 -m unittest discover -s tests -t .                            # all of it
LFPL_FUZZ=300 LFPL_FUZZ_SEED=12 python3 -m unittest tests.test_fuzz   # a longer run
```

## Where the code parts ways with the model

Turning the model into code showed that parts of it cannot be followed literally. The
most common reason is the one behind prefix operators: the compiler only reads forward,
and several rules assume it can see what comes later.

Section 4.1 wants the operand that needs the most registers evaluated first, but that
requires knowing the second operand's needs before writing the first one's code, so
operands run in source order. It also wants to spill the value used furthest in the
future, which nothing here can know, so the oldest value spills instead. Scalars raise
the same problem from another side. A read cannot tell whether it was the last, so
writing to a location erases the scalar living there, and using it afterwards is a
compile error, unless an enclosing construct is already waiting on it, in which case it
moves out first. Handing a cell to a callee counts as writing to it, because the callee now
owns it, so a field you still want has to be copied first, with `+(x, 0)`. Section 5.6 needs to know that a recursive call returns its argument's
own cell, which depends on exits the compiler has not reached yet, and that is why the
`@l` annotation exists: the signature declares it, and each exit is checked against it.

Two rules from the later LFPL-O specification are already in force, because a running program
must not hang or depend on hardware accidents. Every function has to terminate: for some fixed
set of linear arguments, not counting `Free`, the total number of cells must strictly fall at
every recursive call. So `fn spin(x: Int64) : Int64 = spin(x);` is refused, while a walk that
shrinks one argument on some calls and another on the rest is fine as long as their sum falls.
And shifts are total: a shift by the width or more gives 0, whatever x86 would do with the count.

A few other places simply did not survive contact. Section 5.7 evacuates live registers
before moving arguments into place, but taken literally that spills a value and reloads
it while its register still holds it, the very round trip Theorem 7.5 says cannot
happen. So both now happen as one parallel move, and a register freed by a spill
remembers what it held. Section 2.5 budgets one cell per frame for Lower, which join
points overflow, so the budget is a larger constant, computed as `S_LOWER`. Section 4.2's
`sub rsp` needs a frame size that is not known until the function ends, so a function
that spills is entered through a small thunk, costing it one extra call's worth of stack.
The model's trees need a value and two children, which do not fit in 16 bytes, so a tree
takes two cell types that refer to each other, and `data Pair;` declares one ahead of its
definition.

When a program runs into one of these limits, the compiler does not work around it. It
stops and reports the rule along with the exact state of the registers, the homes of
live values and the destination it was trying to fill. A deeply right-nested sum that
pins all fifteen registers fails this way, and naming its intermediate values with `let`
is usually all it takes to fix it. The other limits work the same way: a partial result
that would have to survive a call clobbering its register, a home that would change after
a jump under a condition has already been written, and a value moved out of a cell in one
arm of a conditional while it is still needed after the join.

## What is not here yet

The two handlers ask for less than the specification offers. Bounds are derived and certified,
but only `R(n)` refuses anything: deadlines cannot be declared yet, so admission works out `W(n)` and
only adds it to a counter. Nothing relates a step to a number of machine instructions either, so
`W(n)` is a count of the checker's steps and not a time. Arrays hold scalars, have a power-of-two
length until division exists, and can be read but not written, because `swap` and state slots are
not built. Tuples are flat, and at most six wide, because they are only ever registers. There is one
handler and one ingress: one datagram per turn on the loopback address, or one queue of one card. Admission
checks size, pool and queue, but not a program's own predicate, which cannot be declared yet. The control
port and pausing egress are not in the specification: without a way to look inside and a way to hold egress
still, on loopback where sends never block, none of the rest could be shown.

The IPv4 forwarder has no table, because that has not been written, and not because the language lacks
it: read-only arrays are enough. It has only ever run on a virtual pair, where the kernel copies every
frame. Zero-copy and every rate are unmeasured, and so is what `tools/speed.py` does on a real card, since
only its plumbing has been run. Its `port` is checked against the one port there is, and any other is
counted and refused. A frame's fate after it reaches a ring is the kernel's and the card's, so the
guarantee stops there. On a card with zero-copy, running out of frames shows in the card's own counters
and not in `kernel_dropped`.

On the hardware side, `docs/rt-layer.md` lists what in the RT layer turned out to have come from x86:
a pointer's 8 bytes, which is why a chain cell holds 7 bytes of payload, six arguments, and a cost model
in which reading a cell is free until a field is named.

Two limits come from the bound's own arithmetic. A handler's linear program stops at 1,024
inequalities, and because each call gets its own copy of what it calls, those multiply with the depth
of the call graph, so a deep program can be refused for that even though a bound exists. And a
program cannot keep a free list of its own: rule R3 does not count a `Free`, so nothing could ever
get rid of a second one, which is why the arm for running out cannot be reached.

Two parts deserve less trust than the rest. The runtime in `src/runtime.py` is assembly that no
check reads, which section 11.3 of the specification accepts on the grounds that it is small and
fixed, and the tests judge it only by what it does. And the symbolic check builds whole syntax
trees: it runs after the compiler has finished, so the memory bound is the compiler's and not
its own.

`src/lower.py` is still there because `tests/test_invariants.py` was written against its exact
output and has not moved, and `python3 -m src.lower` still runs the first examples. The new path
passes every generator, but it does not yet match the old one's output everywhere: a constructor
bound by `let` and then matched is built before it is taken apart, a `match` on a known null still
tests it, `pack` does not fold constants, and every function adjusts its frame even when the frame
is empty, because the frame's size is not known until the right-to-left pass exists.

A scalar read after its cell was rebuilt or handed over is refused, with the rule and a way around
it, because reading forward nothing says at that point that it is wanted later. A good share of
generated programs stop there, so it is not a corner case: the right-to-left pass in the
specification is what removes it. A diamond that passes through a join counts as handed over.

The largest gap in the language itself is non-tail recursion over a structure, the model's DESCEND
and ASCEND, which is rejected for now. A tree therefore cannot be walked by recursing into both
children, although the reader in `tests/test_fuzz.py` shows that a tail-recursive walk
with a worklist built from the tree's own cells does the same job in constant stack.
Tail recursion modulo cons exists only in the `@` form, where the result is the
argument's own cell.

The language is also narrower than the model in smaller ways. Functions take at most six
arguments and must be defined before they are called. On the verification
side, the model's term-level rules are realized by the emitted code rather than run by
an interpreter, the stages themselves are Python rather than LFPL, and the stack bound of
Theorem 7.3 is computed but has not been measured.

The checks have edges of their own too. The symbolic check trusts the linear type checker,
because it treats distinct pointers as distinct cells, and it takes each callee to mean
what that callee's own check established, so it says nothing about termination. It also
cannot see an instruction encoding, which is why NASM warnings are fatal. The redundancy
validator matches locations by their text within straight-line blocks, so it cannot see
one location reached through a copied pointer, and a clean result from it is evidence
rather than proof.
