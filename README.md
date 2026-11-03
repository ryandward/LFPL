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
when demand exceeds what it can guarantee. Three programs of that kind now run here on real
sockets, a packet forwarder, a checksum whose work depends on the packet, and an IPv4 router whose header
is one record in network byte order, and they come later in this page.

## What a program looks like

Here is a function that adds one to every element of a list, in place:

```
data List = Nil | Cons(h: Int64, t: List);

fn inc(l: List) : List =
  match l with
  | Cons(d, h, t) => Cons@d(+(h, 1), inc(t))
  | Nil => Nil
  end;
```

Matching on a `Cons` takes the cell apart and gives you what was in it, `h` and `t`,
along with `d`, the emptied cell itself. `Cons@d(...)` is the only way to build a
`Cons`, and it has to spend a cell like `d` to do it, so a cell can never be conjured
out of nothing. The compiler works out that the result is `l`'s own cell, because every
exit from the function returns that cell, or the null that `l` was, and it holds every exit
to that. You can also say so yourself, as `: List @l`, and then it is a promise, which is
checked the same way. Operators are written prefix, as in `+(h, 1)`, for a reason that will
come up shortly.

That source compiles to this:

```
fn_inc$body:
    sub rsp, F_inc
    mov rax, rdi
X1:
    test rdi, rdi
    je L2_f0
L3_f0:
    add qword [rdi], 1
    mov rdi, qword [rdi+8]
    jmp X1
L2_f0:
    add rsp, F_inc
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

You need Python 3, `nasm` and `ld`, and no C compiler is involved at any point. The hardware target
also uses `iverilog`, `yosys` and `verilator`, and its tests skip themselves without them.

```
python3 -m src.lfplo examples/inc.lfpl --nasm  # print the assembly
python3 -m src.lfplo examples/sum.lfpl --run   # assemble, link and run it: prints 9
```

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
handler checksum: R(n) = 0 cells, W(n) = 28 n + 4 steps, 14256 cells of 128 bits, 14-bit pointers, 18 states
walk add: 2 clocks per iteration, 2 to leave
walk reverse: 2 clocks per iteration, 2 to leave
clocks(n) <= 4 n + 14, certified, from the edge that samples start to the edge that raises done
```

Every RT register becomes flip-flops, and every cell a word of block RAM. The program becomes its own circuit:
straight-line code is logic that settles in one clock. The emitter writes a state for each instruction, and then
joins each state that reads no memory and falls through with the one after it, unless something else can enter
that one, and a clock touches the block RAM once at most. So a clock ends only at a read, whose word comes back in
the next clock, at a branch, a call or a return, and before a label. Nothing is reordered. Here is `add`'s loop,
as the compiler wrote it, with the comment on the second state cut short:

```
// br.null r2, L2, L3
add_s0: state <= (add_r2 == 14'd0) ? add_s1 : add_s2;
...
// pack r5, <8>, r2[1:1]; pack r6, <8>, r2[3:1]; ... take r25, r2[8:8]; build r2, c6, none, =, =, =, =, =, =, =, =, r3; tail f0, r23, r24, r25, r2
add_s2: begin
    add_r0 <= add_w18;
    add_r1 <= add_w19;
    add_r2 <= add_w20;
    add_r3 <= add_r2;
    state <= add_s0;
end
```

The twenty-three instructions of an iteration after the null test are one clock: what each computes for the next
is a wire, `add_w0` to `add_w20`, and only what the next iteration reads is a register. The `tail` is the part x86
finds hard. Four registers change at once, and `add_r3` gets the value `add_r2` held before the clock edge, because
that is what `<=` means, so the parallel move that takes the x86 backend a scheduler and a scratch register does not
exist. The `build` is in the same clock with no assignment at all, because its work is on the block RAM's write port,
where `b_we = 16'hff00` enables the eight bytes of the link and no others. Eight of its nine fields say `=`, and a field
with no enable cannot be written. Nothing checks for that: it is what the instruction means. And `add_s0`, the null
test, is also the read of the next cell, whose word is at the port in the next clock, which is why the walk is two
clocks a cell and not one.

The emitter charges each clock to the first instruction in it, and `src/rtcheck.py` reads the stream again with those
charges in place of steps, so the bound in clocks is a certificate, as W(n) is, and the report says it before
anything runs. `tests/test_rtverilog.py` then holds the machine to it, on the datagrams of the live UDP run, 240 random
ones and those that take each path all the way along, with each output compared with the x86 binary's on the same
bytes. These are the five the UDP run showed, with the steps counted by an x86 build:

| Payload bytes | n | W(n) | x86 steps | Clocks | Clocks / steps |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 32 | 32 | 18 | 0.5625 |
| 7 | 2 | 60 | 60 | 22 | 0.3667 |
| 8 | 3 | 88 | 88 | 26 | 0.2955 |
| 1000 | 144 | 4036 | 4036 | 590 | 0.1462 |
| 1505 | 216 | 6052 | 6052 | 878 | 0.1451 |

Every run takes exactly what the certificate says, since this handler has one path for each n. That is tested parity
on those inputs, and not a proof. The random pipelines of `tests/test_walks.py`, the forwarding handler and the
generated handlers run on both targets as well, which is what covers joins, tagged types, arrays and tail recursion
modulo cons. `yosys` maps the cell store to block RAM, `verilator -Wall` has nothing to say, and
`tools/place_and_route.py` routes the checksum on an iCE40 HX8K, with an image of 432 cells, at no worse than
49.89 MHz over five seeds, by `nextpnr`'s timing analysis. One clock does more than it did when it was one instruction,
so the clock is slower, and the largest item is faster: its bound in time is 878 clocks, or 17.60 microseconds. There
are no caches or interrupts in the machine for that to depend on, which is what x86 cannot say.

Around a handler by itself, ingress, egress and admission are test code driving the block RAM between items. The
runtime's turn, below, is what puts a handler in a machine that takes frames in and sends them out.

## A program whose header is bigger than a cell

An Ethernet and IPv4 header is 34 bytes, and a cell has 16. `examples/ipv4.lfpl` declares it as one record anyway,
in the order the frame has it, and a field of type `Be16` or `Be32` lies in network byte order:

```
slot Payload;
record Frame {
  dst_hi: Be32, dst_lo: Be16, src_hi: Be32, src_lo: Be16, ethertype: Be16,
  version: Int8, tos: Int8, total: Be16, id: Be16, frag: Be16,
  ttl: Int8, proto: Int8, sum: Be16, source: Be32, destination: Be32,
  length: Int16,
  body: Payload
};
enum Verdict = Malformed | Expired | Send(port: Int8);
```

The rest of the frame, after its headers, is in a slot the frame owns, below.

The compiler lays that across five cells, each keeping 8 bytes for a link to the next unless what is left fits, and
nothing after the first stage of it knows: a match on `Frame` is five matches by the time Check sees it, and building a
`Frame` is five builds, so what stays in a cell stays there and is not written. What a pattern binds from a `Be16` is the
integer, so the code says `!=(ethertype, 0x0800)` and `+(pack<16>(total), 14)` and nobody swaps a constant or a byte. The
fields leave and arrive as they lie, so the runtime scatters and gathers them without looking, and the only swaps that
happen are the ones a read or a store of a changed value needs.

The handler decides where the frame goes and hands the frame back. `Verdict` is an `enum`, a sum whose every value fits
one register, with its tag in the low byte and its fields above it, so the decision costs no cell. `Frame@d { ... }` is the
frame rebuilt in its own five cells with the fields it names changed and every other field left where it is:

```
    if !(well_formed(ethertype, version, total, length, words)) then (Malformed, Frame@d {}, free)
    else if <=(ttl, 1) then (Expired, Frame@d {}, free)
    else
      let hop = next_hop(destination) in
      (Send(hop), Frame@d { dst_hi = 0x02000000, dst_lo = low16(+(pack<8>(hop), 0x10)),
                            src_hi = 0x02000000, src_lo = low16(+(pack<8>(hop), 0x110)),
                            ttl = -(ttl, 1), sum = patched(sum, ttl, proto) }, free)
```

A refusal is one line, and it writes nothing: `Frame@d {}` is five builds in which every field stays and every link goes
back where it was, and x86 makes no instruction of any of them. Forwarding stores the six fields it names, and the runtime
sends the frame to the port the verdict says, its record's fields and then its slot's bytes. Nothing is drawn from the
reserve, and the handler never reads its payload, so its work is the same for every frame:

```
$ python3 -m src.lfplo examples/ipv4.lfpl -o ipv4 --in-port 9000 --out-port 9100 --ctl-port 9001 --n-max 217
handler ipv4: R(n) = 0 cells, W(n) = 94 steps, stack 96 bytes, 330 cells, image 143360 bytes
```

`d`, from a pattern on `Frame`, stands for all five cells, and each still has to go somewhere. `Frame@d(...)` and
`Frame@d { ... }` build in them, `More@d(free)` gives them back, and a constructor of another type that needs one takes it
with `let (c, rest) = peel(d) in`. A record built from nothing names its cells, each of which came from a draw on the
reserve, as `Frame@(a, b, c, d, e)(...)`. `length`
is not on the wire: admission fills it in from how long the datagram was, because a program that does not walk its body
has no other way to find out. `tests/test_ipv4.py` runs every kind of frame through a binary on live sockets, against a model
that shares nothing with it, and `tests/test_rtverilog.py` holds the hardware target to the binary's bytes on a sample of them.

## The runtime, in LFPL, on the hardware target

On x86 the loop around a handler is assembly the compiler writes out: admission, handing the item over, and
taking back what the handler returns. On the hardware target that loop is LFPL. `src/turn.lfpl` is one turn of
it, and `src/turn.py` appends it to the handler's program, so the turn is typed, every cell and every slot it moves is
accounted for by linearity, and its bound is certified, as the handler's is. Admission takes the cells a frame needs out
of the pool by matching it, and when the pool runs dry it gives back what it took and refuses the frame: there is no
count of free cells to trust, because there is none at all.

```
$ python3 -m src.machine examples/ipv4.lfpl -o machine
turn serve for handler ipv4: W = 224 steps for any frame of up to 2082 bytes, with 4 places in the queue; clocks <= 96 a turn, certified; 41 cells of 128 bits and 5 slots, 10-bit pointers
```

That is said before anything runs, and it holds of every turn, whatever the frame and whatever came before. A turn
holds two things that grow with frames, the frame in ingress and the output at the head of the queue. The head is a
parameter of its own, so the frames queued behind it cost the bound nothing: a turn releases one output, and that is
what it pays for. The longest admission measured took 59 cycles, and the longest release 24, against the bound of 96.

A frame's payload is in a slot, 2048 bytes in 128 cells after the rest of the store, and the record holds the slot's
number and how many bytes it holds. Ingress holds a record and a slot. The writer puts a frame's header into the record,
which is registers, and the rest into the slot, a byte to its place, and writes the slot's length. When the verdict sends
the frame, what goes into the queue is a copy of the record, in cells from the pool, and it takes the slot with it:
ingress takes one of the spare slots in its place. When egress has sent the frame, the turn after gives its cells back
to the pool and its slot back to the spares. Nothing copies a payload, so a turn's work does not grow with the frame,
and `examples/ipv4.lfpl`'s decision, which reads and writes only the record, touches no block RAM at all: its 94 steps
are at most 21 clocks.

What runs is `machine.v`, with `serve_core.v` inside it. A frame comes in as bytes, with valid and ready, and leaves the
same way, the record's fields and then the bytes its slot says it holds. The writer never stores a link or a slot's
number, so the hardware cannot make one cell point at another or a record name another slot: those are the turn's to
make. `tests/test_machine.py` sends the whole corpus through it and through the x86 build, for `examples/ipv4.lfpl`,
`examples/firewall.lfpl` and `examples/udp_verify.lfpl`. Every frame comes out the same, to the same port, each verdict
is counted as x86 counts it, every turn stays under its frame's bound, and afterwards every cell and every slot is in
exactly one place.

A hang is stopped by the design, not caught by a test. The writer stores no byte past the slot, and egress sends no
more than a slot holds, whatever a record says. A turn that runs past its frame's bound is stopped, and a turn whose
results put an output at the head of the queue that no outcome accounts for, or more outputs in it than it has places,
is not taken: the machine counts it and writes every cell back as it was before the first turn, which is what a reset
does. With those, the specification's T10 says how soon a frame goes: once egress is ready, every frame admitted is
sent or released within five turns, and the testbench holds every frame it sends to the clocks T10 counts for it. That
second stop exists because a deliberately broken build, with every argument of the turn wired to its first, never ran a
turn past its bound: its first turn ended in 47 cycles and every one after it in 5, and egress sent the same head
forever. `tests/test_machine.py` breaks the generated Verilog that way and others, and each time the machine stops,
counts it and goes on.

### A frame's bytes, owned whole

A checksum over a whole payload needs a program that can read every byte of it, and know that doing so ends. A slot is
what makes that possible: `slot Payload;` names the type, and a record holds one like any other linear field.
`body[i]` reads byte i, and 0 for a byte at or past the length, whatever a frame before left there, and `len(body)` says
how many bytes the runtime put there. Neither uses the slot up. `match within(body, x) with | inside(i) => ... |
outside => ... end` says whether x is below the length, and gives it as a Fin, `i`, that a match counts down, `| 0 =>
... | down(j) => ...`, which is how the compiler knows the walk ends and what it costs. `examples/udp_verify.lfpl`
checks every UDP datagram's checksum that way, over the pseudo-header, the UDP header and every byte of the payload:

```
$ python3 -m src.lfplo examples/udp_verify.lfpl -o udp --in-port 9000 --out-port 9100 --ctl-port 9001
handler udp_verify: R(n) = 0 cells, W(n) = 11 n + 112 steps, stack 168 bytes, 396 cells, image 145408 bytes
$ python3 -m src.machine examples/udp_verify.lfpl -o udp_machine
turn serve for handler udp_verify: W = 11 n + 256 steps, n being a frame's 6 cells and the bytes of its slot, with 4 places in the queue; clocks <= 7 n + 91 a turn, certified; 47 cells of 128 bits and 5 slots, 10-bit pointers
```

It draws no cell, and its slot is counted apart from the cells: nothing can make or drop one, so each frame's one comes
back. Its bound grows with the frame, since a slot carries an amount for each of its bytes, and `within` hands it to the
walk: a frame's `n` is its record's cells and its payload's bytes, and admission works it out from the frame's length,
as it already does for a chain. A frame of 64 bytes is bounded at 287 clocks in the machine, and one of 1518 at 10465.

## How the compiler holds itself to the same rules## How the compiler holds itself to the same rules

Most compilers read a whole file into a syntax tree, which costs memory in proportion
to the file. This one never builds that tree. Its stages sit in a closed ring and pass
16-byte cells around, and every cell a stage finishes with goes back to the start:

```
    +------------------- empty cells ------------------+
    v                                                  |
  Ingest --> Lex --> Parse --> Right-to-left --> Check --> Lower --> Egress
   bytes    tokens   events    the future        types     RT        written
                               written in                  lines     out
```

In place of a tree, `src/dyck.py` emits a stream of bracketed events: open a construct,
then its children, then close it, like balanced parentheses. This is the reason for the
prefix operators. The later stages need to know what a construct is before they see its
children, and an infix `a + b` would have to announce the `+` before reading an `a` of
any size. Each stage that reads the stream keeps one small frame per open construct, so
its memory follows how deeply the source nests rather than how long it is. Only the
table of global names grows with the program, and nesting past a declared limit is a
syntax error, which is what lets the ring be sized in advance.

Reading only forward leaves two things that depend on what comes later: whether a scalar is wanted
after its cell is rebuilt, and which parameter's cell every exit returns. So `src/righttoleft.py` sits
between the parser and the checker, and goes through each function first, in source order. Going forward it
writes the function's events to a file, working out what each name refers to and which parameter's cell each
value is. Then it reads the file backward, holding the set of names with a use still to come, plus two sets
for each open branch, and it writes that set into the file at the start of each arm and scope. Then it sends the
events on as if the parser had just written them, with those facts among them. The file is the only thing
that grows with the length of a function, and it is on disk.

`src/check.py` does the linear typing as a fold over those events, and Lower turns them into
instructions. Lower is built on one idea: every expression is compiled knowing where its value
is wanted, whether that is a particular register, a field of a cell, or a pair of jump targets.
Values therefore appear where they will be used, and much of what other compilers need an
optimizer for never happens at all. A comparison under an `if` becomes a compare and a branch
with no boolean ever materialized, and a constructor taken apart immediately by a `match`
produces no code. `src/ring.py` is the machine that moves cells between stages.

The most important thing that has happened to this code is that Lower stopped knowing about x86. It
used to write assembly directly, so a single pass over the program had to decide ownership, register
allocation and instruction selection all at once, and it kept miscompiling in ways that were hard
to see. `src/rtlower.py` writes [a small typed language of register transfers](docs/rt-layer.md)
instead, with as many registers as it likes and moves that consume their source, and you can read
what it wrote:

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
that hand-written tests and fuzzing found while Lower wrote x86 are local type errors at this
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
writes the forbidden copy by hand, and both the RT checker and the check of the assembly
have to notice.

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

Running every generator through Lower when it was new found six bugs in it the moment it was
tried on anything but the handler. Then an independent review, told only
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
future, which nothing here can know, so the oldest value spills instead. Both of those are
for the backend to settle, and have not been. Scalars raised the same problem from another
side, because a read cannot tell whether it was the last, and so did section 5.6, which needs
to know that a recursive call returns its argument's own cell, depending on exits the compiler
has not reached yet. Those two are answered ahead of time now, by the pass described above: a scalar moves out of
its cell just before the cell is rebuilt or handed to a callee, exactly when something reads it
later.

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
`sub rsp` needs a frame size that is not known until the function ends, so it names a constant
that an `equ` defines there. The model's trees need a value and two children, which do not fit in 16 bytes, so a tree
takes two cell types that refer to each other, and `data Pair;` declares one ahead of its
definition.

When a program runs into a limit, the compiler does not work around it. It stops and says which
limit, from the list in section 8.2 of the specification. None of them is about registers any
more: a deeply right-nested sum compiles, because everything can spill, and a value moved out of
a cell in one arm of a conditional and read after the join is in one register whichever arm ran.

## What is not here yet

The handlers ask for less than the specification offers. Bounds are derived and certified,
but only `R(n)` refuses anything: deadlines cannot be declared yet, so admission works out `W(n)` and
only adds it to a counter. Nothing relates a step to a number of machine instructions either, so
`W(n)` is a count of the checker's steps and not a time. Arrays hold scalars, have a power-of-two
length until division exists, and can be read but not written, because `swap` and state slots are
not built. Tuples are flat, and at most six wide, because they are only ever registers. There is one
handler and one ingress, one datagram per turn, bound to the loopback address. Admission checks size,
pool and queue, but not a program's own predicate, which cannot be declared yet. The control port and
pausing egress are not in the specification: without a way to look inside and a way to hold egress
still, on loopback where sends never block, none of the rest could be shown.

On the hardware side, `docs/rt-layer.md` lists what in the RT layer turned out to have come from x86:
a pointer's 8 bytes, which is why a chain cell holds 7 bytes of payload, six arguments, and a cost model
in which reading a cell is free until a field is named.

Two limits come from the bound's own arithmetic. A handler's linear program stops at 1,024
inequalities, and because each call gets its own copy of what it calls, those multiply with the depth
of the call graph, so a deep program can be refused for that even though a bound exists. And a
program cannot keep a free list of its own: rule R3 does not count a `Free` anywhere a handler can
reach, so nothing could ever get rid of a second one, which is why the arm for running out cannot be
reached. Past the handler it does, because the runtime's turn gives a reserve back by walking it.

Two parts deserve less trust than the rest. The runtime in `src/runtime.py` is assembly that no
check reads, which section 11.3 of the specification accepts on the grounds that it is small and
fixed, and the tests judge it only by what it does. And the symbolic check builds whole syntax
trees: it runs after the compiler has finished, so the memory bound is the compiler's and not
its own.

The code has lost a few things to the simpler Lower, none of them about what a program means. A
constructor bound by `let` and then matched is built before it is taken apart, which section 12 of the
specification lists against Theorem 7.4, a `match` on a known
null still tests it, a value with arms joins in a register and is stored from there, where it used to be
stored in each arm, and a value computed for a `ret` is computed elsewhere and moved. Every function also
adjusts its frame even when the frame is empty, because the backend writes its prologue before it knows
the frame's size.

A diamond that passes through a join counts as handed over, so a scalar of that cell that is read later
moves out, even when all that happens to it is that it is stored back where it was.

A record's headers still get copied, by the kernel, into cells. Reading a field in place from the frame would save that, but
it changes the resource model, so it comes later. Other things show in `examples/ipv4.lfpl`: an update names its fields in
the order the record declares them, an enum cannot be stored in a cell, and a 48-bit address is two fields. The
turn on the hardware target serves only a handler shaped as this one is: no state slots, nothing drawn from its
reserve, a payload in one slot, and a verdict beside the record it was given.

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
