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
when demand exceeds what it can guarantee. The first program of that kind now runs here, a
packet forwarder on real sockets, and it comes later in this page.

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

You need Python 3, `nasm` and `ld`, and no C compiler is involved at any point.

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
handler forward: R(n) = 0 cells, W(n) = 11 steps (23 instructions), stack 8 bytes, 14256 cells, image 233216 bytes
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
refusal count and nothing else. `tests/test_forward.py` checks each of these against a
running binary, from outside it, including pausing egress so that a queue really does fill.

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
in it the moment it was tried on anything but the handler. For a handler, the checks are not
something you run afterwards. `src/lfplo.py` runs the type check on every line, the symbolic check
on the result and an independent recomputation of the stack on the assembly, and any failure
removes the binary, including one left by an earlier build. The test suite damages the compiled
handler one instruction at a time, in every way that still parses, and requires the check to
catch all of it, apart from a short list of changes that do not alter what it means, each with
the reason written beside it.

```
python3 -m unittest discover -s tests -t .
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

The forwarding handler is the first LFPL-O program, and it asks for less than the specification
offers. Its bounds are constants, measured from the code: a handler is refused if recursion can
be reached from it, because the resource analysis that would give a recursive body a bound
`c·n + d` does not exist. So a handler cannot yet walk its own body, and a checksum is out of
reach. Arrays hold scalars, have a power-of-two length until division exists, and can be read
but not written, because `swap` and state slots are not built. There is one handler and one
ingress, one datagram per turn, bound to the loopback address. Admission checks size, pool and
queue, but not deadlines or a program's own predicate, since neither can be declared yet. The
control port and pausing egress are not in the specification: without a way to look inside and
a way to hold egress still, on loopback where sends never block, none of the rest could be shown.

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
