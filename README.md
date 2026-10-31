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

## How the compiler holds itself to the same rules

Most compilers read a whole file into a syntax tree, which costs memory in proportion
to the file. This one never builds that tree. Its stages sit in a closed ring and pass
16-byte cells around, and every cell a stage finishes with goes back to the start:

```
    +------------------- empty cells ------------------+
    v                                                  |
  Ingest --> Lex --> Parse --> Check --> Lower --> Egress
   bytes    tokens   events    types     machine   written
                                         code      out
```

In place of a tree, `src/dyck.py` emits a stream of bracketed events: open a construct,
then its children, then close it, like balanced parentheses. This is the reason for the
prefix operators. The later stages need to know what a construct is before they see its
children, and an infix `a + b` would have to announce the `+` before reading an `a` of
any size. Each stage that reads the stream keeps one small frame per open construct, so
its memory follows how deeply the source nests rather than how long it is. Only the
table of global names grows with the program, and nesting past a declared limit is a
syntax error, which is what lets the ring be sized in advance.

`src/check.py` does the linear typing as a fold over those events, and `src/lower.py`
turns them into instructions. Lower is built on one idea: every expression is compiled
knowing where its value is wanted, whether that is a particular register, a field of a
cell, or a pair of jump targets. Values therefore appear where they will be used, and
much of what other compilers need an optimizer for never happens at all. A comparison
under an `if` becomes a compare and a branch with no boolean ever materialized, and a
constructor taken apart immediately by a `match` produces no code. The registers, slots,
flags and cell capabilities it trades in are modeled in `src/tokens.py`, and
`src/ring.py` is the machine that moves cells between stages.

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

`src/validate.py` is a second opinion that shares no code with the lowerer. It reads the
emitted assembly as text and rejects the redundant moves the model says cannot occur.

`tests/test_fuzz.py` has done the most to earn trust, by repeatedly proving that trust
premature. It generates random programs, works out each expected answer in Python from
what the language means, then runs the binary and compares. It found miscompiles that
every hand-written test had missed. In one, a value joined after an `if` kept the
provenance of its last arm alone, so a store the other arm needed was silently dropped.
In another, a nested constructor overwrote a location an outer constructor was still
waiting to read. Each one now has a named regression test.

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
moves out first. Section 5.6 needs to know that a recursive call returns its argument's
own cell, which depends on exits the compiler has not reached yet, and that is why the
`@l` annotation exists: the signature declares it, and each exit is checked against it.

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
is usually all it takes to fix it.

## What is not here yet

The largest gap is non-tail recursion over a structure, the model's DESCEND and ASCEND,
which is rejected for now. A tree therefore cannot be walked by recursing into both
children, although the reader in `tests/test_fuzz.py` shows that a tail-recursive walk
with a worklist built from the tree's own cells does the same job in constant stack.
Tail recursion modulo cons exists only in the `@` form, where the result is the
argument's own cell.

The language is also narrower than the model in smaller ways. Pairs are left out because
section 1 gives them no term form, functions take at most six arguments and must be
defined before they are called, and shift counts must be literals. On the verification
side, the model's term-level rules are realized by the emitted code rather than run by
an interpreter, the stages themselves are Python rather than LFPL, the stack bound of
Theorem 7.3 is computed but has not been measured, and the validator reasons only within
straight-line blocks, so it says nothing about a pair of moves on either side of a label.
