# The RT layer

Lower currently writes x86 directly. That forces every value onto 15 fixed registers
through a `mov` that copies, so the output cannot say who owns what, and ownership has to
be tracked beside the code and rebuilt afterwards from text. This layer puts one typed
language between Check and every backend. Lower writes register transfers with unbounded
registers and moves that consume their source. A small checker types each instruction with
the first specification's token rules. x86 and RTL each lower from the result.

Nothing here is built yet. This is the design to agree on first.

## What it is, and is not

- **A flat stream.** Lower's fold emits it one event at a time in O(D) state. There is no
  SSA, no CFG and no whole-function tree, so the original directive's ban still holds.
- **Unbounded registers per function.** Each is one location, fixed by the program text.
  Spilling, clobber sets, frames, `rcx` and two-operand forms all belong to the x86 backend.
- **One atomic constructor.** A constructor is a single instruction that reads all its
  sources and then writes: a parallel move, plus the tag, plus the seal. Lower cannot get
  the order of those steps wrong, because at this level they have no order.
- **Not a fix for reading only forward.** Last-use knowledge still comes from the
  right-to-left pass of LFPL-O section 8.1. Until then a scalar read after its cell is gone
  is refused here just as it is today.

## Tokens

Every rule moves tokens and creates none. A register is always exactly one of:

    free r              holds nothing
    r = w : t           holds a value of type t, whose provenance word is w

A cell is opaque, owned as a whole through the pointer that holds it, until a dispatch
opens it. An open cell `c` has a tag fact and one token per location:

    c.tag = k           the tag is known       c.tag not in K     these are ruled out
    c[o:s] = w : t      full: it holds w       c[o:s] ~ w         hole: its bits are still w

A **word** names a value, not bits in a place. `arg i` is parameter i, `c.o` is what offset
o of an unbuilt cell `c` held on entry, `null` is null, and every computed value gets a
fresh word. A move keeps the word. A built constructor's word is its cell's, whatever its
fields are. The result of a call to `f @k` has the word of argument k.

Types `t` are `Int w`, `Bool`, a named type `D`, or `Cap`, which is a cell's capability: the
source language's diamond.

## Operands

Where a rule says operand `o`, it may be a register, a literal `#n`, or a location `p[o:s]`
of an open cell. Reading a scalar operand consumes nothing. Using a linear one consumes it:
a register becomes free, and a location becomes a hole with the same word.

## Instructions and their rules

Each rule reads: tokens required `-->` tokens left. Tokens not named are untouched.

### Values

    const d, #n : t     free d                  -->  d = fresh : t
                        t is scalar, or a tagless D with n = 0, in which case the word is null
    move d, s           s = w : t, free d       -->  free s, d = w : t
    copy d, s           s = w : t, free d       -->  s = w : t, d = w : t         t scalar only
    drop s              s = w : t               -->  free s                       t scalar only
    op d, f/W, a, b     a, b : Int W, free d    -->  d = fresh : Int W
    cmp d, cc, a, b     a, b same scalar, free d -->  d = fresh : Bool
    trunc d, W, a       a : Int V, V >= W, free d -->  d = fresh : Int W
    pack d, <W...>, a...  a_i scalar, free d    -->  d = fresh : Int 64

`f` is add, sub, mul, and, or, xor, shl or shr, with the meaning LFPL-O section 4.4 gives
them at width W, so `shl` and `shr` are 0 when the count reaches W. No instruction has a
flags side effect, because there are no flags here.

### Cells

    read d, p[o:s]      p = c : Cap, c[o:s] = w : t, free d  -->  c[o:s] = w : t, d = w : t
                        t scalar only
    take d, p[o:s]      p = c : Cap, c[o:s] = w : t, free d  -->  c[o:s] ~ w, d = w : t
                        t linear only: this is EVAC

    build p, C, tag, s_1 ... s_n
        p = c : Cap, and for each field i of C at location l_i, source s_i is one of:

          =             c[l_i] = w : t                     STAY: nothing moves
          r!            r = w : t, c[l_i] ~ w              RESTORE: nothing moves, r goes free
          r             r = w : t, and l_i does not already hold w    STORE
          #n            a literal                          STORE
          [l]           c[l] = w : t, another location     STORE, read before anything is written
          f([l], b)     c[l] scalar, b an operand          op on the old value; l = l_i is in place

        tag is `keep`, which needs c.tag = C's tag, or `write`, which needs that it is not.
        A tagless constructor or a record has no tag and says neither.

        Every location of the old layout that a new field or the tag overlaps must be a hole
        or a full scalar, which it erases. A full linear location anywhere in the cell that
        no source names is a type error: it has to be taken first.

        -->  p = c : D, and every token of the open cell is gone, which closes the stale-read
        window. Reading from c afterwards has no token to use.

`r` where `r!` would apply is a type error, and so is `r!` where it would not. That is
Theorem 6.1 as a local check: writing a value back to where it came from is ill-typed, and
so is claiming a store was unnecessary when it was needed.

### Control

    label L             the merge rule below
    jump L              an edge to L carrying the current tokens
    br cc, a, b, Lt, Lf  a, b same scalar              an edge to each, tokens unchanged
    br.null p, Ln, Lc   p = w : D, D tagless
                        Ln: free p, and w is known to be null on this edge
                        Lc: p = w : Cap, with every field of the full constructor full,
                            each with word w.o
    br.tag p, k, Ly, Ln  p = w : D, D tagged
                        Ly: as Lc, for the constructor with tag k, and w.tag = k
                        Ln: p = w : D, and w.tag not in K, with k added to K
    open p, C           p = w : D, and C is the only constructor left: a record, a one-
                        constructor type, or all the others ruled out. No test is emitted.

**The merge rule.** At a label, a word or tag fact survives only if every incoming edge
agrees on it. A word that disagrees becomes a fresh one, which can never satisfy `r!`, and a
tag that disagrees becomes unknown, which forces `write`. Scalar registers free on any edge
are free at the label. Any other disagreement, such as a linear register or location that is
full on one edge and not another, is a type error. At this level no moves are needed: both
arms already deliver into the same register. A backend whose homes can differ from edge to
edge still needs them; see `label` in the x86 lowering.

### Calls and exits

    call d, f, o_1 ... o_n   free d  -->  d = w : ret(f)
        w is the word of o_k if f is declared `@k`, otherwise fresh.
        A linear operand is consumed. A Cap operand also needs every linear location of its
        cell to be a hole, and every token of that cell goes away: the callee owns it now.
        All other tokens survive, because registers here are not clobbered.

    tail f, o_1 ... o_n      the same consumption, then control leaves
    ret s                    s = w : t  -->  control leaves
    tailmod p, C, tag, s_1 ... s_n, f, o_1 ... o_m
        in this order: consume o_1 ... o_m, then build, then consume p, then jump to f

In `tailmod`, exactly one source is `*`, and the order above matters twice. The call's
operands are consumed first, because an operand may be a location of the cell being built,
such as `r0[8:8]` in `inc`, and `build` removes every token of that cell. Consuming that
operand is also what leaves the hole that `*` needs: its location must be a hole holding the
word of f's argument k, where f is declared `@k`, so RESTORE applies. After the build, `p`
holds the new value, and `tailmod` consumes it, because the link to it already exists: the
previous iteration's `*` field points at this cell, or on the first iteration the root does,
which is the word `@k` returns. Without that step `p` would still be full when control
leaves, and the exit rule below would refuse every `tailmod`. This is LFPL-O's tail recursion
modulo cons, with no store to that field.

Every exit needs all linear tokens gone. The checker therefore repeats exactly-once
linearity on its own, one level below Check. In a function declared `@k`, the value leaving
must have the word of parameter k, or be null where that word is known to be null, and the
caller receives parameter k's original word however many iterations ran.

A non-tail recursive `call` is a type error until DESCEND and ASCEND exist.

Functions are written `fn f(r0 : t0, ...) : t [@k]`, closed by `end`. Parameters start full
with word `arg i`, and every other register starts free.

## Phantom construction

There is no instruction for it. A constructor that a match consumes at once produces only
the field computations, and the pattern's names refer to those registers. So LFPL-O's
Theorem 7.4 holds because nothing exists to emit, not because anything is optimized away.

## Lowering to x86-64

The backend is a forward fold holding one table from RT registers to homes, which are
physical registers or frame slots. Consuming tokens tells it when a home is free. Two
registers stay out of allocation as scratch, which is what makes LFPL-O's T8 claim true: at
any instruction at most 5 of the 15 are pinned, so a spill victim always exists, and no
partial result is ever special. All the failures Phase 1 reports about registers go away.

    const, move, copy      mov, or xor for zero, or nothing when the homes coincide
    drop, r!, =            nothing
    op                     mov d, a; f d, b. shl and shr by a register compare with W first,
                           borrow rcx, and give 0 past the width. By a literal past W, xor.
    cmp                    cmp a, b; setcc; movzx
    trunc                  mov r32 or movzx
    pack                   Horner: shl, then or into the subregister
    read, take             a sized mov
    build                  parallel move over the fields that are not `=` or `r!`, with cycles
                           broken through the scratch registers. f([l], b) with l = l_i is one
                           read-modify-write instruction. A 64-bit literal goes through scratch.
                           `write` is mov byte [p], tag, after every source has been read.
    br                     cmp; jcc, with the fall-through chosen by the next label
    br.null                test p, p; jcc
    br.tag                 cmp byte [p], k; jne
    open                   nothing
    label                  the label, after every incoming edge's homes are made equal. The
                           label's home table is the one at the branch point the arms start
                           from, and each arm that spilled or moved a value ends with one
                           parallel move back to it, before its jump or fall-through. This is
                           Phase 1's _normalize. Without it, a value spilled in one arm would
                           be read from the wrong home after the join.
    call                   one parallel move covering the arguments and the evacuation of
                           every live value in the callee's clobber set; call; result in rax
    tail                   parallel move to the parameter registers; jmp
    tailmod                one parallel move covering the fields and the parameter registers;
                           jmp. The rule's order is about tokens, not emission: an operand in
                           the cell must be read before any store lands, and the cell's pointer
                           must last until the last store through it
    ret                    mov rax, s; ret. In an `@k` function, the entry word saved at the
                           prologue goes in rax instead.

The prologue is `sub rsp, F_f`, with F_f defined by an `equ` at `end`, so thunks go away. The
annotations of LFPL-O section 8.1 that exist only for registers, namely need(e), next-use
distance and frame size, belong to this backend. Last use and the result cell stay in the
front end, because every backend needs them.

## Lowering to RTL

Each function is one state machine. Each RT register is a flip-flop bank at its type's width,
so nothing spills. This is LFPL-O section 9.1 with its inputs named.

    const, move, copy      a latch, or a wire when it can be renamed
    drop, r!, =, open      nothing
    op, cmp, trunc         combinational. Verilog's << and >> are already 0 past the width.
    pack                   wires: {a, b, c}
    read, take             a block-RAM read state, then a latch
    build                  one block-RAM write whose byte enables cover exactly the sources that
                           are not `=` or `r!`, and the tag on `write`. RTL-3 is then a property
                           of this rule, not something to check for afterwards.
    br, br.null, br.tag    the next state, chosen by a comparison, with no boolean register
    call                   start and done on the callee's unit, inlined or shared
    tail, tailmod          latch the parameters, then return to the entry state
    ret                    done, with the result on the unit's output

Each instruction takes at most `k_rtl` states, and the schedule keeps to the block RAM's port
limit. `inc` becomes the machine in LFPL-O section 9.2 directly.

## Worked example: `inc`

    fn inc(r0 : List) : List @0
    L1: br.null r0, L2, L3
    L3: tailmod r0, Cons, add([0:8], #1), *, inc, r0[8:8]
    L2: const r1, #0 : List
        ret r1
    end

On the `L3` edge, `r0` is a capability with `[0:8] = arg0.0` and `[8:8] = arg0.8`. Passing
`r0[8:8]` to `inc` leaves a hole holding `arg0.8`, which is exactly the word `@0` says comes
back, so `*` is well typed and nothing is stored there. At `L2`, `arg0` is known to be null, so
returning null satisfies `@0`. On x86 this is the sequence in the first specification's section
5.6: `add qword [rdi], 1`, `mov rdi, qword [rdi+8]`, `jmp`.

## What this would have caught

Of the 12 wrong-code bugs in Phase 1 so far, these are local type errors at this level, found
without running anything or executing symbolically:

| Bug | Rule that refuses it |
| --- | --- |
| a join's word taken from its last arm, dropping a store | merge rule: the word is fresh, so `r!` fails |
| a literal taken for a provenance word | `const` has a fresh word, so `r!` fails |
| a stale read after a rebuild | `build` removes the cell's tokens |
| a read of a cell given to a callee | `call` removes the cell's tokens |
| a tag write trusted by the other arm | tag facts are per edge, so `keep` fails |
| a tag written over a pointer still owed | `build` refuses a full linear location no source names |
| a read deferred past a nested rebuild | the nested `build` removes the tokens that read needs |
| arithmetic width guessed from operands | `op` states W and types both operands |

The other four are x86 facts, and they move into the backend, where `symcheck.py` still has
to find them: a truncated immediate, a spilled operand, the result register lost across a
back-edge, and a masked shift count. So this layer does not replace `symcheck.py`. It takes
the ownership bugs away from it, and leaves it checking that the backend did what the RT
stream says, and that the stream means what the source says.

## Where it goes

    Lex -> Parse -> Check -> Lower -> RT check -> x86 backend -> symcheck, validate
                                              \-> RTL backend -> differential simulation

`src/rtcheck.py` shares no code with Lower, and a failure means no output. A build order: emit
RT from Lower, check it, and send it through an x86 backend that must keep every current test
green. Then build the RTL backend. Arrays and division are new instructions with total
meanings; unboxed pairs are a second result on `call` and `ret`; DESCEND and ASCEND are a pair
that saves live registers into the destructured cell.
