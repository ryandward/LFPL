# LFPL-O: A Specification for Programs That Refuse Rather Than Crash

Oct 31, 2026 · @Ryan

## Summary

LFPL-O is the online profile of LFPL, a first-order linear language, for routers and other long-running services that handle a stream of data items. A program that has started never crashes, hangs or leaks. When demand exceeds what it can guarantee, it refuses work at its boundary instead of degrading in ways no one predicted.

**The guarantee.** For every input stream, a started LFPL-O program does one of two things with each item. Either it completes the item within statically known bounds on steps and cells, or it refuses the item at admission, with no effect on its state. In both cases it never reaches an undefined state, never touches memory outside its fixed image, and always returns to accept the next item.

**Refusal happens in exactly four places, in this order:**

1. **Compile time.** The program is refused: a type error, a declared limit exceeded, or a handler with no derivable bound.
2. **Startup.** The process refuses to start: its image cannot be locked into memory.
3. **Admission.** One item is refused: too large, the reserve is short, or the output queue is full. Refusing costs a constant number of steps and allocates nothing.
4. **After admission:** never. An admitted item has its cells and its step budget reserved before it starts.

**What this adds to the first specification** ([Pure LFPL on Bare Silicon](https://claude.ai/artifact/BEfbkxeqyLUA8WUMdE1zJk)):

- termination and time bounds, from structural recursion and resource typing;
- admission control as the single point where running code can refuse anything;
- fixed-size persistent state, including linear arrays for tables such as routes and flows;
- a fixed image that includes the stack, plus a contract with the operating system;
- a compiler that accepts every program within the declared limits, using two passes in opposite directions;
- a register-transfer-level target alongside x86-64.

**Out of scope:**

- hardware faults beyond what ECC corrects, and kernel bugs;
- cycle-exact latency on x86, where the bounds are counted in instructions and memory accesses (the RTL target gives cycles);
- availability under attack. Refusal protects the runtime, not the service: an adversary can force refusals, and quotas against that are a policy choice, not a language guarantee.

## 1. What "cannot crash" means

A program **crashes** if, after starting, any of three things happens:

- it terminates other than by an explicit shutdown;
- it reaches a state its semantics does not define;
- it stops returning to its input for longer than its declared worst case.

LFPL-O rules out all three, under three stated assumptions about the machine underneath.

### 1.1 Failure modes and what removes each

|  | Failure mode | Typical cause elsewhere | LFPL-O mechanism | Property |
| --- | --- | --- | --- | --- |
| F1 | Undefined state | null dereference, bad index, unhandled case | addresses only as capabilities; exhaustive matching; array indices of type Fin N | C1 |
| F2 | Arithmetic trap | division by zero, INT\_MIN / −1, oversized shift | every primitive is total (section 4.4) | C1 |
| F3 | Out of memory | allocation failure, the OS killing the process | a fixed image, locked and prefaulted at startup | C2 |
| F4 | Stack overflow | deep or unbounded recursion | the stack lives in the image; its size S\_max is a static bound | C3 |
| F5 | Hang or unbounded work | infinite loop, pathological input | structural recursion and a resource-typed bound W(n) | C4 |
| F6 | Collapse under overload | queues grow, latency explodes | admission control and fixed-capacity queues | C5, C6 |
| F7 | Failure halfway | a resource runs out after partial effects | everything the item needs is reserved at admission | C6, C7 |
| F8 | Slow leak over long uptime | memory not returned | per-transaction conservation, enforced by linearity | C8 |
| F9 | Miscompilation | compiler bug | translation validation of every build | A3 |
| F10 | Environment failure | signals, I/O errors, the OS killing the process | the OS contract (section 7.3) | A2 |

### 1.2 Assumptions (the fault model)

- **A1 Hardware.** The processor executes the ISA, or the FPGA executes the netlist, as specified. Memory errors are corrected by ECC; uncorrected ones are out of scope.
- **A2 Operating system** (x86-64 only; the RTL target has none). The OS maps the image, honors `mlockall`, exempts the process from out-of-memory killing (`oom_score_adj = -1000`), and returns a defined result from every I/O system call.
- **A3 Toolchain.** The assembler, linker and synthesis tools are correct, and the compiler's output for this program has passed translation validation (section 8.4).

### 1.3 The properties

Let P be a program with declared limits L, image 𝓘 and per-item bounds W (steps) and R (cells), each a function of an item's size n in cells. For every input stream, once P has started:

- **C1 Safety.** Every step is defined by the semantics: well-typed configurations never get stuck.
- **C2 Memory.** Every memory access falls inside 𝓘, whose size is fixed at link time.
- **C3 Stack.** The stack never exceeds S\_max, which is reserved inside 𝓘.
- **C4 Bounded work.** An admitted item of size n completes within W(n) steps.
- **C5 Liveness.** The serving loop returns to its input within W(N\_max) + r + c\_poll + c\_drain steps of leaving it. Here N\_max is the declared largest item, r is the constant cost of a refusal, and c\_poll and c\_drain are the declared costs of polling ingress and draining egress.
- **C6 No refusal after admission.** An admitted item never fails for lack of cells or steps.
- **C7 Atomic refusal.** A refused item changes nothing except a refusal counter, and its cells go back to the pool.
- **C8 Conservation.** After each item, every cell is in exactly one place: the pool, the persistent state, or a queue. Nothing leaks, however long the program runs.
- **C9 Totality.** Every primitive operation returns a defined value for every input.

In temporal-logic notation, where **G** is "always" and **F**≤k is "within k steps":

```latex
\mathrm{Start}(P) \implies \mathbf{G}\big(\mathrm{Defined} \wedge \mathrm{Mem} \subseteq \mathcal{I} \wedge \mathrm{Stack} \le S_{\max}\big)
\;\wedge\; \forall i.\; \mathrm{Admitted}(i) \implies \mathbf{F}_{\le W(|i|)}\, \mathrm{Done}(i)
\;\wedge\; \forall i.\; \mathrm{Refused}(i) \implies \Sigma_{\text{after}} = \Sigma_{\text{before}}
```

## 2. Where refusal happens

Every refusal happens before any work on what it refuses has begun: the program before it runs, the process before it reads input, and the item before its handler starts. Once a handler starts, it finishes.

&#91;embedded content: where refusal can happen · three refusal points, then none\]

Each refusal is also cheaper than the work it turns away. A refused program costs a compile, and a refused start costs one failed system call. A refused item costs r steps and no allocation, because refusing never needs a resource that might itself have run out.

## 3. The online machine

An LFPL-O program is a **transducer**: a handler applied to one item at a time, threading a fixed-size persistent state from item to item. The only unbounded loop is the runtime's serving loop, and programs cannot write one of their own.

### 3.1 Handlers and items

```latex
h \;:\; \Sigma \otimes \mathsf{Item} \otimes \mathsf{Free} \;\multimap\; \Sigma \otimes \mathsf{Out} \otimes \mathsf{Free}
```

- An **item** is a chain of n cells filled at ingress, with n ≤ N\_max (declared; for packets, set by the MTU).
- **Σ** is the persistent state: fixed-size linear arrays plus a fixed set of cell slots. It never grows (section 4.5).
- **Free** is the reserve of R(n) cells handed to this item at admission. The handler returns whatever it did not use.
- **Out** is a bounded list of at most O\_max messages, each of at most N\_out cells.

The compiler derives W(n) and R(n) for every handler (section 5) and writes them into the image as an admission table.

### 3.2 Configurations

```latex
\mathcal{M} \;=\; \langle\, F,\ \Lambda_{\mathsf{in}},\ Q_{\mathsf{in}},\ Q_{\mathsf{out}},\ \Sigma,\ T,\ \mathit{cnt} \,\rangle
```

| Component | Meaning | Size |
| --- | --- | --- |
| F | the pool of free cells | variable, inside Λ\_cell |
| Λ\_in | cells owned by ingress: K\_in slots of N\_max cells | constant K\_in · N\_max |
| Q\_in | items waiting in ingress slots | at most K\_in items |
| Q\_out | messages waiting in egress slots | at most K\_out messages |
| Σ | the persistent state | at most S\_cap cells, plus arrays |
| T | the transaction in flight: ⊥, or a term configuration (σ, K, e) from the first specification | at most n + R(n) cells |
| cnt | refusal counters, one per reason | constant |

### 3.3 Transition rules

The admission predicate checks the item's size, the pool, and room for the outputs:

```latex
\mathsf{adm}(n) \;\equiv\; n \le N_{\max} \;\wedge\; |F| \ge R(n) + n \;\wedge\; \mathrm{free}(Q_{\mathsf{out}}) \ge O_{\max}
```

```latex
\begin{array}{ll}
\textsf{Arrive} & \mathrm{free}(Q_{\mathsf{in}}) > 0,\ \text{device writes } i \text{ into a slot of } \Lambda_{\mathsf{in}}
  \;\longrightarrow\; Q_{\mathsf{in}} := Q_{\mathsf{in}} \cdot i\\[8pt]
\textsf{Admit} & T = \bot,\ Q_{\mathsf{in}} = i \cdot q,\ n = |i|,\ \mathsf{adm}(n),\ F = F' \uplus F_R \uplus F_n,\ |F_R| = R(n),\ |F_n| = n\\
  & \quad\longrightarrow\; T := (\Sigma, i, F_R),\ F := F',\ \Lambda_{\mathsf{in}} := (\Lambda_{\mathsf{in}} \setminus \mathrm{cells}(i)) \uplus F_n,\ Q_{\mathsf{in}} := q\\[8pt]
\textsf{Refuse} & T = \bot,\ Q_{\mathsf{in}} = i \cdot q,\ \neg\,\mathsf{adm}(|i|)
  \;\longrightarrow\; Q_{\mathsf{in}} := q,\ \mathit{cnt}[\mathit{why}] := \mathit{cnt}[\mathit{why}] + 1\\[8pt]
\textsf{Step} & T = (\sigma, K, e) \longrightarrow (\sigma', K', e') \quad \text{(term-level rules, first specification §3.3, plus arrays)}\\[8pt]
\textsf{Done} & T = (\Sigma', o, b) \text{ is a value}
  \;\longrightarrow\; \Sigma := \Sigma',\ Q_{\mathsf{out}} := Q_{\mathsf{out}} \cdot o,\ F := F \uplus b,\ T := \bot\\[8pt]
\textsf{Drain} & Q_{\mathsf{out}} = o \cdot q,\ \text{device sends } o
  \;\longrightarrow\; Q_{\mathsf{out}} := q,\ F := F \uplus \mathrm{cells}(o)
\end{array}
```

Three points about these rules:

- **ADMIT replenishes ingress.** The item's cells move into the transaction, and n fresh cells from the pool take their place, so Λ\_in stays at exactly K\_in · N\_max cells.
- **REFUSE touches neither Σ nor F.** The refused item's cells simply stay with ingress, and its slot is reused. Refusing allocates nothing and takes constant steps r.
- **When ingress is full, ARRIVE cannot fire,** so the device drops at its own ring. That is refusal in hardware, outside the program.

**Scheduling.** The loop runs one transaction at a time. Between transactions it drains egress and polls ingress, each for at most a declared constant number of steps. Several cores run share-nothing copies, each with its own image partition; state shared across cores is out of scope here.

**Timers** are an ingress source like any other, whose items are a single cell.

### 3.4 The conservation invariant

```latex
\Lambda_{\mathsf{cell}} \;=\; F \;\uplus\; \Lambda_{\mathsf{in}} \;\uplus\; \mathrm{cells}(Q_{\mathsf{out}}) \;\uplus\; \mathrm{cells}(T) \;\uplus\; \mathrm{cells}(\Sigma)
```

Every rule above moves cells between these components and creates none, so the partition holds in every reachable configuration (Theorem T7). The image is sized so that one maximal transaction always fits once the queues have drained:

```latex
C \;\ge\; K_{\mathsf{in}} N_{\max} \;+\; \big(R(N_{\max}) + N_{\max}\big) \;+\; K_{\mathsf{out}}\, O_{\max} N_{\mathsf{out}} \;+\; S_{\mathsf{cap}}
```

## 4. The language

LFPL-O is the first specification's LFPL with seven additions:

- recursion restricted to structural recursion, so every function terminates;
- every primitive total;
- fixed-size linear arrays and state slots;
- unboxed pairs, so a function can return two things without a cell;
- handlers as the only entry points;
- records that span cells;
- fields in network byte order.

Everything else, including ◇, exactly-once linearity and in-place construction, is unchanged.

### 4.1 Syntax

```latex
\begin{array}{rcl}
\tau &::=& \mathsf{Int}_w \ (w \in \{8,16,32,64\}) \mid \mathsf{Bool} \mid \Diamond \mid D \mid R \mid \mathsf{Array}[N]\,\tau \mid \mathsf{Fin}\,N \mid \mathsf{Free} \mid \tau \otimes \tau\\[4pt]
\mathit{field} &::=& \tau \mid \mathsf{Be}_w \ (w \in \{16,32,64\})\\[4pt]
e &::=& \cdots \text{(first specification §1)} \mid \langle e_1, e_2 \rangle \mid \mathsf{let}\ \langle x, y \rangle = e_1\ \mathsf{in}\ e_2\\
&\mid& \mathsf{idx}(e, N) \mid \mathsf{get}(a, i) \mid \mathsf{swap}(a, i, e) \mid e_1 \,/\, e_2 \mid e_1 \,\%\, e_2\\
&\mid& R@(d_1, \ldots, d_m)(\bar e) \mid \mathsf{let}\ \langle c, r \rangle = \mathsf{peel}(d)\ \mathsf{in}\ e\\[4pt]
\mathit{prog} &::=& \overline{\mathit{data}} \;\; \overline{\mathsf{state}\ x : \tau} \;\; \overline{\mathsf{fn}} \;\; \overline{\mathsf{handler}}
\end{array}
```

- D is a data type and R a record. Each constructor of a data type must fit one cell of B bytes: 1 tag byte plus B − 1 payload bytes. A record is B payload bytes per cell, and takes as many cells as its fields need (section 4.8). B is a link-time constant: 16 in Phase 1, and 64 matches a cache line and most packet headers.
- τ ⊗ τ is an **unboxed** pair. It can be returned and bound, but never stored in a cell, so it costs registers, not memory. This fills the gap the Phase 1 implementation found: the first specification's types had ⊗ but no term for it.
- A program has no `main`. Its entry points are `handler` declarations of the type in section 3.1.

### 4.2 Linearity

Values of type ◇, D, R, Array\[N\] τ and Free are linear: each is used exactly once on every path. Int\_w, Bool and Fin N are erasable. Linearity has one consequence that recursion depends on: **linear data are acyclic.** A cycle would need a pointer to a cell inside the value built in that cell, and the only such pointer is the constructed value itself, which is consumed by being stored.

### 4.3 Exhaustive matching

Every `match` must cover every constructor of its type, or the program is refused. Phase 1 already enforces this.

### 4.4 Total primitives

| Operation | Result for every input |
| --- | --- |
| +, −, × at width w | the result mod 2^w |
| x / y, unsigned | x / 0 = 0 |
| x % y, unsigned | x % 0 = x |
| x / y, signed | x / 0 = 0, and INT\_MIN / −1 = INT\_MIN |
| x % y, signed | x % 0 = x, and INT\_MIN % −1 = 0 |
| shl, shr by k at width w | 0 when k ≥ w; the usual shift otherwise |
| comparisons, and, or, xor, not | the usual, always defined |
| idx(e, N) : Fin N | e mod N, which is a mask when N is a power of two |
| get(a, i), for scalar τ | ⟨a, a\[i\]⟩ |
| swap(a, i, v), for any τ | ⟨a\[i := v\], a\[i\]⟩ |

The x86 lowering pays for totality where the hardware would trap or mask:

- Division tests the divisor and branches around `div`; signed division also tests for INT\_MIN / −1, which faults on x86.
- A shift by a non-literal count compares the count with w first, because x86 masks a 64-bit shift count to 6 bits. Phase 1 compiles `<<(x, 70)` to `shl rax, 70`, which shifts by 6.

### 4.5 Persistent state and arrays

State is declared, fixed in size, and threaded through each handler as a linear value.

- **Array\[N\] τ** has exactly N elements, stored in their own region of the image. Scalar elements can be read with `get`; any element can be exchanged with `swap`. There is no operation that copies a linear element or leaves a hole, so arrays obey the same conservation law as cells.
- **A state slot** holds one linear value of at most N\_max cells. A handler can keep an item's cells across items (for example, a reassembly buffer) only by swapping them into a slot and taking the old contents out. The state therefore holds at most S\_cap cells in total, and the image reserves them.
- **Nothing grows.** A table of flows is an array of fixed size, and inserting into a full table is a policy expressed in the program (evict, overwrite, or forward without state), not a runtime failure.

### 4.6 Structural recursion

**Rule R3 (summed measure).** For each recursive function f, the compiler must find one fixed set S of f's linear parameters such that every recursive call f(ē) in f's body strictly decreases the total μ over S. Alternatively, a parameter x\_k of type Fin N decreases: e\_k = x\_k − 1 inside an arm where x\_k ≠ 0.

μ composes, so the check is linear arithmetic inside the checker:

- Matching C(d, ȳ) gives μ(x) = 1 + Σ μ(y\_i) over its linear fields.
- Building gives μ(C@d(ā)) = 1 + Σ μ(a\_i).
- The result of a call to any other function has unknown μ, so it proves nothing. That holds even for an `@` function, which keeps its root cell but may change its size.

The set is chosen per function. In `rev(l, acc)` only {l} works, because every cell l loses, acc gains. The tree `walk(t, stack, …)` needs {t, stack}: one call shrinks t, another shrinks stack, and every call sends one cell to the free list. Any linear parameter may belong to S, including a free list, since all S has to do is shrink, and a natural number cannot shrink forever. Leaving free lists out of S, as the step 1 implementation does, is conservative but not required for soundness.

For now, recursive calls must be tail calls. Non-tail structural calls, compiled with continuations in the destructured cells (DESCEND and ASCEND in the first specification), are refused until that lowering exists, and section 8.2 lists this as a declared limit. Functions must be defined before use, as in Phase 1, so there is no mutual recursion.

```latex
\mu(v) = \#\{\text{cells reachable from } v\} \qquad \exists\, S \subseteq \mathrm{lin}(f).\;\; \forall\, f(\bar e) \text{ in the body of } f:\;\; \sum_{k \in S} \mu(e_k) \;<\; \sum_{k \in S} \mu(x_k)
```

**Lemma 4.1 (termination).** Every well-typed LFPL-O function terminates on every input. *Proof:* by induction on definition order, then on Σ\_{k∈S} μ(x\_k), or on the Fin N value. That number is finite because linear data are acyclic, and each recursive call strictly decreases it. Every call to an earlier function terminates by the outer induction. ∎

The step 1 implementation was tested against 28 combinations of recursive arms whose behavior was simulated independently. It accepted no non-terminating combination, and refused 4 that terminate, so it is conservative there. That is a test, not a proof. Adversarial probes behave the same way: rebuilding the same cell, swapping two arguments, and recursing on the result of an `@` function that grows its argument are all refused.

R3 rules out the scalar loops Phase 1 accepted. At `9593438`, `fn spin(x: Int64) : Int64 = spin(x);` compiled to a bare `jmp`. It is now refused, because spin has no linear parameter to shrink.

### 4.7 Example: a forwarding handler

The item is a header cell followed by its body chain. The handler drops packets whose TTL has run out, and otherwise looks up an output port in a fixed table:

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

The verdict is built in the header's own cell, so R(n) = 0. The handler never walks the body, so W(n) is a constant. Fwd fits one 16-byte cell: 1 tag byte, plus 1 + 1 + 4 + 8 bytes of fields.

### 4.8 Records that span cells

The source declares one record, and the compiler lays it across as many cells as it needs.

```
record Frame { dst_hi: Be32, dst_lo: Be16, src_hi: Be32, src_lo: Be16, ethertype: Be16, ... };
```

**Layout.** Fields go in declaration order, at their natural alignment. A cell takes all the fields that are left if they fit it. Otherwise it takes those that fit its first 8 bytes, and its second 8 bytes link to the next cell. So every field fits some cell, and a record that fits one cell is laid out as it always was.

**A pattern.** `R(d, f1, ..., fk)` binds every field by name, wherever it is, and `d` stands for all m cells of the record. Exactly-once linearity holds of each of the m, so a cell cannot be left behind. `d` can be used in three ways, and no others:

| Use | Means |
| --- | --- |
| `R@d(...)` | build a record of exactly m cells in them. Fields that do not change stay where they are, in each cell |
| `More@d(free)` | give all m back |
| `let (c, rest) = peel(d) in e` | c is the first cell, for a constructor that takes one, and rest stands for the others |

**Building from loose cells.** `R@(a, b, c)(...)` names one cell per cell of the record, in order. This is how a record is made that did not arrive as one.

**Meaning.** One record means exactly the chain of one-cell records a person could write by hand, with one link each. `src/spans.py` rewrites it into that chain ahead of Check, so that rewrite is all that is new: Check, Lower, the register-transfer layer, its checker and both backends have no idea, and each theorem of section 10 holds of the chain. `src/symcheck.py` does the rewrite again on its own, with its own layout, sharing no code with it, so if they disagree the build is refused. A `$` in a name in the register-transfer stream is one the rewrite made: the lexer has no such character, so it can never clash with a name in the source.

**Resources.** Every cell is a cell. A record that arrives with an item is part of that item: n counts its m cells, so admission's R(n) + n covers them and they are replaced in ingress. A record built from cells drawn from the reserve costs one draw per cell, so R(n) counts them there. μ counts every cell for rule R3, and each cell carries potential and releases it when it is opened, as any other does.

**Limits,** which section 8.2 lists: a record has at most 64 fields, and each cell after its first counts as two open constructs toward D\_max, because a match on it opens the next. `{ e with f = v }` reaches only the first cell, because a pattern is what opens the others. A group is not a value, so it cannot go to a function or through a join. A record that spans cannot be the `*` of tail recursion modulo cons, since the recursive call is then not in tail position. A tagged constructor still has to fit one cell.

### 4.9 Fields in network byte order

Be16, Be32 and Be64 are field types, and only field types. A field of one lies in its cell in network byte order, so the runtime can scatter a datagram into it and gather it out again without looking at it.

What a pattern binds from one is the integer, of type Int16, Int32 or Int64, and what a constructor stores in one is given as the integer. So a program has no bytes to think about, and a constant is written as the value it is: `==(ethertype, 2048)`, for what is 08 00 on the wire.

At the register-transfer level the two are different types, and `bswap` is the only way between them, so a swap that was forgotten is a type error and not a wrong answer. Lower swaps as late as it can, and never when it does not have to:

| The value | Going to | What happens |
| --- | --- | --- |
| a field, as it lies | something that reads the integer | `bswap`, one step |
| a field, as it lies | a Be field of the same width | moved as it lies, and `=` if it stays where it is, so nothing is swapped or stored |
| an integer | a Be field | `bswap`, then a store |
| a literal | a Be field | swapped by the compiler, then a store |

A read-modify-write in place is not available on a Be field, because the carry would go the wrong way. `+(total, 1)` swaps, adds, swaps and stores.

## 5. Resource typing

Every handler gets two linear bounds, derived by the compiler: R(n) = a·n + b cells drawn from its reserve, and W(n) = c·n + d steps. The method is the potential method of amortized analysis in the type system of Hofmann and Jost (POPL 2003). Linearity makes it simpler here than in their setting, because no value is ever shared.

### 5.1 The two resources and the cost model

- **Cells** count draws from Free, meaning matches on its `More` arm. Building a value in a ◇ that came from destructuring costs zero cells: in-place reuse is free, which is the point of LFPL.
- **Steps** count transitions of the term machine. Each construct has a fixed cost c\_e, set by the translation: one step lowers to at most k\_x86 instructions and m memory accesses on x86-64 (section 8), or at most k\_rtl cycles on the RTL target (section 9).

Resource amounts are pairs q = (q\_c, q\_t) of non-negative rationals, compared componentwise.

### 5.2 Annotated types

A data type carries potential on each cell: D^(p,s) means every cell of the value holds p cell-units and s step-units, available to whoever destructures it. Scalars, ◇ and arrays carry none.

```latex
\Phi(v : D^{(p,s)}) \;=\; \big(p \cdot \mu(v),\; s \cdot \mu(v)\big) \qquad \Phi(x_1{:}A_1, \dots, x_k{:}A_k) \;=\; \textstyle\sum_i \Phi(x_i : A_i)
```

The judgment reads: with context Γ and q units free beforehand, e evaluates to a value of type A and leaves q′ units free, the value's own potential not counted.

```latex
\Gamma \;\vdash^{q}_{q'}\; e : A \qquad\text{meaning}\qquad \Phi(\Gamma) + q \;\ge\; \mathrm{cost}(e) + \Phi(\mathit{result}) + q'
```

### 5.3 Rules (selection)

```latex
\begin{array}{ll}
\textsf{Match} &
\dfrac{\Gamma,\ d{:}\Diamond,\ \bar y{:}\bar B[D^{(p,s)}] \;\vdash^{\,q + (p,s) - (0,c_{\mathsf{m}})}_{q'}\; e_C : A \quad \text{for every constructor } C}
{\Gamma,\ x{:}D^{(p,s)} \;\vdash^{q}_{q'}\; \mathsf{match}\ x\ \mathsf{with}\ \overline{C(d,\bar y) \Rightarrow e_C} : A}
\\[22pt]
\textsf{Build} &
\dfrac{q \;\ge\; q' + (p,s) + (0, c_{\mathsf{b}})}
{d{:}\Diamond,\ \bar y{:}\bar B[D^{(p,s)}] \;\vdash^{q}_{q'}\; C_d(\bar y) : D^{(p,s)}}
\qquad \text{(zero cells: the cell is } d\text{)}
\\[22pt]
\textsf{Draw} &
\dfrac{\Gamma,\ d{:}\Diamond,\ f'{:}\mathsf{Free} \;\vdash^{\,q - (1,\,c_{\mathsf{m}})}_{q'}\; e_1 : A}
{\Gamma,\ f{:}\mathsf{Free} \;\vdash^{q}_{q'}\; \mathsf{match}\ f\ \mathsf{with}\ \mathsf{End} \Rightarrow e_0 \mid \mathsf{More}(d, f') \Rightarrow e_1 : A}
\\[22pt]
\textsf{App} &
\dfrac{f : \bar A \xrightarrow{\;q_f / q'_f\;} B \qquad q \ge q_f + (0, c_{\mathsf{call}}) \qquad q' \le q'_f}
{\bar x{:}\bar A \;\vdash^{q}_{q'}\; f(\bar x) : B}
\\[22pt]
\textsf{Let} &
\dfrac{\Gamma_1 \vdash^{q}_{r} e_1 : B \qquad \Gamma_2,\ x{:}B \vdash^{r}_{q'} e_2 : A}
{\Gamma_1, \Gamma_2 \;\vdash^{q}_{q'}\; \mathsf{let}\ x = e_1\ \mathsf{in}\ e_2 : A}
\qquad
\textsf{Relax}\ \dfrac{\Gamma \vdash^{p}_{p'} e : A \quad q \ge p \quad q' \le p'}{\Gamma \vdash^{q}_{q'} e : A}
\end{array}
```

Two things differ from Hofmann and Jost:

- **There is no sharing rule.** Linear data are never duplicated, so potential never has to be split between two uses of one value.
- **DRAW types only the `More` arm against the budget.** The `End` arm must still type-check, but Theorem 5.1 shows it is unreachable when the reserve is large enough. Its cost therefore never enters the bound. This rests on every Free a program holds being that reserve, and the certificate checks it: linearity keeps there from being two of one, so a handler that makes a Free on a path that can run gets no bound. A compiled program should never reach that refusal, because it could not get rid of a second one: a `More` arm leaves as many as it found, R3 does not count a Free, an output cannot hold one, and a handler returns exactly one.

### 5.4 Soundness

**Theorem 5.1 (bounds).** Suppose ⊢^(b,d) h : Σ ⊗ Item^(a,c) ⊗ Free ⊸ … is derivable, and an item i has μ(i) = n. Run h with a Free of length at least a·n + b. Then it draws at most a·n + b cells, never reaches an `End` arm of its reserve, and halts within c·n + d steps. *Proof sketch:* by induction on the evaluation derivation, following Hofmann and Jost's soundness proof for the potential method. Each rule preserves the inequality of section 5.2. A draw is possible exactly when the remaining budget is at least 1, and the budget starts at a·n + b. Lemma 4.1 guarantees the evaluation derivation is finite. ∎

### 5.5 Inference

Every annotation is an unknown non-negative rational, and every rule instance contributes linear inequalities. The compiler solves one linear program per handler, minimizing R(N\_max) and then W(N\_max), and rounds the coefficients up to integers.

A call of a finished function is typed by its own copy of that function's unknowns and inequalities, which is App with a fresh annotation of f. One annotation per function would refuse linear programs, such as reversing what a reversal returned: the function's result would have to carry the amount its parameter does, which pays for the walk and has the result's amount left, so that amount would have to exceed itself. A recursive call stays at the annotation of the call it is part of, since otherwise the inequalities would say nothing about the walk. Rule R3's decreasing set, the parameters whose total falls, shares one amount per cell, since the cell that leaves can be any of theirs. Other linear parameters may carry amounts of their own, and they have to: a walk that moves cells onto an accumulator hands the next walk's budget over with them.

- **Feasible:** the solution gives a, b, c and d, written into the admission table.
- **Infeasible:** the handler has no linear bound, and the program is refused, which is a declared limit (section 8.2).

The LP solver need not be trusted. The typing derivation with its numbers filled in is a certificate, and a small checker verifies it in one pass. Polynomial bounds are a known extension (Hoffmann, Aehlig and Hofmann, POPL 2011) that still reduces to linear programming. Version 1 keeps linear bounds, since per-item work in a router should be linear in the item's size.

### 5.6 Examples

| Handler | What it does | R(n) cells | W(n) steps |
| --- | --- | --- | --- |
| `forward` (section 4.7) | rebuilds the header cell, reads one table entry | 0 | a constant |
| checksum over the body | walks every body cell once | 0 | c·n + d |
| duplicate to two ports | copies the body into fresh cells | n + 1 | c·n + d |

Only the copy needs a reserve. Each copied cell is one DRAW, so the body's type is inferred as Bytes^(1, s), with one cell of potential per cell.

## 6. Admission control

Admission is the only place where running code refuses anything. It decides in constant time, from the item's size and a table of four numbers per handler, and it reserves everything the item will need before any of the handler runs.

### 6.1 The decision

For an item of n cells bound for handler h, admission reads (a\_h, b\_h, c\_h, d\_h) from the admission table, computes R(n) and W(n), and checks these conditions in a fixed order. The first one that fails names the refusal.

| Order | Condition | Refusal reason |
| --- | --- | --- |
| 1 | n ≤ N\_max | TooLarge |
| 2 | the pool holds at least R(n) + n cells | Short |
| 3 | egress has room for O\_max messages | Full |
| 4a | the item's age ≤ its deadline D\_h on the wall clock, if one is declared | Late |
| 4b | the steps of the work queued ahead of it + W(n) ≤ a step budget S\_h, if one is declared | Behind |
| 5 | the program's own admission predicate holds, if one is declared | Policy |

Conditions 4a and 4b are separate because they measure different things, and steps and seconds are never added together. Age is a time a clock read, and W(n) is a count the compiler proved. No constant turns one into the other on x86-64, where a step's time depends on caches and on whatever else the machine is doing. So 4a says how long this item has already waited, and 4b says how much work it would still be, and each is compared with a limit in its own units.

On success, R(n) cells become the item's Free argument, n cells replenish ingress (rule ADMIT), and the handler starts. By Theorem 5.1 it cannot run out.

### 6.2 Refusing

A refusal does exactly two things: it increments `cnt[reason]`, and it frees the ingress slot, leaving the item's cells where they were. It allocates nothing and costs a constant r steps.

Each ingress source declares one of two refusal actions:

- **Drop:** the item is discarded and counted. This is the default for network ports.
- **Hold:** the item stays in its slot, and the source stops accepting more until it is admitted. This is backpressure. It is valid only where the producer can wait: a local producer, or the ready/valid handshake of the RTL target (section 9).

Answering a refused sender, for example with an ICMP message or a NACK, needs cells, so refusal itself cannot do it. Such a reply is a separate small handler with its own reserve, admitted or refused like any other item.

### 6.3 The admission predicate

A program can declare one predicate that sees the state and the item's header:

```latex
\mathsf{admit} \;:\; \Sigma \otimes \mathsf{Hdr} \;\multimap\; \Sigma \otimes \mathsf{Hdr} \otimes \mathsf{Bool} \qquad R_{\mathsf{admit}} = 0,\quad W_{\mathsf{admit}} = \text{a constant}
```

The compiler checks both bounds: no draws, and a step bound independent of n. The predicate's cost is part of r. Per-source token buckets kept in an array are the typical use: they bound what one sender can make the program do.

### 6.4 Behavior under overload

Each turn of the serving loop either completes one admitted item, at most W(N\_max) steps, or refuses one item, r steps. Drain and poll add at most a declared constant between turns. It follows that, for any arrival pattern:

- **Memory never moves.** The image is fixed, and Theorem T7 holds on every turn.
- **Admitted items finish on time.** An item admitted from position j of ingress starts within j turns, and the serving loop's share of its latency is at most

```latex
L(n) \;\le\; K_{\mathsf{in}} \cdot \big(W(N_{\max}) + c_{\mathsf{poll}} + c_{\mathsf{drain}}\big) \;+\; W(n)
```

- **Overload changes only the refusal rate.** Excess arrivals become refusals, at ingress or at the device's own ring, at a cost of r per refusal. The program keeps serving at its full rate instead of thrashing.

What this does not give is fairness between senders, or service to any particular sender during an attack. Those are policies, written in the admission predicate.

## 7. The fixed image

All the memory a program will ever use is laid out at link time, locked into RAM at startup, and touched once before the first item arrives. After that, the program never asks the operating system for memory, and has no way to.

### 7.1 Layout

| Region | Size | Fixed by |
| --- | --- | --- |
| code | the program | the compiler |
| cells | C · B bytes, with C from section 3.4 | K\_in, N\_max, the largest R(N\_max), K\_out, O\_max, N\_out, S\_cap |
| arrays | the sum of N\_k · size(τ\_k) over the state declarations | the program's `state` declarations |
| queues | K\_in + K\_out descriptors | declared capacities |
| stack | S\_max, plus one guard page below it | the compiler's stack bound (first specification, Theorem 7.3) plus the runtime's fixed frames |
| counters and admission table | constant | the program's handlers and refusal reasons |

The total |𝓘| is written in the ELF program headers. Nothing is mapped after startup. The stack is part of the image, not the stack the OS provides: frame sizes and depth are fixed by the program text, so `rsp` only computes addresses inside a reserved region.

### 7.2 Startup

`_start` runs these steps in order. If any step fails, the process exits with a distinct status naming the step, before it reads any input: it **refuses to start**.

1. Block every asynchronous signal, and ignore SIGPIPE.
2. Call `mlockall(MCL_CURRENT | MCL_FUTURE)`. This needs CAP\_IPC\_LOCK, or RLIMIT\_MEMLOCK of at least |𝓘|.
3. Write to every page of the cells, arrays and stack, so no page fault can happen during service.
4. Make the guard page inaccessible with `mprotect(PROT_NONE)`, and point `rsp` at the top of the image's stack.
5. Thread the free list, zero the state, and load the admission table.
6. Open the ingress and egress endpoints.
7. Enter the serving loop.

Step 3 moves every memory failure the OS could cause to startup. A machine without enough memory refuses to run the program instead of killing it later under load.

### 7.3 The contract with the operating system

Assumption A2 in detail. At runtime the program makes only these calls:

| Call | Results | Handling |
| --- | --- | --- |
| ingress read (`read` or `recvmmsg`) | items; no item (0 or EAGAIN); an error | items arrive by rule ARRIVE; no item means idle; an error is counted and treated as no item |
| egress write (`write` or `sendmmsg`) | sent; would block; an error | sent runs rule DRAIN; would block leaves the message queued, and a full queue makes admission refuse with Full; an error is counted, and the message is dropped with its cells returned |
| clock read (`clock_gettime`, or `rdtsc`), only when a deadline is declared | a time | used by condition 4a of admission |

The runtime never calls `mmap`, `brk` or `clone`, and installs no signal handlers. Synchronous faults (SIGSEGV, SIGFPE) cannot arise unless assumption A1 or A3 is violated. The guard page is defense in depth: if a compiler bug overruns the stack, the result is a clean fault, not silent corruption.

Two requirements are deployment settings, not language guarantees:

- **Out-of-memory killing.** Locked memory does not exempt a process from the kernel's OOM killer, so the deployment must set `oom_score_adj = -1000`.
- **Locking permission.** The process needs CAP\_IPC\_LOCK, or a large enough RLIMIT\_MEMLOCK.

On bare metal and on the RTL target, none of this section applies: the image is the machine's memory.

## 8. Compilation

The compiler refuses a program only for a type error or a declared limit, never because of its own internal choices. Achieving that requires information flowing in both directions through the program, so compilation makes two passes over it in opposite directions, and keeps the first specification's O(D) memory bound in each.

### 8.1 Pipeline

1. **Lex and parse** forward, as in the first specification. Each event goes through the rewrite of section 4.8, and then to a file on disk.
2. **The right-to-left pass** reads that file backward with O(D) state and writes one annotation record at each construct's opening position.
3. **Check:** linear typing, exhaustive matching, rule R3, and the arity and size limits.
4. **Resource analysis** generates constraints as it goes and solves one linear program per handler at the handler's end (section 5.5). A handler's inequalities, with the copies for what it calls, are capped by a declared limit, so each linear program is bounded.
5. **Lower** forward with destination passing, reading the annotations in the same order as the source.
6. **Validate:** translation validation, the resource certificate, and the stack bound (section 8.4). Any failure refuses the program, and no binary is written.
7. **Emit** the assembly, the admission table and the image layout.

The annotations are the facts a forward pass cannot see, which Phase 1 had to approximate:

| Attribute | Flows | Replaces in Phase 1 |
| --- | --- | --- |
| last use of each binder | right to left | erasing a scalar on overwrite, then refusing a later read |
| need(e), Sethi–Ullman | upward, delivered at e's opening | evaluating operands in source order |
| distance to next use | right to left | spilling the oldest value |
| frame size F\_f | from the function's end to its start | entering spilling functions through a thunk |
| the cell every exit returns | from the exits to the entry | the mandatory `@` annotation |

With exact last uses, a scalar is moved out before its location is overwritten, or before its cell's ◇ goes to a callee, exactly when it is read later. Otherwise it is simply dropped. Nothing is speculative and nothing is refused.

### 8.2 Why a program can be refused

**(i) Type errors:** linearity, exhaustiveness, rule R3, or a mismatch.

**(ii) Declared limits.** Every limit can be checked from the source text alone:

| Limit | Default | Checked by |
| --- | --- | --- |
| nesting depth D\_max, counting two for each cell of a record that spans after its first | 64 | parser, and the right-to-left pass for what spans add |
| global names G\_max | 2,048 | checker |
| binders in one function | 128 | lowerer; this also bounds snapshots |
| inequalities in one handler's linear program, with what it calls | 1,024 | resource analysis |
| pivots to solve one linear program | 10,240, about ten times what the largest has needed | resource analysis |
| a constructor of a data type fits one cell | B bytes | checker |
| the fields of one record | 64 | right-to-left pass |
| arguments per function | 6 | checker |
| frame slots per function F\_max | declared | lowerer |
| stack S\_max | declared | lowerer |
| N\_max, K\_in, K\_out, O\_max, N\_out, S\_cap | declared per program | linker |
| the image, which includes R(N\_max) cells | 1 GiB, because the code reaches it with 32-bit displacements | linker |
| W(N\_max) | below 2^63, because admission computes it in 64 bits | linker |
| every handler has a linear bound | — | resource analysis |
| the admission predicate draws nothing and takes constant steps | — | resource analysis |
| non-tail recursion | refused until DESCEND and ASCEND are built | lowerer |

**(iii) Never:** a refusal caused by the compiler's own choices, such as evaluation order, which register to use, or which value to spill. Theorem T8 states this as completeness, and the fuzzer tests it: any refusal of a generated program that is not (i) or (ii) is a compiler bug.

### 8.3 Lowering to x86-64

Lowering follows the first specification's translation calculus, with the corrections recorded there:

- §4.1: a register is reserved only while its value is being computed;
- §5.3: a join keeps a fact about memory only if every arm agrees;
- §6.2: writability is tracked per field, and a cell's read window closes when it is rebuilt or its ◇ leaves.

LFPL-O adds:

- **The frame size is known at the prologue,** from the annotations, so the prologue is a plain `sub rsp, 8F_f` inside the image's stack.
- **Total primitives** lower as in section 4.4.
- **Arrays** are addressed as base + index × element size. The index has type Fin N, and `idx` reduces it mod N, so no bounds check is emitted and none can fail. `swap` is a load and a store.
- **Steps become instructions:** every term-machine step lowers to at most k\_x86 instructions. k\_x86 is a constant of the translation, since parallel moves are bounded by the fields of one cell and spills by F\_max. So W\_insn(n) = k\_x86 · W(n).

### 8.4 Validation of every build

Assumption A3 becomes a check on each binary, not a trust in the compiler:

- **Meaning.** A symbolic checker that shares no code with the compiler (`src/symcheck.py`) parses the source itself and executes the assembly symbolically. At every `ret`, call and tail jump, it requires each field of every built constructor to hold what the source says.
- **Bounds.** The resource certificate is checked against the typing rules of section 5.3.
- **Stack.** The deepest stack depth over the call graph is recomputed and compared with S\_max.
- **Redundant moves.** The identity-morphism validator runs, as a check on efficiency, not on correctness.

If any of these fails, the compiler emits no binary. The guarantee is then about this binary, validated, rather than about every possible output of an unverified compiler.

### 8.5 The register-transfer layer

Lower does not write x86 directly. `src/rtlower.py` emits a flat, typed register-transfer stream, and an independent checker, `src/rtcheck.py`, types every instruction with the first specification's token rules before any backend runs. The design is in `docs/rt-layer.md`.

- **Explicit ownership.** Registers are unbounded per function, and moves consume their source, so ownership is written into the code instead of being reconstructed beside it.
- **Atomic constructors.** A constructor is one instruction: it reads all its sources, then writes, and each field's source says STAY, RESTORE or STORE. Writing a value back where it came from is a local type error, and so is claiming RESTORE when a store is needed. That is the first specification's Theorem 6.1, checked locally.
- **Joins and calls.** Labels apply the merge rule of section 8.3. A call removes every token of a cell whose capability it receives.
- **Two backends.** x86 becomes register allocation over this stream, which is `src/x86.py`, and RTL maps almost one instruction to one state, which is `src/rtverilog.py`.

Of the 12 wrong-code bugs found in Phase 1, 8 are local type errors at this level. The other 4 are facts about x86: a truncated immediate, a spilled operand, the result register lost across a back-edge, and a masked shift count. Those stay with the symbolic checker. The layer keeps the O(D) bound: control flow is structured, so the checker holds token states only for labels still open, at most one per enclosing construct.

## 9. The RTL target

LFPL's operational semantics is already a register-transfer semantics: a fixed set of storage locations, where every step reads some, writes some, and conserves the set. On x86-64 that has to squeeze through 15 registers and a call stack. On an FPGA or ASIC it maps directly, and the per-item bound becomes a bound in clock cycles.

### 9.1 Mapping

| LFPL-O | Register-transfer level |
| --- | --- |
| a cell of B bytes | one word of block RAM, 8B bits wide |
| ◇ | that word's address, log₂ C bits |
| a scalar binder | a register; the number of registers is fixed by the program text, so nothing spills |
| `match` | a block-RAM read, then a tag compare that selects the next state |
| building into d | a block-RAM write whose byte enables cover only the fields that changed |
| phantom construction | wires, with nothing stored |
| a conditional jump | a state transition, with no boolean register |
| tail recursion | a loop in the state machine |
| DESCEND and ASCEND | a pointer-reversal walk through the cells; no hardware stack exists or is needed |
| a function | a unit (a state machine plus its datapath), inlined, or shared by sequential call sites |
| Array\[N\] τ | a block RAM of N words; `swap` is a read, then a write |
| Free | a chain of free addresses through the cells, as on x86 |
| ingress and egress | AXI4-Stream ports with valid and ready signals |
| admission | a check of n against the pool counter, done before `ready` is raised; R(n) = a·n + b needs one small multiplier |
| Hold, as a refusal action | keeping `ready` low |
| Drop, as a refusal action | accepting the item and discarding it, counted |
| the serving loop | the top-level state machine |

The split between static and input-dependent memory becomes physical. Whatever the program text fixes is flip-flops; whatever grows with the input is block-RAM cells addressed by ◇. Nothing else exists.

### 9.2 Example: `inc` as a state machine

The list cell is tagless: h in bytes 0–7 and t in bytes 8–15. One transition per state, with p the current cell's address:

| State | Action | Next |
| --- | --- | --- |
| S0 | if p = 0, the walk is done | DONE, or S1 |
| S1 | issue a block-RAM read of word p | S2 |
| S2 | latch h and t from the read data | S3 |
| S3 | write h + 1 to word p with byte enables 0x00FF; set p := t | S0 |

The tail is never written, because its byte enables are never set. The first specification's Theorem 6.1, that unchanged fields are not copied, becomes a property of the netlist anyone can inspect. That costs four cycles per element as written; overlapping the next read with the current write brings it down to one.

### 9.3 Cycle bounds

Each term-machine step maps to at most k\_rtl states. The schedule respects the block RAM's ports: at most two accesses per cycle for dual-port RAM. So:

```latex
\mathrm{cycles}(h, n) \;\le\; k_{\mathsf{rtl}} \cdot W(n) \qquad\qquad \mathrm{latency}(h, n) \;\le\; \frac{k_{\mathsf{rtl}} \cdot W(n)}{f_{\max}}
```

Here f\_max is the clock frequency synthesis achieves, set by the longest combinational path within one state. There are no caches and no interrupts, so this is a true upper bound in time. x86-64 has no equivalent.

### 9.4 Properties specific to this target

- **RTL-1 (storage is the image).** All storage is the block RAM and registers sized at synthesis. The netlist has no other memory.
- **RTL-2 (cycle bound).** An admitted item finishes within k\_rtl · W(n) cycles.
- **RTL-3 (write masks).** No write enable ever covers an unchanged field.

### 9.5 How to check it

Two checks follow, and both are available in the current workspace:

- **Differential simulation.** Simulate the generated Verilog with Icarus Verilog, and compare outputs with the x86-64 binaries on the same fuzz corpus.
- **Synthesis.** Run Yosys to confirm register and block-RAM counts against the image layout.

## 10. Theorems and proof sketches

The guarantee rests on nine theorems. T1–T7 are about the abstract machine of section 3. T8 is about the compiler. T9 carries the abstract results down to each binary.

**Main theorem.** Under assumptions A1–A3, every started LFPL-O program satisfies C1–C9. *Proof:* T1 gives C1 and C9. T2 gives C2 and C3. T3 gives C4, T6 gives C5, T4 gives C6, T5 gives C7, and T7 gives C8. T9 and A3 carry these from the abstract machine to the binary, and A1 and A2 carry them from the binary to the machine. ∎

### T1 Type safety (C1, C9)

*A well-typed configuration is either a value or takes a defined step to another well-typed configuration.*

Proof sketch: progress and preservation by induction on typing derivations. The cases that could get stuck elsewhere all reduce to a rule of this specification:

- `match` has an arm for every constructor (section 4.3).
- Every primitive is total (section 4.4).
- An array index has type Fin N, so it lies in \[0, N).
- Every address is a capability: ◇ has no constructor, pointers support no arithmetic, and the null pointer exists only as a nullary constructor, which is matched before any field is read.

### T2 Memory stays in the image (C2, C3)

*Every location accessed is in 𝓘, and stack depth never exceeds S\_max.*

Proof sketch:

- **Cells.** No rule of section 3.3, and no term-level rule, creates a location. This is the first specification's Theorem 7.1, extended to arrays, which are fixed regions.
- **Stack.** Rule R3 makes every recursive call either a tail call or a structural call whose frames live in the destructured cells. With no mutual recursion, the call graph minus those edges is acyclic, and every frame size F\_f is fixed. The first specification's Theorem 7.3 then bounds the depth by S\_max, and the image reserves exactly that.

### T3 Bounded work (C4)

*An admitted item of size n finishes within W(n) = c·n + d steps.*

Proof sketch: Lemma 4.1 gives termination, and Theorem 5.1 gives the bound, given a reserve of R(n) cells, which ADMIT provides.

### T4 No refusal after admission (C6)

*Once admitted, an item's transaction reaches DONE.*

Proof sketch: a transaction could fail in only four ways, and each is excluded:

- running out of cells, which Theorem 5.1 excludes with a reserve of R(n);
- running out of steps, which T3 excludes;
- having nowhere to put its output, which admission condition 3 excludes;
- getting stuck, which T1 excludes.

No other rule fires on a transaction in flight.

### T5 Atomic refusal (C7)

*A refused item changes only `cnt` and frees its ingress slot. Admission is the only point at which running code refuses.*

Proof sketch: REFUSE touches only Q\_in and `cnt`; it leaves Σ and F unchanged. Inspecting the rules of section 3.3, REFUSE is the only rule that discards an item, and it fires only when T = ⊥, before a handler has run.

### T6 Liveness and latency (C5)

*The serving loop returns to ingress within W(N\_max) + r + c\_poll + c\_drain steps. An item admitted from ingress position j finishes within the bound L(n) of section 6.4.*

Proof sketch: each turn runs one transaction to completion, at most W(N\_max) by T3, or one refusal, r steps. Drain and poll are bounded by declaration. Position j waits at most j turns.

### T7 Conservation (C8)

*The partition of section 3.4 holds in every reachable configuration, so no cell leaks, however many items are processed.*

Proof sketch: by induction over transitions. Each rule moves cells between components and creates none. At DONE, the handler must account for every cell it received, as Σ′, as output o, or as returned reserve b, because discarding a linear value is a type error. So the pool after each item equals the pool before it, minus two amounts: the cells held by queued output, which DRAIN returns, and any net growth of the state's slots, which S\_cap bounds.

### T8 Completeness of compilation

*Every program that type-checks and satisfies the declared limits of section 8.2 compiles.*

Proof sketch: with exact annotations (section 8.1), every lowering rule's premises can be met.

- **Destinations are hints, not reservations.** A register holding a finished value can always be spilled and released. At any instruction, the only pinned registers are that instruction's operands plus two scratch registers excluded from allocation. That is at most 5 of the 15, so a spill victim always exists. Spill slots are bounded by F\_max, a declared limit.
- **Parallel moves always succeed.** At joins and calls they need at most the two scratch registers, which are always free.
- **What a forward pass lacks is supplied.** The frame size is known at the prologue, last uses are exact, and the result cell is inferred.

So none of the internal failures Phase 1 reports can occur.

### T9 Refinement, per binary

*The emitted binary behaves as the abstract machine does on this program.*

This theorem is established per build by the checks of section 8.4, not by a proof of the compiler. A build that fails them produces no binary. On the RTL target, RTL-1 to RTL-3 are checked the same way, by differential simulation and by synthesis reports.

## 11. Limits and non-goals

The guarantee is narrow on purpose: a started program stays defined, bounded and responsive. It does not promise that the program's logic is right, that every item gets served, or that x86 timing is exact.

### 11.1 What programs give up

- **General recursion.** Irregular traversals must be restructured, for example as a tail-recursive walk with a worklist built from the structure's own cells, which the Phase 1 tree tests already do.
- **Superlinear handlers.** A handler with no linear bound is refused. Polynomial bounds are a known extension (section 5.5) but are left out of version 1.
- **Growing state.** Tables are fixed-size arrays, and holding data across items uses fixed slots. Programs that need unbounded state must decide what to evict.
- **Higher-order functions and closures.** The language is first-order, as in the first specification.
- **More than six arguments**, until arguments in memory homes are added.
- **Unframed streams.** Input must arrive as items of at most N\_max cells. A byte stream such as TCP has to be framed by the program, using state slots.

### 11.2 What the guarantee does not cover

- **Functional correctness.** LFPL-O guarantees that a program stays defined and within its bounds, not that its forwarding logic is right.
- **Precision of bounds.** Derived bounds may be loose, so admission may refuse items the program could have handled. That costs throughput, never safety. The compiler should report every handler's R and W so operators can see what they are paying.
- **Cycle-exact timing on x86-64.** Bounds there count instructions and memory accesses. Caches, TLB misses, interrupts, simultaneous multithreading and frequency scaling make cycles vary. Core isolation helps but guarantees nothing. Only the RTL target bounds time.
- **Availability under attack.** Refusal protects the runtime, not the service. An adversary who floods the input forces refusals; per-source quotas in the admission predicate are the countermeasure, and they are policy.
- **State shared across cores.** Version 1 runs share-nothing copies, and flows are spread across cores by the NIC.
- **Hardware faults and kernel bugs,** beyond assumptions A1 and A2. An external watchdog remains sensible as defense in depth.

### 11.3 What must be trusted

| Trusted | Why it cannot be avoided |
| --- | --- |
| the assembler and linker, or the synthesis tools | they produce the artifact that runs |
| the kernel, on x86-64 | it maps the image and performs I/O |
| the runtime: `_start`, the serving loop, admission | fixed code, written once; small enough to review line by line |
| the symbolic checker and the certificate checker | they are the oracle for every build |

The compiler and the LP solver are deliberately not on this list. A wrong output from either is caught by validation, and no binary is written.

## 12. Status against the repository

The section 4.7 handler now compiles unchanged from `examples/forward.lfpl` and serves UDP sockets on the loopback address, and `examples/checksum.lfpl` walks its item under a certified bound, through the whole of the pipeline below except the parts this table marks as missing. This is `ryandward/LFPL` main at commit `b5078db`, where all 454 tests pass with Python 3.14.8.

| Requirement | Status at `b5078db` | Needed |
| --- | --- | --- |
| exactly-once linear typing | present in Check, and repeated one level down by `src/rtcheck.py` | nothing |
| exhaustive matching | present | nothing |
| structural recursion (R3) | present, with the summed measure, not counting Free, so `spin` is refused | the Fin N case |
| total primitives | shifts are total, and `idx` and `get` cannot go out of range. There is no division yet. | division with the semantics of section 4.4 |
| non-tail recursion | refused, as a declared limit | DESCEND and ASCEND |
| arrays and Fin N | `Array[N] t` of scalars, read with `idx` and `get`. N must be a power of two. A Fin N may live in a cell. | `swap`, linear elements, and any N once division exists |
| unboxed pairs | present, as flat tuples of at most six, returned and bound and never stored | nothing |
| records that span cells, and fields in network order | present, for records: `src/spans.py` rewrites one record into its chain of cells ahead of Check, and `examples/ipv4.lfpl` is one record of five cells. A field is still read from its cell, which the runtime copied it into. | reading a field in place from the frame, which changes the resource model; `{ with }` past the first cell; a record built from loose cells has to name each one |
| handlers, serving loop, admission | one handler and one ingress, over UDP on the loopback address, one datagram per turn. Admission works out n from the datagram's length, then checks TooLarge, Short and Full. Late, Behind and Policy cannot be declared yet, so W(n) is only added to a counter, and every source refuses by Drop. | several handlers, recvmmsg or AF_XDP, Hold, deadlines, the admission predicate, and state slots of cells |
| resource analysis and certificates | present on every build. `src/rtcheck.py` writes section 5's inequalities as it types each function, `src/simplex.py` proposes numbers, and the checker evaluates every inequality with them, so the solver is not trusted. A handler can walk its body. | polynomial bounds, and k\_x86 of section 8.3, which has not been computed, so W(n) counts steps and is not yet a time |
| stack in the image | present: S_max comes from the call graph, the stack is in the image with a guard page below it, and the depth is recomputed from the assembly | nothing |
| startup: lock, prefault, signals | present. Each step that can fail exits with its own number, before any input. | nothing |
| register-transfer layer | present: `src/rtlower.py`, `src/rtcheck.py` and `src/x86.py`, and it is the only path | nothing |
| two-pass compilation | present for last uses and the result cell: `src/righttoleft.py`, between Parse and Check, so that a scalar read after its cell closes moves out first, and `@` is optional. need(e), the distance to the next use and frame sizes are the x86 backend's, which does not compute the first two yet. | those two, in the backend |
| translation validation | present on every build: a failure in `src/symcheck.py` leaves no binary. It covers the handler and its helpers, and it does not read the runtime. | admission |
| identity-morphism validator | present, but **only in the test suite**: `src/lfplo.py` does not run it, so a build does not yet make the fourth check of section 8.4 | run it on every build |
| heap-flatness test | present, and it holds when run on its own | nothing |
| cell conservation in the compiler ring | present, with Theorem 7.2's bound checked at every step | nothing |
| RTL backend | the handler and everything it calls, for every RT instruction: `src/rtverilog.py`, which needed no change to anything before it. Its bound in cycles is derived by the emitter and held to in simulation, its outputs are compared with the x86 binary's on the same bytes, and yosys maps its cell store to block RAM. | hardware ingress, egress and admission, which are test code on this target so far, and `swap` |

The implementation decides some things this text leaves open, or does not mention:

- **Literals.** A literal must be a value of the type it meets, or the program is refused. A shift count is the exception, because section 4.4 gives every count a meaning.
- **The wire format.** An item is a record, whose fields are scalars, other records, and at most one field of a type declared and never defined. On the wire it is those scalars in declaration order, going into a record where one is met, packed, followed by the bytes of that last field, which the runtime holds as a chain of cells. An IntN is little endian and a BeN is in network order. So n is the record's cells and the chain's. A Fin N cannot be a field of an item, because a sender could then choose an address.
- **Length.** A field named `length` of type Int16 is on the wire neither way. Admission fills it in with how long the datagram was, because a program that does not walk its body has no other way to find out.
- **Output.** A constructor with a field `port: Int8` is sent to a base port plus that number, as its other scalars followed by its chain, going into a record where one is met. Any other constructor sends nothing, and all of its cells return to the pool.
- **Free.** A program may declare its own, for Phase 1's sake, but a handler's must be exactly `End | More(next: Free)`, because the pool is threaded through that field.
- **Malformed.** A datagram shorter than an item's scalars is counted as malformed. That is a fifth reason at ingress, beside those of section 6.1, because it never was an item.
- **The control port.** It is one more ingress source, bound to the loopback address. `S` answers with the counters, `P` pauses egress, `R` resumes it and `Q` shuts down. Pausing exists because loopback sends never block, so without it nothing could show that a full queue refuses.
- **System calls.** Beyond section 7.3's list, ingress and egress use `recvmsg` and `sendmsg`, scattering a datagram straight into cells, and the control port uses `recvfrom` and `sendto`. Idling is a blocking ingress read with a 1 ms timeout, so `poll` is not needed. `setsockopt` asks for control messages on the ingress socket, and in a build that keeps `max_age_ns`, `clock_gettime` is called once per admitted item.
- **What the kernel says about ingress.** `SO_RXQ_OVFL` gives the socket's count of datagrams it dropped, on each datagram it delivers, and `kernel_dropped` adds the unsigned 32-bit differences. The kernel stamps a datagram with the count when it queues it, so while the socket is overflowing nothing can say so, and the count lags until the next datagram gets in. A build that asks for it with `--measure-age` also has `SO_TIMESTAMPNS`, which gives each datagram's arrival, kept while that item is being served, and `max_age_ns` is the largest wait from there to admission, with a negative age taken as 0. Nothing refuses on it yet: it is what section 6.1's condition 4a will compare with a deadline.
- **Link-time options.** `--state` gives a state array its initial contents, and `--cells` can make the image smaller than section 3.4's C, which is what makes Short reachable. `--count-steps` validates the build without its counters and then rebuilds with one at each step the checker charged for, so that tests can hold running code to W(n). That rebuild is for measuring, and is not itself validated.
- **The chain.** An item's chain may be abstract, like `data Bytes;`, or declared with the exact layout the runtime builds, `Last | Chunk(used, b0, ..., b6: Int8, next)`, which lets a handler walk it. Cells are reused, so admission clears what follows the used bytes of a chain's last cell, and a program can read nothing an earlier item left. A program can now also write the used count and grow the chain, so egress clamps each count to 7. N\_out is not declared: it is N\_max + R(N\_max), because an output can hold every cell of its item and of its reserve. One message gathers at most 1,024 chain entries, which is what `sendmsg` takes, and a longer chain counts as a send error, with its cells returned.
- **The right-to-left pass.** It works one function at a time, in source order, and not on the whole file: what a function returns can depend on what its callees return, and a callee always comes first. It goes forward as well as backward, because a let's name comes before the expression it is bound to, so reading backward a use cannot yet be told from a use of an outer name. Going forward it writes the events to the file and resolves names, and works out each value's word. Going backward, it holds the set of names with a use still to come, and two sets per open branch. The annotation is two things in the event word Check and Lower already read: `leaf(VARLAST)` in place of `leaf(VAR)` at a last use, and `leaf(LIVE, mask)` after each separator that starts an arm or a scope. A name past the 120th in scope is taken to be wanted, which costs a read and is right.
- **The inferred cell.** It is conservative, and exact about nothing but what `rtcheck` will accept: the rule is at the head of `src/righttoleft.py`, and `src/symcheck.py` works it out again without sharing any code, so a disagreement refuses the build. Where it finds nothing, the function simply has no `@`, as if none were written. A `@` that is written is held to every exit as before, and what the pass finds is held to the same.
- **Narrowing.** `low8`, `low16` and `low32` keep the low bits of an integer. `pack` only widens, and without these a sum worked out in Int64 could not be stored in a narrower field.
- **Steps.** A step is one instruction of the register-transfer stream, except `move`, `drop` and `open`, which lower to nothing on x86. On the RTL target `open` can take a clock, and so can the read in front of `br.tag`, because a block RAM returns a word a clock after it is asked. `docs/rt-layer.md` lists what else in the stream came from x86.
- **k\_rtl.** It is not a constant of the translation. The emitter derives k and c for each program from the states it wrote, and section 9.3's bound becomes cycles(n) <= k * W(n) + c, counted from the edge that samples `start` to the edge that raises `done`. For `examples/checksum.lfpl`, k = 1 and c = 1.
- **Read during write.** No state of the machine enables both of the cell store's ports, and a host that writes does not read, so a clock never reads and writes one address. In the Verilog such a read would return the old word, but no block RAM is asked to say so: yosys proves from the state decode that the two enables never meet, and maps the store with no logic for it. If the host's write did not hold off its read, yosys could not prove it, and on iCE40 it spends about 300 flip-flops delaying the write to keep the old-word answer.

The plan, with what is done marked. Arrays and tuples came ahead of resource analysis, because the handler needs them:

1. **R3 and total shifts.** Done. This closed the hang (F5) and the undefined shift (F2).
2. **The register-transfer layer.** Done. This turned 8 of the 12 Phase 1 bug classes into type errors, and the first Lower is gone.
3. **Stack in the image, and the startup sequence.** Done. This closes F3, F4 and F10 on x86-64.
4. **A serving loop with admission.** Done, with measured constant bounds instead of hand-written ones. This closes F6 and F7 for a handler that does not recurse.
5. **Resource analysis,** which replaces the measured bounds with derived and certified ones. Done. `examples/checksum.lfpl` is the first handler that needs it, with R(n) = 0 and W(n) = c n + d.
6. **Arrays, Fin N with R3's Fin N case, unboxed pairs and division,** for router tables. Arrays, Fin N and tuples are done, to the extent the table says. R3's Fin N case, division and `swap` are not.
7. **DESCEND and ASCEND,** which lift the limit on non-tail recursion.
8. **The right-to-left pass,** for exact last uses and the inferred result cell. Done, for those two. This removes the refusal of a scalar read after its cell closes (T8).
9. **The RTL backend,** built from the register-transfer layer and checked by differential simulation against x86-64.

## 13. Related work and references

Each part of LFPL-O has prior work behind it. What this specification adds is the combination: in-place linear memory with no allocator, derived per-item bounds enforced at admission, a fixed image, and a hardware target, aimed at programs that run forever on untrusted input.

### Linear types and resource analysis

- **LFPL** — Martin Hofmann, *Linear types and non-size-increasing polynomial time computation*, LICS 1999; *A type system for bounded space and functional in-place update*, Nordic Journal of Computing 7(4), 2000. ([papers](https://www.dcs.ed.ac.uk/home/resbnd/papers/martin.html))
- **The potential method for heap bounds** — Hofmann and Jost, *Static prediction of heap space usage for first-order functional programs*, POPL 2003. This is the basis of section 5. ([listed here](https://www.dcs.ed.ac.uk/home/resbnd/papers/martin.html))
- **Polynomial bounds by linear programming** — Hoffmann, Aehlig and Hofmann, *Multivariate Amortized Resource Analysis*, POPL 2011, and the TOPLAS version ([paper](https://www.cs.yale.edu/homes/hoffmann/papers/HAH12Toplas.pdf)); implemented in Resource Aware ML ([CAV 2012](https://link.springer.com/chapter/10.1007/978-3-642-31424-7_64)).
- **Fully in-place functions with constant stack** — Lorenzen, Leijen and Swierstra, *FP²: Fully in-Place Functional Programming*, ICFP 2023. ([MSR report](https://www.microsoft.com/en-us/research/publication/fp2-fully-in-place-functional-programming/))

### Languages with bounded resources

- **Hume** — Hammond and Michaelson, *Hume: a Domain-Specific Language for Real-Time Embedded Systems*, GPCE 2003 ([record](https://research-portal.st-andrews.ac.uk/en/publications/hume-a-domain-specific-language-for-real-time-embedded-systems/)). This is the closest in purpose: a functional language for embedded systems with bounded time and space, whose processes are boxes that each have bounded cost. Worst-case execution time analysis for it came out of the EmBounded project ([paper](https://typeset.io/pdf/towards-formally-verifiable-wcet-analysis-for-a-functional-5e9vmzcgqw.pdf)). LFPL-O differs in using in-place linear memory with ◇, and in refusing work at admission at runtime.
- **The eBPF verifier** refuses, at load time, any program it cannot prove safe, including ones that might not terminate ([kernel docs](https://docs.ebpf.io/linux/concepts/verifier/)). Bounded loops arrived in Linux 5.3 ([LWN](https://lwn.net/Articles/794934)). It is the nearest operational precedent for refusing programs rather than crashing at runtime. It derives no per-item bounds and has no admission step.

### Packet processing

- **P4** — Bosshart et al., *P4: Programming Protocol-Independent Packet Processors*, ACM SIGCOMM CCR 44(3), 2014 ([arXiv](https://arxiv.org/abs/1312.1719)). It programs the match-action pipelines of switch chips. LFPL-O's RTL target aims at the same hardware with a general, structurally recursive language.

### Hardware synthesis from functional languages

- **SAFL** — Mycroft and Sharp, *A Statically Allocated Parallel Functional Language*, ICALP 2000, and its silicon compiler to RTL Verilog ([project page](https://www.cl.cam.ac.uk/~am21/research/sa)).
- **SASL** — Frankau and Mycroft, static allocation through linearity and stability constraints ([HICSS 2003](https://www.cl.cam.ac.uk/~am21/papers/hicss_2003.pdf)).
- **Geometry of Synthesis** — Ghica, POPL 2007 ([record](https://research.birmingham.ac.uk/en/publications/geometry-of-synthesis-a-structured-approach-to-vlsi-design/)), and GoS IV, compiling affine recursion into static hardware, ICFP 2011 ([paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2011/09/Compiling-Affine-Recursion-into-Static-Hardware.pdf)).
- **FHW** — Zhai, Townsend, Lairmore, Kim and Edwards, *Hardware Synthesis from a Recursive Functional Language*, CODES 2015 ([report](https://www.cs.columbia.edu/~sedwards/papers/zhai2015hardware-tr.pdf)). It supports recursion and dynamic allocation in hardware.

As far as these sources show, none of them keeps recursive data in a fixed block RAM with no allocator, which is what ◇ provides.

### Compiler theory used in section 8

- Jazayeri and Walter, *Alternating semantic evaluator*, ACM Annual Conference 1975 ([DOI](https://doi.org/10.1145/800181.810328)). Engelfriet and Filè, *Passes, sweeps and visits in attribute grammars*, ICALP 1981 ([paper](https://rd.springer.com/content/pdf/10.1007/3-540-10843-2_16.pdf)).

### This project

- The first specification: [Pure LFPL on Bare Silicon: A Formal Execution Model](https://claude.ai/artifact/BEfbkxeqyLUA8WUMdE1zJk).
- The implementation: [ryandward/LFPL](https://github.com/ryandward/LFPL). Phase 1 is commit `9593438`, and section 12 describes commit `f827822`.

The links point to pages found while writing this specification. Claims about each work are limited to what those pages state.
