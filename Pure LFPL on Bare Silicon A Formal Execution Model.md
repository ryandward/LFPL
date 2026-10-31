# Pure LFPL on Bare Silicon: A Formal Execution Model

Oct 31, 2026 · @Ryan

## Summary

The paradox goes away once ◇ is read as a **location capability**, not a value. What is conserved is storage locations (cells, registers, frame slots, the flags register), never bits. Every location is minted exactly once, by the linker when it lays out the ELF image. After that, the running system only transfers locations. The OS never creates tokens: read() and write() are *cell transformers* that borrow owned cells and hand them back.

Read that way, this specification delivers:

- **A closed ring** of six stages (Ingest → Lex → Parse → Check → Lower → Egress → Ingest) whose cells recycle continuously. The syntax tree is a well-bracketed event stream that is never materialized. Peak occupancy is W + c·D cells, independent of source length N.
- **A destination-passing translation** 𝒟⟦e⟧(δ) with δ ∈ {reg r, mem a, cond L\_t L\_f, fields δ̄, pack r}. In this calculus, phantom construction, jumping-code conditionals, tail-recursion-modulo-cons and the linear mutation law each fall out of one or two rules.
- **A provenance-indexed hole token** ◇ₐʸ (location a, last evacuated value y). It turns writing an unchanged field back to its own address into a type error.
- **Proofs** of cell conservation, O(D) occupancy, a static machine-stack bound, phantom construction, and the absence of identity morphisms.

Four of the directive's claims cannot be proven as stated, so the last section weakens them:

1. A finite machine must reject inputs nested deeper than a declared D\_max. Running out of memory at runtime becomes impossible, but only because the grammar rejects such inputs; it is not impossible unconditionally.
2. The global signature table costs O(G) cells, not O(D).
3. A compiler's output is larger than its input. Pure LFPL cannot express a size-increasing function, so the extra size is turned into time by draining output through Egress.
4. With more than 15 scalars live at once, spill traffic is unavoidable. What gets eliminated is *redundant* memory traffic, not all memory traffic.

## 1. Resource algebra

Every executable resource is drawn from a finite location universe Λ fixed at link time. No rule ever adds an element to it.

```latex
\Lambda \;=\; \Lambda_{\mathsf{cell}} \uplus \Lambda_{\mathsf{reg}} \uplus \Lambda_{\mathsf{slot}} \uplus \{\varphi\}
```

| Component | Elements | Size | Notes |
| --- | --- | --- | --- |
| Λ\_cell | ℓ₀ … ℓ\_{C−1} in .bss | C cells × 16 B | 1 tag byte + 15 payload bytes. The longest x86-64 instruction encoding is 15 bytes, so one encoded instruction fits one cell. |
| Λ\_reg | rax rbx rcx rdx rsi rdi rbp r8–r15 | 15 tokens | rsp is not a token: it is the origin of the frame manifold. |
| Λ\_slot | rsp + 8k, k < F\_f, per function f | static F\_f | Fixed coordinates, numbered at compile time. |
| φ | rflags | 1 token | Free (φ) or holding a comparison (φ\_cc(a,b)). |

### Token forms

```latex
\begin{array}{rcll}
\delta_{\mathsf{tok}} &::=& \Diamond_a^{\,y} & \text{hole at } a \text{, last evacuee } y \ (y = \bot \text{ if fresh})\\
&\mid& a \mapsto v & \text{full location}\\
&\mid& \rho_r & \text{free register } r\\
&\mid& \varphi \;\mid\; \varphi_{cc}(o_1,o_2) & \text{flags: free / holding a comparison}
\end{array}
```

Hofmann's ◇ is exactly the hole token ◇ₐ on a cell: a pointer to storage you own whose contents you may overwrite. The superscript provenance y is the one addition this spec makes to LFPL; section 6 explains why it is needed.

### Source language (first-order LFPL)

```latex
\begin{array}{rcl}
\tau &::=& \mathsf{Int}_w \mid \mathsf{Bool} \mid \Diamond \mid L(\tau) \mid T(\tau) \mid R\{\overline{f:\tau}\} \mid \tau \otimes \tau\\[4pt]
e &::=& x \mid n \mid e_1 \oplus e_2 \mid e_1 \bowtie e_2 \mid \neg e \mid e_1 \mathbin{\&\&} e_2 \mid e_1 \mathbin{||} e_2\\
&\mid& \mathsf{let}\ x = e_1\ \mathsf{in}\ e_2 \mid \mathsf{if}\ e\ \mathsf{then}\ e_1\ \mathsf{else}\ e_2\\
&\mid& C_d(\bar e) \mid \mathsf{match}\ e\ \mathsf{with}\ \overline{C_j(d_j,\bar x_j) \Rightarrow e_j}\\
&\mid& \{\, e \ \mathsf{with}\ f = e' \,\} \mid f(\bar e) \mid \mathsf{pack}\langle \bar w \rangle(\bar e)
\end{array}
```

The linearity discipline:

- Variables of type ◇, L, T or R are **linear**: used exactly once.
- Scalars (Int\_w, Bool) are **erasable**. Their *content* may be discarded, because what is conserved is the register or slot holding them, not the bits.
- There is no term of type ◇ other than a variable bound by a match. In particular there is no `fresh`.

### Homes

At every program point, the lowerer keeps a partial map from live variables to locations:

```latex
H : \mathrm{dom}(\Gamma) \rightharpoonup \mathsf{Loc}, \qquad \mathsf{Loc} ::= \mathsf{reg}\ r \mid \mathsf{mem}\ a \mid \mathsf{imm}\ n \mid \mathsf{virt}(C, \bar u) \mid \mathsf{hole}(p)
```

**Invariant H (home uniqueness).** Each live variable has exactly one home, and no two live variables share a home. `virt(C, ū)` is a constructor that has not been materialized: its fields have homes ū, and it occupies no storage of its own.

### Judgment forms

```latex
\begin{array}{ll}
\Gamma \vdash e : \tau & \text{LFPL typing (Hofmann, unchanged)}\\[3pt]
\Delta;\, H \vdash e \leadsto \delta \dashv \Delta';\, H' \;\triangleright\; I & \text{lowering: emits instruction sequence } I\\[3pt]
(\sigma, K, e) \longrightarrow (\sigma', K', e') & \text{token machine (section 3)}
\end{array}
```

𝒟⟦e⟧(δ) in section 5 is shorthand for the I produced by the unique derivation of the lowering judgment. Lowering is syntax-directed, so the derivation is unique once the destination hints are fixed.

**Mint-freedom (meta-property).** In every rule, the multiset of locations in (Δ, H, δ) before equals the multiset in (Δ′, H′, result) after. No location appears only on the right side of any rule.

## 2. Environmental reification and the closed cyclic ring

Raw bytes never need to become tokens. They land in cells that already *are* tokens. The baptism happens once, at link time, and the environment only ever fills or drains cells it has been lent.

### 2.1 The single mint

The ELF image reserves C cells in .bss. When the loader maps that region, `_start` receives the whole of Λ\_cell as one token set Δ₀ = {◇\_ℓ | ℓ ∈ Λ\_cell}. This is the only mint. It sits outside the operational semantics and happens exactly once per process. No rule in sections 3–6 creates a cell.

### 2.2 The environment as a cell transformer

The OS (or, on bare metal, a device) is typed as a function on cells it is handed, never as a source of cells:

```latex
\begin{array}{l}
\mathsf{read}_k \;:\; \Diamond^{k} \;\multimap\; \exists j \le k.\; (\mathsf{Byte} \otimes \Diamond)^{j} \otimes \Diamond^{k-j}\\[4pt]
\mathsf{write}_k \;:\; (\mathsf{Byte} \otimes \Diamond)^{k} \;\multimap\; \Diamond^{k}
\end{array}
```

The ◇ inside (Byte ⊗ ◇) is the cell's own capability, exactly as in Hofmann's list cells. read changes the contents of locations you own and gives every location back. Hardware already works this way: a NIC receive ring has the driver post buffers it owns, and the device returns the same buffers filled. The syscall ABI fits the model too. `syscall` clobbers rcx and r11, so at the register level it is a token exchange that consumes ρ\_rcx and ρ\_r11 for its duration.

### 2.3 The ring

The pipeline is six coroutine stages joined by channels that hold at most one cell each:

```latex
F \xrightarrow{\ \mathsf{read}\ } \mathsf{Lex} \xrightarrow{c_2} \mathsf{Parse} \xrightarrow{c_3} \mathsf{Check} \xrightarrow{c_4} \mathsf{Lower} \xrightarrow{c_5} \mathsf{Egress} \xrightarrow{\ \mathsf{write}\ } F
```

F is the **return edge**, the set of empty cells waiting to be refilled. Every stage that finishes with a cell (whitespace, a closing delimiter, a popped frame, a written output cell) sends its ◇ back to F. Scheduling is **pull-driven**: a stage fires only when its output channel is empty. So at most five cells are ever in flight between stages.

&#91;embedded content: the closed ring · six stages, one return edge\]

Solid arrows are the forward channels c₂–c₅, each holding one cell at most. Dashed arrows are cells returning to F. Only frames, one per open construct, stay outside F for long.

### 2.4 The syntax tree as a Dyck stream

The parser never builds a tree. It emits a well-bracketed event word over the alphabet

```latex
\Sigma_{\mathsf{ev}} = \{\, \mathsf{open}(k,\pi),\ \mathsf{sep}(k,i),\ \mathsf{leaf}(k,\pi),\ \mathsf{close}(k) \,\}
```

where k is a construct kind and π is an inline payload of up to 15 bytes. Downstream stages consume this word as a fold. They see each construct's opening event before its children (so a binder is in scope while its body is checked) and its closing event after them (so the type synthesized for the children is available).

**Theorem 2.1 (streaming tree).** If the parser accepts, the emitted word w is the bracketed preorder serialization of the unique parse tree t. Every node of t exists, at any moment, only as an event in flight or as one frame on some stage's stack. *Sketch:* the parser is a deterministic pushdown transducer whose stack symbols are frames. Each push emits `open` and each pop emits `close`, so w is in the Dyck language over the construct kinds, and parsing it back gives t.

"Verified" therefore means the stream is certified well-formed by construction. Downstream stages may assume bracket balance without checking it.

### 2.5 Bounding active memory by depth

Each stage that needs to remember something across a construct keeps **one frame per open construct**, stored in a cell:

| Stage | Frame contents | Cells per open construct |
| --- | --- | --- |
| Lex | the pending lexeme chain | at most ⌈L\_max / 15⌉, not per construct |
| Parse | production state | 1 (the opener token's own cell) |
| Check | binder name, type, usage state; or the pending operand type | 1, plus ⌈\|x\| / 15⌉ for a binder's name chain |
| Lower | binder home, pending labels, join homes | 1 |

Frames are pushed by `open` and popped by the matching `close`, so a stage's stack depth equals the current nesting depth D\_t. When a stage needs a cell for a frame, it receives one from F. Section 7 proves that F is never empty when that happens.

**Depth limit.** The parser checks nesting depth when it pushes a frame. If depth would exceed D\_max, it rejects the input with a *syntax* diagnostic. This is a language rule, like C's translation limits: it depends only on the input text, never on timing or allocation state. With C ≥ bound(D\_max) (Theorem 7.2), receiving from F is total, and the machine code for that receive has no failure branch.

**Remark (opener payment).** Suppose the grammar gives every construct at least as many delimiter tokens as there are stages that need a frame for it. For example, `let x = e₁ in e₂` has the tokens `let`, `x`, `=` and `in`. Then each stage can keep one delimiter cell as its frame instead of receiving from F. The occupancy bound is unchanged; only who pays for the frame changes.

## 3. Abstract Token Machine

The machine has two levels. The **ring level** moves cells between stages. The **term level** runs each stage's LFPL program. Every rule at both levels only moves locations, which is what Theorem 7.1 checks.

### 3.1 Configurations

```latex
\mathcal{C} \;=\; \langle\, F \;\|\; c_2, c_3, c_4, c_5 \;\|\; S_{\mathsf{lex}}, S_{\mathsf{par}}, S_{\mathsf{chk}}, S_{\mathsf{low}}, S_{\mathsf{egr}} \,\rangle,
\qquad S \;=\; (\sigma_S,\, K_S,\, e_S)
```

- F ⊆ Λ\_cell: empty cells on the return edge.
- c\_i ∈ Λ\_cell ∪ {⊥}: channel contents. Capacity one.
- σ\_S : Λ\_cell ⇀ Content: cells stage S owns, with their contents. Content ::= byte b | tok(…) | ev(…) | frame(…) | insn(bytes ≤ 15) | ◇.
- K\_S ∈ Λ\_cell\*: S's frame stack. **Each frame is a cell.**
- e\_S: S's control state, i.e. the residual LFPL program of that stage.

Each stage S has a deterministic step function over its control state and one input content:

```latex
\mathsf{step}_S(e, v) \in \{\ \mathsf{pass}(e', v'),\ \mathsf{push}(e', \phi, v'),\ \mathsf{pop}(e', v'),\ \mathsf{drop}(e'),\ \mathsf{expand}(e', v_1 \dots v_k)\ \}
```

### 3.2 Ring-level rules

Notation: c\_in and c\_out are S's input and output channels. Every rule except INGEST and EGRESS has the side condition c\_out = ⊥ (pull discipline).

```latex
\begin{array}{ll}
\textsf{Ingest} & F \uplus \{\ell\},\ c_2 = \bot,\ \mathsf{read}_1(\Diamond_\ell) = (b, \Diamond_\ell)
  \;\longrightarrow\; F,\ c_2 = \ell,\ \sigma(\ell) := \mathsf{byte}\ b\\[8pt]
\textsf{Pass} & c_{in} = \ell,\ \mathsf{step}_S(e_S, \sigma(\ell)) = \mathsf{pass}(e', v)
  \;\longrightarrow\; \sigma(\ell) := v,\ c_{in} = \bot,\ c_{out} = \ell,\ e_S := e'\\[8pt]
\textsf{Push} & c_{in} = \ell,\ \mathsf{step}_S = \mathsf{push}(e', \phi, v),\ F = F' \uplus \{m\},\ |K_S| < D_{\max}\\
  & \quad\longrightarrow\; \sigma(\ell) := \mathsf{frame}\ \phi,\ K_S := \ell \cdot K_S,\ \sigma(m) := v,\ c_{out} = m,\ F := F'\\[8pt]
\textsf{Depth} & c_{in} = \ell,\ \mathsf{step}_{\mathsf{par}} = \mathsf{push}(\dots),\ |K_{\mathsf{par}}| = D_{\max}
  \;\longrightarrow\; \mathsf{reject}(\textit{nesting too deep})\\[8pt]
\textsf{Pop} & c_{in} = \ell,\ \mathsf{step}_S = \mathsf{pop}(e', v),\ K_S = f \cdot K'
  \;\longrightarrow\; \sigma(\ell) := v,\ c_{out} = \ell,\ K_S := K',\ F := F \uplus \{f\}\\[8pt]
\textsf{Drop} & c_{in} = \ell,\ \mathsf{step}_S = \mathsf{drop}(e')
  \;\longrightarrow\; c_{in} = \bot,\ F := F \uplus \{\ell\}\\[8pt]
\textsf{Expand} & c_{in} = \ell,\ \mathsf{step}_{\mathsf{low}} = \mathsf{expand}(e', v_1 \dots v_k),\ F = F' \uplus \{m_2..m_k\}\\
  & \quad\longrightarrow\; \sigma(\ell) := v_1,\ \sigma(m_i) := v_i,\ \text{queue } \ell, m_2..m_k \text{ on } c_5 \text{ in order}\\[8pt]
\textsf{Egress} & c_5 = \ell,\ \mathsf{write}_1(\sigma(\ell)) = \Diamond_\ell
  \;\longrightarrow\; c_5 = \bot,\ F := F \uplus \{\ell\}
\end{array}
```

The lexer's work fits these schemas. Whitespace is DROP. A byte that extends a lexeme is PUSH onto its pending chain, but with no forwarded cell. On lexeme completion, the lexer packs the k bytes into the first m = ⌈k/15⌉ cells of the chain, forwards the head as the token (PASS), and DROPs the other k − m cells. A token never needs more cells than its lexeme had bytes, so **lexing is non-size-increasing** in Hofmann's exact sense. Stages after Lex are non-size-increasing too, with one exception: Lower is the only stage with EXPAND.

### 3.3 Term-level rules (inside a stage)

These are Hofmann's in-place semantics, plus one rule that stores continuations in destructured cells:

```latex
\begin{array}{ll}
\textsf{Match} & \sigma(\ell) = C_k(\bar v),\ e = \mathsf{match}\ \ell\ \mathsf{with} \dots C_k(d,\bar x) \Rightarrow e_k \dots\\
  & \quad (\sigma, K, e) \longrightarrow (\sigma[\ell \mapsto \Diamond^{\bar v}],\ K,\ e_k[\ell/d,\ \bar v/\bar x])\\[8pt]
\textsf{Cons} & \sigma(\ell) = \Diamond^{\_},\ (\sigma, K, C_\ell(\bar v)) \longrightarrow (\sigma[\ell \mapsto C(\bar v)],\ K,\ \ell)\\[8pt]
\textsf{Descend} & e = E[f(\ell')],\ \ell \text{ destructured in } e,\ \mathrm{live}(E) \le 15\,\mathrm{B}\\
  & \quad (\sigma, K, e) \longrightarrow (\sigma[\ell \mapsto \mathsf{frame}(E)],\ \ell \cdot K,\ f(\ell'))\\[8pt]
\textsf{Ascend} & (\sigma, \ell \cdot K, v),\ \sigma(\ell) = \mathsf{frame}(E) \longrightarrow (\sigma[\ell \mapsto \Diamond],\ K,\ E[v][\ell / d_E])
\end{array}
```

In MATCH, the cell becomes a hole ◇^{v̄} whose old contents stay readable until something overwrites them. This **stale-read window** is what makes zero-copy in-place reconstruction possible, and section 6 types it.

In DESCEND, the pending continuation of a non-tail structural recursion is stored *in the cell being destructured*. That cell's ◇ is in scope in E anyway, so storing the frame there costs nothing. K is therefore a chain of input cells (a pointer-reversed spine, as in Schorr–Waite), not a separately allocated stack. If E's live state exceeds 15 bytes, the program is rejected with a type error, *frame does not fit in ◇*.

## 4. Physical silicon as conserved ◇ manifolds

Registers, frame slots and rflags follow the same discipline as cells: a lowering rule may use a location only if it holds that location's token, and it must return every token it does not pass on.

### 4.1 Registers as tokens

Δ\_reg ⊆ Λ\_reg is the set of free register tokens. A destination `reg r` carries ρ\_r, and the rule that receives it must deliver its value into r. A register named by an enclosing destination is **pinned**: no subcomputation may spill it. Register demand follows Sethi–Ullman, but against a *finite* pool:

```latex
\begin{array}{rcl}
\mathsf{need}(x) &=& 0 \quad (\text{any home}) \qquad \mathsf{need}(n) = 0\\
\mathsf{need}(e_1 \oplus e_2) &=& \begin{cases} \max(\mathsf{need}(e_1), \mathsf{need}(e_2)) & \mathsf{need}(e_1) \ne \mathsf{need}(e_2)\\ \mathsf{need}(e_1) + 1 & \text{otherwise} \end{cases}
\end{array}
```

The binary-operation rule evaluates the needier operand first into the destination register. It uses a second register only if the other operand cannot be an instruction operand directly, that is, if it is not a variable homed in memory, a register variable, or an imm32:

```latex
\frac{\mathsf{need}(e_1) \ge \mathsf{need}(e_2) \quad \rho_{r'} \in \Delta}
{\mathcal{D}[\![e_1 \oplus e_2]\!](\mathsf{reg}\ r) \;=\; \mathcal{D}[\![e_1]\!](\mathsf{reg}\ r) \cdot \mathcal{D}[\![e_2]\!](\mathsf{reg}\ r') \cdot \mathtt{op}\ r, r' \quad (\rho_{r'} \text{ returned})}
\qquad
\frac{e_2 \text{ is an operand } o}
{\mathcal{D}[\![e_1 \oplus e_2]\!](\mathsf{reg}\ r) \;=\; \mathcal{D}[\![e_1]\!](\mathsf{reg}\ r) \cdot \mathtt{op}\ r, o}
```

**SPILL is the only rule that writes a frame slot.** It fires only when a destination demands a register and Δ\_reg is empty:

```latex
\frac{\Delta_{\mathsf{reg}} = \emptyset \quad H(v) = \mathsf{reg}\ s \quad s \notin \mathsf{pinned} \quad \Diamond_k \in \Delta_{\mathsf{slot}}}
{\Delta;\, H \vdash \mathsf{spill}(v) \dashv (\Delta \setminus \Diamond_k) \cup \{\rho_s\};\ H[v \mapsto \mathsf{mem}\,[\mathtt{rsp}+8k]] \;\triangleright\; \mathtt{mov}\ [\mathtt{rsp}+8k],\, s}
```

The victim is the unpinned value whose next use is furthest away (Belady). Lowering is syntax-directed, so next-use distances are known statically.

### 4.2 Stack slots as invariant coordinates

A function's frame is a fixed manifold of F\_f coordinates. F\_f is the maximum number of slot tokens held at any point of f's lowering derivation, so it is computed statically. The prologue `sub rsp, 8F_f` and the matching epilogue are the only stack-pointer arithmetic in f.

Arguments arrive in one of three ways under the internal convention:

- as **register tokens** transferred to the callee;
- as **memory homes** inside a cell, where the caller passes the pointer p and the callee binds x ↦ mem \[p + off\];
- in the caller's outgoing area \[rbp + 16 + 8i\], when there are more arguments than the register budget allows.

The callee treats every one of these as the argument's *home*, not as a value to be copied in.

**Lemma 4.1 (no argument mirroring).** No derivation stores a value whose home is `mem a` into a frame slot. *Proof:* stores to slots arise only from SPILL, whose premise requires H(v) = reg s. No rule moves memory to memory, except relocation for a destination at a *different* address, and that relocation targets a constructor field, never a slot. ∎

**Calls are token exchanges.** A callee's signature declares the clobber set κ\_f ⊆ Λ\_reg it consumes. Before `call f`, every live value homed in κ\_f moves to a free register outside κ\_f, or is spilled if there is none. Either move is a real relocation, because f will overwrite that register.

### 4.3 Flags as a linear token

```latex
\begin{array}{lll}
\mathtt{cmp}\ o_1, o_2 &:& \varphi \;\multimap\; \varphi_{cc}(o_1, o_2)\\
\mathtt{j}cc\ L &:& \varphi_{cc}(o_1,o_2) \;\multimap\; \varphi \quad (\text{branch on } cc)\\
\mathtt{set}cc\ r_8 &:& \varphi_{cc}(o_1,o_2) \otimes \rho_r \;\multimap\; \varphi \otimes (r \mapsto o_1 \mathbin{cc} o_2)\\
\mathtt{add},\mathtt{sub},\mathtt{and},\mathtt{or},\mathtt{xor},\mathtt{shl},\dots &:& \varphi \;\multimap\; \varphi \quad (\text{clobber})\\
\mathtt{mov},\ \mathtt{lea},\ \mathtt{movzx} &:& \text{flag-neutral: } \varphi \text{ not required}
\end{array}
```

While φ\_cc is outstanding, the type system forbids every flag-clobbering instruction. So nothing that destroys the comparison can be scheduled between cmp and jcc. A concrete consequence: zeroing a register with `xor r32, r32` needs φ free, so under a pending comparison the lowerer must use `mov r32, 0`.

### 4.4 The machine stack is statically bounded

Recursive calls take one of two forms, and anything else is a type error:

- **Tail calls** reuse the frame and compile to `jmp`.
- **Structural non-tail calls** use DESCEND and ASCEND (section 3.3). The continuation goes into the destructured cell, and the call becomes a descent loop followed by an ascent loop.

The call graph with these edges removed is acyclic, so the hardware stack is bounded by the heaviest acyclic path (Theorem 7.3). Tail-recursion modulo cons, as in `map`, is a third, cheaper case: it is just the destination `mem a` passed around a loop (section 5.6).

### 4.5 ALU folds as in-place Horner accumulation

For pack⟨w₁..w\_k⟩(e₁..e\_k), with e₁ most significant, the accumulator is the destination register, which is pinned:

```latex
\mathcal{D}[\![\mathsf{pack}\langle \bar w\rangle(e_1..e_k)]\!](\mathsf{reg}\ r)
\;=\; \mathcal{D}[\![e_1]\!](\mathsf{reg}\ r) \cdot \prod_{i=2}^{k} \Big( \mathtt{shl}\ r, w_i \cdot \mathtt{or}\ r_{|w_i|},\, \mathsf{opnd}(e_i) \Big)
```

Here r\_{|w|} is the subregister of width w (al, ax or rax). 8-bit and 16-bit writes preserve the upper bits of r, so they merge a field without disturbing the accumulator. Leading constant fields are folded at compile time.

Worked example, a 64-bit word laid out as tag:8 | kind:8 | x:16 | y:32. Here tag = 3, kind is in cl, x is homed in mem \[rdi+2\], and y is in edx, which is zero-extended by the x86-64 rule that a 32-bit write clears the upper half:

```asm
mov  eax, 0x300          ; tag<<8, constant-folded
or   al,  cl             ; kind
shl  rax, 16
or   ax,  word [rdi+2]   ; x, read directly from its home
shl  rax, 32
or   rax, rdx            ; y
```

The dilation factors are defined against a reference sequence:

```latex
\mathcal{D}_I \;=\; \frac{|\,\mathcal{D}[\![\mathsf{pack}(\bar e)]\!]\,|}{|\,\mathrm{fold}(\mathrm{Horner}(\bar e))\,|} \;=\; 1,
\qquad
\mathcal{D}_M \;=\; \frac{\#\text{memory operands emitted}}{\#\{e_i \text{ homed in memory}\} + [\delta = \mathsf{mem}]} \;=\; 1
```

Both equal 1 because the accumulator is pinned, so it cannot be spilled. Each memory-homed field is read exactly once as an `or` operand: it is linear, so it has exactly one use. And the result is stored at most once, only if δ = mem a.

What D\_I = 1 does *not* claim is absolute optimality; section 8 lists the encoding-level exceptions.

## 5. Translation calculus 𝒟⟦e⟧(δ)

Every term is compiled *into* the place its consumer wants it, so values are produced where they will be used. Moves arise only at genuine relocations: call boundaries, join points with differing homes, and permuting parallel moves.

### 5.1 Destinations and operands

```latex
\begin{array}{rcll}
\delta &::=& \mathsf{reg}\ r & \text{deliver the value into } r \ (\text{carries } \rho_r)\\
&\mid& \mathsf{mem}\ a & \text{deliver into location } a \ (\text{carries } \Diamond_a^{y})\\
&\mid& \mathsf{cond}(L_t, L_f) & \text{transfer control; deliver nothing}\\
&\mid& \mathsf{fields}(\delta_1..\delta_n) & \text{deliver a constructor's fields, unboxed}\\
&\mid& \mathsf{eff} & \text{evaluate for effect only}\\[4pt]
o &::=& r \mid [a] \mid \mathit{imm}_{32} & \text{x86-64 operand}
\end{array}
```

Notation: L\_f = • means "fall through". `jmp •` emits nothing.

### 5.2 Variables, literals, let

```latex
\begin{array}{lcl}
\mathcal{D}[\![x]\!](\mathsf{reg}\ r) &=& \varepsilon \ \text{ if } H(x) = \mathsf{reg}\ r; \quad \mathtt{mov}\ r, H(x) \ \text{ otherwise}\\
\mathcal{D}[\![x]\!](\mathsf{mem}\ a) &=& \varepsilon \ \text{ if } H(x) = \mathsf{mem}\ a \ (\textsf{Stay}), \text{ or } \Diamond_a^{x} \ (\textsf{Restore}, \S 6)\\
&& \mathtt{mov}\ [a], s \ \text{ if } H(x) = \mathsf{reg}\ s \ (\textsf{Store}, \S 6)\\
\mathcal{D}[\![x]\!](\delta) &=& \mathcal{D}[\![C(\bar u)]\!](\delta) \ \text{ if } H(x) = \mathsf{virt}(C, \bar u) \quad (\text{materialize on demand})\\[4pt]
\mathcal{D}[\![n]\!](\mathsf{reg}\ r) &=& \mathtt{xor}\ r_{32}, r_{32} \ \text{ if } n = 0 \wedge \varphi \in \Delta; \quad \mathtt{mov}\ r, n \ \text{ otherwise}\\[4pt]
\mathcal{D}[\![\mathsf{let}\ x = e_1\ \mathsf{in}\ e_2]\!](\delta) &=& \mathcal{D}[\![e_1]\!](\mathsf{home}(x)) \cdot \mathcal{D}[\![e_2]\!](\delta)\\
\mathsf{home}(x) &=& \begin{cases} \text{alias: } H(x) := H(y), \ \varepsilon & e_1 = y \ \ (\text{let is renaming})\\ \mathsf{virt}(C, \bar u), \ \mathcal{D}[\![\bar e]\!](\bar u) & e_1 = C_d(\bar e)\\ \mathsf{reg}\ r \text{ hinted by } x\text{'s use in } e_2 & \text{otherwise} \end{cases}
\end{array}
```

A side condition holds throughout: `mov r, H(x)` is emitted only when H(x) ≠ reg r. There is no rule that can emit `mov r, r`.

### 5.3 Conditionals under cond: jumping code with no flags at rest

```latex
\begin{array}{lcl}
\mathcal{D}[\![e_1 \bowtie e_2]\!](\mathsf{cond}(L_t, L_f)) &=& \mathcal{D}[\![e_1]\!](\mathsf{reg}\ r) \cdot \mathtt{cmp}\ r, \mathsf{opnd}(e_2) \cdot \mathtt{j}_{\bowtie}\ L_t \cdot \mathtt{jmp}\ L_f\\
\mathcal{D}[\![\neg e]\!](\mathsf{cond}(L_t, L_f)) &=& \mathcal{D}[\![e]\!](\mathsf{cond}(L_f, L_t))\\
\mathcal{D}[\![e_1 \mathbin{\&\&} e_2]\!](\mathsf{cond}(L_t, L_f)) &=& \mathcal{D}[\![e_1]\!](\mathsf{cond}(\bullet, L_f)) \cdot \mathcal{D}[\![e_2]\!](\mathsf{cond}(L_t, L_f))\\
\mathcal{D}[\![e_1 \mathbin{||} e_2]\!](\mathsf{cond}(L_t, L_f)) &=& \mathcal{D}[\![e_1]\!](\mathsf{cond}(L_t, L)) \cdot L{:} \cdot \mathcal{D}[\![e_2]\!](\mathsf{cond}(L_t, L_f))\\
\mathcal{D}[\![\mathsf{true}]\!](\mathsf{cond}(L_t, L_f)) &=& \mathtt{jmp}\ L_t \qquad \mathcal{D}[\![\mathsf{false}]\!](\mathsf{cond}(L_t, L_f)) = \mathtt{jmp}\ L_f\\
\mathcal{D}[\![x{:}\mathsf{Bool}]\!](\mathsf{cond}(L_t, L_f)) &=& \mathtt{test}\ r, r \cdot \mathtt{jnz}\ L_t \cdot \mathtt{jmp}\ L_f \quad (H(x) = \mathsf{reg}\ r)\\[4pt]
\mathcal{D}[\![\mathsf{if}\ c\ \mathsf{then}\ e_1\ \mathsf{else}\ e_2]\!](\delta) &=& \mathcal{D}[\![c]\!](\mathsf{cond}(\mathsf{entry}(e_1), \mathsf{entry}(e_2))) \cdot \mathsf{blk}(e_1, \delta) \cdot \mathsf{blk}(e_2, \delta) \cdot L_j{:}
\end{array}
```

Two auxiliary functions make the scheme trampoline-free. `entry(e)` gives the label at which e's code starts. For a Boolean literal under a cond destination, that label is the target itself:

```latex
\mathsf{entry}(\mathsf{true}) = L_t, \quad \mathsf{entry}(\mathsf{false}) = L_f, \quad \mathsf{entry}(e) = \text{fresh } L_e \text{ otherwise};
\qquad \mathsf{blk}(e, \delta) = \begin{cases} \varepsilon & e \in \{\mathsf{true}, \mathsf{false}\},\ \delta = \mathsf{cond}\\ L_e{:} \cdot \mathcal{D}[\![e]\!](\delta) \cdot \mathtt{jmp}\ L_j & \text{otherwise} \end{cases}
```

So `if c then true else false` under cond(L\_t, L\_f) compiles to exactly 𝒟⟦c⟧(cond(L\_t, L\_f)). The only rule that emits `setcc` is a comparison in *value* position:

```latex
\mathcal{D}[\![e_1 \bowtie e_2]\!](\mathsf{reg}\ r) \;=\; \mathcal{D}[\![e_1]\!](\mathsf{reg}\ s) \cdot \mathtt{cmp}\ s, \mathsf{opnd}(e_2) \cdot \mathtt{set}_{\bowtie}\ r_8 \cdot \mathtt{movzx}\ r_{32}, r_8
```

**Join points.** Both arms of an `if` or `match` receive the *same* δ, so the result needs no φ-copy at the join. Other live variables must agree on their homes at L\_j. The second arm is lowered with the first arm's exit homes as its target. Where homes differ, one parallel move (section 6.3) ends the second arm.

### 5.4 Constructors and matches

```latex
\begin{array}{lcl}
\mathcal{D}[\![C_d(e_1..e_n)]\!](\mathsf{fields}(\delta_1..\delta_n)) &=& \mathcal{D}[\![e_1]\!](\delta_1) \cdots \mathcal{D}[\![e_n]\!](\delta_n) \qquad (\textsf{Phantom})\\[4pt]
\mathcal{D}[\![C_d(e_1..e_n)]\!](\mathsf{reg}\ r) &=& \mathsf{pmove}\{\, [p{+}\mathit{off}_i] \leftarrow e_i \,\}_{i} \cdot \mathsf{tag}(p, C) \cdot \mathtt{mov}\ r, p^{?} \qquad (H(d) = \mathsf{hole}(p))\\[4pt]
\mathcal{D}[\![\mathsf{match}\ e\ \mathsf{with}\ \overline{C_j(d_j, \bar x_j) \Rightarrow e_j}]\!](\delta) &=&
\begin{cases}
\mathcal{D}[\![\bar e']\!](\mathsf{fields}(\bar u)) \cdot \mathcal{D}[\![e_k]\!](\delta)[\bar u/\bar x_k,\ d'/d_k] & e \equiv C_k{}_{d'}(\bar e') \text{ or } H(e) = \mathsf{virt}(C_k, \bar u)\\[4pt]
\mathcal{D}[\![e]\!](\mathsf{reg}\ p) \cdot \mathsf{dispatch}(p) \cdot \prod_j L_j{:}\ \mathcal{D}[\![e_j]\!](\delta)\big[\overline{\mathsf{mem}[p{+}\mathit{off}_i]}/\bar x_j,\ \mathsf{hole}(p)/d_j\big] & \text{otherwise}
\end{cases}
\end{array}
```

The rules work as follows:

- **Known constructor.** If the scrutinee is syntactically a constructor, or a let-bound variable with a `virt` home, the first case applies. The fields are delivered straight into the homes ū that the branch wants, and the branch's ◇ variable d\_k is renamed to d′. Nothing is written, so this is phantom construction (Theorem 7.4).
- **Unknown constructor.** `dispatch(p)` is `cmp byte [p], tagⱼ; jne …` (or a jump table). For a type with one nullary constructor it is `test p, p; jz L_Nil`, using the tagless layout in which all 16 bytes are payload.
- **Lazy field binding.** Fields are bound to their memory homes *lazily*: no load is emitted at the match. Each field is fetched only at its single use, directly into that use's destination or as a memory operand.
- **Reusing the cell.** d\_j is bound to hole(p): the cell is now a ◇ whose old contents remain readable (the stale-read window). `tag(p, C)` emits `mov byte [p], tag_C`, unless the cell already held tag C and is being rebuilt with the same constructor; then it is a self-move and STAY applies. `mov r, p` is omitted when the hint already chose p = r.
- **In-place arithmetic.** If H(x) = mem a and δ = mem a, then 𝒟⟦x ⊕ e⟧(mem a) = `op [a], opnd(e)`. That is one read-modify-write instruction: no load, no store-back.

### 5.5 Record update (derived, not primitive)

```latex
\{\, e \ \mathsf{with}\ f_k = v \,\} \;\triangleq\; \mathsf{match}\ e\ \mathsf{with}\ R(d, x_1..x_n) \Rightarrow R_d(x_1, .., x_{k-1}, v, x_{k+1}, .., x_n)
```

By the unknown-constructor rule, each x\_i is homed at mem \[p + off\_i\]. By the constructor rule, it is delivered to mem \[p + off\_i\], the same location. That is STAY, which emits nothing. The tag is unchanged, which is also STAY. Only field k emits code. Section 6 shows this is not an optimization but the only typable derivation.

### 5.6 Tail-recursion modulo cons is a destination

A recursive call in the last field of a constructor is compiled with destination mem \[p + off\_last\], and the destination is threaded through as a hidden argument. For an in-place increment over a list:

```latex
\mathsf{inc}(l) = \mathsf{match}\ l\ \mathsf{with}\ \mathsf{Nil} \Rightarrow \mathsf{Nil} \mid \mathsf{Cons}(d, h, t) \Rightarrow \mathsf{Cons}_d(h + 1,\ \mathsf{inc}(t))
```

```asm
inc:                        ; rdi = l (tagless cell: [p+0]=h, [p+8]=t)
    mov   rax, rdi          ; result is the root cell itself
.loop:
    test  rdi, rdi          ; dispatch: Nil = null
    jz    .done
    add   qword [rdi], 1    ; h+1 into mem[p+0], h homed at mem[p+0]: in-place RMW
    mov   rdi, [rdi+8]      ; evacuate t, leaving hole [p+8] with provenance t
    jmp   .loop             ; inc(t) delivers into mem[p+8]
.done:
    ret
```

There is no store to \[p+8\]. The recursive result is rebuilt in t's own cell, so its pointer value *is* t, or null if t was null. The destination hole \[p+8\] has provenance t, so RESTORE applies and the location reverts with zero instructions. A naive TRMC lowering would emit `mov [p+8], rdi` on every iteration; under section 6 that store is a type error.

### 5.7 Calls

```latex
\begin{array}{lcl}
\mathcal{D}[\![f(\bar e)]\!]_{\mathsf{tail}} &=& \mathsf{pmove}\{\, \mathsf{param}_i(f) \leftarrow e_i \,\} \cdot \mathtt{jmp}\ f\\
\mathcal{D}[\![f(\bar e)]\!](\delta) &=& \mathsf{evacuate}(\kappa_f) \cdot \mathsf{pmove}\{\, \mathsf{param}_i(f) \leftarrow e_i \,\} \cdot \mathtt{call}\ f \cdot \mathcal{D}[\![\mathtt{rax}]\!](\delta)
\end{array}
```

The callee's preferred result register is used as the hint for δ whenever δ is free to choose, so the final move is usually ε.

## 6. The linear mutation law

Plain linearity is **not enough** to forbid copying an unchanged field. The sequence `mov r, [a]; mov [a], r` is perfectly linear: it consumes the full location, produces a hole and a register, then consumes both to refill the location. The law needs one more piece of information, namely *what used to be there*. So every hole remembers its evacuee, and no rule writes the evacuee back.

### 6.1 Provenance words

Each hole carries a symbolic word ν describing the bits it last held:

```latex
\nu \;::=\; x \ (\text{scalar variable}) \;\mid\; \mathsf{addr}(c) \ (\text{pointer to cell } c) \;\mid\; \mathsf{null} \;\mid\; n \ (\text{literal}) \;\mid\; \bot \ (\text{never written})
```

The word of a value is defined structurally. For a scalar variable it is its name. For a constructor built in place into cell c it is addr(c), *whatever its fields are*, because the pointer bits do not change. Moves change a value's home but never its word.

### 6.2 The four location rules

```latex
\begin{array}{ll}
\textsf{Evac} &
\dfrac{H(y) = \mathsf{mem}\ a \qquad \rho_r \in \Delta}
{(a \mapsto y) \otimes \rho_r \;\leadsto\; \Diamond_a^{\,\nu(y)} \otimes (r \mapsto y) \;\triangleright\; \mathtt{mov}\ r, [a]}
\\[18pt]
\textsf{Store} &
\dfrac{H(v) = \mathsf{reg}\ r \qquad \nu(v) \neq \mu}
{\Diamond_a^{\,\mu} \otimes (r \mapsto v) \;\leadsto\; (a \mapsto v) \otimes \rho_r \;\triangleright\; \mathtt{mov}\ [a], r}
\\[18pt]
\textsf{Stay} &
\dfrac{H(v) = \mathsf{mem}\ a \qquad \delta = \mathsf{mem}\ a}
{(a \mapsto v) \;\leadsto\; (a \mapsto v) \;\triangleright\; \varepsilon}
\\[18pt]
\textsf{Restore} &
\dfrac{\nu(v) = \mu \qquad \delta = \mathsf{mem}\ a}
{\Diamond_a^{\,\mu} \otimes (\text{any home of } v) \;\leadsto\; (a \mapsto v) \otimes (\text{that home freed}) \;\triangleright\; \varepsilon}
\end{array}
```

RESTORE is sound because of the stale-read invariant (Lemma 6.2): a hole ◇ₐ^μ still physically holds μ until a STORE overwrites it. Delivering μ back to a therefore costs nothing.

**Theorem 6.1 (unchanged-field copies are untypable).** No lowering derivation emits a pair `mov r, [a]` … `mov [a], r'` in which the stored value has the same provenance word as the evacuated one, through any chain of intermediate registers or slots. *Proof:* the load must come from EVAC, producing ◇ₐ^{ν(y)}. Intermediate moves (register to register, SPILL to a slot) change homes but never the word ν. Filling a with a value of word ν(y) is covered by STORE only if ν(v) ≠ ν(y), which is false. The one rule that matches is RESTORE, and it emits nothing. ∎

**Lemma 6.2 (stale-read invariant).** If a value y has home mem a and a later instruction reads \[a\] as y, then no STORE to a lies between y's binding and that read. *Proof:* a STORE to a needs ◇ₐ, and a hole at a can only exist if y has already been evacuated (EVAC changes H(y) away from mem a). After that, every read of y goes through its new home. ∎

### 6.3 In-place reconstruction as a parallel move

Rebuilding constructor C′ into the cell p of a just-matched C produces a set of moves M = {\[p + off\_i\] ← ν\_i}. Lowering it takes three steps:

1. **Drop trivial moves**, where ν\_i is the word currently in \[p + off\_i\]. These are STAY or RESTORE and emit nothing. The tag move is trivial when C′ = C.
2. **Emit the acyclic moves** in reverse topological order of the read-before-write graph. Lemma 6.2 guarantees that every source is read before its location is overwritten.
3. **Break each cycle.** A cycle of k memory locations costs k loads and k stores using two scratch register tokens, the minimum on x86-64, which has no memory-to-memory `mov`. A locked `xchg` with memory is excluded on cost.

**Corollary 6.3 (record update).** For 𝒞⟦{ r with f = v }⟧, every x\_i with i ≠ f is trivial by step 1. The emitted code is therefore exactly one of three forms:

- `mov [p + off(f)], opnd(v)` when v is already an operand (register or imm32), which is the directive's law verbatim;
- 𝒟⟦v⟧(mem \[p + off(f)\]) when v must be computed, so the computation lands directly in the field;
- `op [p + off(f)], opnd(e)` when v = x\_f ⊕ e, a single read-modify-write.

In all three, no unchanged field is touched, and Theorem 6.1 makes any derivation that touches one ill-typed rather than merely suboptimal.

## 7. Invariant proofs

Five theorems carry the specification. Each proof is by induction over machine steps or over lowering derivations, and each relies only on rules stated above.

### Theorem 7.1 (conservation)

In every reachable configuration, the cells partition exactly:

```latex
F \;\uplus\; \{c_2,c_3,c_4,c_5\}\!\setminus\!\{\bot\} \;\uplus\; \biguplus_S \big(\mathrm{dom}(\sigma_S) \cup K_S\big) \;=\; \Lambda_{\mathsf{cell}}
```

At every lowering point, the registers partition exactly into free tokens, register homes and pinned destination registers:

```latex
\Delta_{\mathsf{reg}} \;\uplus\; \{\, r \mid H(x) = \mathsf{reg}\ r \,\} \;\uplus\; \mathsf{pinned} \;=\; \Lambda_{\mathsf{reg}}
```

*Proof.* Base case: at `_start`, F = Λ\_cell and Δ\_reg = Λ\_reg. Step: check each rule in sections 3.2, 3.3, 4 and 6.

- INGEST and EGRESS move ℓ between F and a channel; read and write return the same ℓ.
- PASS moves ℓ from c\_in to c\_out.
- PUSH moves ℓ from c\_in to K\_S, and m from F to c\_out.
- POP moves ℓ from c\_in to c\_out, and f from K\_S to F.
- DROP moves ℓ from c\_in to F.
- EXPAND moves m₂..m\_k from F to the c₅ queue.
- MATCH, CONS, DESCEND and ASCEND change σ(ℓ) but not which component owns ℓ.
- At register level, SPILL, EVAC, STORE and RESTORE are written as token-for-token exchanges.

No rule has a location only on its right side. ∎

### Theorem 7.2 (O(D) occupancy; receiving from F never fails)

Let D\_t be the current nesting depth, L\_max the longest lexeme, s the number of cells per frame (a language constant covering name chains and type annotations), E the largest expansion of a single lowering rule, and W the ingest batch size. Then:

```latex
|\Lambda_{\mathsf{cell}} \setminus F| \;\le\; B(D_t) \;=\; \underbrace{W + 4 + (E{-}1) + L_{\max}}_{\text{constant}} \;+\; 3s \cdot D_t \;+\; s \cdot G
```

If C ≥ B(D\_max) + 1 and the program has at most G\_max global declarations, then every PUSH and EXPAND premise F ≠ ∅ holds. So the code for receiving from F has no failure branch.

*Proof.* Check each stage's invariant:

- **Lex** holds at most L\_max cells: one pending lexeme before packing.
- **Parse, Check and Lower** each push a frame only on `open` and pop it only on the matching `close` (Theorem 2.1 guarantees the matching). So |K\_S| equals the number of constructs S has opened but not yet closed, which is at most D\_t at S's own position in the stream. Each frame has at most s cells.
- **Check** additionally holds the global signature table: G entries of at most s cells each.
- **Channels** hold at most 4 cells, and EXPAND queues at most E − 1 more.
- **Egress** returns each cell in the same step.

Summing gives B(D\_t). The parser's DEPTH rule makes D\_t ≤ D\_max, so C − B(D\_t) ≥ 1 whenever a premise needs a cell. ∎

**Corollary.** Peak working memory is independent of source length N. A one-gigabyte file with nesting depth 40 runs in the same C cells as a one-line file.

### Theorem 7.3 (static stack bound)

Let G₀ be the call graph with tail-call edges and structural-recursion edges removed. Then G₀ is a DAG, and the hardware stack never exceeds:

```latex
S_{\max} \;=\; \max_{\pi \,\in\, \mathrm{paths}(G_0)} \;\sum_{f \in \pi} \big(8F_f + 8\big)
```

*Proof.* Every recursive edge, including edges inside a mutually recursive group, is either a tail call (`jmp`, the frame is reused) or a structural call compiled to descent and ascent loops whose frames live in destructured cells (DESCEND). Any other recursive edge is rejected by the typing in section 4.4. So every cycle of the call graph is cut, and each function's frame size F\_f is static. ∎

### Theorem 7.4 (phantom construction)

If C\_d(ē) is consumed by a match, either directly or through a let-bound variable, then the emitted code is 𝒟⟦ē⟧(fields ū) · 𝒟⟦e\_k⟧(δ). It contains no instruction whose operand is home(d) or the tag of C. So the construction emits zero instructions and writes zero bytes.

*Proof.* The first case of the match rule applies. Its output consists of the field computations, which the source program requires in any case, followed by the branch. The ◇ variable d\_k is substituted by d′ as a compile-time renaming, so the register or hole token passes through unchanged. In the let-bound case, H(x) = virt(C, ū), and linearity guarantees x has exactly one use. That use is the match, so deferring the construction neither duplicates the code of ē nor reorders any effect. In a non-linear language this deferral would be unsound or would duplicate work, which is why linearity matters here. The cell d remains a hole and passes on to its next consumer. ∎

### Theorem 7.5 (no identity morphisms)

Define three kinds of identity morphism in an emitted sequence:

- **(I1)** `mov r, r`.
- **(I2) Mem→Reg→Mem:** a load from a, then a store to a of a value with the same provenance word.
- **(I3) Reg→Mem→Reg:** a store of v to a, then a load from a while some register still holds v, owned and not overwritten.

The translation emits none of them within a procedure.

*Proof.*

- **(I1)** The variable rule's move premise requires H(x) ≠ reg r, and parallel-move resolution drops self-moves.
- **(I2)** This is Theorem 6.1.
- **(I3)** A store comes from SPILL or from STORE into a constructor field.
  - *After SPILL*, ρ\_s goes to the destination that demanded it, and that destination writes s before any later use of v. By home uniqueness, no register holds v at its reload.
  - *After STORE*, v has been linearly consumed into the constructed value. That value has exactly one consumer. A match is impossible, since Theorem 7.4 would have applied and no STORE would exist. The remaining consumers are a call or return, where registers are clobbered or out of scope, or a TRMC loop back-edge, where the next iteration reads a different cell.

So no reload ever finds v intact in a register. ∎

**Scope of 7.5.** The guarantee is intraprocedural. Suppose a convention keeps r callee-saved, v is stored into a cell passed to f, and f returns that cell to be matched. Then the field reload happens while r may still hold v, and only an interprocedural analysis would notice. An implementation should also include a linear-time *validator* over the emitted code. It would track a provenance word per register and location, and reject I1–I3 as a check independent of the lowering rules.

## 8. Where the directive's premises must be weakened

The architecture holds, but ten of the directive's statements are stronger than anything a finite machine can prove. The table gives the provable replacement for each.

| Directive's claim | Why it fails as stated | Provable replacement |
| --- | --- | --- |
| "Bit-slots are never created or destroyed, only permuted." | Every `mov` overwrites and erases bits; a physical register file destroys information constantly. | *Locations* are conserved. Contents are not. LFPL's ◇ is a location capability, and scalars are erasable (section 1). |
| Arena exhaustion is "permanently impossible". | Finite C and unbounded input depth cannot both hold. | Exhaustion at runtime is impossible. Inputs deeper than D\_max are rejected by a grammar rule that depends only on the input text (Theorem 7.2). |
| Peak memory is O(D). | The checker needs every global signature, and binders retain their names. | O(D + G), with constants depending on L\_max and the frame size s. O(D) holds only if globals count toward nesting, or if signatures are re-read from disk: a second pass costs 2× time, and rescanning per lookup costs O(N·G) time. |
| The parser "emits a verified syntax tree" in O(D). | A materialized tree has O(N) nodes. | It emits a certified Dyck *stream* of the tree; the tree is never materialized (Theorem 2.1). |
| The pipeline is pure LFPL end to end. | LFPL defines exactly the non-size-increasing functions. Assembly output is longer than source, so the compiler, as a function, is not expressible in LFPL. | Each stage is a non-size-increasing *transducer* per step. Lower's EXPAND takes up to E − 1 cells from F, and Egress returns them in the same step: output size is converted into time. This is the one place a stage takes cells it was not handed. |
| Zero identity morphisms means no Reg→Mem→Reg traffic. | With more than 15 simultaneously live scalars, or values live across a call that clobbers their registers, memory traffic is necessary. | No *redundant* round trip within a procedure (Theorem 7.5). Spills are real relocations. |
| D\_I ≡ 1 and D\_M ≡ 1. | These are only meaningful relative to some reference sequence. | Equal to 1 against the folded Horner reference. The absolute minimum can be lower: for w ∈ {1,2,3}, `lea r, [o + r·2^w]` fuses shift and or. It can also be higher: `or r32, m32` zero-extends, so a 32-bit memory field needs one extra load, and an 8- or 16-bit merge can cost a merge micro-op on some cores. |
| 16 GPRs serve as tokens. | rsp is the stack pointer. | 15 register tokens, or 14 if rbp is kept as a frame pointer. |
| Caller arguments live at \[rbp + offset\]. | The System V ABI passes the first six integer arguments in rdi, rsi, rdx, rcx, r8 and r9. | \[rbp + 16 + 8i\] homes arise only for further arguments, or under the internal convention. The OS boundary uses the syscall ABI, which clobbers rcx and r11. |
| Polynomial time comes for free. | Hindley–Milner unification needs a substitution store that grows with N. | The checker must be first-order and monomorphic, with annotations on binders. Inference beyond local synthesis breaks the O(D) bound. |

**The real cost of zero allocation is expressiveness.** Hofmann's LFPL does not bound the stack. Here, a non-tail recursion whose live state across the call exceeds 15 bytes is a type error, *frame does not fit in ◇*. The program must be restructured, for example by destructuring two cells per step, which doubles the frame budget to 30 bytes, or by rewriting the recursion with an explicit zipper. FP² makes the same trade for its fully in-place fragment.

## 9. Related work and references

The novel pieces here are the ring framing, provenance-indexed holes, and treating flags and frame slots as LFPL tokens. Everything else extends established work:

- **LFPL and in-place update.** Hofmann, *Linear types and non-size-increasing polynomial time computation*, LICS 1999. Hofmann, *A type system for bounded space and functional in-place update*, Nordic Journal of Computing 7(4):258–289, 2000; extended abstract at ESOP 2000, LNCS 1782. This is the original "malloc()-free C" compilation. [Hofmann's resource-bounded papers](https://www.dcs.ed.ac.uk/home/resbnd/papers/martin.html)
- **Usage aspects beyond pure linearity.** Aspinall & Hofmann, *Another type system for in-place update*, ESOP 2002. Aspinall & Compagnoni, *Heap-bounded assembly language*, J. Automated Reasoning 2003, the closest prior work to typing at the machine level. [Edinburgh resource-bounded project](https://www.dcs.ed.ac.uk/home/resbnd/)
- **Fully in-place, constant-stack execution.** Lorenzen, Leijen & Swierstra, *FP²: Fully in-Place Functional Programming*, ICFP 2023, PACMPL 7(ICFP):198. Their FIP calculus proves allocation-free, constant-stack execution for a linear fragment, which is the practical counterpart of sections 3.3 and 4.4. [MSR technical report](https://www.microsoft.com/en-us/research/publication/fp2-fully-in-place-functional-programming/)
- **Continuations in the data.** Schorr & Waite, *An efficient machine-independent procedure for garbage collection in various list structures*, CACM 1967 (pointer reversal, used by DESCEND and ASCEND).
- **Destinations.** Minamide, *A functional representation of data structures with a hole*, POPL 1998. Bour, Clément & Scherer, *Tail Modulo Cons*, JFLA 2021.
- **Register demand and jumping code.** Sethi & Ullman, *The generation of optimal code for arithmetic expressions*, JACM 1970. Appel, *Modern Compiler Implementation*, 1998, on the conditional-expression translation that section 5.3 sharpens to remove trampolines.

Entries without links are cited from memory, not from pages opened for this document.
