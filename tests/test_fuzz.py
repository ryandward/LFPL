"""Differential fuzzing: native code against an independent Python semantics.

Each generator builds a program text and, alongside it, computes the answer from
the language's meaning, without any part of the compiler. Translation correctness
is judged by running the binary. Every program also runs with the cell partition and
Theorem 7.2's bound checked, through the redundancy validator, and through the symbolic
check of meaning, which has to agree with the binary without running it.

A declared limit is a legal outcome for a program that really hits it, so each is
counted by rule and bounded: a generator whose programs mostly get rejected would
prove nothing. Anything else that stops the compiler fails the campaign.

Set LFPL_FUZZ to raise the number of programs per generator.
"""
import collections
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lfplo import build_and_run, compile_program  # noqa: E402
from src import rtlower  # noqa: E402
from src.rtlower import Limit  # noqa: E402
from src.spans import parts_of  # noqa: E402
from src.symcheck import check_translation  # noqa: E402
from src.validate import validate  # noqa: E402
from tests.reference.lang import without_annotations  # noqa: E402

COUNT = int(os.environ.get("LFPL_FUZZ", "60"))
SEED = int(os.environ.get("LFPL_FUZZ_SEED", "0"))
WIDTHS = {8: "Int8", 16: "Int16", 32: "Int32", 64: "Int64"}


def compile_rt(source: bytes) -> str:
    def on_step(ring) -> None:
        passed, check = ring.stages[2], ring.stages[3]
        in_use = ring.store.C - len(ring.F)
        depth = max(stage.depth() for stage in ring.stages)
        g_cells = (len(check.G.cells) + len(check.TA.cells) + len(check.TF.cells) + len(check.TT.cells)
                   + passed.table_cells())
        limit = rtlower.occupancy_bound(depth, g_cells)
        if in_use > limit:
            raise AssertionError(f"Theorem 7.2: {in_use} cells at depth {depth}, bound {limit}")
        if ring.counts["egress"] % 97 == 0:
            ring.check_conservation()

    return compile_program(source, on_step=on_step)


class Scalars:
    """Expressions of one width. `env` maps a name to its value."""

    def __init__(self, rng: random.Random, bits: int):
        self.rng = rng
        self.bits = bits
        self.mask = (1 << bits) - 1
        self.names = 0

    def literal(self) -> tuple[str, int]:
        value = self.rng.choice((0, 1, 2, 7, self.mask, self.mask // 2,
                                 self.rng.randrange(self.mask + 1)))
        return str(value), value

    def fresh(self) -> str:
        self.names += 1
        return f"v{self.names}"

    def value(self, depth: int, env: dict) -> tuple[str, int]:
        roll = self.rng.random()
        if depth <= 0 or roll < 0.2:
            if env and self.rng.random() < 0.7:
                name = self.rng.choice(sorted(env))
                return name, env[name]
            return self.literal()
        if roll < 0.55:
            return self.binop(depth, env)
        if roll < 0.62 and self.bits == 64:
            return self.pack(depth, env)
        if roll < 0.8:
            name = self.fresh()
            bound_text, bound = self.value(depth - 1, env)
            inner = dict(env)
            inner[name] = bound
            body_text, result = self.value(depth - 1, inner)
            return f"let {name} = {bound_text} in {body_text}", result
        cond_text, cond = self.boolean(depth - 1, env)
        then_text, then_value = self.value(depth - 1, env)
        else_text, else_value = self.value(depth - 1, env)
        return (f"if {cond_text} then {then_text} else {else_text}",
                then_value if cond else else_value)

    def pack(self, depth: int, env: dict) -> tuple[str, int]:
        """Horner over a 64-bit word; a field keeps only its low bits."""
        widths: list[int] = []
        while not widths or (sum(widths) < 64 and self.rng.random() < 0.7):
            room = [w for w in (8, 16, 32) if sum(widths) + w <= 64]
            if not room:
                break
            widths.append(self.rng.choice(room))
        texts, result = [], 0
        for width in widths:
            text, value = self.value(depth - 1, env)
            texts.append(text)
            result = result << width | value & (1 << width) - 1
        codes = ",".join(str(w) for w in widths)
        return f"pack<{codes}>({', '.join(texts)})", result & self.mask

    def binop(self, depth: int, env: dict) -> tuple[str, int]:
        op = self.rng.choice(("+", "-", "*", "&", "|", "^", "<<", ">>"))
        typed = sorted(name for name in env if not name.startswith("v"))
        if self.bits < 64 and typed:
            # Arithmetic on two literals is Int64 by rule, and a let-bound literal
            # is still a literal, so a narrow operation takes a parameter or field
            # on the left to fix its width.
            left_text = self.rng.choice(typed)
            left = env[left_text]
        else:
            left_text, left = self.value(depth - 1, env)
        if op in ("<<", ">>"):
            # Section 4.4: 0 once the count reaches the width. The counts around the
            # width and the ones x86 would mask are the ones worth trying.
            if self.rng.random() < 0.5:
                count = self.rng.choice((self.rng.randrange(self.bits), self.bits - 1, self.bits,
                                         self.bits + 1, 32, 64, 70, 255, 1 << 32))
                right_text = str(count)
            else:
                right_text, count = self.value(depth - 1, env)
            if count >= self.bits:
                result = 0
            elif op == "<<":
                result = (left << count) & self.mask
            else:
                result = left >> count
            return f"{op}({left_text}, {right_text})", result
        right_text, right = self.value(depth - 1, env)
        table = {"+": left + right, "-": left - right, "*": left * right,
                 "&": left & right, "|": left | right, "^": left ^ right}
        return f"{op}({left_text}, {right_text})", table[op] & self.mask

    def boolean(self, depth: int, env: dict) -> tuple[str, bool]:
        roll = self.rng.random()
        if depth <= 0 or roll < 0.5:
            op = self.rng.choice(("<", "<=", ">", ">=", "==", "!="))
            left_text, left = self.value(max(depth - 1, 0), env)
            right_text, right = self.value(max(depth - 1, 0), env)
            table = {"<": left < right, "<=": left <= right, ">": left > right,
                     ">=": left >= right, "==": left == right, "!=": left != right}
            return f"{op}({left_text}, {right_text})", table[op]
        if roll < 0.6:
            value = self.rng.random() < 0.5
            return ("true" if value else "false"), value
        if roll < 0.7:
            text, value = self.boolean(depth - 1, env)
            return f"!({text})", not value
        if roll < 0.85:
            left_text, left = self.boolean(depth - 1, env)
            right_text, right = self.boolean(depth - 1, env)
            return f"&&({left_text}, {right_text})", left and right
        left_text, left = self.boolean(depth - 1, env)
        right_text, right = self.boolean(depth - 1, env)
        return f"||({left_text}, {right_text})", left or right


def widen(text: str, bits: int) -> str:
    return text if bits == 64 else f"pack<{bits}>({text})"


def scalar_program(rng: random.Random) -> tuple[bytes, int]:
    bits = rng.choice((8, 16, 32, 64, 64))
    gen = Scalars(rng, bits)
    # Four parameters, so that a shift's count has to work around a live value in rcx.
    params = {name: rng.choice((rng.randrange(gen.mask + 1), rng.randrange(min(gen.mask, 80) + 1)))
              for name in ("a", "b", "c", "d")}
    text, value = gen.value(rng.randrange(2, 6), params)
    ty = WIDTHS[bits]
    source = (f"fn f(a: {ty}, b: {ty}, c: {ty}, d: {ty}) : Int64 = {widen(text, bits)};\n"
              f"fn main() : Int64 = f({params['a']}, {params['b']}, {params['c']}, {params['d']});\n")
    return source.encode(), value


def call_program(rng: random.Random) -> tuple[bytes, int]:
    """Calls force evacuation of live values across clobbered registers."""
    # A fixed helper keeps the mirror honest: its meaning is written here once.
    helper_source = "fn h(x: Int64, y: Int64) : Int64 = if <(x, y) then +(*(x, 3), y) else -(x, y);"
    mask = (1 << 64) - 1

    def h(x: int, y: int) -> int:
        return (x * 3 + y) & mask if x < y else (x - y) & mask

    env = {name: rng.randrange(1 << 16) for name in ("a", "b", "c")}
    lines = []
    for i in range(rng.randrange(1, 7)):
        name = f"t{i}"
        left, right = rng.choice(sorted(env)), rng.choice(sorted(env))
        lines.append(f"let {name} = h({left}, {right}) in ")
        env[name] = h(env[left], env[right])
    a, b = rng.choice(sorted(env)), rng.choice(sorted(env))
    tail = f"+(h({a}, 1), {b})"
    result = (h(env[a], 1) + env[b]) & mask
    return (f"{helper_source}\nfn f(a: Int64, b: Int64, c: Int64) : Int64 = "
            f"{''.join(lines)}{tail};\n"
            f"fn main() : Int64 = f({env['a']}, {env['b']}, {env['c']});\n").encode(), result


def loop_program(rng: random.Random) -> tuple[bytes, int]:
    """Tail recursion: parallel moves over parameter registers. The counter is a list of
    cells, because rule R3 needs a structure that shrinks, and it has a type of its own
    because R3 does not let a function walk the reserve."""
    mask = (1 << 64) - 1
    x, y = rng.randrange(1 << 20), rng.randrange(1 << 20)
    kind = rng.randrange(3)
    if kind == 0:
        step = "y, x"  # a register cycle

        def advance(x, y):
            return y, x
    elif kind == 1:
        step = "+(x, y), x"

        def advance(x, y):
            return (x + y) & mask, x
    else:
        step = "+(y, 1), *(x, 2)"

        def advance(x, y):
            return (y + 1) & mask, (x * 2) & mask
    if rng.random() < 0.2:
        cells, n = [], 1024  # every cell in the binary, so a real loop
        call = f"go(r0, {x}, {y}, End)"
    else:
        n = rng.randrange(1, 9)
        cells = [f"c{i}" for i in range(n)]
        chain = "End"
        for cell in reversed(cells):
            chain = f"More@{cell}({chain})"
        call = f"go({chain}, {x}, {y}, r{n})"
    for _ in range(n):
        x, y = advance(x, y)
    return (f"data Fuel = End | More(next: Fuel);\n"
            f"data Res = Fail | Done(value: Int64, rest: Fuel);\n"
            f"fn go(n: Fuel, x: Int64, y: Int64, back: Fuel) : Res = match n with\n"
            f"  | End => match back with End => Fail | More(c, rest) => Done@c(x, rest) end\n"
            f"  | More(d, r) => go(r, {step}, More@d(back)) end;\n"
            f"fn main(r0: Fuel) : Res = {take_cells(cells, call)};\n").encode(), x


RECORD = """data Free = End | More(next: Free);
data Res = Fail | Done(value: Int64, rest: Free);
record R { a: Int64, b: Int32, c: Int16, e: Int8 };
fn rd(p: R, free: Free) : Res = match p with R(d, a, b, c, e) =>
  Done@d(+(a, +(*(pack<32>(b), 3), +(*(pack<16>(c), 5), *(pack<8>(e), 7)))), free) end;
"""
_FIELDS = (("a", 64), ("b", 32), ("c", 16), ("e", 8))


def record_program(rng: random.Random) -> tuple[bytes, int]:
    """Field updates and rebuilds: STAY, RESTORE, RMW, and parallel moves."""
    state = {name: rng.randrange(1 << bits) for name, bits in _FIELDS}
    initial = dict(state)
    functions = []
    calls = "P"
    for step in range(rng.randrange(1, 5)):
        name = f"u{step}"
        if rng.random() < 0.5:
            field, bits = rng.choice(_FIELDS)
            gen = Scalars(rng, bits)
            text, value = gen.value(rng.randrange(0, 3), {field: state[field]})
            state[field] = value
            functions.append(f"fn {name}(p: R) : R @p = {{ p with {field} = {text} }};")
        else:
            parts = []
            updated = dict(state)
            for field, bits in _FIELDS:
                choice = rng.random()
                if choice < 0.5:
                    parts.append(field)
                else:
                    gen = Scalars(rng, bits)
                    env = {field: state[field]}
                    text, value = gen.value(rng.randrange(0, 3), env)
                    parts.append(text)
                    updated[field] = value
            state = updated
            functions.append(f"fn {name}(p: R) : R @p = match p with R(d, a, b, c, e) => "
                             f"R@d({', '.join(parts)}) end;")
        calls = f"{name}({calls})"
    expected = (state["a"] + state["b"] * 3 + state["c"] * 5 + state["e"] * 7) & ((1 << 64) - 1)
    build = (f"R@c({initial['a']}, {initial['b']}, {initial['c']}, {initial['e']})")
    main = (f"fn main(free: Free) : Res = match free with End => Fail | More(c, rest) => "
            f"rd({calls.replace('P', build)}, rest) end;")
    return (RECORD + "\n".join(functions) + "\n" + main + "\n").encode(), expected


def wire_record_program(rng: random.Random) -> tuple[bytes, int]:
    """The same programs with their fields in network order. What they mean has not changed, so the answer has not."""
    source, expected = record_program(rng)
    return source.replace(b"record R { a: Int64, b: Int32, c: Int16, e: Int8 };",
                          b"record R { a: Be64, b: Be32, c: Be16, e: Int8 };"), expected


SPAN_FIELDS = (("Int8", 8), ("Int16", 16), ("Int32", 32), ("Int64", 64), ("Be16", 16), ("Be32", 32), ("Be64", 64))


def span_program(rng: random.Random) -> tuple[bytes, int]:
    """One record across as many cells as its fields take, built from loose cells, rebuilt, and read. Some fields stay
    and some change, in every cell and at every width, and a field can be anywhere in its cell, or waiting on a call.
    Half the rebuilds are updates, some with a let in front of them taking the name of a field they leave alone."""
    mask = (1 << 64) - 1
    while True:
        fields = [(f"f{i}", *rng.choice(SPAN_FIELDS)) for i in range(rng.randrange(3, 13))]
        parts = parts_of([bits // 8 for _name, _type, bits in fields])
        if len(parts) > 1:
            break
    names = [name for name, _type, _bits in fields]
    state = {name: rng.randrange(1 << bits) for name, _type, bits in fields}
    initial = dict(state)
    functions = ["fn h(x: Int64, y: Int64) : Int64 = +(*(x, 3), y);"]
    cells = [f"c{i}" for i in range(len(parts))]
    calls = f"R@({', '.join(cells)})({', '.join(str(initial[name]) for name in names)})"
    for step in range(rng.randrange(1, 4)):
        updated = dict(state)
        arguments = []
        named = []
        left = []
        for name, _type, bits in fields:
            if rng.random() < 0.6:
                arguments.append(name)
                left.append(name)
            else:
                text, value = Scalars(rng, bits).value(rng.randrange(0, 3), {name: state[name]})
                arguments.append(text)
                named.append(f"{name} = {text}")
                updated[name] = value
        state = updated
        built = f"R@d({', '.join(arguments)})"
        if rng.random() < 0.5:
            built = (" ".join(f"let {name} = 0 in" for name in left if rng.random() < 0.5)
                     + f" R@d {{ {', '.join(named)} }}")
        functions.append(f"fn u{step}(p: R) : R @p = match p with R(d, {', '.join(names)}) => {built} end;")
        calls = f"u{step}({calls})"
    weights = {name: rng.randrange(1, 1 << 16) for name in names}
    total = "0"
    expected = 0
    for name, _type, bits in fields:
        widened = name if bits == 64 else f"pack<{bits}>({name})"
        total = f"+(*({widened}, {weights[name]}), {total})"
        expected = (expected + state[name] * weights[name]) & mask
    fields_text = ", ".join(f"{name}: {type_name}" for name, type_name, _bits in fields)
    source = (f"data Free = End | More(next: Free);\ndata Res = Fail | Done(value: Int64, rest: Free);\n"
              f"record R {{ {fields_text} }};\n" + "\n".join(functions) + "\n"
              f"fn rd(p: R, free: Free) : Res = match p with R(d, {', '.join(names)}) => "
              f"let (c, rest) = peel(d) in Done@c({total}, More@rest(free)) end;\n"
              f"fn main(r0: Free) : Res = {take_cells(cells, f'rd({calls}, r{len(cells)})')};\n")
    return source.encode(), expected


LISTS = """data Free = End | More(next: Free);
data Res = Fail | Done(value: Int64, rest: Free);
data List = Nil | Cons(h: Int64, t: List);
fn addk(l: List, k: Int64) : List @l = match l with
  | Cons(d, h, t) => Cons@d(+(h, k), addk(t, k)) | Nil => Nil end;
fn neg(l: List) : List @l = match l with
  | Nil => Nil | Cons(d, h, t) => Cons@d(-(0, h), neg(t)) end;
fn g(x: Int64, k: Int64) : Int64 = +(*(x, 2), k);
fn dbl(l: List, k: Int64) : List @l = match l with
  | Cons(d, h, t) => Cons@d(g(h, k), dbl(t, k)) | Nil => Nil end;
fn rev(l: List, acc: List) : List = match l with
  | Nil => acc | Cons(d, h, t) => rev(t, Cons@d(h, acc)) end;
fn weigh(l: List, n: Int64, acc: Int64, free: Free) : Res = match l with
  | Cons(d, h, t) => weigh(t, +(n, 1), +(acc, *(h, n)), More@d(free))
  | Nil => match free with End => Fail | More(d, rest) => Done@d(acc, rest) end end;
"""


def take_cells(cells: list, body: str) -> str:
    """Fetch these cells from free list r0, leaving the rest in r<len(cells)>. Running
    out must hand back the cells already taken, because nothing can be dropped."""
    for i in reversed(range(len(cells))):
        held = "End"
        for cell in cells[1:i]:
            held = f"More@{cell}({held})"
        fail = "Fail" if i == 0 else f"Done@{cells[0]}(0, {held})"
        body = f"match r{i} with End => {fail} | More({cells[i]}, r{i + 1}) => {body} end"
    return body


def list_program(rng: random.Random) -> tuple[bytes, int]:
    """TRMC, in-place reversal, and giving every cell back."""
    mask = (1 << 64) - 1
    items = [rng.randrange(1 << 32) for _ in range(rng.randrange(1, 8))]
    chain = "Nil"
    cells = [f"c{i}" for i in range(len(items))]
    for cell, item in zip(reversed(cells), reversed(items)):
        chain = f"Cons@{cell}({item}, {chain})"
    expression = chain
    state = list(items)
    for _ in range(rng.randrange(0, 5)):
        op = rng.randrange(4)
        k = rng.randrange(1 << 16)
        if op == 0:
            expression = f"addk({expression}, {k})"
            state = [(x + k) & mask for x in state]
        elif op == 1:
            expression = f"neg({expression})"
            state = [(-x) & mask for x in state]
        elif op == 2:
            expression = f"dbl({expression}, {k})"  # a call inside the loop
            state = [(x * 2 + k) & mask for x in state]
        else:
            expression = f"rev({expression}, Nil)"
            state = state[::-1]
    expected = sum(x * (i + 1) for i, x in enumerate(state)) & mask
    body = take_cells(cells, f"weigh({expression}, 1, 0, r{len(cells)})")
    return (LISTS + f"fn main(r0: Free) : Res = {body};\n").encode(), expected


def pressure_program(rng: random.Random) -> tuple[bytes, int]:
    """More live values than registers, across calls and conditional arms: SPILL,
    join normalization, frame thunks, and evacuation with nowhere free to go."""
    mask = (1 << 64) - 1
    params = {name: rng.randrange(1 << 16) for name in ("a", "b", "c")}
    env = dict(params)
    lines = []

    def clobber(x: int, y: int) -> int:
        return (x * 3 + y) & mask

    for i in range(rng.randrange(10, 26)):
        name = f"t{i}"
        left, right = rng.choice(sorted(env)), rng.choice(sorted(env))
        if rng.random() < 0.25:
            lines.append(f"let {name} = h({left}, {right}) in ")
            env[name] = clobber(env[left], env[right])
        elif rng.random() < 0.3:
            pick, other = rng.choice(sorted(env)), rng.choice(sorted(env))
            when = env[left] < env[right]
            then_value = (env[pick] + 1) & mask
            else_value = clobber(env[other], 2)
            lines.append(f"let {name} = if <({left}, {right}) then +({pick}, 1) "
                         f"else h({other}, 2) in ")
            env[name] = then_value if when else else_value
        else:
            lines.append(f"let {name} = +({left}, {right}) in ")
            env[name] = (env[left] + env[right]) & mask
    used = rng.sample(sorted(env), min(4, len(env)))
    tail = used[0]
    result = env[used[0]]
    for name in used[1:]:
        tail = f"+({name}, {tail})"
        result = (env[name] + result) & mask
    return (f"fn h(x: Int64, y: Int64) : Int64 = +(*(x, 3), y);\n"
            f"fn f(a: Int64, b: Int64, c: Int64) : Int64 = {''.join(lines)}{tail};\n"
            f"fn main() : Int64 = f({params['a']}, {params['b']}, {params['c']});\n").encode(), result


TAGGED = """data Free = End | More(next: Free);
data Res = Fail | Done(value: Int64, rest: Free);
data S = A(x: Int32) | B(x: Int32, y: Int16) | C(w: Int8) | D;
fn rd(s: S, free: Free) : Res = match s with
  | A(d, x) => Done@d(+(1000000, pack<32>(x)), free)
  | B(d, x, y) => Done@d(+(2000000, +(pack<32>(x), *(pack<16>(y), 3))), free)
  | C(d, w) => Done@d(+(3000000, pack<8>(w)), free)
  | D(d) => Done@d(4000000, free)
  end;
"""


def tagged_program(rng: random.Random) -> tuple[bytes, int]:
    """Tagged cells rebuilt as other constructors: tag writes and STAY on tags,
    and offsets that overlap between layouts, so a stale read would show."""
    def apply(state: tuple, arms: dict) -> tuple:
        return arms[state[0]](state)

    state = rng.choice((("A", rng.randrange(1 << 20)),
                        ("B", rng.randrange(1 << 20), rng.randrange(1 << 16)),
                        ("C", rng.randrange(256)), ("D",)))
    start = {"A": lambda s: f"A@c({s[1]})", "B": lambda s: f"B@c({s[1]}, {s[2]})",
             "C": lambda s: f"C@c({s[1]})", "D": lambda s: "D@c()"}[state[0]](state)
    functions, calls = [], start
    for step in range(rng.randrange(1, 5)):
        k = rng.randrange(1 << 12)
        table = rng.choice((
            ("A(d, x) => B@d(x, 7)", "B(d, x, y) => A@d(+(x, pack32y))", "C(d, w) => C@d(+(w, 1))",
             "D(d) => A@d(9)"),
            ("A(d, x) => A@d(+(x, K))", "B(d, x, y) => B@d(x, +(y, 1))", "C(d, w) => D@d()",
             "D(d) => C@d(3)"),
            ("A(d, x) => D@d()", "B(d, x, y) => B@d(+(x, 1), y)", "C(d, w) => A@d(11)",
             "D(d) => B@d(1, 2)"),
        ))
        mask32, mask16, mask8 = (1 << 32) - 1, (1 << 16) - 1, 255
        if table[0].startswith("A(d, x) => B"):
            new = {"A": ("B", state[1], 7) if state[0] == "A" else None,
                   "B": ("A", state[1]) if state[0] == "B" else None,
                   "C": ("C", (state[1] + 1) & mask8) if state[0] == "C" else None,
                   "D": ("A", 9)}[state[0]]
            arm_b = "B(d, x, y) => A@d(x)"
            arms = [table[0], arm_b, table[2], table[3]]
        elif table[0].startswith("A(d, x) => A"):
            new = {"A": ("A", (state[1] + k) & mask32) if state[0] == "A" else None,
                   "B": ("B", state[1], (state[2] + 1) & mask16) if state[0] == "B" else None,
                   "C": ("D",), "D": ("C", 3)}[state[0]]
            arms = [table[0].replace("K", str(k)), table[1], table[2], table[3]]
        else:
            new = {"A": ("D",),
                   "B": ("B", (state[1] + 1) & mask32, state[2]) if state[0] == "B" else None,
                   "C": ("A", 11), "D": ("B", 1, 2)}[state[0]]
            arms = list(table)
        rng.shuffle(arms)
        functions.append(f"fn s{step}(v: S) : S @v = match v with " + " | ".join(arms) + " end;")
        calls = f"s{step}({calls})"
        state = new
    expected = {"A": lambda s: 1000000 + s[1],
                "B": lambda s: 2000000 + s[1] + s[2] * 3,
                "C": lambda s: 3000000 + s[1], "D": lambda s: 4000000}[state[0]](state)
    main = ("fn main(free: Free) : Res = match free with End => Fail | More(c, rest) => "
            f"rd({calls}, rest) end;")
    return (TAGGED + "\n".join(functions) + "\n" + main + "\n").encode(), expected


CELLS = """data Free = End | More(next: Free);
data Res = Fail | Done(value: Int64, rest: Free);
record Q { a: Int64, b: Int64 };
fn h(x: Int64, y: Int64) : Int64 = +(*(x, 3), y);
fn rd(q: Q, free: Free) : Res = match q with Q(d, a, b) => Done@d(+(*(a, 5), b), free) end;
"""


def cell_call_program(rng: random.Random) -> tuple[bytes, int]:
    """Field scalars read after calls, so the cell pointer itself must survive
    evacuation or a spill while its fields are still homed in memory."""
    mask = (1 << 64) - 1

    def h(x: int, y: int) -> int:
        return (x * 3 + y) & mask

    a, b = rng.randrange(1 << 16), rng.randrange(1 << 16)
    functions, calls = [], f"Q@c({a}, {b})"
    for step in range(rng.randrange(1, 4)):
        env = {"a": a, "b": b}
        lines = []
        for i in range(rng.randrange(0, 18)):
            name = f"t{i}"
            x, y = rng.choice(sorted(env)), rng.choice(sorted(env))
            if rng.random() < 0.35:
                lines.append(f"let {name} = h({x}, {y}) in ")
                env[name] = h(env[x], env[y])
            else:
                lines.append(f"let {name} = +({x}, {y}) in ")
                env[name] = (env[x] + env[y]) & mask
        names = sorted(env)
        first, second = rng.choice(names), rng.choice(names)
        extra = rng.choice(names)
        field_a = rng.choice((first, f"+({first}, {extra})"))
        value_a = env[first] if field_a == first else (env[first] + env[extra]) & mask
        field_b = rng.choice((second, f"h({second}, {extra})"))
        value_b = env[second] if field_b == second else h(env[second], env[extra])
        functions.append(f"fn s{step}(q: Q) : Q @q = match q with Q(d, a, b) => "
                         f"{''.join(lines)}Q@d({field_a}, {field_b}) end;")
        calls = f"s{step}({calls})"
        a, b = value_a, value_b
    main = ("fn main(free: Free) : Res = match free with End => Fail | More(c, rest) => "
            f"rd({calls}, rest) end;")
    expected = (a * 5 + b) & mask
    return (CELLS + "\n".join(functions) + "\n" + main + "\n").encode(), expected


TREES = """data Free = End | More(next: Free);
data Res = Fail | Done(value: Int64, rest: Free);
data Pair;
data T = Leaf | Node(v: Int64, kids: Pair);
record Pair { l: T, r: T };
data Stack = Empty | Push(t: T, s: Stack);
fn swapkids(k: Pair) : Pair @k = match k with Pair(e, l, r) => Pair@e(r, l) end;
fn mirror(t: T) : T @t = match t with
  | Leaf => Leaf
  | Node(d, v, k) => Node@d(v, swapkids(k)) end;
fn bump(t: T, n: Int64) : T @t = match t with
  | Leaf => Leaf
  | Node(d, v, k) => Node@d(+(v, n), k) end;
fn left(t: T, n: Int64) : T @t = match t with
  | Leaf => Leaf
  | Node(d, v, k) => match k with Pair(e, l, r) =>
      Node@d(v, Pair@e(bump(l, n), r)) end end;
fn rotate(t: T) : T = match t with
  | Leaf => Leaf
  | Node(d, v, k) => match k with Pair(e, l, r) => match l with
      | Leaf => Node@d(v, Pair@e(Leaf, r))
      | Node(d2, v2, k2) => Node@d2(+(v2, v), Pair@e(Node@d(1, k2), r))
    end end end;
fn finish(acc: Int64, free: Free) : Res = match free with
  | End => Fail | More(c, rest) => Done@c(acc, rest) end;
fn walk(t: T, stack: Stack, acc: Int64, free: Free) : Res = match t with
  | Node(d, v, k) => match k with Pair(e, l, r) =>
      walk(l, Push@e(r, stack), +(*(acc, 3), v), More@d(free)) end
  | Leaf => match stack with
      | Empty => finish(acc, free)
      | Push(d, next, rest) => walk(next, rest, acc, More@d(free)) end
  end;
"""


def tree_program(rng: random.Random) -> tuple[bytes, int]:
    """Linear fields in nested cells: a matched field becomes a cell pointer, and
    constructors nest, so a pointer built in place flows into another field. The
    reader walks any tree in constant stack, with a worklist made of its own cells."""
    mask = (1 << 64) - 1

    def grow(depth: int):
        if depth == 0 or rng.random() < 0.3:
            return None
        return (rng.randrange(1 << 10), grow(depth - 1), grow(depth - 1))

    def bumped(t, n: int):
        return None if t is None else ((t[0] + n) & mask, t[1], t[2])

    def fold(t, acc: int) -> int:
        if t is None:
            return acc
        return fold(t[2], fold(t[1], (acc * 3 + t[0]) & mask))

    # Each cell costs a match and an arm in main, so this stays within D_max.
    state = (rng.randrange(1 << 10), grow(2), grow(2))
    cells: list[str] = []

    def text(t) -> str:
        if t is None:
            return "Leaf"
        node, pair = f"c{len(cells)}", f"c{len(cells) + 1}"
        cells.extend((node, pair))
        first = text(t[1])
        second = text(t[2])
        return f"Node@{node}({t[0]}, Pair@{pair}({first}, {second}))"

    expression = text(state)
    for _ in range(rng.randrange(0, 6)):
        op = rng.randrange(4)
        n = rng.randrange(1 << 10)
        if op == 0:
            expression = f"mirror({expression})"
            state = (state[0], state[2], state[1])
        elif op == 1:
            expression = f"bump({expression}, {n})"
            state = bumped(state, n)
        elif op == 2:
            expression = f"left({expression}, {n})"
            state = (state[0], bumped(state[1], n), state[2])
        else:
            expression = f"rotate({expression})"
            v, first, second = state
            if first is not None:
                state = ((first[0] + v) & mask, (1, first[1], first[2]), second)
    body = take_cells(cells, f"walk({expression}, Empty, 0, r{len(cells)})")
    return (TREES + f"fn main(r0: Free) : Res = {body};\n").encode(), fold(state, 0)


DIAMONDS = """data Free = End | More(next: Free);
data Res = Fail | Done(value: Int64, rest: Free);
data List = Nil | Cons(h: Int64, t: List);
fn mk(c: Diamond, k: Int64, t: List) : List = Cons@c(k, t);
fn one(c: Diamond, k: Int64) : List = Cons@c(k, Nil);
fn rev(l: List, acc: List) : List = match l with
  | Nil => acc | Cons(d, h, t) => rev(t, Cons@d(h, acc)) end;
fn weigh(l: List, n: Int64, acc: Int64, free: Free) : Res = match l with
  | Cons(d, h, t) => weigh(t, +(n, 1), +(acc, *(h, n)), More@d(free))
  | Nil => match free with End => Fail | More(d, rest) => Done@d(acc, rest) end end;
"""


def diamond_call_program(rng: random.Random) -> tuple[bytes, int]:
    """A matched cell's diamond passed to a call, so the callee writes a cell whose
    fields this function may still want. Reading a scalar after that without a copy has to
    give the right answer, because the right-to-left pass says the scalar is wanted."""
    mask = (1 << 64) - 1
    k = rng.randrange(1 << 16)
    templates = [
        # A copy made before the call survives it.
        (f"let s = +(h, 0) in match mk(d, {k}, t) with "
         f"| Cons(e, h2, t2) => Cons@e(+(s, h2), t2) | Nil => Nil end",
         lambda h, tail: [(h + k) & mask] + tail),
        # A scalar passed as an argument is read before the callee runs.
        (f"mk(d, +(h, {k}), t)", lambda h, tail: [(h + k) & mask] + tail),
        (f"let r = mk(d, +(h, {k}), t) in r", lambda h, tail: [(h + k) & mask] + tail),
        (f"let c = d in let r = mk(c, h, t) in r", lambda h, tail: [h] + tail),
        # A linear field used after the call has to leave the cell first.
        (f"rev(one(d, {k}), t)", lambda h, tail: [k] + tail),
        (f"let m = one(d, h) in rev(m, t)", lambda h, tail: [h] + tail),
        # A read an enclosing frame deferred has to survive the inner call.
        (f"let m = one(d, {k}) in rev(m, t)", lambda h, tail: [k] + tail),
    ]
    items = [rng.randrange(1 << 32) for _ in range(rng.randrange(1, 6))]
    if rng.random() < 0.15:
        # No copy: h has to leave its cell ahead of the call.
        text = (f"match mk(d, {k}, t) with "
                f"| Cons(e, h2, t2) => Cons@e(+(h, h2), t2) | Nil => Nil end")
        action = lambda h, tail: [(h + k) & mask] + tail
    else:
        text, action = rng.choice(templates)
    chain = "Nil"
    cells = [f"c{i}" for i in range(len(items))]
    for cell, item in zip(reversed(cells), reversed(items)):
        chain = f"Cons@{cell}({item}, {chain})"
    state = action(items[0], items[1:])
    expected = sum(x * (i + 1) for i, x in enumerate(state)) & mask
    source = (DIAMONDS + f"fn f(l: List) : List = match l with "
              f"| Cons(d, h, t) => {text} | Nil => Nil end;\n"
              f"fn main(r0: Free) : Res = "
              f"{take_cells(cells, f'weigh(f({chain}), 1, 0, r{len(cells)})')};\n")
    return source.encode(), expected


def tagged_if_program(rng: random.Random) -> tuple[bytes, int]:
    """Tagged constructors built in both arms of an `if`, so each arm has to write or
    keep the tag for itself, whatever the arm before it did."""
    mask32, mask16, mask8 = (1 << 32) - 1, (1 << 16) - 1, 255
    k32, k16, k8 = rng.randrange(1 << 20), rng.randrange(1 << 16), rng.randrange(256)
    table = {
        "A(d, x)": [("A@d(x)", lambda s: ("A", s[1])),
                    (f"A@d(+(x, {k32}))", lambda s: ("A", (s[1] + k32) & mask32)),
                    (f"B@d(x, {k16})", lambda s: ("B", s[1], k16)),
                    (f"B@d(+(x, 1), {k16})", lambda s: ("B", (s[1] + 1) & mask32, k16)),
                    (f"C@d({k8})", lambda s: ("C", k8)), ("D@d()", lambda s: ("D",))],
        "B(d, x, y)": [("B@d(x, y)", lambda s: s),
                       ("B@d(x, +(y, 1))", lambda s: ("B", s[1], (s[2] + 1) & mask16)),
                       ("A@d(x)", lambda s: ("A", s[1])),
                       (f"A@d(+(x, {k32}))", lambda s: ("A", (s[1] + k32) & mask32)),
                       (f"C@d({k8})", lambda s: ("C", k8)), ("D@d()", lambda s: ("D",))],
        "C(d, w)": [("C@d(w)", lambda s: s),
                    ("C@d(+(w, 1))", lambda s: ("C", (s[1] + 1) & mask8)),
                    (f"A@d({k32})", lambda s: ("A", k32)),
                    (f"B@d({k32}, {k16})", lambda s: ("B", k32, k16)), ("D@d()", lambda s: ("D",))],
        "D(d)": [("D@d()", lambda s: s), (f"A@d({k32})", lambda s: ("A", k32)),
                 (f"B@d({k32}, {k16})", lambda s: ("B", k32, k16)),
                 (f"C@d({k8})", lambda s: ("C", k8))],
    }
    state = rng.choice((("A", rng.randrange(1 << 20)),
                        ("B", rng.randrange(1 << 20), rng.randrange(1 << 16)),
                        ("C", rng.randrange(256)), ("D",)))
    start = {"A": lambda s: f"A@c({s[1]})", "B": lambda s: f"B@c({s[1]}, {s[2]})",
             "C": lambda s: f"C@c({s[1]})", "D": lambda s: "D@c()"}[state[0]](state)
    functions, calls = [], start
    for step in range(rng.randrange(1, 4)):
        p, q = rng.random() < 0.5, rng.random() < 0.5
        arms, chosen = [], {}
        for pattern, options in table.items():
            first, second, third = (rng.choice(options) for _ in range(3))
            if rng.random() < 0.3:
                text = f"if p then {first[0]} else if q then {second[0]} else {third[0]}"
                taken = first if p else second if q else third
            else:
                text = f"if p then {first[0]} else {second[0]}"
                taken = first if p else second
            arms.append(f"{pattern} => {text}")
            chosen[pattern[0]] = taken[1]
        rng.shuffle(arms)
        functions.append(f"fn s{step}(v: S, p: Bool, q: Bool) : S @v = match v with "
                         + " | ".join(arms) + " end;")
        calls = f"s{step}({calls}, {str(p).lower()}, {str(q).lower()})"
        state = chosen[state[0]](state)
    expected = {"A": lambda s: 1000000 + s[1], "B": lambda s: 2000000 + s[1] + s[2] * 3,
                "C": lambda s: 3000000 + s[1], "D": lambda s: 4000000}[state[0]](state)
    main = ("fn main(free: Free) : Res = match free with End => Fail | More(c, rest) => "
            f"rd({calls}, rest) end;")
    return (TAGGED + "\n".join(functions) + "\n" + main + "\n").encode(), expected


class Differential(unittest.TestCase):
    def _campaign(self, generator, seed: int, *, min_compiled: float = 0.7) -> None:
        rng = random.Random(seed + 1000 * SEED)
        compiled = 0
        limits: collections.Counter = collections.Counter()
        with tempfile.TemporaryDirectory() as workdir:
            for index in range(COUNT):
                source, expected = generator(rng)
                try:
                    asm = compile_rt(source)
                except Limit as failure:
                    limits[str(failure).split(":")[0]] += 1
                    continue
                self.assertEqual(validate(asm), [], source.decode())
                self.assertEqual([str(p) for p in check_translation(source, asm)], [],
                                 f"#{index}\n{source.decode()}\n{asm}")
                got = build_and_run(asm, workdir)
                self.assertEqual(got, expected, f"#{index}\n{source.decode()}\n{asm}")
                # The pass has to find every `@` that was written.
                self.assertEqual(compile_rt(without_annotations(source)), asm, f"#{index}\n{source.decode()}")
                compiled += 1
        print(f"{generator.__name__}: {compiled} ran, stopped at limits {dict(limits)}",
              file=sys.stderr)
        self.assertGreaterEqual(compiled, COUNT * min_compiled,
                                f"only {compiled} compiled; limits {dict(limits)}")

    def test_scalar_expressions(self):
        self._campaign(scalar_program, 1)

    def test_calls_across_live_values(self):
        self._campaign(call_program, 2)

    def test_tail_recursion(self):
        self._campaign(loop_program, 3)

    def test_records(self):
        self._campaign(record_program, 4)

    def test_records_in_network_order(self):
        self._campaign(wire_record_program, 12)

    def test_records_that_span(self):
        self._campaign(span_program, 13)

    def test_lists(self):
        self._campaign(list_program, 5)

    def test_register_pressure(self):
        self._campaign(pressure_program, 6)

    def test_tagged_cells(self):
        self._campaign(tagged_program, 7)

    def test_cells_across_calls(self):
        self._campaign(cell_call_program, 8)

    def test_nested_cells(self):
        self._campaign(tree_program, 9)

    def test_diamonds_passed_to_calls(self):
        self._campaign(diamond_call_program, 10)

    def test_tagged_constructors_in_both_arms(self):
        self._campaign(tagged_if_program, 11)


if __name__ == "__main__":
    unittest.main()
