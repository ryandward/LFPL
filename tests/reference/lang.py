"""A small model of LFPL-O source: a syntax tree, a printer, and a reference interpreter, written
from the language's meaning and sharing nothing with the compiler. gen.py and hgen.py build programs
from this tree, so each one comes with the answer the interpreter gives, and the binary is judged
against that. An independent reviewer wrote these to find what tests/test_fuzz.py does not generate,
and they did: see tests/test_generated.py.
"""
import re


MASK = {8: 0xFF, 16: 0xFFFF, 32: 0xFFFFFFFF, 64: (1 << 64) - 1}


def show(e):
    k = e[0]
    if k == 'lit':
        return str(e[1])
    if k == 'bool':
        return 'true' if e[1] else 'false'
    if k == 'var':
        return e[1]
    if k == 'bin':
        return f"{e[1]}({show(e[2])}, {show(e[3])})"
    if k == 'cmp':
        return f"{e[1]}({show(e[2])}, {show(e[3])})"
    if k == 'and':
        return f"&&({show(e[1])}, {show(e[2])})"
    if k == 'or':
        return f"||({show(e[1])}, {show(e[2])})"
    if k == 'not':
        return f"!({show(e[1])})"
    if k == 'if':
        return f"if {show(e[1])} then {show(e[2])} else {show(e[3])}"
    if k == 'let':
        return f"let {e[1]} = {show(e[2])} in {show(e[3])}"
    if k == 'lett':
        return f"let ({', '.join(e[1])}) = {show(e[2])} in {show(e[3])}"
    if k == 'tup':
        return '(' + ', '.join(show(x) for x in e[1]) + ')'
    if k == 'call':
        return f"{e[1]}({', '.join(show(x) for x in e[2])})"
    if k == 'ctor':
        return f"{e[1]}@{e[2]}({', '.join(show(x) for x in e[3])})"
    if k == 'null':
        return e[1]
    if k == 'match':
        arms = []
        for c, binders, body in e[2]:
            head = c if not binders else f"{c}({', '.join(binders)})"
            arms.append(f"{head} => {show(body)}")
        return f"match {show(e[1])} with " + " | ".join(arms) + " end"
    if k == 'with':
        return f"{{ {show(e[1])} with {e[2]} = {show(e[3])} }}"
    if k == 'pack':
        return f"pack<{','.join(str(w) for w in e[1])}>({', '.join(show(x) for x in e[2])})"
    if k == 'idx':
        return f"idx({show(e[1])}, {e[2]})"
    if k == 'get':
        return f"get({show(e[1])}, {show(e[2])})"
    raise ValueError(k)


class Interp:
    """Evaluates a program by what it means. `fields` maps a record or constructor to its field names."""

    def __init__(self, fns, fields=None):
        self.fns = fns
        self.fields = fields or {}
        self.steps = 0

    def call(self, name, args):
        params, body = self.fns[name]
        env = dict(zip(params, args))
        return self.ev(body, env)

    def ev(self, e, env):
        self.steps += 1
        k = e[0]
        if k == 'lit':
            return e[1]
        if k == 'bool':
            return e[1]
        if k == 'var':
            return env[e[1]]
        if k == 'bin':
            _, op, a, b, w = e
            x, y = self.ev(a, env), self.ev(b, env)
            m = MASK[w]
            if op in ('<<', '>>'):
                x &= m
                if y >= w:
                    return 0
                return (x << y) & m if op == '<<' else x >> y
            x &= m
            y &= m
            return {'+': x + y, '-': x - y, '*': x * y, '&': x & y, '|': x | y, '^': x ^ y}[op] & m
        if k == 'cmp':
            _, op, a, b = e
            x, y = self.ev(a, env), self.ev(b, env)
            return {'<': x < y, '<=': x <= y, '>': x > y, '>=': x >= y, '==': x == y, '!=': x != y}[op]
        if k == 'and':
            return self.ev(e[1], env) and self.ev(e[2], env)
        if k == 'or':
            return self.ev(e[1], env) or self.ev(e[2], env)
        if k == 'not':
            return not self.ev(e[1], env)
        if k == 'if':
            return self.ev(e[2], env) if self.ev(e[1], env) else self.ev(e[3], env)
        if k == 'let':
            inner = dict(env)
            inner[e[1]] = self.ev(e[2], env)
            return self.ev(e[3], inner)
        if k == 'lett':
            values = self.ev(e[2], env)
            inner = dict(env)
            for n, v in zip(e[1], values):
                inner[n] = v
            return self.ev(e[3], inner)
        if k == 'tup':
            return [self.ev(x, env) for x in e[1]]
        if k == 'call':
            return self.call(e[1], [self.ev(x, env) for x in e[2]])
        if k == 'ctor':
            return (e[1],) + tuple(self.ev(x, env) for x in e[3])
        if k == 'null':
            return (e[1],)
        if k == 'match':
            v = self.ev(e[1], env)
            for c, binders, body in e[2]:
                if c == v[0]:
                    inner = dict(env)
                    if binders:
                        inner[binders[0]] = ('#',)
                        for n, f in zip(binders[1:], v[1:]):
                            inner[n] = f
                    return self.ev(body, inner)
            raise ValueError(f"no arm for {v[0]}")
        if k == 'with':
            v = self.ev(e[1], env)
            names = self.fields[v[0]]
            at = names.index(e[2])
            inner = dict(env)
            inner[e[2]] = v[at + 1]
            new = self.ev(e[3], inner)
            return v[:at + 1] + (new,) + v[at + 2:]
        if k == 'pack':
            _, widths, args = e
            acc = 0
            for w, a in zip(widths, args):
                acc = ((acc << w) | (self.ev(a, env) & MASK[w])) & MASK[64]
            return acc
        if k == 'idx':
            return self.ev(e[1], env) & (e[2] - 1)
        if k == 'get':
            arr = self.ev(e[1], env)
            return [arr, arr[self.ev(e[2], env)]]
        raise ValueError(k)


def without_annotations(source: bytes) -> bytes:
    """The same program with every written `@k` taken out of its signatures: the right-to-left pass has to
    find each of them, so what this compiles to has to be what the original compiles to."""
    return re.sub(rb"(\) : [^=;]*?) @\w+ =", rb"\1 =", source)
