"""Random handlers with a reference meaning: state arrays, helpers, tuples, spills, several outputs.
Every datagram's expected output is computed by lang.Interp from the same AST the source is printed from."""
import random
import struct

from .lang import Interp, show, MASK
from .gen import Gen, SCAL, INTS, BITS, SRC, parse_helpers, params_text, ret_text, type_text, Skip, Program

FMT = {'i8': 'B', 'i16': 'H', 'i32': 'I', 'i64': 'Q', 'bool': 'B'}
WIRE = {'i16': 'Be16', 'i32': 'Be32', 'i64': 'Be64'}  # the same integer, in network order on the wire
SIZE = {'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8, 'bool': 1}
PRELUDE = """data Fuel = Zed | Suc(prev: Fuel);
fn id8(x: Int8) : Int8 = x;
fn id16(x: Int16) : Int16 = x;
fn id32(x: Int32) : Int32 = x;
fn id64(x: Int64) : Int64 = x;
fn idb(x: Bool) : Bool = x;
fn hh(x: Int64, y: Int64) : Int64 = +(*(x, 3), y);
"""


def layout_fits(types, start):
    off = start
    for t in types:
        size = 8 if t == 'Bytes' else SIZE[t]
        off = (off + size - 1) // size * size
        if off + size > 16:
            return False
        off += size
    return True


class H:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.seed = seed
        r = self.r
        # item
        self.has_body = r.random() < 0.6
        # Up to five fields usually fit one cell, and more take as many as they take.
        n = r.randrange(0 if self.has_body else 1, 6 if r.random() < 0.7 else 13)
        self.item = [(f"i{k}", r.choice(SCAL)) for k in range(n)]
        self.wire = {name for name, t in self.item if t in WIRE and r.random() < 0.4}
        self.spans = not layout_fits([t for _n, t in self.item] + (['Bytes'] if self.has_body else []), 0)
        # state
        self.states = []
        for k in range(r.randrange(0, 5)):
            length = 1 << r.randrange(0, 7)
            element = r.choice(SCAL)
            self.states.append((f"s{k}", length, element))
        self.init = {}
        for name, length, element in self.states:
            values = []
            for _ in range(length):
                if element == 'bool':
                    values.append(r.randrange(2))
                else:
                    values.append(r.randrange(MASK[BITS[element]] + 1))
            self.init[name] = values
        # outputs
        self.outs = []
        count = r.randrange(1, 5)
        for k in range(count):
            for _try in range(50):
                sends = r.random() < 0.6
                fields = [(f"o{k}f{j}", r.choice(SCAL)) for j in range(r.randrange(0, 4))]
                layout = (['i8'] if sends else []) + [t for _n, t in fields] + (['Bytes'] if self.has_body else [])
                if layout_fits(layout, 1 if count != 2 else 1):
                    break
            else:
                sends, fields = False, []
            if count == 2 and not sends and not fields and not self.has_body:
                fields = [(f"o{k}f0", 'i8')]  # a tagless null constructor takes no cell, so give it a field
            self.wire |= {name for name, t in fields if t in WIRE and r.random() < 0.4}
            self.outs.append((f"Ou{k}", sends, fields))
        self.g = Gen(r, spill=0.2)
        self.defs = []
        self.nsel = 0

    def helper_fns(self):
        p = Program(self.r)
        p.g = self.g
        for i in range(self.r.randrange(0, 4)):
            p.fn_scalar(f"hs{i}")
        for i in range(self.r.randrange(0, 3)):
            p.fn_tuple(f"ht{i}")
        self.defs = p.defs

    def build(self):
        g = self.g
        r = self.r
        self.helper_fns()
        # the body: thread state arrays through get, compute scalars, then a decision tree of outputs
        env = {}
        d = 'dg'
        binder_names = [n for n, _t in self.item]
        for n, t in self.item:
            env[n] = t
        arrays = {}
        lin_names = {}
        lets = []
        for name, length, element in self.states:
            lin_names[name] = name
        for _ in range(r.randrange(0, 8)):
            kind = r.random()
            if kind < 0.45 and self.states:
                name, length, element = r.choice(self.states)
                cur = lin_names[name]
                newa, val = g.fresh('a'), g.fresh('v')
                index = g.scalar(r.choice(INTS), env, r.randrange(0, 3))[0]
                lets.append(('lett', [newa, val], ('get', ('var', cur), ('idx', index, length))))
                lin_names[name] = newa
                env[val] = element
            elif kind < 0.7:
                tys = [r.choice(SCAL) for _ in range(r.randrange(2, 4))]
                te, acts = g.tuple_expr(tys, [], env, r.randrange(0, 3))
                names = [g.fresh() for _ in tys]
                lets.append(('lett', names, te))
                for n, a in zip(names, acts):
                    env[n] = a
            else:
                n = g.fresh()
                ty = r.choice(SCAL)
                val, act = g.scalar(ty, env, r.randrange(0, 3))
                lets.append(('let', n, val))
                env[n] = act
        body = self.decide(env, lin_names, d, r.randrange(0, 3))
        for head in reversed(lets):
            body = (head[0], head[1], head[2], body)
        if self.spans:  # one of its cells for the output, and the others go back
            body = ('lett', ['dc', 'dr'], ('call', 'peel', [('var', d)]), body)
        pattern = [d] + binder_names + (['bd'] if self.has_body else [])
        self.body = ('match', ('var', 'm'), [('Hd', pattern, body)])

    def decide(self, env, lin_names, d, depth):
        g = self.g
        r = self.r
        if depth > 0 and r.random() < 0.7:
            c = g.boolean(env, 1)[0]
            return ('if', c, self.decide(env, lin_names, d, depth - 1), self.decide(env, lin_names, d, depth - 1))
        ctor, sends, fields = r.choice(self.outs)
        args = []
        if sends:
            args.append(('lit', r.randrange(8)) if r.random() < 0.7 else ('bin', '&', g.scalar('i8', env, 0, typed=True)[0], ('lit', 7), 8))
        for _n, t in fields:
            args.append(g.scalar(t, env, r.randrange(0, 2), big=True)[0])
        if self.has_body:
            args.append(('var', 'bd'))
        state = [('var', lin_names[n]) for n, _l, _e in self.states]
        if self.spans:
            return ('tup', state + [('ctor', ctor, 'dc', args), ('ctor', 'More', 'dr', [('var', 'fr')])])
        return ('tup', state + [('ctor', ctor, d, args), ('var', 'fr')])

    def source_type(self, name, t):
        return WIRE[t] if name in self.wire else SRC[t]

    def packed(self, name, t, value):
        return struct.pack(('>' if name in self.wire else '<') + FMT[t], value)

    def text(self):
        out = [PRELUDE]
        if self.has_body:
            out.append("data Bytes;")
        fields = ', '.join(f"{n}: {self.source_type(n, t)}" for n, t in self.item)
        if self.has_body:
            fields = (fields + ', ' if fields else '') + 'bd: Bytes'
        out.append(f"record Hd {{ {fields} }};")
        ctors = []
        for ctor, sends, fs in self.outs:
            parts = (['port: Int8'] if sends else []) + [f"{n}: {self.source_type(n, t)}" for n, t in fs] + (['bd: Bytes'] if self.has_body else [])
            ctors.append(f"{ctor}({', '.join(parts)})" if parts else ctor)
        out.append("data Out = " + " | ".join(ctors) + ";")
        for name, length, element in self.states:
            out.append(f"state {name} : Array[{length}] {SRC[element]};")
        for name, params, ret, at, body in self.defs:
            out.append(f"fn {name}({params_text(params)}) : {ret_text(ret)} = {show(body)};")
        params = [f"{n}: Array[{l}] {SRC[e]}" for n, l, e in self.states] + ["m: Hd", "fr: Free"]
        rets = [f"Array[{l}] {SRC[e]}" for _n, l, e in self.states] + ["Out", "Free"]
        out.append(f"handler hh0({', '.join(params)}) : ({', '.join(rets)}) = {show(self.body)};")
        return "\n".join(out) + "\n"

    def state_bytes(self):
        res = {}
        for name, length, element in self.states:
            res[name] = b''.join(struct.pack('<' + FMT[element], v) for v in self.init[name])
        return res

    def interp(self):
        fns = {name: ([n for n, _t in params], body) for name, params, _r, _a, body in self.defs}
        fns['hh0'] = ([n for n, _l, _e in self.states] + ['m', 'fr'], self.body)
        interp = Interp(fns, {})
        native = parse_helpers()
        native['peel'] = lambda _cells: [('#',), ('#',)]
        orig = interp.call
        interp.call = lambda name, args: native[name](*args) if name in native else orig(name, args)
        return interp

    def datagram(self):
        r = self.r
        vals = []
        wire = b''
        for _n, t in self.item:
            if t == 'bool':
                v = r.randrange(2)
            else:
                m = MASK[BITS[t]]
                v = r.choice((0, 1, 2, 7, m, m // 2, r.randrange(m + 1), r.randrange(m + 1)))
            vals.append(v)
            wire += self.packed(_n, t, v)
        body = b''
        if self.has_body:
            length = r.choice((0, 1, 6, 7, 8, 14, 15, r.randrange(50)))
            body = bytes(r.randrange(256) for _ in range(length))
        return wire + body, vals, body

    def expect(self, interp, vals, body):
        arrs = [self.init[n] for n, _l, _e in self.states]
        item = ('Hd',) + tuple(vals) + ((('Body', body),) if self.has_body else ())
        result = interp.call('hh0', arrs + [item, ('Fr',)])
        out = result[len(self.states)]
        ctor = out[0]
        spec = next(o for o in self.outs if o[0] == ctor)
        _c, sends, fields = spec
        if not sends:
            return None, None
        port = out[1]
        wire = b''
        for (fname, t), v in zip(fields, out[2:]):
            if t == 'bool':
                v = 1 if v else 0
            wire += self.packed(fname, t, v & MASK[8 * SIZE[t]])
        return port, wire + body
