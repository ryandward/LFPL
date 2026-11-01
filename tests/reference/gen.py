"""A type-directed generator of well-typed, linear LFPL-O programs, with the reference result
computed by lang.Interp. It aims at what tests/test_fuzz.py does not generate."""
import random
from .lang import Interp, show, MASK

SRC = {'i8': 'Int8', 'i16': 'Int16', 'i32': 'Int32', 'i64': 'Int64', 'bool': 'Bool'}
BITS = {'i8': 8, 'i16': 16, 'i32': 32, 'i64': 64}
INTS = ('i8', 'i16', 'i32', 'i64')
SCAL = INTS + ('bool',)

DATA = """data Free = End | More(next: Free);
data Res = Fail | Done(value: Int64, rest: Free);
data Fuel = Zed | Suc(prev: Fuel);
data List = Nil | Cons(h: Int64, t: List);
data Lb = Nb | Cb(ba: Int8, bf: Bool, bn: Int16, bt: Lb);
record R { a: Int64, b: Int32, c: Int16, e: Int8 };
record Q { p: Bool, q: Int8, r: Int16, s: Int32, w: Int64 };
data S = Sa(x: Int32) | Sb(x: Int32, y: Int16) | Sc(z: Int8, f: Bool) | Sd;
record P { l: List, r2: List };
data Tt = Tu(v: Int32, n: List) | Tw(g: Int16, u: List) | Tz(tb: Bool);
"""
# ctor -> (fields as (name, type)), and for a type its constructors in declaration order
TYPES = {
    'Free': [('End', []), ('More', [('next', 'Free')])],
    'Fuel': [('Zed', []), ('Suc', [('prev', 'Fuel')])],
    'List': [('Nil', []), ('Cons', [('h', 'i64'), ('t', 'List')])],
    'Lb': [('Nb', []), ('Cb', [('ba', 'i8'), ('bf', 'bool'), ('bn', 'i16'), ('bt', 'Lb')])],
    'R': [('R', [('a', 'i64'), ('b', 'i32'), ('c', 'i16'), ('e', 'i8')])],
    'Q': [('Q', [('p', 'bool'), ('q', 'i8'), ('r', 'i16'), ('s', 'i32'), ('w', 'i64')])],
    'S': [('Sa', [('x', 'i32')]), ('Sb', [('x', 'i32'), ('y', 'i16')]),
          ('Sc', [('z', 'i8'), ('f', 'bool')]), ('Sd', [])],
    'P': [('P', [('l', 'List'), ('r2', 'List')])],
    'Tt': [('Tu', [('v', 'i32'), ('n', 'List')]), ('Tw', [('g', 'i16'), ('u', 'List')]), ('Tz', [('tb', 'bool')])],
}
FIELDS = {c: [f for f, _t in fs] for ts in TYPES.values() for c, fs in ts}
RECORDS = ('R', 'Q', 'P')
NULLS = {'End', 'Zed', 'Nil', 'Nb'}
CELLISH = ('D', 'R', 'Q', 'S')  # a resource that can become one single-cell value
TAGGED = ('S', 'Tt')

HELPERS = """
fn rev(l: List, acc: List) : List = match l with Nil => acc | Cons(d, h, t) => rev(t, Cons@d(h, acc)) end;
fn revb(l: Lb, acc: Lb) : Lb = match l with Nb => acc | Cb(d, a, f, n, t) => revb(t, Cb@d(a, f, n, acc)) end;
fn dgL(l: List, acc: Int64, free: Free) : (Int64, Free) = match l with
  | Nil => (acc, free) | Cons(d, h, t) => dgL(t, +(*(acc, 31), h), More@d(free)) end;
fn dgB(l: Lb, acc: Int64, free: Free) : (Int64, Free) = match l with
  | Nb => (acc, free)
  | Cb(d, a, f, n, t) => dgB(t, +(*(+(*(acc, 7), pack<8>(a)), 5), +(if f then 3 else 2, *(pack<16>(n), 11))), More@d(free)) end;
fn dgR(v: R, acc: Int64, free: Free) : (Int64, Free) = match v with R(d, a, b, c, e) =>
  (+(*(acc, 13), +(a, +(*(pack<32>(b), 3), +(*(pack<16>(c), 5), *(pack<8>(e), 7))))), More@d(free)) end;
fn dgQ(v: Q, acc: Int64, free: Free) : (Int64, Free) = match v with Q(d, p, q, r, s, w) =>
  (+(*(acc, 17), +(if p then 1 else 0, +(*(pack<8>(q), 3), +(*(pack<16>(r), 5), +(*(pack<32>(s), 7), *(w, 11)))))), More@d(free)) end;
fn dgS(v: S, acc: Int64, free: Free) : (Int64, Free) = match v with
  | Sa(d, x) => (+(*(acc, 19), +(1000000, pack<32>(x))), More@d(free))
  | Sb(d, x, y) => (+(*(acc, 19), +(2000000, +(pack<32>(x), *(pack<16>(y), 3)))), More@d(free))
  | Sc(d, z, f) => (+(*(acc, 19), +(3000000, +(pack<8>(z), if f then 500 else 600))), More@d(free))
  | Sd(d) => (+(*(acc, 19), 4000000), More@d(free)) end;
fn id8(x: Int8) : Int8 = x;
fn id16(x: Int16) : Int16 = x;
fn id32(x: Int32) : Int32 = x;
fn id64(x: Int64) : Int64 = x;
fn idb(x: Bool) : Bool = x;
fn hh(x: Int64, y: Int64) : Int64 = +(*(x, 3), y);
"""


def parse_helpers():
    """The helpers' meaning, written by hand from their text above."""
    def dgL(l, acc, free):
        while l[0] == 'Cons':
            acc = (acc * 31 + l[1]) & MASK[64]
            free = ('More', free)
            l = l[2]
        return [acc, free]

    def dgB(l, acc, free):
        while l[0] == 'Cb':
            _c, a, f, n, t = l
            acc = ((acc * 7 + a) * 5 + (3 if f else 2) + n * 11) & MASK[64]
            free = ('More', free)
            l = t
        return [acc, free]

    def dgR(v, acc, free):
        _c, a, b, c, e = v
        return [(acc * 13 + a + b * 3 + c * 5 + e * 7) & MASK[64], ('More', free)]

    def dgQ(v, acc, free):
        _c, p, q, r, s, w = v
        return [(acc * 17 + (1 if p else 0) + q * 3 + r * 5 + s * 7 + w * 11) & MASK[64], ('More', free)]

    def dgS(v, acc, free):
        c = v[0]
        if c == 'Sa':
            x = 1000000 + v[1]
        elif c == 'Sb':
            x = 2000000 + v[1] + v[2] * 3
        elif c == 'Sc':
            x = 3000000 + v[1] + (500 if v[2] else 600)
        else:
            x = 4000000
        return [(acc * 19 + x) & MASK[64], ('More', free)]

    def rev(l, acc):
        while l[0] == 'Cons':
            acc = ('Cons', l[1], acc)
            l = l[2]
        return acc

    def revb(l, acc):
        while l[0] == 'Cb':
            acc = ('Cb', l[1], l[2], l[3], acc)
            l = l[4]
        return acc

    return {'revb': revb, 'dgL': dgL, 'dgB': dgB, 'dgR': dgR, 'dgQ': dgQ, 'dgS': dgS, 'rev': rev,
            'id8': lambda x: x & 255, 'id16': lambda x: x & 65535, 'id32': lambda x: x & MASK[32],
            'id64': lambda x: x, 'idb': lambda x: x, 'hh': lambda x, y: (x * 3 + y) & MASK[64]}


class Skip(Exception):
    pass


class Gen:
    def __init__(self, rng, *, spill=0.25, tuples=0.3, ban_tail_logic=True):
        self.r = rng
        self.names = 0
        self.fns = []          # (name, params [(n,t)], ret, at, body)
        self.spill = spill
        self.tuples = tuples
        self.ban_tail_logic = ban_tail_logic
        self.scalar_fns = {}   # ret type -> [(name, param types)]
        self.tuple_fns = {}    # tuple of member types -> [(name, param types)]
        self.size = 0

    # --- names
    def fresh(self, stem='v'):
        self.names += 1
        return f"{stem}{self.names}"

    def pick(self, seq):
        return self.r.choice(list(seq))

    # --- literals
    def lit(self, ty, big):
        if ty == 'bool':
            return ('bool', self.r.random() < 0.5), 'bool'
        if big:
            m = MASK[BITS[ty]]
            return ('lit', self.pick((0, 1, 2, 7, m, m // 2, m // 2 + 1, self.r.randrange(m + 1)))), 'lit'
        return ('lit', self.r.randrange(128)), 'lit'

    def vars_of(self, ty, env, allow_lit=True):
        return [n for n, t in env.items() if t == ty or (allow_lit and t == 'lit' and ty != 'bool')]

    # --- scalars. Returns (ast, actual type)
    def scalar(self, ty, env, d, tail=False, typed=False, big=False):
        self.size += 1
        if self.size > 900:
            d = 0
        roll = self.r.random()
        if d <= 0 or roll < 0.18:
            names = self.vars_of(ty, env, allow_lit=not typed)
            if names and (typed or self.r.random() < 0.75):
                n = self.pick(names)
                return ('var', n), env[n]
            if not typed:
                return self.lit(ty, big)
            return self._typed_base(ty, env)
        if roll < 0.28:
            return self._if(lambda dd, t, b: self.scalar(ty, env, dd, t, t and typed, b),
                            ty, env, d, tail, typed, big)
        if roll < 0.38:
            return self._let_scalar(ty, env, d, tail, typed, big)
        if roll < 0.38 + self.tuples * 0.4:
            return self._let_tuple_scalar(ty, env, d, tail, typed, big)
        if roll < 0.74:
            if ty == 'bool':
                return self._bool_op(env, d, tail)
            return self._binop(ty, env, d)
        if roll < 0.82 and ty == 'i64':
            return self._pack(env, d)
        if roll < 0.92:
            call = self._call_scalar(ty, env, d)
            if call is not None:
                return call
        if ty == 'bool':
            return self._bool_op(env, d, tail)
        return self._binop(ty, env, d)

    def _typed_base(self, ty, env):
        if ty == 'bool':
            return ('bool', self.r.random() < 0.5), 'bool'
        lit = self.lit(ty, True)[0]
        name = {'i8': 'id8', 'i16': 'id16', 'i32': 'id32', 'i64': 'id64'}[ty]
        return ('call', name, [lit]), ty

    def _if(self, sub, ty, env, d, tail, typed, big):
        c, _ = self.boolean(env, d - 1)
        a, ta = sub(d - 1, tail, big)
        b, tb = sub(d - 1, tail, big)
        if typed and ta == 'lit' and tb == 'lit':
            a, ta = self._typed_base(ty, env)
        return ('if', c, a, b), (tb if ta == 'lit' else ta)

    def boolean(self, env, d):
        roll = self.r.random()
        if d <= 0 or roll < 0.3:
            names = self.vars_of('bool', env)
            if names and self.r.random() < 0.5:
                return ('var', self.pick(names)), 'bool'
            return self._cmp(env, max(d, 0))
        if roll < 0.55:
            return self._cmp(env, d)
        return self.scalar('bool', env, d)

    def _cmp(self, env, d):
        ty = self.pick(INTS)
        op = self.pick(('<', '<=', '>', '>=', '==', '!='))
        if self.r.random() < 0.5:
            a, _ = self.scalar(ty, env, d - 1, typed=True)
            b, _ = self.scalar(ty, env, d - 1, big=True)
            if self.r.random() < 0.5:
                a, b = b, a
        else:
            a, _ = self.scalar(ty, env, d - 1, big=False)
            b, _ = self.scalar(ty, env, d - 1, big=False)
        # both sides must agree on a type: only ever `ty` or a literal, so this always holds
        return ('cmp', op, a, b), 'bool'

    def _bool_op(self, env, d, tail):
        if tail and self.ban_tail_logic:
            return self._cmp(env, d)
        roll = self.r.random()
        if roll < 0.25:
            a, _ = self.boolean(env, d - 1)
            return ('not', a), 'bool'
        if roll < 0.55:
            a, _ = self.boolean(env, d - 1)
            b, _ = self.boolean(env, d - 1)
            return ('and', a, b), 'bool'
        if roll < 0.85:
            a, _ = self.boolean(env, d - 1)
            b, _ = self.boolean(env, d - 1)
            return ('or', a, b), 'bool'
        return self._cmp(env, d)

    def _binop(self, ty, env, d):
        op = self.pick(('+', '-', '*', '&', '|', '^', '<<', '>>'))
        w = BITS[ty]
        typed_left = ty != 'i64' or self.r.random() < 0.7
        left, lt = self.scalar(ty, env, d - 1, typed=typed_left, big=False)
        if op in ('<<', '>>'):
            if self.r.random() < 0.5:
                right = ('lit', self.pick((0, 1, w - 1, w, w + 1, 32, 64, 70, 255, 1 << 32, self.r.randrange(w))))
            else:
                right, _ = self.scalar(ty, env, d - 1, big=False)
            return ('bin', op, left, right, w), ty
        right, _ = self.scalar(ty, env, d - 1, big=True if lt != 'lit' else False)
        if lt != 'lit' and op in ('+', '*', '&', '|', '^') and self.r.random() < 0.3:
            left, right = right, left
        if lt != 'lit' and op == '-' and self.r.random() < 0.3:
            left, right = self.lit(ty, True)[0], left
        return ('bin', op, left, right, w), ty

    def _pack(self, env, d):
        widths = []
        while not widths or (sum(widths) < 64 and self.r.random() < 0.6):
            room = [w for w in (8, 16, 32) if sum(widths) + w <= 64]
            if not room:
                break
            widths.append(self.pick(room))
        args = []
        for w in widths:
            ty = self.pick(INTS)
            args.append(self.scalar(ty, env, d - 1)[0])
        return ('pack', widths, args), 'i64'

    def _call_scalar(self, ty, env, d):
        if ty == 'i64' and self.r.random() < 0.2:
            return ('call', 'hh', [self.scalar('i64', env, d - 1, big=True)[0],
                                   self.scalar('i64', env, d - 1, big=True)[0]]), 'i64'
        options = self.scalar_fns.get(ty, [])
        if not options:
            return None
        name, ptys = self.pick(options)
        return ('call', name, [self.scalar(t, env, d - 1, big=True)[0] for t in ptys]), ty

    def _let_scalar(self, ty, env, d, tail, typed, big):
        name = self.fresh()
        bty = self.pick(SCAL)
        bound, actual = self.scalar(bty, env, d - 1)
        inner = dict(env)
        inner[name] = actual
        body, bt = self.scalar(ty, inner, d - 1, tail, typed, big)
        return ('let', name, bound, body), bt

    def _let_tuple_scalar(self, ty, env, d, tail, typed, big):
        tys = [self.pick(SCAL) for _ in range(self.r.randrange(2, 5))]
        te, actuals = self.tuple_expr(tys, [], env, d - 1)
        names = [self.fresh() for _ in tys]
        inner = dict(env)
        for n, t in zip(names, actuals):
            inner[n] = t
        body, bt = self.scalar(ty, inner, d - 1, tail, typed, big)
        return ('lett', names, te, body), bt

    # --- tuples: scalar members, and data members fed by `lin`
    def tuple_expr(self, tys, lin, env, d):
        self.size += 1
        if self.size > 900:
            d = 0
        roll = self.r.random()
        if d > 0 and roll < 0.3:
            c, _ = self.boolean(env, d - 1)
            a, ta = self.tuple_expr(tys, lin, env, d - 1)
            b, tb = self.tuple_expr(tys, lin, env, d - 1)
            return ('if', c, a, b), [y if x == 'lit' else x for x, y in zip(ta, tb)]
        if d > 0 and roll < 0.42:
            name = self.fresh()
            bty = self.pick(SCAL)
            bound, actual = self.scalar(bty, env, d - 1)
            inner = dict(env)
            inner[name] = actual
            body, actuals = self.tuple_expr(tys, lin, inner, d - 1)
            return ('let', name, bound, body), actuals
        if d > 0 and roll < 0.52 and not lin:
            options = self.tuple_fns.get(tuple(tys), [])
            if options:
                name, ptys = self.pick(options)
                return ('call', name, [self.scalar(t, env, d - 1, big=True)[0] for t in ptys]), list(tys)
        if d > 0 and roll < 0.62 and not lin:
            # a tuple-returning let whose result is reordered
            order = list(range(len(tys)))
            self.r.shuffle(order)
            inner_tys = [tys[i] for i in order]
            te, actuals = self.tuple_expr(inner_tys, [], env, d - 1)
            names = [self.fresh() for _ in tys]
            back = [None] * len(tys)
            acts = [None] * len(tys)
            for position, i in enumerate(order):
                back[i] = ('var', names[position])
                acts[i] = actuals[position]
            return ('lett', names, te, ('tup', back)), acts
        if lin and d > 0 and roll < 0.8:
            return self.opened_tuple(tys, lin, env, d)
        groups = self.distribute(tys, lin)
        members, actuals = [], []
        for t, group in zip(tys, groups):
            if t in SCAL:
                m, a = self.scalar(t, env, d - 1, big=False)
            else:
                m, a = self.data(t, group, env, d - 1)
            members.append(m)
            actuals.append(a)
        return ('tup', members), actuals

    def opened_tuple(self, tys, lin, env, d):
        """Match on a linear resource, around a tuple."""
        res = self.pick(lin)
        rest = [x for x in lin if x is not res]
        name, rty = res
        if rty == 'D' or rty not in TYPES:
            groups = self.distribute(tys, lin)
            members, actuals = [], []
            for t, group in zip(tys, groups):
                if t in SCAL:
                    m, a = self.scalar(t, env, d - 1)
                else:
                    m, a = self.data(t, group, env, d - 1)
                members.append(m)
                actuals.append(a)
            return ('tup', members), actuals
        arms = []
        joined = None
        for c, fields in TYPES[rty]:
            inner = dict(env)
            newlin = list(rest)
            binders = []
            if c not in NULLS:
                dn = self.fresh('d')
                binders.append(dn)
                newlin.append((dn, 'D'))
                for f, ft in fields:
                    b = self.fresh('f')
                    binders.append(b)
                    if ft in SCAL:
                        inner[b] = ft
                    else:
                        newlin.append((b, ft))
            body, actuals = self.tuple_expr(tys, newlin, inner, d - 1)
            joined = actuals if joined is None else [y if x == 'lit' else x for x, y in zip(joined, actuals)]
            arms.append((c, binders, body))
        self.r.shuffle(arms)
        return ('match', ('var', name), arms), joined

    def distribute(self, tys, lin):
        """Give each data member the resources it can absorb."""
        groups = [[] for _ in tys]
        pool = list(lin)
        self.r.shuffle(pool)
        for i, t in enumerate(tys):
            if t in ('R', 'Q', 'S'):
                found = next((x for x in pool if x[1] in CELLISH), None)
                if found is None:
                    raise Skip('no cell for a single-cell member')
                pool.remove(found)
                groups[i].append(found)
        free_at = [i for i, t in enumerate(tys) if t == 'Free']
        fuel_at = [i for i, t in enumerate(tys) if t == 'Fuel']
        for x in list(pool):
            if x[1] == 'Free':
                if not free_at or any(y[1] == 'Free' for y in groups[free_at[0]]):
                    raise Skip('a Free with nowhere to go')
                groups[free_at[0]].append(x)
                pool.remove(x)
            elif x[1] == 'Fuel':
                if not fuel_at or any(y[1] == 'Fuel' for y in groups[fuel_at[0]]):
                    raise Skip('a Fuel with nowhere to go')
                groups[fuel_at[0]].append(x)
                pool.remove(x)
        sinks = [i for i, t in enumerate(tys) if t in ('List', 'Lb')]
        for x in pool:
            if sinks and not (x[1] == 'Lb' and not any(tys[i] == 'Lb' for i in sinks)) and not (
                    x[1] in ('List', 'P', 'Tt') and not any(tys[i] == 'List' for i in sinks)):
                candidates = [i for i in sinks if tys[i] == 'List'] if x[1] in ('List', 'P', 'Tt') else (
                    [i for i in sinks if tys[i] == 'Lb'] if x[1] == 'Lb' else sinks)
                groups[self.pick(candidates)].append(x)
            elif x[1] == 'D' and (free_at or fuel_at):
                groups[self.pick(free_at + fuel_at)].append(x)
            else:
                raise Skip(f'no sink for {x}')
        return groups

    # --- data: an expression of a data type that consumes exactly `lin`
    def data(self, ty, lin, env, d):
        self.size += 1
        if self.size > 900:
            d = 0
        if ty in ('List', 'Lb', 'Free', 'Fuel'):
            return self.sink(ty, lin, env, d)
        return self.single(ty, lin, env, d)

    def wrap(self, ty, lin, env, d, build):
        """Constructs any data expression can sit under."""
        roll = self.r.random()
        if d > 0 and roll < 0.12:
            c, _ = self.boolean(env, d - 1)
            a, _t = build(lin, env, d - 1)
            b, _t = build(lin, env, d - 1)
            return ('if', c, a, b), ty
        if d > 0 and roll < 0.24:
            name = self.fresh()
            bty = self.pick(SCAL)
            bound, actual = self.scalar(bty, env, d - 1)
            inner = dict(env)
            inner[name] = actual
            body, _t = build(lin, inner, d - 1)
            return ('let', name, bound, body), ty
        if d > 0 and roll < 0.32 and lin:
            res = self.pick(lin)
            name = self.fresh('y')
            newlin = [(name, res[1]) if x is res else x for x in lin]
            body, _t = build(newlin, env, d - 1)
            return ('let', name, ('var', res[0]), body), ty
        if d > 0 and roll < 0.42 and len(lin) >= 2:
            # both resources pass through a tuple let, sometimes swapped at a join
            r1, r2 = self.r.sample(lin, 2)
            rest = [x for x in lin if x is not r1 and x is not r2]
            n1, n2 = self.fresh('y'), self.fresh('y')
            s = self.fresh()
            sty = self.pick(SCAL)
            sval, sact = self.scalar(sty, env, d - 1)
            tup = ('tup', [('var', r1[0]), ('var', r2[0]), sval])
            if r1[1] == r2[1] and self.r.random() < 0.6:
                c, _ = self.boolean(env, d - 1)
                sval2, sact2 = self.scalar(sty, env, d - 1)
                tup = ('if', c, tup, ('tup', [('var', r2[0]), ('var', r1[0]), sval2]))
                sact = sact2 if sact == 'lit' else sact
            inner = dict(env)
            inner[s] = sact
            body, _t = build(rest + [(n1, r1[1]), (n2, r2[1])], inner, d - 1)
            return ('lett', [n1, n2, s], tup, body), ty
        if d > 0 and roll < 0.6 and lin:
            opened = [x for x in lin if x[1] in TYPES]
            if opened:
                return self.opened(ty, lin, opened, env, d, build)
        if d > 0 and roll < 0.68 and lin:
            recs = [x for x in lin if x[1] in ('R', 'Q')]
            if recs:
                res = self.pick(recs)
                fields = TYPES[res[1]][0][1]
                f, ft = self.pick(fields)
                inner = dict(env)
                inner[f] = ft
                val, _a = self.scalar(ft, inner, d - 1, big=True)
                name = self.fresh('y')
                newlin = [(name, res[1]) if x is res else x for x in lin]
                body, _t = build(newlin, env, d - 1)
                return ('let', name, ('with', ('var', res[0]), f, val), body), ty
        return None

    def opened(self, ty, lin, candidates, env, d, build):
        res = self.pick(candidates)
        rest = [x for x in lin if x is not res]
        arms = []
        for c, fields in TYPES[res[1]]:
            inner = dict(env)
            newlin = list(rest)
            binders = []
            if c not in NULLS:
                dn = self.fresh('d')
                binders.append(dn)
                newlin.append((dn, 'D'))
                for f, ft in fields:
                    b = self.fresh('f')
                    binders.append(b)
                    if ft in SCAL:
                        inner[b] = ft
                    else:
                        newlin.append((b, ft))
            body, _t = build(newlin, inner, d - 1)
            arms.append((c, binders, body))
        self.r.shuffle(arms)
        return ('match', ('var', res[0]), arms), ty

    def sink(self, ty, lin, env, d):
        def build(l, e, dd):
            return self.sink(ty, l, e, dd)
        wrapped = self.wrap(ty, lin, env, d, build)
        if wrapped is not None:
            return wrapped
        if not lin:
            null = {'List': 'Nil', 'Lb': 'Nb', 'Free': 'End', 'Fuel': 'Zed'}[ty]
            return ('null', null), ty
        # take a D if there is one, because it is what grows the structure
        cells = [x for x in lin if x[1] == 'D']
        same = [x for x in lin if x[1] == ty]
        if len(lin) == 1 and same:
            return ('var', same[0][0]), ty
        if cells:
            res = self.pick(cells)
            rest = [x for x in lin if x is not res]
            tail, _t = self.sink(ty, rest, env, d - 1)
            if ty == 'List':
                h, _a = self.scalar('i64', env, d - 1, big=True)
                return ('ctor', 'Cons', res[0], [h, tail]), ty
            if ty == 'Lb':
                a, _a = self.scalar('i8', env, d - 1, big=True)
                f, _a = self.scalar('bool', env, d - 1)
                n, _a = self.scalar('i16', env, d - 1, big=True)
                return ('ctor', 'Cb', res[0], [a, f, n, tail]), ty
            if ty == 'Free':
                return ('ctor', 'More', res[0], [tail]), ty
            return ('ctor', 'Suc', res[0], [tail]), ty
        if same and ty in ('List', 'Lb'):
            res = self.pick(same)
            rest = [x for x in lin if x is not res]
            tail, _t = self.sink(ty, rest, env, d - 1)
            return ('call', 'rev' if ty == 'List' else 'revb', [('var', res[0]), tail]), ty
        # open whatever is left: it yields a cell
        opened = [x for x in lin if x[1] in TYPES and x[1] != 'Free' and x[1] != 'Fuel']
        if not opened:
            raise Skip(f'sink {ty} cannot take {lin}')
        return self.opened(ty, lin, opened, env, d, build)

    def single(self, ty, lin, env, d):
        """R, Q, S: exactly one resource, which is or becomes one cell."""
        if len(lin) != 1 or lin[0][1] not in CELLISH:
            raise Skip(f'single {ty} from {lin}')
        def build(l, e, dd):
            return self.single(ty, l, e, dd)
        wrapped = self.wrap(ty, lin, env, d, build)
        if wrapped is not None:
            return wrapped
        res = lin[0]
        if res[1] == ty and self.r.random() < 0.3:
            return ('var', res[0]), ty
        if res[1] == 'D':
            ctor, fields = self.pick(TYPES[ty])
            args = []
            for _f, ft in fields:
                args.append(self.scalar(ft, env, d - 1, big=True)[0])
            return ('ctor', ctor, res[0], args), ty
        return self.opened(ty, lin, [res], env, d, build)


# --- functions

def params_text(params):
    return ', '.join(f"{n}: {SRC.get(t, 'Diamond' if t == 'D' else t)}" for n, t in params)


def type_text(t):
    return SRC.get(t, 'Diamond' if t == 'D' else t)


def ret_text(ret):
    if isinstance(ret, (list, tuple)):
        return '(' + ', '.join(type_text(t) for t in ret) + ')'
    return type_text(ret)


class Program:
    def __init__(self, rng, **kw):
        self.r = rng
        self.g = Gen(rng, **kw)
        self.defs = []       # (name, params, ret, at, body)

    def add(self, name, params, ret, body, at=None):
        self.defs.append((name, params, ret, at, body))

    def with_lets(self, env, make_body, low=8, high=22):
        """Many named scalars around a body: RT keeps each live until its scope ends, so this is
        what forces spills across whatever the body does."""
        g = self.g
        if self.r.random() > 0.3:
            return make_body(env)
        env = dict(env)
        lets = []
        for _ in range(self.r.randrange(low, high)):
            n = g.fresh()
            ty = self.r.choice(SCAL)
            g.size = min(g.size, 400)
            val, act = g.scalar(ty, env, self.r.randrange(0, 2))
            lets.append((n, val))
            env[n] = act
        body = make_body(env)
        for n, val in reversed(lets):
            body = ('let', n, val, body)
        return body

    def fn_scalar(self, name):
        g = self.g
        n = self.r.randrange(1, 7)
        params = [(g.fresh('p'), self.r.choice(SCAL)) for _ in range(n)]
        ret = self.r.choice(SCAL)
        env = dict(params)
        g.size = 0
        if self.r.random() < g.spill:
            body = self.pressure(ret, env)
        else:
            body = g.scalar(ret, env, self.r.randrange(2, 6), tail=True)[0]
        self.add(name, params, ret, body)
        g.scalar_fns.setdefault(ret, []).append((name, [t for _n, t in params]))
        return name

    def pressure(self, ret, env):
        """More live values than registers, carried across calls, joins and tuple lets, all used at the end."""
        g = self.g
        env = dict(env)
        lets = []
        names = []
        for _ in range(self.r.randrange(10, 24)):
            kind = self.r.random()
            if kind < 0.2:
                n1, n2 = g.fresh(), g.fresh()
                ty = self.r.choice(INTS)
                te, acts = g.tuple_expr([ty, 'i64'], [], env, self.r.randrange(0, 3))
                lets.append((('lett', [n1, n2], te), None))
                names += [(n1, acts[0], ty), (n2, acts[1], 'i64')]
                env[n1], env[n2] = acts
            else:
                n = g.fresh()
                ty = self.r.choice(INTS)
                val, act = g.scalar(ty, env, self.r.randrange(0, 3))
                lets.append((('let', n, val), None))
                names.append((n, act, ty))
                env[n] = act
        # the end: fold every name, so all of them stay live to here
        acc = ('lit', 1)
        for n, act, ty in names:
            acc = ('bin', '+', ('bin', '*', acc, ('lit', 3), 64), ('pack', [BITS[ty] if BITS[ty] != 64 else 32], [('var', n)]), 64)
        if ret == 'bool':
            final = ('cmp', '>', acc, ('lit', self.r.randrange(1 << 20)))
        elif ret == 'i64':
            final = acc
        else:
            # a narrow result has to come from a typed name
            final = g.scalar(ret, env, 1, tail=True, typed=True)[0]
            # keep the fold alive through a comparison that picks between two arms
            other = g.scalar(ret, env, 0, tail=True)[0]
            final = ('if', ('cmp', '>', acc, ('lit', 7)), final, other)
        for (head, _x) in reversed(lets):
            if head[0] == 'let':
                final = ('let', head[1], head[2], final)
            else:
                final = ('lett', head[1], head[2], final)
        return final

    def fn_tuple(self, name):
        g = self.g
        n = self.r.randrange(1, 7)
        params = [(g.fresh('p'), self.r.choice(SCAL)) for _ in range(n)]
        ret = tuple(self.r.choice(SCAL) for _ in range(self.r.randrange(2, 7)))
        g.size = 0
        body = self.with_lets(dict(params), lambda e: g.tuple_expr(list(ret), [], e, self.r.randrange(1, 5))[0])
        self.add(name, params, list(ret), body)
        g.tuple_fns.setdefault(ret, []).append((name, [t for _n, t in params]))
        return name

    def fn_linear(self, name):
        """Takes some linear values and scalars, returns a tuple or a data value that absorbs them."""
        g = self.g
        lin_tys = [self.r.choice(('List', 'List', 'R', 'Q', 'S', 'P', 'Tt', 'D', 'Lb'))
                   for _ in range(self.r.randrange(1, 4))]
        scal = [self.r.choice(SCAL) for _ in range(self.r.randrange(1, 7 - len(lin_tys)))]
        params = [(g.fresh('p'), t) for t in lin_tys] + [(g.fresh('p'), t) for t in scal]
        self.r.shuffle(params)
        lin = [(n, t) for n, t in params if t not in SCAL]
        env = {n: t for n, t in params if t in SCAL}
        # results: a sink for everything, one extra scalar, sometimes a dedicated single-cell result
        ret = []
        cells = [t for t in lin_tys if t in CELLISH]
        if cells and self.r.random() < 0.5:
            ret.append(self.r.choice(('R', 'Q', 'S')))
        if 'Lb' in lin_tys:
            ret.append('Lb')
        ret.append('List')
        for _ in range(self.r.randrange(0, 3)):
            ret.append(self.r.choice(SCAL))
        self.r.shuffle(ret)
        g.size = 0
        depth = self.r.randrange(2, 6)
        if len(ret) == 1:
            body = self.with_lets(env, lambda e: g.data(ret[0], lin, e, depth)[0])
            ret_val = ret[0]
        else:
            body = self.with_lets(env, lambda e: g.tuple_expr(ret, lin, e, depth)[0])
            ret_val = list(ret)
        self.add(name, params, ret_val, body)
        return name, params, ret_val

    def fn_loop(self, name):
        """A self tail call whose arguments permute and recompute the parameters."""
        g = self.g
        ty = self.r.choice(INTS)
        a, b, c = g.fresh('p'), g.fresh('p'), g.fresh('p')
        n, back, d, r = g.fresh('n'), g.fresh('b'), g.fresh('d'), g.fresh('r')
        params = [(n, 'Fuel'), (a, ty), (b, ty), (c, ty), (back, 'Fuel')]
        ret = [ty, ty, ty, 'Fuel']
        env = {a: ty, b: ty, c: ty}
        def step():
            order = [a, b, c]
            self.r.shuffle(order)
            args = []
            for name2 in order:
                if self.r.random() < 0.3:
                    args.append(g._binop(ty, env, 1)[0])
                else:
                    args.append(('var', name2))
            return args
        def tail():
            call = ('call', name, [('var', r)] + step() + [('ctor', 'Suc', d, [('var', back)])])
            return call
        body = tail()
        if self.r.random() < 0.5:
            body = ('if', g.boolean(env, 1)[0], tail(), tail())
        if self.r.random() < 0.3:
            v = g.fresh()
            val, act = g.scalar(ty, env, 1)
            body = ('let', v, val, body)
        body = self.with_lets(env, lambda e: body)
        order = [a, b, c]
        self.r.shuffle(order)
        base = ('tup', [('var', x) for x in order] + [('var', back)])
        full = ('match', ('var', n), [('Zed', [], base), ('Suc', [d, r], body)])
        self.add(name, params, ret, full)
        return name, ty

    def fn_trmc(self, name):
        """Tail recursion modulo cons, with the scalar arguments permuted at each step."""
        g = self.g
        if self.r.random() < 0.5:
            l, x, y, z = g.fresh('l'), g.fresh('x'), g.fresh('y'), g.fresh('z')
            d, h, t = g.fresh('d'), g.fresh('h'), g.fresh('t')
            params = [(l, 'List'), (x, 'i64'), (y, 'i64'), (z, 'i64')]
            order = [x, y, z]
            self.r.shuffle(order)
            env = {x: 'i64', y: 'i64', z: 'i64', h: 'i64'}
            head = self.r.choice((
                ('var', h),
                ('bin', self.r.choice(('+', '-', '&', '|', '^')), ('var', h), ('var', x), 64),
                ('bin', '+', ('var', h), ('lit', self.r.choice((1, 1 << 40, (1 << 64) - 1))), 64),
                g.scalar('i64', env, 2)[0]))
            args = [('var', t)]
            for v in order:
                args.append(('var', v) if self.r.random() < 0.7 else g._binop('i64', {x: 'i64', y: 'i64', z: 'i64', h: 'i64'}, 1)[0])
            arm = self.with_lets(env, lambda e: ('ctor', 'Cons', d, [head, ('call', name, args)]))
            body = ('match', ('var', l), [('Nil', [], ('null', 'Nil')), ('Cons', [d, h, t], arm)])
            self.add(name, params, 'List', body, at=l)
            return name, 'List', ['i64', 'i64', 'i64']
        l, x, y = g.fresh('l'), g.fresh('x'), g.fresh('y')
        d, a, f, n, t = g.fresh('d'), g.fresh('a'), g.fresh('f'), g.fresh('n'), g.fresh('t')
        params = [(l, 'Lb'), (x, 'i8'), (y, 'i8')]
        env = {x: 'i8', y: 'i8', a: 'i8', f: 'bool', n: 'i16'}
        args = [('var', t)] + ([('var', y), ('var', x)] if self.r.random() < 0.5 else [('var', x), ('var', y)])
        fa = self.r.choice((('var', a), ('bin', '+', ('var', a), ('var', x), 8), g.scalar('i8', env, 2)[0]))
        ff = self.r.choice((('var', f), ('not', ('var', f)), ('and', ('var', f), ('cmp', '<', ('var', x), ('var', y))), g.scalar('bool', env, 2)[0]))
        fn = self.r.choice((('var', n), ('bin', '+', ('var', n), ('lit', 1), 16), g.scalar('i16', env, 2)[0]))
        arm = self.with_lets(env, lambda e: ('ctor', 'Cb', d, [fa, ff, fn, ('call', name, args)]))
        body = ('match', ('var', l), [('Nb', [], ('null', 'Nb')), ('Cb', [d, a, f, n, t], arm)])
        self.add(name, params, 'Lb', body, at=l)
        return name, 'Lb', ['i8', 'i8']

    # --- main
    def literal(self, ty):
        if ty == 'bool':
            return ('bool', self.r.random() < 0.5)
        m = MASK[BITS[ty]]
        return ('lit', self.r.choice((0, 1, 2, 7, m, m // 2, self.r.randrange(m + 1))))

    def build(self):
        g = self.g
        for i in range(self.r.randrange(1, 4)):
            self.fn_scalar(f"fs{i}")
        for i in range(self.r.randrange(1, 3)):
            self.fn_tuple(f"ft{i}")
        calls = []
        lin_fns = []
        for i in range(self.r.randrange(0, 3)):
            try:
                lin_fns.append(self.fn_linear(f"fl{i}"))
            except Skip:
                pass
        loops = [self.fn_loop(f"lp{i}") for i in range(self.r.randrange(0, 2))]
        trmcs = [self.fn_trmc(f"tm{i}") for i in range(self.r.randrange(0, 2))]
        # a function that exercises the scalar ones
        env = {}
        body = ('lit', 1)
        fold = []
        for name, params, ret, _at, _b in self.defs:
            if name.startswith('fs'):
                n = g.fresh('o')
                fold.append(('let', n, ('call', name, [self.literal(t) for _n, t in params]), ret))
            elif name.startswith('ft'):
                names = [g.fresh('o') for _ in ret]
                fold.append(('lett', names, ('call', name, [self.literal(t) for _n, t in params]), ret))
        return fold, lin_fns, loops, trmcs

    def text(self, main_text):
        out = [DATA, HELPERS]
        for name, params, ret, at, body in self.defs:
            at_text = f" @{at}" if at else ""
            out.append(f"fn {name}({params_text(params)}) : {ret_text(ret)}{at_text} = {show(body)};")
        out.append(main_text)
        return "\n".join(out) + "\n"


def to_int(name, ty):
    if ty == 'bool':
        return f"if {name} then 1 else 0"
    if ty == 'i64':
        return name
    return f"pack<{BITS[ty]}>({name})"


def value_of(v, ty):
    if ty == 'bool':
        return 1 if v else 0
    return v


def build_value_text(rng, ty, cells, names):
    """Source for a constructed input and its value, taking cells from `names`."""
    def lit(t):
        if t == 'bool':
            b = rng.random() < 0.5
            return ('true' if b else 'false'), b
        m = MASK[BITS[t]]
        v = rng.choice((0, 1, 2, 7, m, m // 2, rng.randrange(m + 1)))
        return str(v), v
    if ty in SCAL:
        return lit(ty)
    if ty == 'D':
        return names.pop(0), ('#',)
    if ty == 'List':
        n = rng.randrange(0, 3)
        text, value = 'Nil', ('Nil',)
        for _ in range(n):
            if not names:
                break
            t, v = lit('i64')
            text, value = f"Cons@{names.pop(0)}({t}, {text})", ('Cons', v, value)
        return text, value
    if ty == 'Lb':
        n = rng.randrange(0, 3)
        text, value = 'Nb', ('Nb',)
        for _ in range(n):
            if not names:
                break
            a, av = lit('i8')
            f, fv = lit('bool')
            k, kv = lit('i16')
            text, value = f"Cb@{names.pop(0)}({a}, {f}, {k}, {text})", ('Cb', av, fv, kv, value)
        return text, value
    if ty in ('R', 'Q', 'S'):
        ctor, fields = rng.choice(TYPES[ty])
        parts = [lit(ft) for _f, ft in fields]
        return (f"{ctor}@{names.pop(0)}({', '.join(p[0] for p in parts)})", (ctor,) + tuple(p[1] for p in parts))
    if ty == 'P':
        c = names.pop(0)
        l, lv = build_value_text(rng, 'List', cells, names)
        m, mv = build_value_text(rng, 'List', cells, names)
        return f"P@{c}({l}, {m})", ('P', lv, mv)
    if ty == 'Tt':
        ctor, fields = rng.choice(TYPES['Tt'])
        c = names.pop(0)
        parts = []
        for _f, ft in fields:
            parts.append(build_value_text(rng, ft, cells, names))
        return (f"{ctor}@{c}({', '.join(p[0] for p in parts)})", (ctor,) + tuple(p[1] for p in parts))
    raise ValueError(ty)


def cells_needed(ty):
    return {'D': 1, 'List': 2, 'Lb': 2, 'R': 1, 'Q': 1, 'S': 1, 'P': 5, 'Tt': 3}.get(ty, 0)


def program(seed):
    """A whole program: the source text and the Int64 it must print."""
    rng = random.Random(seed)
    p = Program(rng)
    fold, lin_fns, loops, trmcs = p.build()
    # main: every function is called with literals and fresh structures, and every result digested
    interp_fns = {}
    for name, params, ret, _at, body in p.defs:
        interp_fns[name] = ([n for n, _t in params], body)
    interp = Interp(interp_fns, FIELDS)
    native = parse_helpers()
    orig_call = interp.call

    def call(name, args):
        if name in native:
            return native[name](*args)
        return orig_call(name, args)
    interp.call = call

    acc = 1
    lines = []
    results = []

    def mix(a, v):
        return (a * 1000003 + v) & MASK[64]

    # scalar and tuple functions: each is called from a small function of its own, which keeps main shallow
    body_lets = []
    acc_terms = []
    use_fns = []
    use_expected = []
    for position, kind_entry in enumerate(fold):
        kind = kind_entry[0]
        uname = f"use{position}"
        if kind == 'let':
            _k, n, call_ast, ret = kind_entry
            value = interp.ev(call_ast, {})
            use_fns.append(f"fn {uname}(acc: Int64) : Int64 = let {n} = {show(call_ast)} in "
                           f"+(*(acc, 1000003), {to_int(n, ret)});")
            use_expected.append([value_of(value, ret)])
        else:
            _k, names_, call_ast, ret = kind_entry
            values = interp.ev(call_ast, {})
            text = "acc"
            for n, t in zip(names_, ret):
                text = f"+(*({text}, 1000003), {to_int(n, t)})"
            use_fns.append(f"fn {uname}(acc: Int64) : Int64 = let ({', '.join(names_)}) = {show(call_ast)} in {text};")
            use_expected.append([value_of(v, t) for v, t in zip(values, ret)])

    # linear functions: gather cells for the inputs
    plan = []
    total_cells = 0
    for name, params, ret in lin_fns:
        total_cells += sum(cells_needed(t) for _n, t in params)
    for name, ty in loops:
        total_cells += 3
    for name, ty, _s in trmcs:
        total_cells += 3
    if total_cells > 24:
        raise Skip('too many cells for main')
    cell_names = [f"c{i}" for i in range(total_cells)]
    names = list(cell_names)

    state = {'free': 'fr', 'acc': None}
    code = []
    digest_acc = 'a0'
    code.append(f"let {digest_acc} = 1 in ")
    acc_value = 1
    fr_value = ('More', ('End',))  # opaque; only threaded

    def digest(text_expr, ty, value, step):
        nonlocal digest_acc, acc_value
        newacc, newfr = f"a{step}", f"q{step}"
        fn = {'List': 'dgL', 'Lb': 'dgB', 'R': 'dgR', 'Q': 'dgQ', 'S': 'dgS'}[ty]
        code.append(f"let ({newacc}, {newfr}) = {fn}({text_expr}, {digest_acc}, {state['free']}) in ")
        got = native[fn](value, acc_value, ('End',))
        acc_value = got[0]
        digest_acc = newacc
        state['free'] = newfr

    step = 1
    scalar_terms = []
    for name, params, ret in lin_fns:
        args_text, args_val = [], []
        for _n, t in params:
            text, val = build_value_text(rng, t, None, names)
            args_text.append(text)
            args_val.append(val)
        result = interp.call(name, args_val)
        call_text = f"{name}({', '.join(args_text)})"
        if isinstance(ret, list):
            rn = [f"w{step}_{i}" for i in range(len(ret))]
            code.append(f"let ({', '.join(rn)}) = {call_text} in ")
            for n, t, v in zip(rn, ret, result):
                if t in SCAL:
                    scalar_terms.append((to_int(n, t), value_of(v, t)))
                else:
                    digest(n, t, v, step)
                    step += 1
        else:
            digest(call_text, ret, result, step)
            step += 1
    for name, ty in loops:
        n = rng.randrange(0, 4)
        chain_text, chain_val = 'Zed', ('Zed',)
        for _ in range(min(n, 3)):
            chain_text, chain_val = f"Suc@{names.pop(0)}({chain_text})", ('Suc', chain_val)
        args = [p.literal(ty) for _ in range(3)]
        result = interp.call(name, [chain_val] + [a[1] for a in args] + [('Zed',)])
        rn = [f"w{step}_{i}" for i in range(4)]
        code.append(f"let ({', '.join(rn)}) = {name}({chain_text}, {', '.join(show(a) for a in args)}, Zed) in ")
        for n2, v in zip(rn[:3], result[:3]):
            scalar_terms.append((to_int(n2, ty), value_of(v, ty)))
        # the Fuel that comes back goes to the free list through a helper
        code.append(f"let {state['free']}x = fuelback({rn[3]}, {state['free']}) in ")
        state['free'] = state['free'] + 'x'
        step += 1
    for name, ty, scal in trmcs:
        text, val = build_value_text(rng, ty, None, names)
        args = [p.literal(t) for t in scal]
        result = interp.call(name, [val] + [a[1] for a in args])
        digest(f"{name}({text}, {', '.join(show(a) for a in args)})", ty, result, step)
        step += 1

    # cells no input needed go back to the free list
    for leftover in names:
        code.append(f"let {state['free']}y = More@{leftover}({state['free']}) in ")
        state['free'] = state['free'] + 'y'
    expected = acc_value
    final = digest_acc
    for position, values in enumerate(use_expected):
        n = f"u{position}"
        code.append(f"let {n} = use{position}({final}) in ")
        final = n
        for value in values:
            expected = (expected * 1000003 + value) & MASK[64]
    for position, (term, value) in enumerate(scalar_terms):
        n = f"k{position}"
        code.append(f"let {n} = +(*({final}, 1000003), {term}) in ")
        final = n
        expected = (expected * 1000003 + value) & MASK[64]
    tail = (f"match {state['free']} with End => Fail | More(zc, zr) => "
            f"Done@zc({final}, zr) end")
    body = "".join(code) + "".join(body_lets) + tail
    # take the cells from the free list
    for i in reversed(range(len(cell_names))):
        held = "End"
        for cell in cell_names[1:i]:
            held = f"More@{cell}({held})"
        fail = "Fail" if i == 0 else f"Done@{cell_names[0]}(0, {held})"
        nxt = f"r{i + 1}" if i + 1 < len(cell_names) else 'fr'
        body = f"match r{i} with End => {fail} | More({cell_names[i]}, {nxt}) => {body} end"
    if not cell_names:
        body = f"let fr = r0 in {body}"
    extra = ("fn fuelback(n: Fuel, free: Free) : Free = match n with Zed => free "
             "| Suc(d, r) => fuelback(r, More@d(free)) end;")
    main = extra + "\n" + "\n".join(use_fns) + "\n" + f"fn main(r0: Free) : Res = {body};"
    return p.text(main).encode(), expected
