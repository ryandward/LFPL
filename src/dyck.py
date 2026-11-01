"""Lex and Parse: bytes to the certified Dyck event word (spec section 2.4).

    Sigma_ev = { open(k, pi), sep(k, i), leaf(k, pi), close(k) }

Both stages are deterministic pushdown transducers on the ring. Neither builds
a tree: every node exists only as an event in flight or as one parser frame.

Concrete syntax is prefix for every operator, `+(a, b)`, because the spec asks
for a preorder word and a top-down destination, and an O(D) transducer cannot
emit open(+) before reading an e1 of unbounded size. Every other construct
already begins with a keyword or a name, so one token of lookahead decides it.

    program ::= decl*
    decl    ::= ('fn'|'handler') id '(' [id ':' ty {',' id ':' ty}] ')' ':' res ['@' id] '=' e ';'
              | 'data' Name ['=' ctor {'|' ctor}] ';'
              | 'record' Name '{' id ':' ty {',' id ':' ty} '}' ';'
              | 'state' id ':' ty ';'
    ctor    ::= Name ['(' id ':' ty {',' id ':' ty} ')']
    ty      ::= Name | 'Array' '[' int ']' Name | 'Fin' int
    res     ::= ty | '(' ty ',' ty {',' ty} ')'
    e       ::= id | int | 'true' | 'false'
              | ('+'|'-'|'*'|'&'|'|'|'^'|'<<'|'>>') '(' e ',' e ')'
              | ('<'|'<='|'>'|'>='|'=='|'!=') '(' e ',' e ')'
              | '&&' '(' e ',' e ')' | '||' '(' e ',' e ')' | '!' '(' e ')'
              | 'let' id '=' e 'in' e | 'if' e 'then' e 'else' e
              | 'let' '(' id ',' id {',' id} ')' '=' e 'in' e
              | '(' e ',' e {',' e} ')' | 'idx' '(' e ',' int ')' | 'get' '(' e ',' e ')'
              | Name | Name '@' id '(' [e {',' e}] ')'
              | 'match' e 'with' ['|'] arm {'|' arm} 'end'
              | '{' e 'with' id '=' e '}'
              | id '(' [e {',' e}] ')'
              | 'pack' '<' int {',' int} '>' '(' e {',' e} ')'
    arm     ::= Name ['(' id {',' id} ')'] '=>' e
"""
from __future__ import annotations

import struct
from typing import Callable, Iterable, Optional

from .ring import BYTE_EOF, Reject, Ring, Stage
from .tokens import PAYLOAD_BYTES

DEFAULT_D_MAX = 64
L_MAX = PAYLOAD_BYTES  # a lexeme fits one cell, so the pending chain is <= 15

# ------------------------------------------------------------------- tokens

_TOKENS = tuple((
    "EOF IDENT NAME INT LET IN IF THEN ELSE MATCH WITH END TRUE FALSE FN DATA "
    "RECORD PACK LPAREN RPAREN LBRACE RBRACE COMMA COLON SEMI AT EQ ARROW BAR "
    "PLUS MINUS STAR AMP CARET SHL SHR LT LE GT GE EQEQ NE ANDAND OROR BANG "
    "HANDLER STATE LBRACKET RBRACKET "
).split())
(
    T_EOF, T_IDENT, T_NAME, T_INT, T_LET, T_IN, T_IF, T_THEN,
    T_ELSE, T_MATCH, T_WITH, T_END, T_TRUE, T_FALSE, T_FN, T_DATA,
    T_RECORD, T_PACK, T_LPAREN, T_RPAREN, T_LBRACE, T_RBRACE, T_COMMA, T_COLON,
    T_SEMI, T_AT, T_EQ, T_ARROW, T_BAR, T_PLUS, T_MINUS, T_STAR,
    T_AMP, T_CARET, T_SHL, T_SHR, T_LT, T_LE, T_GT, T_GE,
    T_EQEQ, T_NE, T_ANDAND, T_OROR, T_BANG,
    T_HANDLER, T_STATE, T_LBRACKET, T_RBRACKET,
) = range(len(_TOKENS))

_KEYWORDS = {b"let": T_LET, b"in": T_IN, b"if": T_IF, b"then": T_THEN,
             b"else": T_ELSE, b"match": T_MATCH, b"with": T_WITH, b"end": T_END,
             b"true": T_TRUE, b"false": T_FALSE, b"fn": T_FN, b"data": T_DATA,
             b"record": T_RECORD, b"pack": T_PACK, b"handler": T_HANDLER, b"state": T_STATE}
_ONE = {ord("("): T_LPAREN, ord(")"): T_RPAREN, ord("{"): T_LBRACE,
        ord("}"): T_RBRACE, ord(","): T_COMMA, ord(":"): T_COLON,
        ord(";"): T_SEMI, ord("@"): T_AT, ord("="): T_EQ, ord("|"): T_BAR,
        ord("+"): T_PLUS, ord("-"): T_MINUS, ord("*"): T_STAR, ord("&"): T_AMP,
        ord("^"): T_CARET, ord("<"): T_LT, ord(">"): T_GT, ord("!"): T_BANG,
        ord("["): T_LBRACKET, ord("]"): T_RBRACKET}
_TWO = {b"=>": T_ARROW, b"==": T_EQEQ, b"!=": T_NE, b"<=": T_LE, b">=": T_GE,
        b"<<": T_SHL, b">>": T_SHR, b"&&": T_ANDAND, b"||": T_OROR}
_TWO_FIRST = {pair[0] for pair in _TWO}

BINOPS = {T_PLUS: "add", T_MINUS: "sub", T_STAR: "imul", T_AMP: "and",
          T_BAR: "or", T_CARET: "xor", T_SHL: "shl", T_SHR: "shr"}
OP_NAMES = ("add", "sub", "imul", "and", "or", "xor", "shl", "shr")
CMPS = {T_LT: "b", T_LE: "be", T_GT: "a", T_GE: "ae", T_EQEQ: "e", T_NE: "ne"}
CC_NAMES = ("b", "be", "a", "ae", "e", "ne")

# ------------------------------------------------------------------- events

OPEN, SEP, LEAF, CLOSE = range(4)
_KINDS = tuple((
    "FN PARAM TYPE RESWORD DATA CTORDECL FIELD RECORD LET BINDER IF MATCH ARM "
    "WITH BINOP CMP AND OR NOT CALL CTOR DIAMOND PACK VAR INT BOOL NULLCTOR "
    "EOF HANDLER STATE TARRAY TFIN TTUPLE TUPLE IDX GET RTDECL RTDECLMORE "
).split())
(
    K_FN, K_PARAM, K_TYPE, K_RESWORD, K_DATA, K_CTORDECL, K_FIELD, K_RECORD,
    K_LET, K_BINDER, K_IF, K_MATCH, K_ARM, K_WITH, K_BINOP, K_CMP,
    K_AND, K_OR, K_NOT, K_CALL, K_CTOR, K_DIAMOND, K_PACK, K_VAR,
    K_INT, K_BOOL, K_NULLCTOR, K_EOF,
    K_HANDLER, K_STATE, K_TARRAY, K_TFIN, K_TTUPLE, K_TUPLE, K_IDX, K_GET, K_RTDECL, K_RTDECLMORE,
) = range(len(_KINDS))

# What an array's element can be, as Check and the payload of leaf(TARRAY) name it.
ELEMENT_CODES = {b"Int8": 1, b"Int16": 2, b"Int32": 3, b"Int64": 4, b"Bool": 5}

def ev_tag(cls: int, kind: int) -> int:
    return cls << 6 | kind


def ev_class(tag: int) -> int:
    return tag >> 6


def ev_kind(tag: int) -> int:
    return tag & 63


def show_event(tag: int, payload: bytes) -> str:
    cls = ("open", "sep", "leaf", "close")[ev_class(tag)]
    kind = _KINDS[ev_kind(tag)]
    if ev_class(tag) == SEP:
        return f"sep({kind},{payload[0]})"
    if ev_class(tag) == CLOSE:
        return f"close({kind})"
    text = payload.rstrip(b"\0")
    return f"{cls}({kind},{text.decode('latin-1')})" if text else f"{cls}({kind})"


# -------------------------------------------------------------------- Lex

_M_NONE, _M_WORD, _M_NUMBER, _M_SYMBOL, _M_COMMENT = range(5)


class Lexer(Stage):
    """Bytes to tokens. It never needs more cells than its lexeme had bytes."""

    name = "lex"

    def __init__(self) -> None:
        super().__init__()
        self.mode = _M_NONE
        self.chain: list[int] = []  # the pending lexeme's cells, in sigma
        self.value = 0
        self.first = 0
        self.held: Optional[int] = None
        self.line = 1

    def has_work(self) -> bool:
        return self.held is not None

    def fire(self, cell: Optional[int]) -> None:
        if cell is None:
            cell = self.held
            self.held = None
        assert cell is not None
        ring = self.ring
        if ring.store.tag(cell) == BYTE_EOF:
            if self.mode in (_M_WORD, _M_NUMBER, _M_SYMBOL):
                self._complete()
                self.held = cell
            else:
                ring.pass_(self, cell, T_EOF)
            return
        byte = ring.store.payload(cell)[0]
        if self.mode == _M_COMMENT:
            if byte == 10:
                self.line += 1
                self.mode = _M_NONE
            ring.drop(self, cell)
        elif self.mode == _M_WORD:
            if _is_word(byte):
                if len(self.chain) == L_MAX:
                    raise Reject(f"line {self.line}: name longer than {L_MAX} bytes")
                self.chain.append(cell)
            else:
                self._complete()
                self.held = cell
        elif self.mode == _M_NUMBER:
            if 48 <= byte <= 57:
                self.value = self.value * 10 + byte - 48
                if self.value >= 1 << 64:
                    raise Reject(f"line {self.line}: integer exceeds 64 bits")
                ring.drop(self, cell)
            elif _is_word(byte):
                raise Reject(f"line {self.line}: malformed number")
            else:
                self._complete()
                self.held = cell
        elif self.mode == _M_SYMBOL:
            pair = bytes((self.first, byte))
            if pair in _TWO:
                head = self.chain.pop()
                self.mode = _M_NONE
                ring.pass_(self, head, _TWO[pair])
                ring.drop(self, cell)
            else:
                self._complete()
                self.held = cell
        else:
            self._begin(cell, byte)

    def _begin(self, cell: int, byte: int) -> None:
        ring = self.ring
        if byte in b" \t\r\n":
            if byte == 10:
                self.line += 1
            ring.drop(self, cell)
        elif byte == ord("#"):
            self.mode = _M_COMMENT
            ring.drop(self, cell)
        elif 48 <= byte <= 57:
            self.mode, self.value = _M_NUMBER, byte - 48
            self.chain = [cell]
        elif _is_word(byte):
            self.mode = _M_WORD
            self.chain = [cell]
        elif byte in _TWO_FIRST:
            self.mode, self.first = _M_SYMBOL, byte
            self.chain = [cell]
        elif byte in _ONE:
            ring.pass_(self, cell, _ONE[byte])
        else:
            raise Reject(f"line {self.line}: unexpected byte {byte:#x}")

    def _complete(self) -> None:
        """Pack the lexeme into its head cell and DROP the rest of the chain."""
        ring = self.ring
        head, rest = self.chain[0], self.chain[1:]
        if self.mode == _M_NUMBER:
            ring.pass_(self, head, T_INT, struct.pack("<Q", self.value))
        elif self.mode == _M_SYMBOL:
            ring.pass_(self, head, _ONE[self.first])
        else:
            text = bytes(ring.store.payload(c)[0] for c in self.chain)
            tag = _KEYWORDS.get(text)
            if tag is None:
                tag = T_NAME if 65 <= text[0] <= 90 else T_IDENT
            ring.pass_(self, head, tag, text)
        for extra in rest:
            ring.drop(self, extra)
        self.chain = []
        self.mode = _M_NONE


def _is_word(byte: int) -> bool:
    return 48 <= byte <= 57 or 65 <= byte <= 90 or 97 <= byte <= 122 or byte == 95


# ------------------------------------------------------------------ Parse

# A frame is (state, aux). `tag` is the construct kind; declarations use their
# own event kinds. Each state says what the next token must be.
S_NAME, S_LPAREN, S_PARAM_OR_END, S_PARAM_COLON, S_PARAM_TYPE, S_PARAM_NEXT = range(1, 7)
S_RET_COLON, S_RET_TYPE, S_RET_NEXT, S_RES_NAME, S_BODY_EQ, S_BODY, S_SEMI = range(7, 14)
S_DATA_EQ, S_DATA_CTOR, S_DATA_NEXT = range(14, 17)
S_CD_NEXT, S_CD_FIELD, S_CD_COLON, S_CD_TYPE, S_CD_AFTER = range(17, 22)
S_REC_LBRACE, S_REC_FIELD, S_REC_COLON, S_REC_TYPE, S_REC_NEXT, S_REC_SEMI = range(22, 28)
S_BINDER, S_LET_EQ, S_LET_E1, S_LET_IN, S_LET_BODY = range(28, 33)
S_IF_COND, S_IF_THEN_KW, S_IF_THEN, S_IF_ELSE_KW, S_IF_ELSE = range(33, 38)
S_M_SCRUT, S_M_WITH, S_M_ARM, S_M_NEXT = range(38, 42)
S_ARM_OPEN, S_ARM_BINDER, S_ARM_BNEXT, S_ARM_ARROW, S_ARM_BODY = range(42, 47)
S_W_E, S_W_KW, S_W_FIELD, S_W_EQ, S_W_V, S_W_END = range(47, 53)
S_OP_LP, S_OP_E1, S_OP_COMMA, S_OP_E2, S_OP_END = range(53, 58)
S_NOT_LP, S_NOT_E, S_NOT_END = range(58, 61)
S_ARGS_FIRST, S_ARG, S_ARGS_NEXT = range(61, 64)
S_CTOR_D, S_CTOR_LP = range(64, 66)
S_PACK_LT, S_PACK_W, S_PACK_NEXT, S_PACK_LP = range(66, 70)
S_TA_LB, S_TA_N, S_TA_RB, S_TA_ELEM, S_TF_N, S_TT_TYPE, S_TT_NEXT = range(70, 77)
S_LTB, S_LTB_NEXT, S_ST_COLON, S_ST_TYPE, S_ST_SEMI = range(77, 82)

# Expression-expecting states, and what follows the expression.
_AFTER = {S_BODY: S_SEMI, S_LET_E1: S_LET_IN, S_IF_COND: S_IF_THEN_KW,
          S_IF_THEN: S_IF_ELSE_KW, S_M_SCRUT: S_M_WITH, S_W_E: S_W_KW,
          S_W_V: S_W_END, S_OP_E1: S_OP_COMMA, S_OP_E2: S_OP_END,
          S_NOT_E: S_NOT_END, S_ARG: S_ARGS_NEXT, S_ARGS_FIRST: S_ARGS_NEXT}
# Expression-expecting states whose construct has no closing token: when the
# child completes, the frame's own cell carries the close event.
_IMPLICIT_CLOSE = {S_LET_BODY: K_LET, S_IF_ELSE: K_IF, S_ARM_BODY: K_ARM}
_EXPECTS_EXPR = frozenset(_AFTER) | frozenset(_IMPLICIT_CLOSE)


_PACK_FIELDS = 8  # 64 bits at no fewer than 8 bits a field


class Parser(Stage):
    """Tokens to the Dyck word. Stack depth is the current nesting depth."""

    name = "parse"

    def __init__(self, d_max: int = DEFAULT_D_MAX):
        super().__init__()
        self.d_max = d_max
        self.held: Optional[int] = None    # one token of pushback
        self.name_cell: Optional[int] = None   # the pending identifier
        self.unwinding = False

    def attach(self, ring: Ring) -> None:
        super().attach(ring)
        self.frames = self.stack()

    def depth(self) -> int:
        return len(self.frames)

    def has_work(self) -> bool:
        return self.unwinding or self.held is not None

    # --- frame helpers

    def _push(self, kind: int, state: int, opener: Optional[int]) -> None:
        """Open a frame, paid by the opener's own cell or, without one, by F."""
        if len(self.frames) >= self.d_max:
            raise Reject(f"nesting too deep: more than {self.d_max} open "
                         f"constructs (the Depth rule)")
        self.frames.push(kind, bytes((state, 0)) + bytes(_PACK_FIELDS), first=opener)

    def _state(self) -> int:
        return self.frames.peek()[1][0]

    def _set(self, state: int, aux: Optional[int] = None) -> None:
        tag, data = self.frames.peek()
        self.frames.rewrite(
            len(self.frames) - 1, tag,
            bytes((state, data[1] if aux is None else aux)) + data[2:2 + _PACK_FIELDS])

    def _fail(self, what: str, tag: Optional[int] = None) -> Reject:
        line = self.ring.stages[0].line
        got = "" if tag is None else f", got {_TOKENS[tag]}"
        return Reject(f"near line {line}: expected {what}{got}")

    def _emit(self, cell: int, cls: int, kind: int, payload: bytes = b"") -> None:
        self.ring.pass_(self, cell, ev_tag(cls, kind), payload)

    def _emit_new(self, cls: int, kind: int, payload: bytes = b"") -> None:
        self.ring.emit_fresh(self, ev_tag(cls, kind), payload)

    # --- completion of an expression

    def _complete(self) -> None:
        """A child expression ended. Nothing to read: no postfix form exists."""
        if not len(self.frames):
            raise self._fail("a declaration")
        self.unwinding = True

    def _unwind(self) -> None:
        state = self._state()
        kind = self.frames.peek()[0]
        if state in _IMPLICIT_CLOSE:
            self.frames.pop()
            self._emit_new(CLOSE, _IMPLICIT_CLOSE[state])
            if kind == K_ARM:
                self.unwinding = False  # the match frame decides what is next
        else:
            self._set(_AFTER[state])
            self.unwinding = False

    # --- the transition function

    def fire(self, cell: Optional[int]) -> None:
        if self.unwinding:
            self._unwind()
            return
        if cell is None:
            cell = self.held
            self.held = None
        assert cell is not None
        tag = self.ring.store.tag(cell)
        payload = self.ring.store.payload(cell)
        if self.name_cell is not None:
            self._resolve_name(cell, tag)
            return
        if not len(self.frames):
            self._declaration(cell, tag)
            return
        state = self._state()
        if state in _EXPECTS_EXPR:
            self._expression(cell, tag, payload)
        else:
            self._structural(cell, tag, payload, state)

    def _resolve_name(self, cell: int, tag: int) -> None:
        """The one token of lookahead: a name followed by '(' or '@' opens."""
        name_cell = self.name_cell
        assert name_cell is not None
        self.name_cell = None
        name_tag = self.ring.store.tag(name_cell)
        text = self.ring.store.payload(name_cell)
        if name_tag == T_IDENT and tag == T_LPAREN:
            self._push(K_CALL, S_ARGS_FIRST, cell)
            self._emit(name_cell, OPEN, K_CALL, text)
        elif name_tag == T_NAME and tag == T_AT:
            self._push(K_CTOR, S_CTOR_D, cell)
            self._emit(name_cell, OPEN, K_CTOR, text)
        else:
            kind = K_VAR if name_tag == T_IDENT else K_NULLCTOR
            self._emit(name_cell, LEAF, kind, text)
            self.held = cell
            self._complete()

    def _declaration(self, cell: int, tag: int) -> None:
        if tag == T_EOF:
            self._emit(cell, LEAF, K_EOF)
        elif tag == T_FN:
            self._push(K_FN, S_NAME, cell)
        elif tag == T_HANDLER:
            self._push(K_HANDLER, S_NAME, cell)
        elif tag == T_STATE:
            self._push(K_STATE, S_NAME, cell)
        elif tag == T_DATA:
            self._push(K_DATA, S_NAME, cell)
        elif tag == T_RECORD:
            self._push(K_RECORD, S_NAME, cell)
        else:
            raise self._fail("a declaration", tag)

    def _expression(self, cell: int, tag: int, payload: bytes) -> None:
        if tag == T_INT:
            self._emit(cell, LEAF, K_INT, payload[:8])
            self._complete()
        elif tag in (T_TRUE, T_FALSE):
            self._emit(cell, LEAF, K_BOOL, bytes((tag == T_TRUE,)))
            self._complete()
        elif tag in (T_IDENT, T_NAME):
            self.name_cell = cell  # stays in sigma until the next token
        elif tag == T_LET:
            self._push(K_LET, S_BINDER, cell)
            self._emit_new(OPEN, K_LET)
        elif tag == T_IF:
            self._push(K_IF, S_IF_COND, cell)
            self._emit_new(OPEN, K_IF)
        elif tag == T_MATCH:
            self._push(K_MATCH, S_M_SCRUT, cell)
            self._emit_new(OPEN, K_MATCH)
        elif tag == T_LPAREN:
            self._push(K_TUPLE, S_ARG, cell)
            self._emit_new(OPEN, K_TUPLE)
        elif tag == T_LBRACE:
            self._push(K_WITH, S_W_E, cell)
            self._emit_new(OPEN, K_WITH)
        elif tag in BINOPS:
            self._push(K_BINOP, S_OP_LP, cell)
            self._emit_new(OPEN, K_BINOP, bytes((OP_NAMES.index(BINOPS[tag]),)))
        elif tag in CMPS:
            self._push(K_CMP, S_OP_LP, cell)
            self._emit_new(OPEN, K_CMP, bytes((CC_NAMES.index(CMPS[tag]),)))
        elif tag == T_ANDAND:
            self._push(K_AND, S_OP_LP, cell)
            self._emit_new(OPEN, K_AND)
        elif tag == T_OROR:
            self._push(K_OR, S_OP_LP, cell)
            self._emit_new(OPEN, K_OR)
        elif tag == T_BANG:
            self._push(K_NOT, S_NOT_LP, cell)
            self._emit_new(OPEN, K_NOT)
        elif tag == T_PACK:
            self._push(K_PACK, S_PACK_LT, cell)
        elif tag == T_RPAREN and self._state() == S_ARGS_FIRST:
            self._close_args(cell)
        else:
            raise self._fail("an expression", tag)

    def _close_args(self, cell: int) -> None:
        kind = self.frames.peek()[0]
        self.frames.pop()
        self._emit(cell, CLOSE, kind)
        self._complete()

    def _structural(self, cell: int, tag: int, payload: bytes, state: int) -> None:
        kind = self.frames.peek()[0]
        ring = self.ring

        def require(wanted: int, what: str) -> None:
            if tag != wanted:
                raise self._fail(what, tag)

        def next_state(state_after: int, what: str, wanted: int) -> None:
            require(wanted, what)
            self._set(state_after)
            ring.drop(self, cell)

        if state == S_NAME:
            if kind in (K_FN, K_HANDLER):
                require(T_IDENT, "a function name")
                self._set(S_LPAREN)
                self._emit(cell, OPEN, kind, payload)
            elif kind == K_STATE:
                require(T_IDENT, "a state name")
                self._set(S_ST_COLON)
                self._emit(cell, OPEN, K_STATE, payload)
            else:
                require(T_NAME, "a type name")
                self._set(S_DATA_EQ if kind == K_DATA else S_REC_LBRACE)
                self._emit(cell, OPEN, kind, payload)
        elif state == S_LPAREN:
            next_state(S_PARAM_OR_END, "'('", T_LPAREN)
        elif state in (S_PARAM_OR_END, S_REC_FIELD, S_CD_FIELD):
            if tag == T_RPAREN and state == S_PARAM_OR_END:
                self._emit(cell, SEP, kind, b"\0")
                self._set(S_RET_COLON)
            else:
                require(T_IDENT, "a field or parameter name")
                self._set({S_PARAM_OR_END: S_PARAM_COLON, S_REC_FIELD: S_REC_COLON,
                           S_CD_FIELD: S_CD_COLON}[state])
                self._emit(cell, LEAF,
                           K_PARAM if state == S_PARAM_OR_END else K_FIELD, payload)
        elif state in (S_PARAM_COLON, S_REC_COLON, S_CD_COLON, S_RET_COLON):
            next_state({S_PARAM_COLON: S_PARAM_TYPE, S_REC_COLON: S_REC_TYPE,
                        S_CD_COLON: S_CD_TYPE, S_RET_COLON: S_RET_TYPE}[state],
                       "':'", T_COLON)
        elif state in (S_PARAM_TYPE, S_REC_TYPE, S_CD_TYPE, S_RET_TYPE, S_ST_TYPE, S_TT_TYPE):
            if tag == T_LPAREN and state == S_RET_TYPE:
                self._set(S_RET_NEXT)
                self._push(K_TTUPLE, S_TT_TYPE, None)
                self._emit(cell, OPEN, K_TTUPLE)
                return
            require(T_NAME, "a type")
            self._set({S_PARAM_TYPE: S_PARAM_NEXT, S_REC_TYPE: S_REC_NEXT, S_CD_TYPE: S_CD_AFTER,
                       S_RET_TYPE: S_RET_NEXT, S_ST_TYPE: S_ST_SEMI, S_TT_TYPE: S_TT_NEXT}[state])
            name = payload.rstrip(b"\0")
            if name == b"Array":
                self._push(K_TARRAY, S_TA_LB, cell)
            elif name == b"Fin":
                self._push(K_TFIN, S_TF_N, cell)
            else:
                self._emit(cell, LEAF, K_TYPE, payload)
        elif state == S_TA_LB:
            next_state(S_TA_N, "'['", T_LBRACKET)
        elif state == S_TA_N:
            require(T_INT, "an array length")
            frame_tag, data = self.frames.peek()
            self.frames.rewrite(len(self.frames) - 1, frame_tag,
                                bytes((S_TA_RB, 0)) + payload[:_PACK_FIELDS])
            ring.drop(self, cell)
        elif state == S_TA_RB:
            next_state(S_TA_ELEM, "']'", T_RBRACKET)
        elif state == S_TA_ELEM:
            require(T_NAME, "an array's element type")
            code = ELEMENT_CODES.get(payload.rstrip(b"\0"))
            if code is None:
                raise self._fail("a scalar element type: an array of linear values waits for swap")
            length = self.frames.peek()[1][2:2 + _PACK_FIELDS]
            self.frames.pop()
            self._emit(cell, LEAF, K_TARRAY, length + bytes((code,)))
        elif state == S_TF_N:
            require(T_INT, "the bound of a Fin")
            self.frames.pop()
            self._emit(cell, LEAF, K_TFIN, payload[:_PACK_FIELDS])
        elif state == S_TT_NEXT:
            count = self.frames.peek()[1][1]
            if tag == T_COMMA:
                self._set(S_TT_TYPE, min(count + 1, 255))
                ring.drop(self, cell)
            else:
                require(T_RPAREN, "',' or ')'")
                if count == 0:
                    raise self._fail("a second type: a tuple has at least two members")
                self.frames.pop()
                self._emit(cell, CLOSE, K_TTUPLE)
        elif state == S_ST_COLON:
            next_state(S_ST_TYPE, "':'", T_COLON)
        elif state == S_ST_SEMI:
            require(T_SEMI, "';'")
            self.frames.pop()
            self._emit(cell, CLOSE, K_STATE)
        elif state == S_PARAM_NEXT:
            if tag == T_COMMA:
                self._set(S_PARAM_OR_END)
                ring.drop(self, cell)
            else:
                require(T_RPAREN, "',' or ')'")
                self._emit(cell, SEP, kind, b"\0")
                self._set(S_RET_COLON)
        elif state == S_RET_NEXT:
            if tag == T_AT:
                self._set(S_RES_NAME)
                ring.drop(self, cell)
            else:
                require(T_EQ, "'=' or '@'")
                self._emit(cell, SEP, kind, b"\1")
                self._set(S_BODY)
        elif state == S_RES_NAME:
            require(T_IDENT, "a parameter name after '@'")
            self._emit(cell, LEAF, K_RESWORD, payload)
            self._set(S_BODY_EQ)
        elif state == S_BODY_EQ:
            require(T_EQ, "'='")
            self._emit(cell, SEP, kind, b"\1")
            self._set(S_BODY)
        elif state == S_SEMI:
            require(T_SEMI, "';'")
            self.frames.pop()
            self._emit(cell, CLOSE, kind)
        elif state == S_DATA_EQ:
            if tag == T_SEMI:  # a forward declaration, for mutually recursive types
                self.frames.pop()
                self._emit(cell, CLOSE, K_DATA)
            else:
                next_state(S_DATA_CTOR, "'='", T_EQ)
        elif state == S_DATA_CTOR:
            require(T_NAME, "a constructor")
            self._set(S_DATA_NEXT)
            self._push(K_CTORDECL, S_CD_NEXT, None)
            self._emit(cell, OPEN, K_CTORDECL, payload)
        elif state == S_DATA_NEXT:
            if tag == T_BAR:
                self._set(S_DATA_CTOR)
                ring.drop(self, cell)
            else:
                require(T_SEMI, "'|' or ';'")
                self.frames.pop()
                self._emit(cell, CLOSE, K_DATA)
        elif state == S_CD_NEXT:
            if tag == T_LPAREN:
                self._set(S_CD_FIELD)
                ring.drop(self, cell)
            else:
                self.frames.pop()
                self.held = cell
                self._emit_new(CLOSE, K_CTORDECL)
        elif state == S_CD_AFTER:
            if tag == T_COMMA:
                self._set(S_CD_FIELD)
                ring.drop(self, cell)
            else:
                require(T_RPAREN, "',' or ')'")
                self.frames.pop()
                self._emit(cell, CLOSE, K_CTORDECL)
        elif state == S_REC_LBRACE:
            next_state(S_REC_FIELD, "'{'", T_LBRACE)
        elif state == S_REC_NEXT:
            if tag == T_COMMA:
                self._set(S_REC_FIELD)
                ring.drop(self, cell)
            else:
                next_state(S_REC_SEMI, "',' or '}'", T_RBRACE)
        elif state == S_REC_SEMI:
            require(T_SEMI, "';'")
            self.frames.pop()
            self._emit(cell, CLOSE, K_RECORD)
        elif state == S_BINDER:
            if tag == T_LPAREN:
                self._set(S_LTB)
                ring.drop(self, cell)
                return
            require(T_IDENT, "a binder")
            self._set(S_LET_EQ)
            self._emit(cell, LEAF, K_BINDER, payload)
        elif state == S_LTB:
            require(T_IDENT, "a binder")
            self._set(S_LTB_NEXT)
            self._emit(cell, LEAF, K_BINDER, payload)
        elif state == S_LTB_NEXT:
            count = self.frames.peek()[1][1]
            if tag == T_COMMA:
                self._set(S_LTB, min(count + 1, 255))
                ring.drop(self, cell)
            else:
                if count == 0:
                    raise self._fail("a second name: a tuple has at least two members")
                next_state(S_LET_EQ, "',' or ')'", T_RPAREN)
        elif state == S_LET_EQ:
            next_state(S_LET_E1, "'='", T_EQ)
        elif state == S_LET_IN:
            require(T_IN, "'in'")
            self._set(S_LET_BODY)
            self._emit(cell, SEP, K_LET, b"\1")
        elif state == S_IF_THEN_KW:
            require(T_THEN, "'then'")
            self._set(S_IF_THEN)
            self._emit(cell, SEP, K_IF, b"\1")
        elif state == S_IF_ELSE_KW:
            require(T_ELSE, "'else'")
            self._set(S_IF_ELSE)
            self._emit(cell, SEP, K_IF, b"\2")
        elif state == S_M_WITH:
            require(T_WITH, "'with'")
            self._set(S_M_ARM)
            self._emit(cell, SEP, K_MATCH, b"\1")
        elif state == S_M_ARM:
            if tag == T_BAR:
                ring.drop(self, cell)  # the optional leading bar
            else:
                require(T_NAME, "a constructor pattern")
                self._set(S_M_NEXT)
                self._push(K_ARM, S_ARM_OPEN, None)
                self._emit(cell, OPEN, K_ARM, payload)
        elif state == S_M_NEXT:
            if tag == T_BAR:
                self._set(S_M_ARM)
                ring.drop(self, cell)
            else:
                require(T_END, "'|' or 'end'")
                self.frames.pop()
                self._emit(cell, CLOSE, K_MATCH)
                self._complete()
        elif state == S_ARM_OPEN:
            if tag == T_LPAREN:
                self._set(S_ARM_BINDER)
                ring.drop(self, cell)
            else:
                require(T_ARROW, "'(' or '=>'")
                self._set(S_ARM_BODY)
                self._emit(cell, SEP, K_ARM, b"\1")
        elif state == S_ARM_BINDER:
            require(T_IDENT, "a pattern binder")
            self._set(S_ARM_BNEXT)
            self._emit(cell, LEAF, K_BINDER, payload)
        elif state == S_ARM_BNEXT:
            if tag == T_COMMA:
                self._set(S_ARM_BINDER)
                ring.drop(self, cell)
            else:
                next_state(S_ARM_ARROW, "',' or ')'", T_RPAREN)
        elif state == S_ARM_ARROW:
            require(T_ARROW, "'=>'")
            self._set(S_ARM_BODY)
            self._emit(cell, SEP, K_ARM, b"\1")
        elif state == S_W_KW:
            require(T_WITH, "'with'")
            self._set(S_W_FIELD)
            self._emit(cell, SEP, K_WITH, b"\1")
        elif state == S_W_FIELD:
            require(T_IDENT, "a field name")
            self._set(S_W_EQ)
            self._emit(cell, LEAF, K_FIELD, payload)
        elif state == S_W_EQ:
            require(T_EQ, "'='")
            self._set(S_W_V)
            self._emit(cell, SEP, K_WITH, b"\2")
        elif state == S_W_END:
            require(T_RBRACE, "'}'")
            self.frames.pop()
            self._emit(cell, CLOSE, K_WITH)
            self._complete()
        elif state == S_OP_LP:
            next_state(S_OP_E1, "'('", T_LPAREN)
        elif state == S_OP_COMMA:
            require(T_COMMA, "','")
            self._set(S_OP_E2)
            self._emit(cell, SEP, kind, b"\1")
        elif state == S_OP_END:
            require(T_RPAREN, "')'")
            self.frames.pop()
            self._emit(cell, CLOSE, kind)
            self._complete()
        elif state == S_NOT_LP:
            next_state(S_NOT_E, "'('", T_LPAREN)
        elif state == S_NOT_END:
            require(T_RPAREN, "')'")
            self.frames.pop()
            self._emit(cell, CLOSE, K_NOT)
            self._complete()
        elif state == S_ARGS_NEXT:
            if tag == T_COMMA:
                aux = self.frames.peek()[1][1] + 1
                if aux > 255:
                    raise self._fail("')': a construct takes at most 256 arguments")
                self._set(S_ARG, aux)
                self._emit(cell, SEP, kind, bytes((aux,)))
            else:
                require(T_RPAREN, "',' or ')'")
                if kind == K_TUPLE and self.frames.peek()[1][1] == 0:
                    raise self._fail("a second member: a tuple has at least two members")
                self._close_args(cell)
        elif state == S_CTOR_D:
            require(T_IDENT, "the diamond variable after '@'")
            self._set(S_CTOR_LP)
            self._emit(cell, LEAF, K_DIAMOND, payload)
        elif state == S_CTOR_LP:
            next_state(S_ARGS_FIRST, "'('", T_LPAREN)
        elif state == S_PACK_LT:
            next_state(S_PACK_W, "'<'", T_LT)
        elif state == S_PACK_W:
            require(T_INT, "a field width")
            width = struct.unpack_from("<Q", payload)[0]
            if width not in (8, 16, 32):
                raise self._fail("a width of 8, 16 or 32")
            frame_tag, data = self.frames.peek()
            count = data[1]
            if count == _PACK_FIELDS:
                raise self._fail(f"at most {_PACK_FIELDS} pack fields")
            codes = bytearray(data[2:2 + _PACK_FIELDS])
            codes[count] = width
            self.frames.rewrite(len(self.frames) - 1, frame_tag,
                                bytes((S_PACK_NEXT, count + 1)) + bytes(codes))
            ring.drop(self, cell)
        elif state == S_PACK_NEXT:
            if tag == T_COMMA:
                self._set(S_PACK_W)
                ring.drop(self, cell)
            else:
                next_state(S_PACK_LP, "',' or '>'", T_GT)
        elif state == S_PACK_LP:
            require(T_LPAREN, "'('")
            data = self.frames.peek()[1]
            codes = data[2:2 + data[1]]
            self._set(S_ARGS_FIRST, 0)
            self._emit(cell, OPEN, K_PACK, codes)
        else:
            raise self._fail(f"state {state}", tag)


# ---------------------------------------------------------------- harness


class DyckError(Exception):
    """The event word is not well bracketed."""


class DyckValidator:
    """An independent O(D) check of the word, written against the alphabet only.

    Theorem 2.1 lets downstream stages assume balance. This exists so tests can
    confirm the assumption without trusting the parser's own frames.
    """

    def __init__(self) -> None:
        self.stack: list[int] = []
        self.peak = 0

    def feed(self, tag: int) -> None:
        cls, kind = ev_class(tag), ev_kind(tag)
        if cls == OPEN:
            self.stack.append(kind)
            self.peak = max(self.peak, len(self.stack))
        elif cls == CLOSE:
            if not self.stack or self.stack.pop() != kind:
                raise DyckError(f"close({_KINDS[kind]}) does not match")
        elif cls == SEP:
            if not self.stack or self.stack[-1] != kind:
                raise DyckError(f"sep({_KINDS[kind]}) outside its construct")

    def finish(self) -> None:
        if self.stack:
            raise DyckError(f"{len(self.stack)} constructs left open")


class EventSink(Stage):
    """A terminal stage for tests: hands each event to a callback."""

    name = "sink"

    def __init__(self, on_event: Callable[[int, bytes], None]):
        super().__init__()
        self.on_event = on_event

    def fire(self, cell: Optional[int]) -> None:
        assert cell is not None
        self.on_event(self.ring.store.tag(cell), self.ring.store.payload(cell))
        self.ring.drop(self, cell)


def parse_events(source: Iterable[int], *, cells: int = 256,
                 d_max: int = DEFAULT_D_MAX,
                 on_event: Callable[[int, bytes], None],
                 on_step: Optional[Callable[[Ring], None]] = None) -> Ring:
    """Run Ingest, Lex and Parse only, streaming events to `on_event`."""
    stages = [Lexer(), Parser(d_max), EventSink(on_event)]
    ring = Ring(cells, iter(source), stages, lambda tag, payload: None, on_step)
    ring.run()
    return ring
