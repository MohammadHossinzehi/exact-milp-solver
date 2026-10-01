"""Reader for a practical subset of the CPLEX LP text format.

    \\ comments start with a backslash
    Maximize
     profit: 5 x + 4 y + 3 z
    Subject To
     labour:   2 x + 3 y + z <= 5
     material: 4 x + y + 2 z <= 11
     range:    1 <= x + y <= 3
    Bounds
     0 <= x <= 2
     z free
     y >= -1
    General
     x
    Binary
     b
    End

One statement per line. Coefficients may be integers, decimals or
fractions written ``3/4``; all are read exactly. Variables default to
``0 <= v < inf`` as in CPLEX.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, List, Tuple

from .model import LinExpr, Model

_TOKEN = re.compile(r"\s*(?:(<=|>=|=<|=>|<|>|=)|(\d+(?:\.\d*)?(?:/\d+)?|\.\d+)|([A-Za-z_][\w.\[\]#]*)|([+\-:]))")
_SECTIONS = {
    "maximize": "max", "maximise": "max", "maximum": "max", "max": "max",
    "minimize": "min", "minimise": "min", "minimum": "min", "min": "min",
    "subject to": "st", "such that": "st", "st": "st", "s.t.": "st",
    "bounds": "bounds", "bound": "bounds",
    "general": "int", "generals": "int", "gen": "int", "integer": "int", "integers": "int",
    "binary": "bin", "binaries": "bin", "bin": "bin",
    "end": "end",
}
_INF = {"inf", "infinity", "+inf", "+infinity"}


class LPFormatError(ValueError):
    def __init__(self, msg, line_no=None):
        super().__init__(f"line {line_no}: {msg}" if line_no else msg)


def _tokenize(text: str, line_no: int) -> List[Tuple[str, str]]:
    out, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise LPFormatError(f"unexpected character {text[pos:].strip()[:10]!r}", line_no)
        pos = m.end()
        op, num, ident, sym = m.groups()
        if op:
            out.append(("op", {"=<": "<=", "=>": ">=", "<": "<=", ">": ">="}.get(op, op)))
        elif num:
            out.append(("num", num))
        elif ident:
            out.append(("id", ident))
        else:
            out.append(("sym", sym))
    return out


class _Reader:
    def __init__(self):
        self.model = Model()
        self.declared_bounds: Dict[str, Tuple] = {}

    def var(self, name):
        try:
            return self.model.get_var(name)
        except KeyError:
            return self.model.var(name)

    def expr(self, toks, line_no) -> LinExpr:
        """Parse a sum of signed terms: [+-] [coef] [var]."""
        e = LinExpr(self.model)
        i = 0
        if not toks:
            raise LPFormatError("empty expression", line_no)
        while i < len(toks):
            sign = Fraction(1)
            while i < len(toks) and toks[i] in (("sym", "+"), ("sym", "-")):
                if toks[i][1] == "-":
                    sign = -sign
                i += 1
            coef = None
            if i < len(toks) and toks[i][0] == "num":
                coef = Fraction(toks[i][1])
                i += 1
            if i < len(toks) and toks[i][0] == "id" and toks[i][1].lower() not in _INF:
                term = self.var(toks[i][1]) * (sign * (coef if coef is not None else 1))
                i += 1
            elif coef is not None:
                term = sign * coef
            else:
                raise LPFormatError("expected a number or variable", line_no)
            e = e + term
        return e

    @staticmethod
    def split_name(toks):
        if len(toks) >= 2 and toks[0][0] == "id" and toks[1] == ("sym", ":"):
            return toks[0][1], toks[2:]
        return None, toks

    @staticmethod
    def split_rel(toks):
        parts, cur, ops = [], [], []
        for t in toks:
            if t[0] == "op":
                parts.append(cur), ops.append(t[1])
                cur = []
            else:
                cur.append(t)
        parts.append(cur)
        return parts, ops

    def constraint(self, toks, line_no):
        name, toks = self.split_name(toks)
        parts, ops = self.split_rel(toks)
        if len(ops) == 1:
            lhs, rhs = self.expr(parts[0], line_no), self.expr(parts[1], line_no)
            con = {"<=": lhs <= rhs, ">=": lhs >= rhs, "=": lhs == rhs}[ops[0]]
            self.model.add(con, name)
        elif len(ops) == 2 and ops[0] == ops[1] and ops[0] in ("<=", ">="):
            lo, mid, hi = (self.expr(p, line_no) for p in parts)
            base = name or f"r{len(self.model.constraints)}"
            if ops[0] == "<=":
                self.model.add(lo <= mid, base + "_lo")
                self.model.add(mid <= hi, base + "_hi")
            else:
                self.model.add(lo >= mid, base + "_hi")
                self.model.add(mid >= hi, base + "_lo")
        else:
            raise LPFormatError("expected 'expr op expr' or a ranged 'a <= expr <= b'", line_no)

    def number(self, toks, line_no):
        sign = Fraction(1)
        i = 0
        while i < len(toks) and toks[i][0] == "sym" and toks[i][1] in "+-":
            if toks[i][1] == "-":
                sign = -sign
            i += 1
        rest = toks[i:]
        if len(rest) == 1 and rest[0][0] == "id" and rest[0][1].lower() in _INF:
            return None if sign > 0 else "-inf"
        if len(rest) == 1 and rest[0][0] == "num":
            return sign * Fraction(rest[0][1])
        raise LPFormatError("expected a number in bounds", line_no)

    def bound(self, toks, line_no):
        if len(toks) == 2 and toks[0][0] == "id" and toks[1][0] == "id" and toks[1][1].lower() == "free":
            v = self.var(toks[0][1])
            self.model.lb[v.index] = self.model.ub[v.index] = None
            return
        parts, ops = self.split_rel(toks)
        if len(ops) == 2:
            lo, (vname,), hi = parts[0], parts[1], parts[2]
            v = self.var(vname[1])
            self._set(v, ops[0], self.number(lo, line_no), "left", line_no)
            self._set(v, ops[1], self.number(hi, line_no), "right", line_no)
        elif len(ops) == 1 and len(parts[0]) == 1 and parts[0][0][0] == "id":
            v = self.var(parts[0][0][1])
            self._set(v, ops[0], self.number(parts[1], line_no), "right", line_no)
        elif len(ops) == 1 and len(parts[1]) == 1 and parts[1][0][0] == "id":
            v = self.var(parts[1][0][1])
            self._set(v, ops[0], self.number(parts[0], line_no), "left", line_no)
        else:
            raise LPFormatError("unrecognised bound", line_no)

    def _set(self, v, op, val, side, line_no):
        # Normalise to "v op val".
        if side == "left":
            op = {"<=": ">=", ">=": "<=", "=": "="}[op]
        m, i = self.model, v.index
        if op == "=":
            if val in (None, "-inf"):
                raise LPFormatError("cannot fix a variable to infinity", line_no)
            m.lb[i] = m.ub[i] = val
        elif op == ">=":
            m.lb[i] = None if val == "-inf" else val
        else:
            m.ub[i] = None if val in (None, "-inf") else val


def parse_lp(text: str) -> Model:
    r = _Reader()
    section = None
    objective_seen = False
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.split("\\", 1)[0].strip()
        if not line:
            continue
        key = " ".join(line.lower().split())
        if key in _SECTIONS:
            section = _SECTIONS[key]
            if section == "end":
                break
            continue
        toks = _tokenize(line, line_no)
        if section in ("max", "min"):
            if objective_seen:
                raise LPFormatError("objective must fit on one line", line_no)
            _, toks = r.split_name(toks)
            e = r.expr(toks, line_no)
            (r.model.maximize if section == "max" else r.model.minimize)(e)
            objective_seen = True
        elif section == "st":
            r.constraint(toks, line_no)
        elif section == "bounds":
            r.bound(toks, line_no)
        elif section in ("int", "bin"):
            for kind, name in toks:
                if kind != "id":
                    raise LPFormatError("expected variable names", line_no)
                v = r.var(name)
                if v.index not in r.model.integer:
                    r.model.integer.append(v.index)
                if section == "bin":
                    r.model.lb[v.index], r.model.ub[v.index] = Fraction(0), Fraction(1)
        else:
            raise LPFormatError("content before any section header", line_no)
    if not objective_seen:
        raise LPFormatError("no objective section found")
    return r.model


def read_lp(path: str) -> Model:
    with open(path, encoding="utf-8") as fh:
        m = parse_lp(fh.read())
    m.name = path
    return m
