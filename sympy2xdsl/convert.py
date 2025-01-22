from typing import Callable, Iterable
import sympy
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.ir import SSAValue
import itertools

from enum import Flag, auto


class ExprKind(Flag):
    # arithmetic
    ADD = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()

    # variable
    VAR = auto()

    # functions
    FUN = auto()

    # comparisons
    COMPARISON = auto()

    # literals
    LITERAL = auto()


class LitKind(Flag):
    INT = auto()
    FLOAT = auto()
    RATIONAL = auto()
    BOOL = auto()
    COMPLEX = auto()


class FunKind(Flag):
    """
    Various math functions
    """

    LOG = auto()
    # trig
    SIN = auto()
    COS = auto()
    TAN = auto()
    ASIN = auto()
    ACOS = auto()
    ATAN = auto()
    ATAN2 = auto()
    SINH = auto()
    COSH = auto()
    TANH = auto()
    ASINH = auto()
    ACOSH = auto()
    ATANH = auto()
    # bitwise
    AND = auto()
    OR = auto()
    XOR = auto()
    # misc
    MIN = auto()
    MAX = auto()
    MOD = auto()


_FUN_MAPPER = {
    sympy.log: FunKind.LOG,
    # trig
    sympy.sin: FunKind.SIN,
    sympy.cos: FunKind.COS,
    sympy.tan: FunKind.TAN,
    sympy.asin: FunKind.ASIN,
    sympy.acos: FunKind.ACOS,
    sympy.atan: FunKind.ATAN,
    sympy.atan2: FunKind.ATAN2,
    sympy.sinh: FunKind.SINH,
    sympy.cosh: FunKind.COSH,
    sympy.tanh: FunKind.TANH,
    sympy.asinh: FunKind.ASINH,
    sympy.acosh: FunKind.ACOSH,
    sympy.atanh: FunKind.ATANH,
    # bitwise
    sympy.And: FunKind.AND,
    sympy.Or: FunKind.OR,
    sympy.Xor: FunKind.XOR,
    # misc
    sympy.Min: FunKind.MIN,
    sympy.Max: FunKind.MAX,
    sympy.Mod: FunKind.MOD,
}
"""
Map from sympy functions to FunKind
"""


class Converter:
    _var_to_ssa_vars: dict[sympy.Expr, SSAValue]
    _expr: sympy.Expr
    _curr_exp: sympy.Expr

    def __init__(self, expr: sympy.Expr, var_mapping: dict[sympy.Expr, SSAValue]):
        self._expr = expr
        self._curr_exp = expr
        self._var_to_ssa_vars = var_mapping

    def get_curr_expr_kind(self) -> ExprKind:
        if self._curr_exp.is_Add:
            return ExprKind.ADD
        elif self._curr_exp.is_Mul:
            return ExprKind.MUL
        elif self._curr_exp.is_Pow:
            # detect n/x as div instead of pow
            if self._curr_exp.args[1].is_Integer:
                return ExprKind.DIV
            return ExprKind.POW
        elif self._curr_exp.is_Function:
            return ExprKind.FUN
        elif self._curr_exp.is_number:
            return ExprKind.LITERAL
        elif self._curr_exp.is_Relational:
            return ExprKind.COMPARISON
        elif self._curr_exp.is_Symbol:
            return ExprKind.VAR
        else:
            raise ValueError(
                f"Unknown expression kind: {self._curr_exp} ({self._curr_exp.func})"
            )

    def get_curr_fun_kind(self) -> FunKind:
        if not self._curr_exp.is_Function:
            raise ValueError("Current expression is not a function")
        return _FUN_MAPPER[self._curr_exp.func]

    def get_curr_lit_kind(self) -> LitKind:
        lit = LitKind(0)
        if self._curr_exp.is_Integer:
            lit |= LitKind.INT
        if self._curr_exp.is_Float:
            lit |= LitKind.FLOAT
        if self._curr_exp.is_Rational:
            lit |= LitKind.RATIONAL
        if self._curr_exp.is_Boolean:
            lit |= LitKind.BOOL
        return lit

    def ssa_val_for(self, expr: sympy.Expr) -> SSAValue | None:
        return self._var_to_ssa_vars.get(expr, None)

    def visit(self) -> SSAValue:
        pass

    def walk(self):
        for expr in _walk_expr_from_leaves(self._expr):
            self._curr_exp = expr
            val = self.visit()
            self._var_to_ssa_vars[expr] = val
        return val


def _walk_expr_from_leaves(expr: sympy.Expr) -> Iterable[sympy.Expr]:
    for arg in expr.args:
        yield from _walk_expr_from_leaves(arg)
    yield expr
