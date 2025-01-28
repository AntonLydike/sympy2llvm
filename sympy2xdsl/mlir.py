from functools import reduce
from typing import Any
import argparse
import sympy
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import llvm, math, get_all_dialects
from xdsl.ir import SSAValue, Attribute, Block, Region, Operation
from xdsl.parser import Parser, MLContext
from xdsl.dialects.builtin import (
    IntegerAttr,
    i64,
    f64,
    FloatAttr,
    IntegerType,
    AnyFloat,
)

from sympy2xdsl.base import SimpleConverter


FUN_TO_MATH: dict[sympy.Function, Operation | None] = {
    sympy.log: math.LogOp,
    sympy.exp: math.ExpOp,
    # trig
    sympy.sin: math.SinOp,
    sympy.cos: math.CosOp,
    sympy.tan: math.TanOp,
    sympy.asin: None,
    sympy.acos: None,
    sympy.atan: math.AtanOp,
    sympy.atan2: math.Atan2Op,
    sympy.sinh: None,
    sympy.cosh: None,
    sympy.tanh: math.TanhOp,
    sympy.asinh: None,
    sympy.acosh: None,
    sympy.atanh: None,
    # bitwise
    sympy.And: llvm.AndOp,
    sympy.Or: llvm.OrOp,
    sympy.Xor: llvm.XOrOp,
    # misc
    sympy.Min: None,
    sympy.Max: None,
    sympy.Mod: None,
    sympy.Abs: None,
    # rounding:
    sympy.floor: math.FloorOp,
    sympy.ceiling: math.CeilOp,
}


def _wider_type(a: Attribute, b: Attribute) -> Attribute:
    """
    Takes two xdsl int or float types and returns:
    - the wider if both are integer or float
    - the float type if one is float and one is int
    - nonsense otherwise
    """
    # return wider of two integer types
    if isinstance(a, IntegerType) and isinstance(b, IntegerType):
        return a if a.bitwidth > b.bitwidth else b
    # return wider of two float types
    elif isinstance(a, AnyFloat) and isinstance(b, AnyFloat):
        return a if a.bitwidth > b.bitwidth else b
    # return the float type if one is float and one is int
    else:
        return a if isinstance(a, AnyFloat) else b


def parse_string_to_xdsl_type(type_str: str) -> Attribute:
    ctx = MLContext()
    for name, dialect in get_all_dialects().items():
        ctx.register_dialect(name, dialect)
    p = Parser(ctx, type_str, "args")
    return p.parse_type()


class ConvertMLIR(SimpleConverter):
    fun_name: str
    _inp_types: tuple[Attribute, ...]
    _float_t: Attribute
    _int_t: Attribute
    _var_to_ssa_vars: dict[sympy.Basic, SSAValue]

    def __init__(
        self,
        expr: sympy.Expr,
        fun_name: str,
        inp_types: tuple[Attribute, ...],
        /,
        int_t: Attribute = i64,
        float_t: Attribute = f64,
    ):
        super().__init__(expr)
        if len(expr.free_symbols) != len(inp_types):
            raise ValueError("Not enough input types provided for the expression")
        self.fun_name = fun_name
        self._inp_types = inp_types
        self._float_t = float_t
        self._int_t = int_t
        self._var_to_ssa_vars = {}

    @classmethod
    def register_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--int-t",
            default="i64",
            type=parse_string_to_xdsl_type,
            help="An xdsl type expression",
        )
        parser.add_argument(
            "--float-t",
            default="f64",
            type=parse_string_to_xdsl_type,
            help="An xdsl type expression",
        )
        parser.add_argument("function", help="Name of the llvm function to generate")
        parser.add_argument(
            "arg_types",
            nargs="+",
            help="Argument types for the functions arguments, e.g. i64 f64",
            type=parse_string_to_xdsl_type,
        )

    @classmethod
    def args_to_init_args(
        cls, args: argparse.Namespace
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Convert namespace to args and kwargs for __init__
        """
        return (args.function, args.arg_types), {
            "int_t": args.int_t,
            "float_t": args.float_t,
        }

    def expr_to_xdsl(self, expr: sympy.Basic) -> SSAValue | Operation:
        if expr.is_Symbol:
            return self._var_to_ssa_vars[expr]
        elif expr.is_Integer:
            return llvm.ConstantOp(
                IntegerAttr(int(expr.evalf()), self._int_t), self._int_t
            )
        elif expr.is_Float | expr.is_Rational:
            return llvm.ConstantOp(
                FloatAttr(float(expr.evalf()), self._float_t), self._float_t
            )
        elif expr.is_Add:
            # detect a + -1*b
            if expr.args[1].is_Mul and expr.args[1].args[0] == -1:
                return self._emit_var_int_or_float_op(
                    llvm.SubOp, llvm.FSubOp, (expr.args[0], expr.args[1].args[1])
                )
            return self._emit_var_int_or_float_op(llvm.AddOp, llvm.FAddOp, expr.args)
        elif expr.is_Pow:
            base = self.visit(expr.args[0])
            exp = expr.args[1]
            if expr.args[1] == -1:
                # emit 1/x
                return llvm.FDivOp(
                    self._v_as(self.visit(exp), self._float_t),
                    self._v_as(base, self._float_t),
                )
            elif expr.args[1] == 2:
                # emit x*x
                return self._emit_var_int_or_float_op(
                    llvm.MulOp, llvm.FMulOp, (expr.args[0], expr.args[0])
                )
            else:
                exp_val = self.visit(exp)
                if isinstance(exp_val.type, IntegerType):
                    if isinstance(base.type, IntegerType):
                        return math.IPowIOp(base, exp_val)
                    return math.FPowIOp(base, exp_val)
                return math.PowFOp(
                    self._v_as(base, self._float_t),
                    self._v_as(exp_val, self._float_t),
                )
        elif expr.is_Mul:
            # detect a * 1/b (which is a * (b**-1)
            if expr.args[1].is_Pow and expr.args[1].args[1] == -1:
                # print a/b as either sdiv or fdiv
                return self._emit_var_int_or_float_op(
                    llvm.SDivOp, llvm.FDivOp, (expr.args[0], expr.args[1].args[0])
                )
            return self._emit_var_int_or_float_op(llvm.MulOp, llvm.FMulOp, expr.args)
        elif expr.func in FUN_TO_MATH:
            if FUN_TO_MATH.get(expr.func) is None:
                raise NotImplementedError(f"Cannot translate {expr} to math dialect")
            op = FUN_TO_MATH[expr.func]
            if issubclass(op, math.FloatingPointLikeBinaryMathOperation):
                assert len(expr.args) == 2
                return op(
                    self._v_as(self.visit(expr.args[0]), self._float_t),
                    self._v_as(self.visit(expr.args[1]), self._float_t),
                )
            elif issubclass(op, math.FloatingPointLikeUnaryMathOperation):
                assert len(expr.args) == 1
                return op(self._v_as(self.visit(expr.args[0]), self._float_t))
            else:
                raise ValueError(f"Unknown math operation {op}")

    def _v_as(self, val: SSAValue, dest_t: Attribute) -> SSAValue:
        if val.type == dest_t:
            return val
        if isinstance(val.type, IntegerType) and isinstance(dest_t, IntegerType):
            return llvm.SExtOp(val, dest_t).res
        elif isinstance(val.type, IntegerType) and isinstance(dest_t, AnyFloat):
            return llvm.SIToFPOp(val, dest_t).result
        elif isinstance(val.type, AnyFloat) and isinstance(dest_t, AnyFloat):
            return llvm.FPExtOp(val, dest_t).result
        elif isinstance(val.type, FloatAttr) and isinstance(dest_t, IntegerType):
            # not implemented in llvm dialect yet:
            return llvm.FPToSIOp(val, dest_t).result

    def _widest_type_among(self, args: tuple[SSAValue, ...]):
        return reduce(lambda a, x: _wider_type(a, x.type), args, args[0].type)

    def _emit_var_int_or_float_op(
        self,
        i_op: type[Operation],
        f_op: type[Operation],
        args: tuple[sympy.Basic, ...],
        dest_t: Attribute | None = None,
    ):
        ssa_args = tuple(map(self.visit, args))
        if dest_t is None:
            dest_t = self._widest_type_among(ssa_args)

        # choose int/float operation depending on type
        op_t = f_op
        if isinstance(dest_t, IntegerType):
            op_t = i_op

        base = self._v_as(ssa_args[0], dest_t)
        for arg in ssa_args[1:]:
            base = op_t(base, self._v_as(arg, dest_t)).results[0]
        return base

    def visit(self, expr: sympy.Basic) -> SSAValue:
        v = SSAValue.get(self.expr_to_xdsl(expr))

        if expr not in self._var_to_ssa_vars:
            self._var_to_ssa_vars[expr] = v
        return v

    def walk(self):
        return self.visit(self._expr)

    def convert(self) -> str:
        block = Block(arg_types=tuple(self._inp_types))
        for arg, sym in zip(
            block.args, sorted(self._expr.free_symbols, key=lambda x: x.name)
        ):
            arg.name_hint = sym.name
            self._var_to_ssa_vars[sym] = arg

        with ImplicitBuilder(block):
            res = self.walk()
            # return the final result of the expression
            llvm.ReturnOp.build(operands=[res])

        op = llvm.FuncOp(
            self.fun_name,
            llvm.LLVMFunctionType(self._inp_types, res.type),
            body=Region(block),
        )
        op.verify()
        return op
