from typing import Any
import argparse
import sympy
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import llvm, math, get_all_dialects
from xdsl.ir import SSAValue, Attribute, Block, Region, Operation
from xdsl.parser import Parser, MLContext
from xdsl.dialects.builtin import (
    IntegerAttr,
    i1,
    i64,
    f64,
    FloatAttr,
    IntegerType,
    AnyFloat,
)
from sympy2xdsl.convert import Converter, ExprKind, LitKind, FunKind


FUN_TO_MATH: dict[FunKind, Operation | None] = {
    FunKind.LOG: math.LogOp,
    FunKind.EXP: math.ExpOp,
    # trig
    FunKind.SIN: math.SinOp,
    FunKind.COS: math.CosOp,
    FunKind.TAN: math.TanOp,
    FunKind.ASIN: None,
    FunKind.ACOS: None,
    FunKind.ATAN: math.AtanOp,
    FunKind.ATAN2: math.Atan2Op,
    FunKind.SINH: None,
    FunKind.COSH: None,
    FunKind.TANH: math.TanhOp,
    FunKind.ASINH: None,
    FunKind.ACOSH: None,
    FunKind.ATANH: None,
    # bitwise
    FunKind.AND: llvm.AndOp,
    FunKind.OR: llvm.OrOp,
    FunKind.XOR: llvm.XOrOp,
    # misc
    FunKind.MIN: None,
    FunKind.MAX: None,
    FunKind.MOD: None,
    FunKind.ABS: None,
    # rounding:
    FunKind.FLOOR: math.FloorOp,
    FunKind.CEIL: math.CeilOp,
}


def parse_string_to_xdsl_type(type_str: str) -> Attribute:
    ctx = MLContext()
    for name, dialect in get_all_dialects().items():
        ctx.register_dialect(name, dialect)
    p = Parser(ctx, type_str, "args")
    return p.parse_type()


class ConvertMLIR(Converter):
    fun_name: str
    _inp_types: tuple[Attribute, ...]
    _float_t: Attribute
    _int_t: Attribute
    _var_to_ssa_vars: dict[sympy.Symbol, SSAValue]

    def __init__(
        self,
        expr: sympy.Expr,
        fun_name: str,
        inp_types: tuple[Attribute, ...],
        /,
        int_t: Attribute = i64,
        float_t: Attribute = f64,
    ):
        super().__init__(expr, dict())
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

    def visitor(self) -> SSAValue:
        if self._curr_exp in self._var_to_ssa_vars:
            return self._var_to_ssa_vars[self._curr_exp]
        expr = self.get_curr_expr_kind()

        if expr == ExprKind.VAR:
            return self.ssa_val_for(self._curr_exp)
        if expr in ExprKind.ADD | ExprKind.MUL | ExprKind.DIV | ExprKind.POW:
            return self.visit_math(expr)
        elif expr == ExprKind.FUN:
            return self.visit_fun(expr)
        elif expr == ExprKind.COMPARISON:
            return self.visit_comp(expr)
        elif expr == ExprKind.LITERAL:
            return self.visit_lit(expr)

    def visit_lit(self, expr) -> SSAValue:
        kind = self.get_curr_lit_kind()
        if LitKind.INT in kind:
            return llvm.ConstantOp(
                IntegerAttr(int(self._curr_exp), self._int_t), self._int_t
            ).result
        if LitKind.BOOL in kind:
            return llvm.ConstantOp(IntegerAttr(int(expr), i1), i1).result
        if LitKind.FLOAT in kind or LitKind.RATIONAL in kind:
            return llvm.ConstantOp(
                FloatAttr(float(self._curr_exp), self._float_t), self._float_t
            ).result

    def visit_fun(self, expr: sympy.Expr) -> SSAValue:
        kind = self.get_curr_fun_kind()
        if FUN_TO_MATH.get(kind, None) is not None:
            if len(self._curr_exp.args) == 1:
                arg = self.get_val_as_type(self._curr_exp.args[0], self._float_t)
                return FUN_TO_MATH.get(kind)(arg).results[0]
            else:
                # for AND/OR/XOR
                FUN_TO_MATH.get(kind)(
                    *(
                        self.get_val_as_type(arg, self._int_t)
                        for arg in self._curr_exp.args
                    )
                ).results[0]
        raise ValueError(f"Cannot translate function: {kind}")

    def coalesce_args(
        self, args: tuple[SSAValue, ...]
    ) -> tuple[tuple[SSAValue, ...], Attribute]:
        """
        Coalesce ssa values to the same type.

        type widening is applied, i.e. they are all widened to the wides int type seen

        if any of the args is float, they are all cast to float instead
        """
        dest_t: IntegerType | AnyFloat = args[0].type
        is_int = isinstance(dest_t, IntegerType)
        for arg in args[1:]:
            # both are int
            if is_int and isinstance(arg.type, IntegerType):
                if arg.type.bitwidth > dest_t.bitwidth:
                    dest_t = arg.type
            # switching to float
            elif is_int and isinstance(arg.type, AnyFloat):
                dest_t = arg.type
                is_int = False
            # both are float
            elif isinstance(arg.type, AnyFloat):
                # switch to wider type
                if arg.type.bitwidth > dest_t.bitwidth:
                    dest_t = arg.type

        # cast each arg to new type
        new_args = []
        for arg in args:
            # is_int implies all args are int
            if is_int:
                if arg.type.bitwidth == dest_t.bitwidth:
                    new_args.append(arg)
                    continue
                new_args.append(llvm.SExtOp(arg, dest_t).res)
            else:
                if isinstance(arg.type, IntegerType):
                    # cast int to float
                    new_args.append(llvm.SIToFPOp(arg, dest_t).result)
                else:
                    # cast float to float
                    if arg.type.bitwidth == dest_t.bitwidth:
                        new_args.append(arg)
                        continue
                    else:
                        raise NotImplementedError("No llvm.fpext op in xDSL")
        return tuple(new_args), dest_t

    def get_curr_exprs_ssa_args(self) -> tuple[SSAValue, ...]:
        if self.get_curr_expr_kind() == ExprKind.DIV and self._curr_exp.is_Pow:
            # reverse args for division, as div is really b**-1
            return tuple(map(self.ssa_val_for, self._curr_exp.args[::-1]))
        return tuple(map(self.ssa_val_for, self._curr_exp.args))

    def get_curr_expr_args_coalesced(self) -> tuple[SSAValue, ...]:
        return self.coalesce_args(self.get_curr_exprs_ssa_args())

    def get_val_as_type(
        self, exp: sympy.Expr | SSAValue, dest_t: Attribute
    ) -> SSAValue:
        if isinstance(exp, sympy.Expr):
            ssa_val: SSAValue = self.ssa_val_for(exp)
        else:
            ssa_val: SSAValue = exp
        source_t = ssa_val.type
        if isinstance(source_t, IntegerType) and isinstance(dest_t, IntegerType):
            if source_t.bitwidth == dest_t.bitwidth:
                return ssa_val
            return llvm.SExtOp(ssa_val, dest_t).res
        elif isinstance(source_t, AnyFloat) and isinstance(dest_t, AnyFloat):
            if source_t.bitwidth == dest_t.bitwidth:
                return ssa_val
            return llvm.FPExtOp(ssa_val, dest_t).result
        elif isinstance(source_t, IntegerType) and isinstance(dest_t, AnyFloat):
            return llvm.SIToFPOp(ssa_val, dest_t).result
        elif isinstance(source_t, AnyFloat) and isinstance(dest_t, IntegerType):
            return llvm.FPtoSIOp(
                ssa_val, dest_t
            ).result  # TODO: this is not implemented yet
        raise ValueError("This should be unreachable")

    def _instantiate_multi_arg_op(
        self, op_t: Operation, args: tuple[SSAValue, ...]
    ) -> SSAValue:
        start = args[0]
        for arg in args[1:]:
            start = op_t(start, arg).results[0]
        return start

    def visit_math(self, expr) -> SSAValue:
        match expr:
            case ExprKind.ADD:
                ssa_args, dst_type = self.get_curr_expr_args_coalesced()
                base_op = llvm.FAddOp
                if isinstance(dst_type, IntegerType):
                    base_op = llvm.AddOp
                return self._instantiate_multi_arg_op(base_op, ssa_args)
            case ExprKind.MUL:
                # simplify a*(b**-1) to just a/b
                if (
                    self._curr_exp.args[1].is_Pow
                    and self._curr_exp.args[1].args[1] == -1
                ):
                    return self._insert_div(
                        self.ssa_val_for(self._curr_exp.args[0]),
                        self.ssa_val_for(self._curr_exp.args[1].args[0]),
                    )
                ssa_args, dst_type = self.get_curr_expr_args_coalesced()
                base_op = llvm.FMulOp
                if isinstance(dst_type, IntegerType):
                    base_op = llvm.MulOp
                return self._instantiate_multi_arg_op(base_op, ssa_args)
            case ExprKind.DIV:
                return self._insert_div(*self.get_curr_exprs_ssa_args())
            case ExprKind.POW:
                base, exp = self.get_curr_exprs_ssa_args()
                if not isinstance(base.type, AnyFloat):
                    if isinstance(exp.type, IntegerAttr):
                        return math.IPowIOp(*self.coalesce_args((base, exp))).result
                    base = llvm.SIToFPOp(base, exp.type).result
                if isinstance(exp.type, IntegerAttr):
                    return math.FPowIOp(base, exp).result
                return math.PowFOp(base, exp).result
            case wrong:
                raise ValueError(f"Unknown math kind: {wrong}")

    def _insert_div(self, a: SSAValue, b: SSAValue) -> SSAValue:
        ssa_args, dst_type = self.coalesce_args((a, b))
        base_op = llvm.FDivOp
        if isinstance(dst_type, IntegerType):
            base_op = llvm.SDivOp
        return base_op(*ssa_args).res

    def visit(self, expr: sympy.Basic) -> SSAValue:
        v = super().visit(expr)
        assert v is not None, "No SSA value returned"
        self._var_to_ssa_vars[expr] = v
        return v

    def walk(self):
        res_val = super().walk()
        # return the final result of the expression
        llvm.ReturnOp.build(operands=[res_val])
        return res_val

    def convert(self):
        block = Block(arg_types=tuple(self._inp_types))
        for arg, sym in zip(
            block.args, sorted(self._expr.free_symbols, key=lambda x: x.name)
        ):
            arg.name_hint = sym.name
            self._var_to_ssa_vars[sym] = arg

        with ImplicitBuilder(block):
            res = self.walk()

        return llvm.FuncOp(
            self.fun_name,
            llvm.LLVMFunctionType(self._inp_types, res.type),
            body=Region(block),
        )
