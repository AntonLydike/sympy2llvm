import sympy
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import llvm, arith
from xdsl.dialects.experimental import math
from xdsl.ir import SSAValue, Attribute, Block, Region
from xdsl.dialects.builtin import Float64Type, IntegerAttr, i1, i64, f64, FloatAttr, IntegerType, AnyFloat
from sympy2xdsl.convert import Converter, ExprKind, LitKind, FunKind


class ConvertLLM(Converter):
    def __init__(self, expr, var_mapping, int_t: Attribute = i64, float_t: Attribute = f64):
        super().__init__(expr, var_mapping)
        self._float_t = float_t
        self._int_t = int_t

    def visit(self) -> SSAValue:
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
            return llvm.ConstantOp(IntegerAttr(int(self._curr_exp), self._int_t), self._int_t).result
        if LitKind.BOOL in kind:
            return llvm.ConstantOp(IntegerAttr(int(expr), i1), i1).result
        if LitKind.FLOAT in kind or LitKind.RATIONAL in kind:
            return llvm.ConstantOp(FloatAttr(float(self._curr_exp), self._float_t), self._float_t).result

    def coalesce_args(self, args: tuple[SSAValue, ...]) -> tuple[tuple[SSAValue, ...], Attribute]:
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
        new_args  = []
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
                    new_args.append(arith.SIToFPOp(arg, dest_t).result)
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

    def visit_math(self, expr) -> SSAValue:
        match expr:
            case ExprKind.ADD:
                ssa_args, dst_type = self.get_curr_expr_args_coalesced()
                base_op = arith.AddfOp
                if isinstance(dst_type, IntegerType):
                    base_op = llvm.AddOp
                return base_op(*ssa_args).result
            case ExprKind.MUL:
                # simplify a*(b**-1) to just a/b
                if self._curr_exp.args[1].is_Pow and self._curr_exp.args[1].args[1] == -1:
                    return self._insert_div(
                        self.ssa_val_for(self._curr_exp.args[0]),
                        self.ssa_val_for(self._curr_exp.args[1].args[0])
                    )
                ssa_args, dst_type = self.get_curr_expr_args_coalesced()
                base_op = arith.MulfOp
                if isinstance(dst_type, IntegerType):
                    base_op = llvm.MulOp
                return base_op(*ssa_args).result
            case ExprKind.DIV:
                return self._insert_div(*self.get_curr_exprs_ssa_args())
            case ExprKind.POW:
                base, exp = self.get_curr_exprs_ssa_args()
                if not isinstance(base.type, AnyFloat):
                    if isinstance(exp.type, IntegerAttr):
                        return math.IPowIOp(*self.coalesce_args((base, exp))).result    
                    base = arith.SIToFPOp(base, exp.type).result
                if isinstance(exp.type, IntegerAttr):
                    return math.FPowIOp(base, exp).result
                return math.PowFOp(base, exp).result
            case wrong:
                raise ValueError(f"Unknown math kind: {wrong}")


    def _insert_div(self, a: SSAValue, b: SSAValue) -> SSAValue:
        ssa_args, dst_type = self.coalesce_args((a, b))
        base_op = arith.DivfOp
        if isinstance(dst_type, IntegerType):
            base_op = arith.DivSIOp
        return base_op(*ssa_args).result

    def walk(self):
        res_val = super().walk()
        # return the final result of the expression
        llvm.ReturnOp.build(operands=[res_val])
        return res_val

def make_llvm_func_with_args(name: str, args: dict[sympy.Symbol, Attribute], expr: sympy.Expr, int_t= i64, float_t= f64):
    block = Block(arg_types=tuple(args.values()))
    for arg, sym in zip(block.args, args):
        arg.name_hint = sym.name
    with ImplicitBuilder(block):
        res = ConvertLLM(
            expr,
            int_t=int_t,
            float_t=float_t,
            var_mapping={
                sym: block_argg for sym, block_argg in zip(args.keys(), block.args)
            },
        ).walk()

    return llvm.FuncOp(name, llvm.LLVMFunctionType(tuple(args.values()), res.type), body=Region(block))

