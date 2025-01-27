"""
Conversion directly to llvm IR, skipping MLIR and xdsl entirely
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce
from io import TextIOBase, StringIO
from typing import ContextManager, Any, Literal, TypeAlias

import argparse
import sympy

from sympy2xdsl.base import SimpleConverter


class LLVMType:
    kind: Literal["f", "i"]
    width: int

    def __init__(self, kind: Literal["f", "i"], width: int):
        self.kind = kind
        self.width = width

    def __hash__(self):
        return hash((self.kind, self.width))

    def __eq__(self, other):
        if isinstance(other, LLVMType):
            return other.width == self.width and other.kind == self.kind

    @classmethod
    def int(cls, width: int):
        if width < 1:
            raise ValueError(f"Invalid integer width: {width}")
        return cls("i", width)

    @classmethod
    def float(cls, width: int):
        if width not in (16, 32, 64, 128):
            raise ValueError(f"Invalid float width: {width}")
        return cls("f", width)

    @classmethod
    def parse(cls, expr: str):
        for width, name in LLVMType._FLOAT_WIDTH_TO_LLVM_NAME.items():
            if name == expr:
                return cls.float(width)
        if expr[0] == "f":
            return cls.float(int(expr[1:]))
        elif expr[0] == "i":
            return cls.int(int(expr[1:]))
        raise ValueError(f"Invalid type supplied: {expr}")

    _FLOAT_WIDTH_TO_LLVM_NAME = {
        16: "half",
        32: "float",
        64: "double",
        128: "fp128",
    }

    def __str__(self):
        if self.kind == "f":
            return LLVMType._FLOAT_WIDTH_TO_LLVM_NAME[self.width]
        return f"i{self.width}"

    def widest(self, other: LLVMType):
        match (self, other):
            case (
                LLVMType(kind=k1, width=width1),
                LLVMType(kind=k2, width=width2),
            ) if k1 == k2:
                return LLVMType(k1, max(width1, width2))
            case (LLVMType(kind="f", width=width1), LLVMType(kind="i", width=width2)):
                return LLVMType("f", width1)
            case (LLVMType(kind="i", width=width1), LLVMType(kind="f", width=width2)):
                return LLVMType("f", width2)
            case _:
                raise ValueError(f"Unknown LLVMType comparison: {self} vs. {other}")

    def is_int(self) -> bool:
        return self.kind == "i"


LLVMType.i32 = LLVMType.int(32)
LLVMType.i64 = LLVMType.int(64)
LLVMType.f32 = LLVMType.float(32)
LLVMType.f64 = LLVMType.float(64)


@dataclass
class SSAVal:
    id: int | str
    type: LLVMType

    def __format__(self, format_spec):
        if format_spec == "v" or format_spec == "":
            return f"%{self.id}"
        elif format_spec == "t":
            return str(self.type)
        elif format_spec == "tv":
            return f"{self.type} %{self.id}"
        raise ValueError(f"Invalid format string for SSAVal: {format_spec}")

    def __str__(self):
        return format(self, "v")


@dataclass
class Const:
    val: int | float
    type: LLVMType

    def __str__(self):
        return str(self.val)

    def __format__(self, format_spec):
        if format_spec == "" or format_spec == "v":
            if self.type.is_int():
                return str(int(self.val))
            else:
                return str(float(self.val))
        elif format_spec == "t":
            return str(self.type)
        elif format_spec == "tv":
            return f"{self.type} {self.val}"
        raise ValueError(f"Invalid format string for Const: {format_spec}")


ValOrConst: TypeAlias = SSAVal | Const


class PrintLLVMIR:
    _io: TextIOBase
    _indent: str
    _id_counter: int
    _funcs_called: dict[str, tuple[tuple[LLVMType, ...], LLVMType]]
    _funcs_defined: set[str]
    _last_ret_val: ValOrConst | None

    def __init__(
        self,
        stream: TextIOBase,
        /,
        indent: str = "",
        id_counter=1,
    ):
        self._io = stream
        self._indent = indent
        self._id_counter = id_counter
        self._funcs_called = dict()
        self._funcs_defined = set()
        self._last_ret_val = None

    def create_val(self, vtype: LLVMType, name: str | None = None):
        if name is None:
            name = self._id_counter
            self._id_counter += 1
        val = SSAVal(name, vtype)
        return val

    @contextmanager
    def _with_temp_buff(
        self, io: TextIOBase, add_indent: str = ""
    ) -> ContextManager[PrintLLVMIR]:
        tmp_io = self._io
        self._io = io
        tmp_indent = self._indent
        self._indent = tmp_indent + add_indent
        yield self
        self._indent = tmp_indent
        self._io = tmp_io

    def println(self, text: str, /, end: str = "\n", indent: str | None = None):
        if indent is None:
            indent = self._indent
        self._io.write(indent)
        self._io.write(text)
        self._io.write(end)

    @contextmanager
    def fun(
        self,
        name: str,
        args: tuple[SSAVal, ...],
    ) -> ContextManager[PrintLLVMIR]:
        body_buff = StringIO()

        with self._with_temp_buff(body_buff, add_indent="  ") as tmp_buff_printer:
            yield tmp_buff_printer

        args_str = ", ".join(format(v, "tv") for v in args)
        self.println(f"define {self._last_ret_val.type} @{name}({args_str}) {{")
        self.println(body_buff.getvalue(), indent="", end="")
        self.println("}")
        self._funcs_defined.add(name)

    def print_ret(self, retval: ValOrConst):
        self.print_inst("ret", retval.type, retval)
        self._last_ret_val = retval

    def print_label(self, name: str):
        self.println(f"{name}:", indent=self._indent[:-2])

    def print_inst(
        self,
        *instr: str | ValOrConst | LLVMType,
        result_t: LLVMType | None = None,
    ) -> SSAVal | None:
        # convert result to join-friendly string
        result_str = ""
        result = None
        if result_t is not None:
            result = self.create_val(result_t)
            result_str = f"{result:v} = "

        # convert instructions to str
        instr_str = _join_llvmir(map(str, instr))

        self.println(f"{result_str}{instr_str}")
        return result

    def print_call(
        self, ret_type: LLVMType | None, name: str, args: tuple[ValOrConst, ...]
    ) -> SSAVal | None:
        args_str = ", ".join(format(v, "tv") for v in args)

        if ret_type is None:
            self.println(f"call void @{name}({args_str})")
            return None

        retv = self.create_val(ret_type)
        self.println(f"{retv:v} = call {ret_type} @{name}({args_str})")
        self._funcs_called[name] = (tuple(arg.type for arg in args), ret_type)
        return retv

    def print_func_decls(self):
        if not set(self._funcs_called) - self._funcs_defined:
            return

        self.println("\n; Declare external functions:")
        for name, (args, ret) in self._funcs_called.items():
            if name not in self._funcs_defined:
                args_str = f", ".join(str(x) for x in args)
                self.println(f"declare {ret} @{name}({args_str})")


def _join_llvmir(args: Iterable[str]):
    try:
        iter_ = iter(args)
        res = [next(iter_)]
        for arg in iter_:
            if arg in ",(:[@":
                res.append(arg)
            else:
                res.append(" ")
                res.append(arg)
        return "".join(res)
    except StopIteration:
        return ""


class ConvertLLVM(SimpleConverter):
    fun_name: str
    printer: PrintLLVMIR
    _var_to_ssa_vars: dict[sympy.Basic, ValOrConst]
    _float_t: LLVMType
    _int_t: LLVMType
    _inp_types: tuple[LLVMType, ...]

    def __init__(
        self,
        expr: sympy.Expr,
        fun_name: str,
        inp_types: tuple[LLVMType, ...],
        /,
        int_t: LLVMType = LLVMType.i64,
        float_t: LLVMType = LLVMType.f64,
    ):
        super().__init__(expr)
        if len(expr.free_symbols) != len(inp_types):
            raise ValueError("Not enough input types provided for the expression")
        self.fun_name = fun_name
        self._inp_types = inp_types
        self._float_t = float_t
        self._int_t = int_t
        self._io = StringIO()
        self._inp_args = {
            sym.name: SSAVal(sym.name, typ)
            for sym, typ in zip(expr.free_symbols, inp_types)
        }
        self.printer = PrintLLVMIR(self._io)

    @classmethod
    def register_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--int-t",
            default="i64",
            type=LLVMType.parse,
            help="An LLVM type expression",
        )
        parser.add_argument(
            "--float-t",
            default="double",
            type=LLVMType.parse,
            help="An LLVM type expression",
        )
        parser.add_argument("function", help="Name of the llvm function to generate")
        parser.add_argument(
            "arg_types",
            nargs="+",
            help="Argument types for the functions arguments, e.g. i64 f64",
            type=LLVMType.parse,
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

    def widest_type_among_args(self, args: tuple[SSAVal, ...]):
        if not args:
            return self._int_t
        return reduce(lambda a, b: a.widest(b.type), args[1:], args[0].type)

    def _v_as(self, val: ValOrConst, t: LLVMType) -> ValOrConst:
        # casting a constant is free:
        if isinstance(val, Const):
            return Const(val.val, t)
        if val.type == t:
            return val
        match (val.type, t):
            case LLVMType(kind="f", width=width1), LLVMType(
                kind="f", width=width2
            ) if width2 > width1:
                return self.printer.print_inst(
                    "fpext", val.type, val, "to", t, result_t=t
                )
            case LLVMType(kind="i"), LLVMType(kind="f"):
                return self.printer.print_inst(
                    "sitofp", val.type, val, "to", t, result_t=t
                )
            case LLVMType(kind="i", width=width1), LLVMType(
                kind="i", width=width2
            ) if width2 > width1:
                return self.printer.print_inst(
                    "sext", val.type, val, "to", t, result_t=t
                )
            case _:
                raise ValueError(f"Conversion not implemented between {val} and {t}")

    def visit(self, expr: sympy.Basic) -> ValOrConst:
        if expr.is_Symbol:
            return self._inp_args[expr.name]
        elif expr.is_Integer:
            return Const(int(expr.evalf()), self._int_t)
        elif expr.is_Float or expr.is_Rational:
            return Const(float(expr.evalf()), self._float_t)
        elif expr.is_Add:
            # detect a + -1*b
            if expr.args[1].is_Mul and expr.args[1].args[0] == -1:
                # emit a - b
                return self._emit_var_len_args(
                    (expr.args[0], expr.args[1].args[1]),
                    "sub {dest_t} {base}, {arg}",
                    "fsub {dest_t} {base}, {arg}",
                )
            return self._emit_var_len_args(
                expr, "add {dest_t} {base}, {arg}", "fadd {dest_t} {base}, {arg}"
            )
        elif expr.is_Mul:
            # detect a * 1/b (which is a * (b**-1)
            if expr.args[1].is_Pow and expr.args[1].args[1] == -1:
                # print a/b as either sdiv or fdiv
                return self._emit_var_len_args(
                    (expr.args[0], expr.args[1].args[0]),
                    "sdiv {dest_t} {base}, {arg}",
                    "fdiv {dest_t} {base}, {arg}",
                )
            return self._emit_var_len_args(
                expr, "mul {dest_t} {base}, {arg}", "fmul {dest_t} {base}, {arg}"
            )
        elif expr.is_Pow:
            arg = self.visit(expr.args[0])
            if expr.args[1] == -1:
                # emit 1/x
                return self.printer.print_inst(
                    f"fdiv {self._float_t} 1, {self._v_as(arg, self._float_t)}"
                )
            elif expr.args[1] == 2:
                # emit x*x
                if arg.type.is_int():
                    return self.printer.print_inst(
                        f"shl {self._int_t} {arg}, 1", result_t=self._float_t
                    )
                return self.printer.print_inst(
                    f"fmul {self._float_t} {arg}, {arg}", result_t=self._float_t
                )
            float_args = tuple(
                self._v_as(v, self._float_t) for v in map(self.visit, expr.args)
            )
            return self.printer.print_call(self._float_t, "pow", float_args)
        else:
            func_name = str(expr.func)
            float_args = tuple(
                self._v_as(v, self._float_t) for v in map(self.visit, expr.args)
            )
            return self.printer.print_call(self._float_t, func_name, float_args)

    def _emit_var_len_args(
        self,
        expr: sympy.Basic | tuple[sympy.Basic, ...],
        int_name: str,
        float_name: str,
    ):
        """
        Take an expression, (e.g. a+b+c) and emit pairwise instructions for each argument.

        Checks if floating point or integer arithmetic instructions should be used.

        If expr is a tuple of expressions, it is assumed that these are the arguments (e.g. for tricky edge cases
        like a/b, which sympy represents as a * (b^-1))
        """
        if isinstance(expr, tuple):
            args = tuple(map(self.visit, expr))
        else:
            args = tuple(map(self.visit, expr.args))
        dest_arg_t = self.widest_type_among_args(args)
        inst = int_name if dest_arg_t.kind == "i" else float_name
        args = tuple(self._v_as(arg, dest_arg_t) for arg in args)

        base = args[0]
        for arg in args[1:]:
            base = self.printer.print_inst(
                inst.format(base=base, arg=arg, dest_t=dest_arg_t), result_t=dest_arg_t
            )
        return base

    def walk(self):
        return self.visit(self._expr)

    def convert(self):
        with self.printer.fun(
            self.fun_name, tuple(sorted(self._inp_args.values(), key=lambda x: x.id))
        ):
            res = self.walk()
            self.printer.print_ret(res)
        # return the final string
        self.printer.print_func_decls()
        return self._io.getvalue().strip()
