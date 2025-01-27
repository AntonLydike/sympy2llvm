import argparse
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any

import sympy


def _walk_expr_from_leaves(expr: sympy.Basic) -> Iterable[sympy.Expr]:
    for arg in expr.args:
        yield from _walk_expr_from_leaves(arg)
    yield expr



class SimpleConverter:
    def __init__(self, expr: sympy.Expr):
        self._expr = expr

    @abstractmethod
    def visit(self, expr: sympy.Basic) -> Any:
        raise NotImplementedError("Visit not implemented for this expression")

    def walk(self):
        ret_val = None
        for expr in _walk_expr_from_leaves(self._expr):
            ret_val = self.visit(expr)
        return ret_val

    @classmethod
    @abstractmethod
    def register_args(cls, parser: argparse.ArgumentParser):
        """
        Register arguments needed for this converter to the argument parser.
        """
        raise NotImplementedError("Register args not implemented for this expression")

    @classmethod
    @abstractmethod
    def args_to_init_args(
        cls, args: argparse.Namespace
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Convert namespace to args and kwargs for __init__.

        Args are provided based on what is registered in register_args.
        """
        raise NotImplementedError(
            "Args to init args not implemented for this expression"
        )

    @classmethod
    def from_args(cls, expr: sympy.Expr, args: argparse.Namespace) -> "Converter":
        args, kwargs = cls.args_to_init_args(args)
        return cls(expr, *args, **kwargs)

    @abstractmethod
    def convert(self):
        raise NotImplementedError("Convert not implemented for this expression")
