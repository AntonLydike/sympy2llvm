import argparse
from abc import abstractmethod
from typing import Any

import sympy


class SimpleConverter:
    def __init__(self, expr: sympy.Expr):
        self._expr = expr

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
    def convert(self) -> str:
        raise NotImplementedError("Convert not implemented for this expression")
