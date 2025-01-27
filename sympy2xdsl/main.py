import argparse
import sys
import sympy
import sympy.abc

from sympy2xdsl.mlir import ConvertMLIR
from sympy2xdsl.llvm import ConvertLLVM

CONVERTERS = {
    "mlir": ConvertMLIR,
    "llvm": ConvertLLVM
}


class Sympy2XDSLMain:
    _args: argparse.Namespace

    def __init__(self, args: list[str]):
        self._args = self._parse_args(args)

    def _parse_input(self) -> sympy.Expr:
        input: str
        if self._args.input == "-":
            input = sys.stdin.read()
        else:
            with open(self._args.input, "r") as f:
                input_lines = []
                for line in f:
                    if line.strip().startswith("//"):
                        continue
                    input_lines.append(line)
                input = " ".join(input_lines)

        return sympy.parse_expr(input)

    def run(self):
        expr = self._parse_input()

        converter_cls = CONVERTERS[self._args.converter]
        converter = converter_cls.from_args(expr, self._args)

        result = converter.convert()

        if self._args.output == "-":
            print(result)
        else:
            with open(self._args.output, "w") as f:
                f.write(result)

    def _parse_args(self, args: list[str]):
        parser = argparse.ArgumentParser(
            "Sympy2XDSL: Convert SymPy expressions to XDSL"
        )

        parser.add_argument(
            "-i", "--input", help="Input file or - for stdin (default)", default="-"
        )
        parser.add_argument(
            "-o",
            "--output",
            help="Output file or - for stdout, default is stdout",
            default="-",
        )

        subparsers = parser.add_subparsers(dest="converter", required=True)
        for name, converter in CONVERTERS.items():
            local_parser = subparsers.add_parser(name)
            converter.register_args(local_parser)

        return parser.parse_args(args)


def main():
    Sympy2XDSLMain(sys.argv[1:]).run()
