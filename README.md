# sympy2llvm - Convert sympy expressions to SSA form

This projects provides a simple tool to convert sympy expressions to SSA form.

## Usage:

Either from command line:

```
sympy2llvm [-i input_file] {llvm,mlir} fun_name argument_type ...
```

This reads a sympy expression either from stdin or the provided file, and prints the desired output (either MLIRs 
llvm dialect, or llvmir directly).


Or programmatically:

```py
from sympy2llvm.llvm import ConvertLLVM, LLVMType
import sympy

x,y = sympy.symbols("x y")

conv = ConvertLLVM(x/y, "div", (LLVMType.i32, LLVMType.i32), int_t=LLVMType.i32)
llvmir = conv.convert()
```
