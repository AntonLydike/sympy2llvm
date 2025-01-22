from sympy import *
from sympy.abc import *
from sympy2xdsl.llvm import make_llvm_func_with_args
from xdsl.dialects.builtin import f32, i32


fun = make_llvm_func_with_args(
    "test",
    {x: f32, y: f32},
    (x / 2 + y) * 2.1,
    int_t=i32,
    float_t=f32,
)

print(fun)
