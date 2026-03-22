# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple

from sympy import Expr, Symbol

import sympy
from torch._inductor.ir import FixedLayout
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import V

from .ir import FixedTiledLayout
from .views import compute_device_coordinates, compute_coordinates


def _concretize_expr(expr: sympy.Expr) -> sympy.Expr:
    """
    Convert a symbolic expression to a concrete value using PyTorch's size variable system.
    This is needed to avoid symbolic comparison errors in layout analysis.
    """
    if isinstance(expr, (int, sympy.Integer)):
        return expr
    
    # Use PyTorch's size variable system to evaluate symbolic expressions
    if hasattr(V.graph, 'sizevars') and V.graph.sizevars is not None:
        try:
            concrete_value = V.graph.sizevars.size_hint(expr)
            return sympy.Integer(concrete_value)
        except Exception:
            pass
    
    # If expression has free symbols, try to substitute them
    if hasattr(expr, 'free_symbols') and len(expr.free_symbols) > 0:
        subs_dict = {}
        for symbol in expr.free_symbols:
            if hasattr(V.graph, 'sizevars') and V.graph.sizevars is not None:
                try:
                    val = V.graph.sizevars.size_hint(symbol)
                    subs_dict[symbol] = val
                except Exception:
                    continue
        if subs_dict:
            result = expr.subs(subs_dict)
            return sympy.Integer(int(result)) if isinstance(result, (int, sympy.Integer, sympy.Number)) else result
    
    # Last resort: try direct conversion
    try:
        return sympy.Integer(int(expr))
    except (TypeError, ValueError):
        # If we can't concretize, return as-is and let it fail with a better error message
        return expr


def _concretize_var_ranges(var_ranges: dict[sympy.Symbol, sympy.Expr]) -> dict[sympy.Symbol, sympy.Expr]:
    """
    Concretize all values in var_ranges dictionary.
    """
    return {sym: _concretize_expr(val) for sym, val in var_ranges.items()}


class SchedNodeArg(NamedTuple):
    dep: MemoryDep
    layout: FixedTiledLayout


def get_mem_deps(n: SchedulerNode) -> list[SchedNodeArg]:
    res: list[SchedNodeArg] = []
    for arg in n.read_writes.reads:
        if isinstance(arg, MemoryDep):
            buf = V.graph.get_buffer(arg.name)
            layout = buf.get_layout()
            if not isinstance(layout, FixedTiledLayout):
                raise RuntimeError(f"{buf} does not have FixedTiledLayout")
            res.append(SchedNodeArg(arg, layout))
    return res


def wildcard_symbol(dim) -> Symbol:
    return sympy.Symbol(f"*_{dim}")


def is_wildcard(s: Symbol) -> bool:
    return s.name.startswith("*_")


def map_dims_to_vars(layout: FixedLayout, index: Expr) -> dict[int, Symbol]:
    """
    Construct a mapping from the dimensions of layout
    to the free variables of index that correspond to them.
    Dimensions of size 1 are mapped to a wild_card_symbol of `*`

    This works by reversing the algorithm used by torch._inductor.ir. _fixed_indexer to build index.
    """
    result = {}
    for sym in index.free_symbols:
        stride_val = sympy_subs(index, {sym: 1}) - sympy_subs(index, {sym: 0})
        if stride_val in layout.stride:
            idx = layout.stride.index(stride_val)
            result[idx] = sym

    for d in range(len(layout.size)):
        if d not in result:
            # For dynamic shapes, dimensions might not be in the index expression
            # even if they're non-trivial. Use wildcard symbol for unmapped dimensions.
            result[d] = wildcard_symbol(d)

    return result


def host_coordinates(layout: FixedLayout, dep: MemoryDep) -> list[sympy.Expr]:
    # Concretize layout dimensions and var_ranges to avoid symbolic comparison errors
    concrete_size = [_concretize_expr(s) for s in layout.size]
    concrete_stride = [_concretize_expr(s) for s in layout.stride]
    concrete_ranges = _concretize_var_ranges(dep.ranges)
    
    # Also need to concretize the index expression to replace symbolic dimensions
    concrete_index = dep.index
    if hasattr(dep.index, 'free_symbols') and len(dep.index.free_symbols) > 0:
        if hasattr(V.graph, 'sizevars') and V.graph.sizevars is not None:
            subs_dict = {}
            for symbol in dep.index.free_symbols:
                try:
                    val = V.graph.sizevars.size_hint(symbol)
                    subs_dict[symbol] = val
                except Exception:
                    pass
            if subs_dict:
                concrete_index = dep.index.subs(subs_dict)
    
    return compute_coordinates(concrete_size, concrete_stride, concrete_ranges, concrete_index)


def device_coordinates(layout: FixedTiledLayout, dep: MemoryDep) -> list[sympy.Expr]:
    # Concretize all dimensions and var_ranges to avoid symbolic comparison errors
    concrete_size = [_concretize_expr(s) for s in layout.size]
    concrete_stride = [_concretize_expr(s) for s in layout.stride]
    concrete_device_size = [_concretize_expr(s) for s in layout.device_layout.device_size]
    concrete_ranges = _concretize_var_ranges(dep.ranges)
    
    # Also need to concretize the index expression
    concrete_index = dep.index
    if hasattr(dep.index, 'free_symbols') and len(dep.index.free_symbols) > 0:
        if hasattr(V.graph, 'sizevars') and V.graph.sizevars is not None:
            subs_dict = {}
            for symbol in dep.index.free_symbols:
                try:
                    val = V.graph.sizevars.size_hint(symbol)
                    subs_dict[symbol] = val
                except Exception:
                    pass
            if subs_dict:
                concrete_index = dep.index.subs(subs_dict)
    
    return compute_device_coordinates(
        concrete_size,
        concrete_stride,
        concrete_device_size,
        layout.device_layout.dim_map,
        concrete_ranges,
        concrete_index,
    )
