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

# Helper methods to handle views

import sympy
from typing import Sequence


def _rel_state(lhs: sympy.Expr, rhs: sympy.Expr, op: str) -> bool | None:
    """
    Evaluate a symbolic relation safely.
    Returns True/False if decidable, otherwise None.
    """
    try:
        if op == "gt":
            rel = lhs > rhs
        elif op == "lt":
            rel = lhs < rhs
        elif op == "ge":
            rel = lhs >= rhs
        elif op == "le":
            rel = lhs <= rhs
        else:
            raise ValueError(f"unsupported relation op: {op}")
    except Exception:
        return None

    try:
        rel = sympy.simplify(rel)
    except Exception:
        pass

    if rel is True or rel is sympy.true:
        return True
    if rel is False or rel is sympy.false:
        return False
    return None


def _is_true(lhs: sympy.Expr, rhs: sympy.Expr, op: str) -> bool:
    return _rel_state(lhs, rhs, op) is True


def _is_not_false(lhs: sympy.Expr, rhs: sympy.Expr, op: str) -> bool:
    return _rel_state(lhs, rhs, op) is not False


def compute_relative_stride(
    rank: int, device_size: Sequence[sympy.Expr], dim_map: Sequence[int]
) -> list[sympy.Expr]:
    """
    Compute strides of device dimensions with respect to host dimensions
    """
    acc = [sympy.S.One] * rank
    rel_stride = [-1] * len(dim_map)
    for device_dim in range(len(dim_map) - 1, -1, -1):
        dim = dim_map[device_dim]
        if dim != -1:
            rel_stride[device_dim] = acc[dim]
            acc[dim] *= device_size[device_dim]
    return rel_stride


def compute_coordinates(
    size: Sequence[sympy.Expr],
    stride: Sequence[sympy.Expr],
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    index: sympy.Expr,
) -> list[sympy.Expr]:
    """
    Derive an array of coordinate expressions into a tensor from an index
    """
    coordinates = [sympy.S.Zero] * len(size)
    vars = index.free_symbols
    for var in vars:
        if var not in var_ranges:
            continue
        extent = var_ranges[var]
        if extent == 1:
            continue
        term = index.subs({v: 0 for v in vars - {var}})
        step = term.subs(var, 1)
        limit = term.subs(var, extent)
        primary_stride = 0
        primary_dim = -1
        for dim in range(len(size)):
            if size[dim] == 1:
                continue
            st = stride[dim]
            if _is_true(st, step, "gt") and _is_true(st, limit, "lt"):
                coordinates[dim] += var * step // st
            elif _is_not_false(st, step, "le") and _is_not_false(
                st, primary_stride, "gt"
            ):
                primary_stride = st
                primary_dim = dim
        if primary_dim != -1:
            coordinates[primary_dim] += var * step // primary_stride
    return coordinates


def compute_device_coordinates(
    size: Sequence[sympy.Expr],
    stride: Sequence[sympy.Expr],
    device_size: Sequence[sympy.Expr],
    dim_map: Sequence[int],
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    index: sympy.Expr,
) -> list[sympy.Expr]:
    """
    Derive an array of coordinate expressions into a device tensor from an index
    """
    rel_stride = compute_relative_stride(len(size), device_size, dim_map)
    host_coordinates = compute_coordinates(size, stride, var_ranges, index)
    coordinates = [sympy.S.Zero] * len(device_size)
    for dim in range(len(device_size)):
        if dim_map[dim] == -1:
            continue
        expr = host_coordinates[dim_map[dim]]
        vars = expr.free_symbols
        for var in vars:
            if var not in var_ranges:
                continue
            term = expr.subs({v: 0 for v in vars - {var}})
            step = term.subs(var, 1)
            limit = term.subs(var, var_ranges[var])
            if _is_true(limit, rel_stride[dim], "gt") and _is_true(
                step, rel_stride[dim] * device_size[dim], "lt"
            ):
                coordinates[dim] += term // rel_stride[dim]
    return coordinates
