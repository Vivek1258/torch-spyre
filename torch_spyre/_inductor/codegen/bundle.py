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

import json
import logging
import os
from collections.abc import Sequence
from typing import Any, NamedTuple

import sympy

from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.constants import MAX_POOL_SIZE_BYTES
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, format_op_spec_list
from torch_spyre._inductor.op_spec_validation import validate_op_specs


logger = get_inductor_logger("sdsc_compile")
sdsc_log = get_inductor_logger("sdsc")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Compiled SDSC entry: (json_dict, symbol_values, affine_strides, symbol_kinds)
#   symbol_values:  list[int] of registered symbol values for this SDSC,
#                   one per symbol ID.  Values are HBM byte addresses for
#                   derived/pool symbols; arg_index sentinels for kernel
#                   symbols on the symbolic-args path.  Only len() is used
#                   by bundle.py; individual values are resolved via symbols[].
#   affine_strides: list[list[dict]] — per tensor, per loop-nesting level
#                   (outermost first).  Each inner dict maps
#                   tiled_sym -> stride_bytes for that level's symbols.
#                   [{} for _ in tiled_symbols] for non-tiled / lx tensors
#                   (one empty dict per level, preserving the level count).
#   symbol_kinds:   list[SymbolKind] parallel to symbol_values
#   cached_json:    the JSON from the first (canonical) compilation of this SDSC,
#                   used for the sdsc_filename and printed symbol_ids in
#                   sdsc_execute.  Equals sdsc_json on a cache miss; on a hit it
#                   carries the original symbol IDs while sdsc_json carries the
#                   fresh ones used for operand resolution.
_CompiledEntry = tuple[Any, list[int], list[list[dict]], list[SymbolKind], Any]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_bundle(
    kernel_name: str,
    output_dir: str,
    specs: Sequence,
    pool_size: int = 0,
) -> list[SymbolKind]:
    """Output the SDSC Bundle for the OpSpecs in output_dir.

    ``specs`` is a list of ``OpSpec | LoopSpec`` entries (nested ``LoopSpec``
    entries are supported).

    HBM tensor addresses are emitted as runtime symbols (``%sym_N``
    constants) in ``bundle.mlir``. Dimension symbols (from ``mark_dynamic``)
    always produce
    ``!sdscbundle.input_arg<index, granularity=G, max_value=M>`` parameters.

    Requires ``config.bundle_symbolic_args`` to be True: the SDSC path
    (this function) unconditionally emits symbolic addresses, but
    ``spyre_kernel.py``/``hbm_pool_planning.py`` still bake absolute
    addresses into tensor allocations when the flag is False (reserved for
    the KTIR emitter path, which is gated separately). Running this
    function with the flag off would silently miscompile addresses rather
    than error, so it's rejected up front instead.

    ``pool_size`` is the byte count for this bundle's HBM pool (from
    ``hbm_pool_planning.py`` via ``SpyreKernel.pool_size``). Ignored unless
    a pool symbol is present in ``specs``. When a pool symbol is present,
    how ``%pool`` is produced depends on
    ``config.frontend_pool_allocation``:

    - ``False`` (default): emitted as ``%pool = sdscbundle.device_mem_allocate
      <pool_size> bytes : index`` as the first statement of the bundle
      body — there is no ``%pool_base_addr`` function parameter.
    - ``True``: emitted as a ``%pool_base_addr: !sdscbundle.input_arg<index>``
      function parameter (ordered first, before any
      ``kernel_arg_sym_indices``/``dimension_sym_indices`` params) plus an
      ``sdscbundle.input_arg_extract value from %pool_base_addr`` statement
      producing ``%pool`` as the first statement of the bundle body. The
      caller (``SpyreKernel.call_kernel()``) is then responsible for
      allocating a real pool tensor and passing its address as this
      argument.

    Either way, ``%pool`` is an ordinary ``index``-typed SSA value; every
    per-buffer ``arith.addi %pool, <offset>`` emission downstream is
    identical in both modes.
    """
    if not _spyre_config.bundle_symbolic_args:
        raise AssertionError(
            "generate_bundle() requires config.bundle_symbolic_args=True "
            "(BUNDLE_SYMBOLIC_ARGS=1). The SDSC bundle path always emits "
            "symbolic HBM addresses; baked absolute addresses "
            "(bundle_symbolic_args=False) are only supported on the KTIR "
            "emitter path (config.ktir_emitter=True)."
        )

    specs_list: list = list(specs)

    if _spyre_config.validate_op_specs:
        validate_op_specs(specs_list, stage="before_bundle_generation")
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "OP SPECS FOR BUNDLE GENERATION\n%s",
            format_op_spec_list(specs_list),
        )

    # -----------------------------------------------------------------------
    # Pass 1: compile all OpSpecs depth-first.
    # ``symbols`` is indexed by abs(symbol_id)-1: one entry per symbol ID in
    # registration order, values may repeat across SDSCs.  Writes one
    # ``sdsc_N.json`` file per OpSpec.
    # -----------------------------------------------------------------------
    symbols: list[int] = []
    compiled: list[_CompiledEntry] = []
    sdsc_counter = [0]
    symbol_id_offset_counter = [0]

    sdsc_cache_counts: list[int] | None = None
    if _spyre_config.sdsc_cache:
        sdsc_cache_counts = [0, 0]  # [hits, misses]
    _compile_specs(
        specs_list,
        symbols,
        compiled,
        sdsc_counter,
        symbol_id_offset_counter,
        output_dir,
        sdsc_cache={} if _spyre_config.sdsc_cache else None,
        _sdsc_cache_counts=sdsc_cache_counts,
    )
    if sdsc_cache_counts is not None:
        hits, misses = sdsc_cache_counts
        logger.info(
            "sdsc_cache: %d/%d ops reused an existing sdsc file (%d unique)",
            hits,
            hits + misses,
            misses,
        )

    # -----------------------------------------------------------------------
    # Pass 2: emit bundle.mlir.
    # -----------------------------------------------------------------------

    # Collect loop bounds and affine maps needed across the whole tree.
    loop_bounds: list[sympy.Expr] = []
    _collect_loop_bounds(specs_list, loop_bounds)

    # Varying dimensions that drive a symbolic loop bound.
    #
    # These are deliberately kept OUT of the SDSC symbol table. On this route
    # the loop is explicit and the SDSC describes one static tile, so the tile
    # has no symbolic dim and nothing belongs in its dimToSymbolMapping_. The
    # dimension exists only at the bundle level: one input_arg parameter, one
    # extract, and the loop bound derived from it. That also keeps the existing
    # symbol_kinds / symbol_values correspondence untouched.
    loop_dim_bounds = _symbolic_count_bounds_in_specs(specs_list)
    loop_dim_sources = _symbolic_count_sources_in_specs(specs_list)
    # symbol name -> the SSA value its input_arg extracts to.
    loop_dim_ssa: dict[str, str] = {
        sym_name: f"%dim_{sym_name}" for sym_name in loop_dim_bounds
    }
    if loop_dim_bounds:
        logger.info(
            "[symbolic-loop] bundle has %d symbolic loop bound(s); "
            "dimension params=%s bounds=%s",
            sum(1 for b in loop_bounds if getattr(b, "free_symbols", None)),
            loop_dim_ssa,
            loop_dim_bounds,
        )
        # One piece downstream of this file does not exist yet, and it fails
        # AFTER a successful emission, so say it here rather than let a pod run
        # end in a confusing launch error: dispatch does not bind a value for
        # these parameters (#4964), so the launch argument list is one short per
        # dimension.
        #
        # The bound itself is fine. We emit `to <dim> step <G>` and never an
        # authored divide, because arith.ceildivsi is on the backend reject list
        # (dxp.cpp) while a runtime-valued scf.for bound is accepted
        # (LoopUnroll.cpp). Emitting the artifact is the point: it is what we
        # review and hand over.
        logger.warning(
            "[symbolic-loop] EMITTING A BUNDLE WITH %d RUNTIME DIMENSION "
            "PARAMETER(S) %s. No value is bound for them at dispatch yet "
            "(#4964), so expect this kernel to emit correctly and then fail to "
            "launch. That is the known phase-zero boundary, not a regression.",
            len(loop_dim_bounds),
            sorted(loop_dim_ssa.values()),
        )
    else:
        logger.debug(
            "[symbolic-loop] no symbolic loop bounds in this kernel (loop_bounds=%s)",
            loop_bounds,
        )

    # Affine map deduplication: stride_key -> map index (0-based).
    # A stride_key is a tuple of stride values in outermost-first level order.
    # Strides from each level are appended in level order; within a level, in
    # symbol dict insertion order.  The corresponding loop-var indices are built
    # from the explicit level index, so each stride is mapped to the correct
    # loop variable regardless of nesting depth.
    #
    # affine_map_loop_var_indices: parallel to compiled, per op a list of
    # per-tensor loop-var index lists.  Each inner list records which positions
    # in the enclosing loop_vars list correspond to the strides in stride_key,
    # one entry per non-zero stride in outermost-first level order.
    # _emit_specs uses this to pass only the relevant loop vars to affine.apply.
    affine_map_index: dict[tuple, int] = {}
    affine_map_loop_var_indices: list[list[list[int]]] = []
    # One plan per loop level, in _collect_loop_bounds order. Built before the
    # affine maps because a symbolic level changes what its strides mean.
    loop_levels = [
        _loop_level(count, lb_idx, loop_dim_ssa)
        for lb_idx, count in enumerate(loop_bounds)
    ]
    _collect_affine_maps(
        specs_list,
        iter(compiled),
        [],
        affine_map_index,
        affine_map_loop_var_indices,
        [level.stride_scale for level in loop_levels],
    )

    compiled_iter = iter(compiled)
    addr_counter = [0]

    # Flatten symbol kinds from all SDSCs. sym_idx_to_dim_origin records
    # (sdsc_idx, ordinal) for each dimension symbol to generate its MLIR name.
    symbol_kinds: list[SymbolKind] = []
    sym_idx_to_dim_origin: dict[int, tuple[int, int]] = {}
    for sdsc_idx, (_, _, _, local_kinds, _) in enumerate(compiled):
        local_dim_ordinal = 0
        for lk in local_kinds:
            if lk.is_dimension:
                local_dim_ordinal += 1
                sym_idx_to_dim_origin[len(symbol_kinds)] = (
                    sdsc_idx,
                    local_dim_ordinal,
                )
            symbol_kinds.append(lk)

    # Determine whether a pool parameter is needed (any pool symbol present).
    has_pool = any(sk.is_pool for sk in symbol_kinds)
    # Indices of kernel-base symbols that become input_arg parameters.
    # Deduplicated by arg_index: multiple SDSCs operating on different slices of
    # the same logical tensor arg share one function parameter (the first-seen
    # sym_idx, which corresponds to core-0 / the lowest address).  Dedup by
    # address alone is insufficient — different slices have different addresses
    # but the same arg_index and must map to one %arg_{ai}_base_addr param.
    # kernel_arg_sym_indices: list of sym_idx values, one per unique arg_index.
    # kernel_dup_canonical: maps duplicate kernel sym_idx → canonical sym_idx.
    kernel_arg_sym_indices: list[int] = []
    kernel_dup_canonical: dict[int, int] = {}  # duplicate sym_idx → canonical sym_idx
    seen_kernel_arg_index: dict[int, int] = {}  # arg_index → canonical sym_idx
    for i, kind_i in enumerate(symbol_kinds):
        if kind_i.kind == "kernel":
            ai = kind_i.arg_index
            if ai not in seen_kernel_arg_index:
                seen_kernel_arg_index[ai] = i
                kernel_arg_sym_indices.append(i)
            else:
                kernel_dup_canonical[i] = seen_kernel_arg_index[ai]
    # Sort by arg_index so the function signature matches the positional order
    # that call_kernel passes tensors to .run().
    kernel_arg_sym_indices.sort(key=lambda idx: symbol_kinds[idx].arg_index)

    # Deduplicate dimension symbols by pytorch_sym (same shape var may appear
    # in every SDSC with a different local ID).
    dimension_sym_indices: list[int] = []
    dimension_dup_canonical: dict[int, int] = {}  # dup sym_idx → canonical sym_idx
    seen_dim_sym: dict[str, int] = {}  # pytorch_sym → canonical sym_idx
    for i, kind_i in enumerate(symbol_kinds):
        if kind_i.is_dimension:
            dim_sym_key = kind_i.pytorch_sym
            if dim_sym_key not in seen_dim_sym:
                seen_dim_sym[dim_sym_key] = i
                dimension_sym_indices.append(i)
            else:
                dimension_dup_canonical[i] = seen_dim_sym[dim_sym_key]
    # MLIR name for each canonical dimension symbol, e.g. "%sym_0_1".
    dim_param_names: dict[int, str] = {
        sym_idx: (
            f"%sym_{sym_idx_to_dim_origin[sym_idx][0]}"
            f"_{sym_idx_to_dim_origin[sym_idx][1]}"
        )
        for sym_idx in dimension_sym_indices
    }

    with open(os.path.join(output_dir, "bundle.mlir"), "w") as f:
        logger.info(f"Generating {f.name}")

        # Module-level affine map definitions (deduped).
        for stride_key, map_idx in sorted(affine_map_index.items(), key=lambda x: x[1]):
            dims = len(stride_key)
            dim_args = ", ".join(f"d{i}" for i in range(dims))
            terms = " + ".join(f"{stride_key[i]}*d{i}" for i in range(dims))
            f.write(
                f"#map_{map_idx} = affine_map<({dim_args})[s0] -> (s0 + {terms})>\n"
            )

        f.write("module {\n")

        # Function signature:
        #   - when config.frontend_pool_allocation and a pool symbol is
        #     present: one %pool_base_addr !sdscbundle.input_arg<index>
        #     param, emitted first
        #   - one !sdscbundle.input_arg<index> param per kernel tensor arg
        #   - one !sdscbundle.input_arg<index, granularity=G, max_value=M> param
        #     per unique dynamic-shape (mark_dynamic) symbol; emitted whenever
        #     present.
        # Otherwise (default), pool allocation is emitted in the body as
        # device_mem_allocate, not as a function parameter.
        emit_pool_param = has_pool and _spyre_config.frontend_pool_allocation
        # Built in lock-step with the params list so the two can never diverge.
        # Order: pool (when frontend_pool_allocation), kernel addresses, dimensions.
        param_symbol_kinds: list[SymbolKind] = []
        if (
            emit_pool_param
            or kernel_arg_sym_indices
            or dimension_sym_indices
            or loop_dim_bounds
        ):
            params = []
            if emit_pool_param:
                params.append("%pool_base_addr: !sdscbundle.input_arg<index>")
                param_symbol_kinds.append(SymbolKind.pool())
            for sym_idx in kernel_arg_sym_indices:
                ai = symbol_kinds[sym_idx].arg_index
                params.append(f"%arg_{ai}_base_addr: !sdscbundle.input_arg<index>")
                param_symbol_kinds.append(symbol_kinds[sym_idx])
            for sym_idx in dimension_sym_indices:
                dim_sk = symbol_kinds[sym_idx]
                params.append(
                    f"{dim_param_names[sym_idx]}_base: {_dim_input_arg_type(dim_sk)}"
                )
                param_symbol_kinds.append(symbol_kinds[sym_idx])
            # Loop-bound dimensions, emitted last so the positional order of
            # every existing parameter is unchanged.
            #
            # A `loop_dimension` entry IS appended for each, because the runtime
            # derives its argument list from param_symbol_kinds. It is a
            # separate SymbolKind variant from `dimension`: these never enter
            # the SDSC symbol table (see loop_dim_bounds above), they exist only
            # as bundle parameters, and the runtime binds each from
            # inputs_outputs[arg_index].size(dim_index).
            for sym_name, (max_value, granularity) in loop_dim_bounds.items():
                source = loop_dim_sources.get(sym_name)
                if source is None:
                    raise NotImplementedError(
                        f"symbolic loop dimension {sym_name} has bounds "
                        f"{(max_value, granularity)} but no source, so nothing "
                        f"at launch knows which tensor dimension to bind into "
                        f"its parameter. SpyreKernel._resolve_loop_dimension_"
                        f"sources is where that is worked out, and it logs the "
                        f"launch arguments it looked at. Refusing to emit a "
                        f"parameter no one can fill rather than bind a wrong "
                        f"number silently."
                    )
                arg_index, dim_index = source
                params.append(
                    f"{loop_dim_ssa[sym_name]}_base: !sdscbundle.input_arg"
                    f"<index, granularity={granularity}, max_value={max_value}>"
                )
                param_symbol_kinds.append(
                    SymbolKind.loop_dimension(
                        granularity=granularity,
                        max_value=max_value,
                        pytorch_sym=sym_name,
                        arg_index=arg_index,
                        dim_index=dim_index,
                    )
                )
            f.write(f"\tfunc.func @sdsc_bundle({', '.join(params)}) {{\n")
        else:
            f.write("\tfunc.func @sdsc_bundle() {\n")

        assert not has_pool or 0 < pool_size <= MAX_POOL_SIZE_BYTES, (
            f"generate_bundle: pool_size={pool_size} out of range "
            f"(0, {MAX_POOL_SIZE_BYTES}] for a bundle with a pool symbol present"
        )
        if has_pool:
            if _spyre_config.frontend_pool_allocation:
                f.write(
                    "\t\t%pool = sdscbundle.input_arg_extract value from"
                    " %pool_base_addr : !sdscbundle.input_arg<index> -> index\n"
                )
            else:
                f.write(
                    f"\t\t%pool = sdscbundle.device_mem_allocate {pool_size} bytes"
                    " : index\n"
                )

        for sym_idx in kernel_arg_sym_indices:
            ai = symbol_kinds[sym_idx].arg_index
            f.write(
                f"\t\t%arg_{ai} = sdscbundle.input_arg_extract value from"
                f" %arg_{ai}_base_addr : !sdscbundle.input_arg<index> -> index\n"
            )
        for sym_idx in dimension_sym_indices:
            dim_sk = symbol_kinds[sym_idx]
            name = dim_param_names[sym_idx]
            f.write(
                f"\t\t{name} = sdscbundle.input_arg_extract value from"
                f" {name}_base : {_dim_input_arg_type(dim_sk)} -> index\n"
            )

        # Loop-bound dimension extracts, emitted before the loop constants so
        # their SSA values are in scope where %loop_bound_N is derived.
        for sym_name, (max_value, granularity) in loop_dim_bounds.items():
            ssa = loop_dim_ssa[sym_name]
            f.write(
                f"\t\t{ssa} = sdscbundle.input_arg_extract value from"
                f" {ssa}_base : !sdscbundle.input_arg"
                f"<index, granularity={granularity}, max_value={max_value}>"
                " -> index\n"
            )

        # Standard loop constants (only emitted when there are loops).
        if loop_bounds:
            f.write("\t\t%c0 = arith.constant 0 : index\n")
            f.write("\t\t%c1 = arith.constant 1 : index\n")
            for lb_idx, lb in enumerate(loop_bounds):
                level = loop_levels[lb_idx]
                for line in level.setup:
                    f.write(f"\t\t{line}\n")
                logger.info(
                    "[symbolic-loop] loop level %d from count=%s (symbolic=%s): "
                    "scf.for to %s step %s, strides scaled by %d%s",
                    lb_idx,
                    lb,
                    bool(getattr(lb, "free_symbols", None)),
                    level.bound,
                    level.step,
                    level.stride_scale,
                    f", setup={list(level.setup)}" if level.setup else "",
                )

        # Emit one declaration per symbol:
        #   - "kernel"          → skipped; already a function param + extract op above
        #   - "kernel_slice"    → arith.addi %arg_{arg_index}, <slice_offset_bytes>
        #                         deduped by (arg_index, slice_offset) pair;
        #                         produces the SSA "sliced base" that per-core offsets
        #                         and sdsc_execute args reference for sliced tensors
        #   - "kernel_derived"  → arith.addi <sliced_base_ssa>, <per_core_offset>
        #                         deduped by (sliced_base_ssa, per_core_offset)
        #   - "pool"            → arith.addi %pool, <pool_offset>
        #                         deduped by pool offset value
        #   - "dimension"       → skipped; replaced by function parameter above,
        #                         resolved at use-sites via sym_canonical
        #   - anything else     → arith.constant (non-symbolic path)
        # All kernel sym indices to skip during emission (canonical + duplicates).
        kernel_arg_sym_set = set(kernel_arg_sym_indices) | set(kernel_dup_canonical)
        # Map kernel sym_idx → arg_index for SSA name generation.
        # Duplicate kernel sym indices inherit the arg_index of their canonical.
        kernel_sym_to_arg_idx: dict[int, int] = {
            sym_idx: symbol_kinds[sym_idx].arg_index
            for sym_idx in kernel_arg_sym_indices
        }
        for dup_idx, canon_idx in kernel_dup_canonical.items():
            if canon_idx in kernel_sym_to_arg_idx:
                kernel_sym_to_arg_idx[dup_idx] = kernel_sym_to_arg_idx[canon_idx]
        # sym_canonical[sym_idx] → canonical SSA name for derived/pool/slice symbols.
        # Pre-populate duplicate kernel sym_idx entries with their canonical extracted name.
        sym_canonical: dict[int, str] = {
            dup_idx: f"%arg_{kernel_sym_to_arg_idx[dup_idx]}"
            for dup_idx in kernel_dup_canonical
            if dup_idx in kernel_sym_to_arg_idx
        }
        # Dimension symbols resolve to their input_arg_extract result.
        sym_canonical.update(
            (sym_idx, dim_param_names[sym_idx]) for sym_idx in dimension_sym_indices
        )
        sym_canonical.update(
            (dup_idx, dim_param_names[canon_idx])
            for dup_idx, canon_idx in dimension_dup_canonical.items()
        )
        # slice_addi_emitted[(arg_index, slice_offset)] → SSA name for sliced base
        slice_addi_emitted: dict[tuple[int, int], str] = {}
        # derived_addi_emitted[(sliced_base_ssa, per_core_offset)] → SSA name
        derived_addi_emitted: dict[tuple[str, int], str] = {}
        # pool_addi_emitted[pool_offset_value] → SSA name already emitted
        pool_addi_emitted: dict[int, str] = {}

        for sym_idx, value in enumerate(symbols):
            if sym_idx in kernel_arg_sym_set:
                continue  # replaced by function parameter + extract op (or duplicate)
            sk: SymbolKind | None = symbol_kinds[sym_idx] if symbol_kinds else None
            if sk is not None and sk.kind == "kernel_slice":
                ai = sk.arg_index
                sl = sk.offset  # slice offset in bytes
                key = (ai, sl)
                if key not in slice_addi_emitted:
                    slice_offset_ssa = f"%arg_{ai}_slice_offset_{sl}"
                    sliced_base_ssa = f"%arg_{ai}_slice_{sl}"
                    f.write(f"\t\t{slice_offset_ssa} = arith.constant {sl} : index\n")
                    f.write(
                        f"\t\t{sliced_base_ssa} = arith.addi"
                        f" %arg_{ai}, {slice_offset_ssa} : index\n"
                    )
                    slice_addi_emitted[key] = sliced_base_ssa
                sym_canonical[sym_idx] = slice_addi_emitted[key]
            elif sk is not None and sk.is_derived:
                # Resolve the SSA name of the sliced base that this core offset builds on.
                base_sym_idx = sk.base_sym_idx
                if base_sym_idx in sym_canonical:
                    sliced_base_ssa = sym_canonical[base_sym_idx]
                elif base_sym_idx in kernel_arg_sym_indices:
                    # slice_offset == 0: sliced base == raw arg extract (%arg_N)
                    ai = symbol_kinds[base_sym_idx].arg_index
                    sliced_base_ssa = f"%arg_{ai}"
                elif base_sym_idx in kernel_dup_canonical:
                    canon = kernel_dup_canonical[base_sym_idx]
                    ai = kernel_sym_to_arg_idx.get(
                        canon, symbol_kinds[base_sym_idx].arg_index
                    )
                    sliced_base_ssa = f"%arg_{ai}"
                else:
                    sliced_base_ssa = None
                if sliced_base_ssa is not None:
                    key_d = (sliced_base_ssa, sk.offset)
                    if key_d not in derived_addi_emitted:
                        offset_ssa = f"%{sliced_base_ssa[1:]}_core_offset_{sk.offset}"
                        addi_ssa = f"%{sliced_base_ssa[1:]}_core_{sk.offset}"
                        f.write(
                            f"\t\t{offset_ssa} = arith.constant {sk.offset} : index\n"
                        )
                        f.write(
                            f"\t\t{addi_ssa} = arith.addi"
                            f" {sliced_base_ssa}, {offset_ssa} : index\n"
                        )
                        derived_addi_emitted[key_d] = addi_ssa
                    sym_canonical[sym_idx] = derived_addi_emitted[key_d]
                else:
                    f.write(
                        f"\t\t%sym_{sym_idx + 1} = arith.constant {value} : index\n"
                    )
            elif sk is not None and sk.is_pool:
                if value not in pool_addi_emitted:
                    offset_ssa = f"%pool_offset_{value}"
                    addi_ssa = f"%pool_addr_{value}"
                    f.write(f"\t\t{offset_ssa} = arith.constant {value} : index\n")
                    f.write(
                        f"\t\t{addi_ssa} = arith.addi %pool, {offset_ssa} : index\n"
                    )
                    pool_addi_emitted[value] = addi_ssa
                sym_canonical[sym_idx] = pool_addi_emitted[value]
            elif sk is not None and sk.is_dimension:
                continue  # replaced by function parameter; resolved via sym_canonical
            else:
                f.write(f"\t\t%sym_{sym_idx + 1} = arith.constant {value} : index\n")

        # Recursive body emission.
        # affine_map_lv_iter spans the entire spec tree (one entry per OpSpec,
        # in the same depth-first order as compiled_iter) and is consumed by
        # _emit_specs across all recursive calls — not reset per loop level.
        loop_bound_idx = [0]
        affine_map_lv_iter = iter(affine_map_loop_var_indices)
        _emit_specs(
            specs_list,
            compiled_iter,
            loop_levels,
            loop_bound_idx,
            affine_map_index,
            affine_map_lv_iter,
            addr_counter,
            [],
            f,
            indent=2,
            kernel_sym_to_arg_idx=kernel_sym_to_arg_idx,
            sym_canonical=sym_canonical,
        )

        f.write("\t\treturn\n")
        f.write("\t}\n")
        f.write("}\n")

    if sdsc_log.isEnabledFor(logging.DEBUG):
        bundle_path = os.path.join(output_dir, "bundle.mlir")
        with open(bundle_path, "r") as bf:
            sdsc_log.debug("BUNDLE MLIR [bundle.mlir]\n%s", bf.read())

    return param_symbol_kinds


# ---------------------------------------------------------------------------
# Pass 1 helpers
# ---------------------------------------------------------------------------


def _compile_specs(
    specs: list,
    symbols: list[int],
    compiled: list,
    sdsc_counter: list,
    symbol_id_offset_counter: list,
    output_dir: str,
    sdsc_cache: dict | None = None,
    _sdsc_cache_counts: list | None = None,
) -> None:
    """Recursively compile all OpSpecs in specs depth-first.

    Identical op specs (same canonical SDSC at counter 0) reuse the previously
    compiled entry — same sdsc file and same symbol registrations.
    Pass sdsc_cache={} to enable caching; None disables it.
    """
    for entry in specs:
        if isinstance(entry, LoopSpec):
            _compile_specs(
                entry.body,
                symbols,
                compiled,
                sdsc_counter,
                symbol_id_offset_counter,
                output_dir,
                sdsc_cache,
                _sdsc_cache_counts,
            )
        elif isinstance(entry, OpSpec):
            cached = None
            if sdsc_cache is not None:
                # Generate a canonical (counter-0) version as cache key,
                # ignoring debug_handle_ which varies per op but is irrelevant
                # to structural identity.
                canonical_json, _, _, _ = compile_op_spec(0, entry, [], 0)
                top_val = next(iter(canonical_json.values()))
                top_val.pop("debug_handle_", None)
                # arg_indices must be part of the key: the canonical json only
                # records sequential placeholder IDs (-1,-2,-3), not which
                # kernel tensor argument each slot belongs to. Two structurally
                # identical ops on different tensors would otherwise collide.
                arg_indices = tuple(a.arg_index for a in entry.args)
                cache_key = json.dumps(canonical_json, sort_keys=True) + str(
                    arg_indices
                )
                cached = sdsc_cache.get(cache_key)
            if cached is None:
                idx = sdsc_counter[0]
                sdsc_counter[0] += 1
                if _sdsc_cache_counts is not None:
                    _sdsc_cache_counts[1] += 1
            else:
                idx, cached_json = cached
                if _sdsc_cache_counts is not None:
                    _sdsc_cache_counts[0] += 1
            sdsc_json, local_sym_values, affine_strides, local_symbol_kinds = (
                compile_op_spec(
                    idx,
                    entry,
                    symbols,
                    symbol_id_offset_counter[0],
                )
            )
            symbol_id_offset_counter[0] += len(local_sym_values)
            file_name = f"sdsc_{idx}.json"
            if cached is None:
                cached_json = sdsc_json
                if sdsc_cache is not None:
                    sdsc_cache[cache_key] = (idx, cached_json)
                with open(os.path.join(output_dir, file_name), "w") as f:
                    logger.info(f"Generating {f.name}")
                    json.dump(sdsc_json, f, indent=2)
            compiled.append(
                (
                    sdsc_json,
                    local_sym_values,
                    affine_strides,
                    local_symbol_kinds,
                    cached_json,
                )
            )
            if sdsc_log.isEnabledFor(logging.DEBUG):
                sdsc_log.debug(
                    "SDSC JSON [%s]\n%s",
                    file_name,
                    json.dumps(sdsc_json, indent=2),
                )
        # UnimplementedOp and other types are silently skipped.


# ---------------------------------------------------------------------------
# Loop-bound collection
# ---------------------------------------------------------------------------


def _collect_loop_bounds(specs: list, bounds: list) -> None:
    """Collect loop trip counts depth-first (same order as loop var naming)."""
    for entry in specs:
        if isinstance(entry, LoopSpec):
            bounds.append(entry.count)
            _collect_loop_bounds(entry.body, bounds)


def _collect_loop_specs(specs: list, out: list) -> None:
    """Collect LoopSpec objects in the SAME order as ``_collect_loop_bounds``.

    The two walks must stay in lockstep: ``out[i]`` is the LoopSpec whose count
    is ``bounds[i]``, and therefore the LoopSpec carrying the symbol bounds that
    ``%loop_bound_i`` is emitted from.
    """
    for entry in specs:
        if isinstance(entry, LoopSpec):
            out.append(entry)
            _collect_loop_specs(entry.body, out)


def _symbolic_count_bounds_in_specs(specs: list) -> "dict[str, tuple[int, int]]":
    """Union of every LoopSpec's ``count_symbol_bounds`` in the spec tree.

    One entry per varying dimension that has to become a bundle parameter.
    Conflicting bounds for the same symbol mean two loops disagree about the
    same dimension, which would emit one parameter that is wrong for one of
    them, so it raises rather than silently picking one.
    """
    merged: dict[str, tuple[int, int]] = {}
    loops: list = []
    _collect_loop_specs(specs, loops)
    for loop in loops:
        for sym_name, bound in getattr(loop, "count_symbol_bounds", {}).items():
            existing = merged.get(sym_name)
            if existing is not None and existing != bound:
                raise RuntimeError(
                    f"symbolic loop bound conflict for {sym_name}: "
                    f"{existing} vs {bound}. Two loops in one kernel disagree "
                    f"about the same dimension's granularity/max."
                )
            merged[sym_name] = bound
    return merged


def _symbolic_count_sources_in_specs(specs: list) -> "dict[str, tuple[int, int]]":
    """Union of every LoopSpec's ``count_symbol_sources`` in the spec tree.

    Symbol name -> ``(arg_index, dim_index)``, the launch argument and dimension
    the runtime reads that symbol's value from. Conflicting sources for one
    symbol mean two loops disagree about where the same dimension comes from,
    which would bind one parameter from the wrong tensor, so it raises rather
    than picking one.
    """
    merged: dict[str, tuple[int, int]] = {}
    loops: list = []
    _collect_loop_specs(specs, loops)
    for loop in loops:
        for sym_name, source in getattr(loop, "count_symbol_sources", {}).items():
            existing = merged.get(sym_name)
            if existing is not None and existing != tuple(source):
                raise RuntimeError(
                    f"symbolic loop dimension source conflict for {sym_name}: "
                    f"{existing} vs {tuple(source)}. Two loops in one kernel "
                    f"read the same dimension from different launch arguments."
                )
            merged[sym_name] = tuple(source)
    return merged


# ---------------------------------------------------------------------------
# Affine map deduplication
# ---------------------------------------------------------------------------


def _scaled_level_strides(per_level_strides, scale_stack):
    """Yield ``(level_idx, stride)`` in affine-map key order, scaled per level.

    ONE place does this. ``_collect_affine_maps`` BUILDS the map index from this
    and ``_emit_specs`` LOOKS IT UP, so if the two compute the key even slightly
    differently the result is a bare ``KeyError`` at emit time with nothing to
    say why. That has happened once already, ``(8192,)`` looked up against a
    table keyed ``(128,)``, when only the building side applied the scale.

    A symbolic level's loop variable steps by its tile size instead of by 1, so
    every stride multiplying it shrinks by that same factor. Exact by
    construction: the stride IS the tile size times the per-element step.
    """
    for level_idx, level_strides in enumerate(per_level_strides):
        if not level_strides:
            continue
        scale = scale_stack[level_idx] if level_idx < len(scale_stack) else 1
        for stride in level_strides.values():
            if scale != 1:
                if stride % scale != 0:
                    raise AssertionError(
                        f"affine stride {stride} is not divisible by the loop "
                        f"step {scale} at level {level_idx}. A symbolic level "
                        f"steps by its tile size, so its strides must be a "
                        f"multiple of it, or the loop variable and the stride "
                        f"disagree about what one step means."
                    )
                stride = stride // scale
            yield level_idx, stride


def _collect_affine_maps(
    specs: list,
    compiled_iter,
    loop_var_depth: list,
    affine_map_index: dict,
    loop_var_indices_out: list,
    loop_scales: "list[int] | None" = None,
    lb_counter: "list[int] | None" = None,
    scale_stack: "list[int] | None" = None,
) -> None:
    """Walk the spec tree and register unique affine stride keys.

    Populates ``affine_map_index`` (stride_key -> map_idx) and appends one
    entry per OpSpec to ``loop_var_indices_out``.  Each entry is a list of
    per-tensor index lists: ``loop_var_indices_out[op_idx][tensor_idx]`` is
    the list of loop-var positions (into the enclosing ``loop_vars`` list at
    emit time) that correspond to the strides in the tensor's stride_key,
    in outermost-first level order.

    ``affine_strides[tensor_idx]`` is a list of dicts, one per loop-nesting
    level (outermost first).  We iterate over levels explicitly and use
    ``loop_var_depth[level_idx]`` to find the correct loop variable for each
    level's strides — no counting from the end.
    """
    # ``loop_scales[i]`` is the stride divisor for the i-th loop level, in the
    # same depth-first order as _collect_loop_bounds. Only a symbolic level is
    # ever anything but 1, so a concrete kernel's stride keys are unchanged.
    if loop_scales is None:
        loop_scales = []
    if lb_counter is None:
        lb_counter = [0]
    if scale_stack is None:
        scale_stack = []

    for entry in specs:
        if isinstance(entry, LoopSpec):
            lb_idx = lb_counter[0]
            lb_counter[0] += 1
            scale = loop_scales[lb_idx] if lb_idx < len(loop_scales) else 1
            _collect_affine_maps(
                entry.body,
                compiled_iter,
                loop_var_depth + [len(loop_var_depth)],
                affine_map_index,
                loop_var_indices_out,
                loop_scales,
                lb_counter,
                scale_stack + [scale],
            )
        elif isinstance(entry, OpSpec):
            _, _, affine_strides, _, _ = next(compiled_iter)
            per_tensor_lv_indices: list[list[int]] = []
            for per_level_strides in affine_strides:
                # per_level_strides is list[dict], one dict per level (outermost first).
                # Build stride_key and lv_indices by iterating levels explicitly.
                stride_vals: list[int] = []
                lv_idxs: list[int] = []
                for level_idx, stride in _scaled_level_strides(
                    per_level_strides, scale_stack
                ):
                    assert level_idx < len(loop_var_depth), (
                        f"affine_strides has {len(per_level_strides)} levels but "
                        f"only {len(loop_var_depth)} enclosing loop(s); "
                        "create_op_spec built more tiled_syms levels than LoopSpec ancestors"
                    )
                    stride_vals.append(stride)
                    lv_idxs.append(loop_var_depth[level_idx])
                if not stride_vals:
                    per_tensor_lv_indices.append([])
                    continue
                stride_key = tuple(stride_vals)
                if stride_key not in affine_map_index:
                    affine_map_index[stride_key] = len(affine_map_index)
                per_tensor_lv_indices.append(lv_idxs)
            loop_var_indices_out.append(per_tensor_lv_indices)


# ---------------------------------------------------------------------------
# Pass 2 helpers
# ---------------------------------------------------------------------------


def _mlir_count_value(count: sympy.Expr) -> str:
    """Return an MLIR value expression for a loop trip count."""
    if isinstance(count, (sympy.Integer, int)):
        return f"arith.constant {int(count)} : index"
    raise NotImplementedError(
        f"Symbolic loop counts are not yet supported in bundle.mlir generation: {count}"
    )


def _decompose_symbolic_count(count: sympy.Expr):
    """Split a symbolic trip count into ``(symbol_name, divisor)``.

    A tiled loop's trip count is the varying dimension divided by the tile size,
    which Inductor represents as ``FloorDiv(S, G)``. The two pieces become the
    loop's bound and its step, so the device derives the count from the
    dimension rather than the host precomputing it. A bare ``S`` (tile size 1)
    gives a divisor of 1.

    Args:
        count: A trip count with free symbols.

    Returns:
        ``(symbol_name, divisor)``, or ``None`` when the expression is not a
        shape this function recognizes. Callers must treat ``None`` as "cannot
        emit", never as "emit something plausible".
    """
    from torch.utils._sympy.functions import FloorDiv

    if isinstance(count, sympy.Symbol):
        return (str(count), 1)

    if isinstance(count, FloorDiv):
        numerator, denominator = count.args
        if isinstance(numerator, sympy.Symbol) and denominator.is_Integer:
            return (str(numerator), int(denominator))

    # The same count after a round trip through the generated source. That
    # source IS the reload path: op specs are serialized as `sympify('<str>')`,
    # `str(FloorDiv(s, 64))` is `(s//64)`, and sympy parses `//` back into
    # `sympy.floor(s/64)` rather than into torch's FloorDiv. Both spellings mean
    # the same thing for a positive integer, so accept both here, at the single
    # point that interprets the count's shape.
    if isinstance(count, sympy.floor):
        numerator, denominator = count.args[0].as_numer_denom()
        if isinstance(numerator, sympy.Symbol) and denominator.is_Integer:
            return (str(numerator), int(denominator))

    return None


class LoopLevel(NamedTuple):
    """How one ``scf.for`` level is emitted, and what that does to its strides.

    ``stride_scale`` is what every affine stride multiplying this level's loop
    variable must be divided by, which is exact by construction: a level's
    stride is its per-element step times the tile size.
    """

    setup: tuple[str, ...]
    bound: str
    step: str
    stride_scale: int


def _loop_level(
    count: sympy.Expr,
    lb_idx: int,
    loop_dim_ssa: "dict[str, str]",
) -> LoopLevel:
    """Pick the ``scf.for`` form for one loop level.

    A concrete count keeps exactly the form this emitter has always produced: a
    constant upper bound with step 1, loop variable counting TILES.

    A symbolic count becomes ``to <dim> step <G>``, so the loop variable counts
    ELEMENTS along the varying dimension and the device works the trip count out
    as ``(ub - lb) / step`` itself. We deliberately do NOT author the division.
    An ``arith.ceildivsi`` bound is on the backend's reject list (dxp.cpp), while
    a runtime-valued bound is accepted (LoopUnroll.cpp), so the only legal shape
    that keeps the dimension itself in the bundle is bound=S step=G. It is also
    the shape that lets one dimension parameter drive two loops with different
    tile sizes, since each derives its own count.

    Because the loop variable now advances by G rather than 1, this returns
    ``stride_scale=G`` and the caller shrinks that level's affine strides to
    match. Floor versus ceiling does not matter here: G divides S exactly, which
    is guaranteed by the ragged-split refusal plus its torch._check.

    Args:
        count: The loop's trip count.
        lb_idx: Index used to name this level's SSA values.
        loop_dim_ssa: Symbol name to the SSA value its input_arg extracted to.

    Raises:
        NotImplementedError: The count is symbolic but not a shape this emitter
            recognizes, or names a symbol with no input_arg.
    """
    if isinstance(count, (sympy.Integer, int)):
        return LoopLevel(
            setup=(f"%loop_bound_{lb_idx} = arith.constant {int(count)} : index",),
            bound=f"%loop_bound_{lb_idx}",
            step="%c1",
            stride_scale=1,
        )

    decomposed = _decompose_symbolic_count(count)
    if decomposed is None:
        raise NotImplementedError(
            f"symbolic loop count {count!r} (type {type(count).__name__}, "
            f"free_symbols={sorted(map(str, getattr(count, 'free_symbols', [])))}) "
            "is not a recognized trip-count shape. Expected FloorDiv(symbol, "
            "integer) or a bare symbol. Extend _decompose_symbolic_count if this "
            "shape is legitimate."
        )

    sym_name, divisor = decomposed
    if sym_name not in loop_dim_ssa:
        raise NotImplementedError(
            f"symbolic loop count {count} references {sym_name}, which has no "
            f"input_arg parameter. Known dimension params: {sorted(loop_dim_ssa)}. "
            "This means LoopSpec.count_symbol_bounds did not carry bounds for "
            "that symbol, so scheduler/spyre_kernel is where to look."
        )

    dim_ssa = loop_dim_ssa[sym_name]
    if divisor == 1:
        # Tile size 1, so the loop variable already counts single elements and
        # the dimension is the bound as-is.
        return LoopLevel(setup=(), bound=dim_ssa, step="%c1", stride_scale=1)
    return LoopLevel(
        setup=(f"%step_{lb_idx} = arith.constant {divisor} : index",),
        bound=dim_ssa,
        step=f"%step_{lb_idx}",
        stride_scale=divisor,
    )


def _dim_input_arg_type(dim_sk: SymbolKind) -> str:
    """MLIR input_arg type string for a dimension symbol.

    Shared by the function-parameter declaration and the corresponding
    input_arg_extract op so the two can't drift out of sync.
    """
    return (
        f"!sdscbundle.input_arg<index, granularity={dim_sk.granularity}, "
        f"max_value={dim_sk.max_value}>"
    )


def _emit_specs(
    specs: list,
    compiled_iter,
    loop_levels: "list[LoopLevel]",
    loop_bound_idx: list,
    affine_map_index: dict,
    affine_map_lv_iter,
    addr_counter: list,
    loop_vars: list,
    f,
    indent: int,
    kernel_sym_to_arg_idx: dict | None = None,
    sym_canonical: dict | None = None,
    loop_scales: list | None = None,
) -> None:
    """Recursively emit MLIR ops for specs into file f."""
    if kernel_sym_to_arg_idx is None:
        kernel_sym_to_arg_idx = {}
    if sym_canonical is None:
        sym_canonical = {}
    # Parallel to loop_vars: the stride scale of each enclosing loop level, in
    # the same order. Threaded the same way so the two cannot get out of step.
    if loop_scales is None:
        loop_scales = []

    # Map from 0-based symbol index to the short SSA name for kernel-arg symbols.
    # sym_idx → %arg_{arg_index}  (the result of input_arg_extract in the function body)
    kernel_arg_sym_to_name: dict[int, str] = {
        sym_idx: f"%arg_{ai}" for sym_idx, ai in kernel_sym_to_arg_idx.items()
    }

    def _resolve_sym(sid: int) -> str:
        # sid is a negative symbol ID; abs(sid)-1 is the 0-based index into symbols[].
        # Both dicts are safe to check unconditionally — empty when their feature is off.
        sym_idx = abs(sid) - 1
        if sym_idx in kernel_arg_sym_to_name:
            return kernel_arg_sym_to_name[sym_idx]
        if sym_idx in sym_canonical:
            return sym_canonical[sym_idx]
        return f"%sym_{abs(sid)}"

    tab = "\t" * indent
    for entry in specs:
        if isinstance(entry, LoopSpec):
            lb_idx = loop_bound_idx[0]
            loop_bound_idx[0] += 1
            loop_var = f"%i_{lb_idx}"
            level = loop_levels[lb_idx]
            f.write(
                f"{tab}scf.for {loop_var} = %c0 to {level.bound} "
                f"step {level.step} {{\n"
            )
            _emit_specs(
                entry.body,
                compiled_iter,
                loop_levels,
                loop_bound_idx,
                affine_map_index,
                affine_map_lv_iter,
                addr_counter,
                loop_vars + [loop_var],
                f,
                indent + 1,
                kernel_sym_to_arg_idx=kernel_sym_to_arg_idx,
                sym_canonical=sym_canonical,
                loop_scales=loop_scales + [level.stride_scale],
            )
            f.write(f"{tab}}}\n")

        elif isinstance(entry, OpSpec):
            sdsc_json, local_sym_values, affine_strides, _, cached_json = next(
                compiled_iter
            )
            # Per-tensor loop-var index lists: which positions in the enclosing
            # loop_vars list correspond to the strides for each tensor.
            per_tensor_lv_indices: list[list[int]] = next(affine_map_lv_iter)

            # Filename and printed symbol_ids come from the cached (first) JSON so
            # that deduplicated executions reference the same sdsc file and IDs.
            sdsc_name = next(iter(cached_json))
            sdsc_idx = sdsc_name.split("_")[0]
            sdsc_filename = f"sdsc_{sdsc_idx}.json"
            cached_symbol_ids = _extract_symbol_ids(cached_json)

            # Fresh symbol_ids (from sdsc_json) are used only for resolving operands.
            symbol_ids = _extract_symbol_ids(sdsc_json)

            # Build affine.apply ops for tiled tensors, tracking which
            # symbol IDs have been upgraded to per-iteration %addr_N names.
            # affine_strides[tensor_idx] is list[dict] (per level, outermost first).
            sym_id_to_operand: dict[int, str] = {}
            for tensor_idx, per_level_strides in enumerate(affine_strides):
                # Flatten per-level strides to build the stride_key in the same
                # outermost-first order used by _collect_affine_maps.
                flat_strides: list[int] = [
                    stride
                    for _, stride in _scaled_level_strides(
                        per_level_strides, loop_scales
                    )
                ]
                if not flat_strides:
                    continue
                num_cores = _sdsc_num_cores(sdsc_json)
                for c in range(num_cores):
                    base_sym_id = _get_tensor_core_sym_id(sdsc_json, tensor_idx, c)
                    if base_sym_id is None or base_sym_id in sym_id_to_operand:
                        continue
                    stride_key = tuple(flat_strides)
                    map_idx = affine_map_index[stride_key]
                    addr_name = f"%addr_{addr_counter[0]}"
                    addr_counter[0] += 1
                    base_addr_name = _resolve_sym(base_sym_id)
                    # lv_indices[tensor_idx] was built by _collect_affine_maps using
                    # explicit level indexing — each entry is the loop_vars position
                    # for the corresponding stride in stride_key.
                    lv_indices = per_tensor_lv_indices[tensor_idx]
                    apply_loop_vars = [loop_vars[i] for i in lv_indices]
                    loop_var_str = ", ".join(apply_loop_vars)
                    f.write(
                        f"{tab}{addr_name} = affine.apply #map_{map_idx}"
                        f"({loop_var_str})[{base_addr_name}]\n"
                    )
                    sym_id_to_operand[base_sym_id] = addr_name

            # Each operand position matches one symbol_id entry.
            # Tiled sym_ids use the %addr_N computed above; others use %sym_N.
            operands = [
                sym_id_to_operand.get(sid, _resolve_sym(sid)) for sid in symbol_ids
            ]

            operand_str = ", ".join(operands)
            symbol_ids_str = ", ".join(str(i) for i in cached_symbol_ids)
            f.write(
                f"{tab}sdscbundle.sdsc_execute ({operand_str}) "
                f'{{sdsc_filename="{sdsc_filename}", '
                f'"symbol_ids"=[{symbol_ids_str}]}}\n'
            )


def _extract_symbol_ids(sdsc_json: dict) -> list[int]:
    """Extract all negative symbol IDs from an SDSC JSON, dimension IDs first.

    Dimension IDs (``dimToSymbolMapping_``) have lower-magnitude negatives than
    HBM address IDs, so scanning them first keeps ``ids`` sorted naturally.
    """
    ids: list[int] = []
    seen: set[int] = set()
    for top_val in sdsc_json.values():
        for dsc_entry in top_val.get("dscs_", []):
            for op_val in dsc_entry.values():
                for dim_syms in op_val.get("dimToSymbolMapping_", {}).values():
                    for v in dim_syms:
                        sym_id = int(v)
                        if sym_id < 0 and sym_id not in seen:
                            ids.append(sym_id)
                            seen.add(sym_id)
                for node in op_val.get("scheduleTree_", []):
                    # NOTE: "hbm" is an sdsc component field and is
                    # distinct from and NOT to be confused with the internal
                    # layout.allocation dict keys ("hbm"/"lx"/"hbm_pool").
                    if node.get("component_") == "hbm":
                        data = node.get("startAddressCoreCorelet_", {}).get("data_", {})
                        for v in data.values():
                            sym_id = int(v)
                            if sym_id < 0 and sym_id not in seen:
                                ids.append(sym_id)
                                seen.add(sym_id)
    return ids


def _sdsc_num_cores(sdsc_json: dict) -> int:
    """Extract num_cores from the SDSC JSON."""
    for top_val in sdsc_json.values():
        return top_val.get("numCoresUsed_", 1)
    return 1


def _get_tensor_core_sym_id(sdsc_json: dict, tensor_idx: int, core: int) -> int | None:
    """Return the symbol ID (negative int) for (tensor_idx, core), or None if lx."""
    for top_val in sdsc_json.values():
        for dsc_entry in top_val.get("dscs_", []):
            for op_val in dsc_entry.values():
                nodes = op_val.get("scheduleTree_", [])
                if tensor_idx < len(nodes):
                    node = nodes[tensor_idx]
                    # NOTE: "hbm" is an sdsc component field and is
                    # distinct from and NOT to be confused with the internal
                    # layout.allocation dict keys ("hbm"/"lx"/"hbm_pool").
                    if node.get("component_") != "hbm":
                        return None
                    data = node.get("startAddressCoreCorelet_", {}).get("data_", {})
                    key = f"[{core}, 0, 0]"
                    if key in data:
                        return int(data[key])
    return None


# ---------------------------------------------------------------------------
# Helpers re-exported for tests
# ---------------------------------------------------------------------------


def _collect_op_specs(specs: list, result: list) -> None:
    """Collect all OpSpec leaves depth-first (for tests / async_compile)."""
    for entry in specs:
        if isinstance(entry, LoopSpec):
            _collect_op_specs(entry.body, result)
        elif isinstance(entry, OpSpec):
            result.append(entry)


def _collect_loop_counts(specs: list) -> list:
    """Return loop counts in depth-first order (for tests)."""
    counts: list = []
    for entry in specs:
        if isinstance(entry, LoopSpec):
            counts.append(entry.count)
            counts.extend(_collect_loop_counts(entry.body))
    return counts
