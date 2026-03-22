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
import tempfile
import sympy
from sympy import Expr
from typing import Any, Union
import os
import subprocess

from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._C import convert_artifacts
from torch_spyre._inductor.codegen.superdsc import generate_sdsc
from torch_spyre._inductor.constants import SEGMENT_OFFSETS
from torch_spyre._inductor.logging_utils import get_inductor_logger, _get_env_bool
from torch_spyre._inductor.op_spec import OpSpec, UnimplementedOp
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner

logger = get_inductor_logger("sdsc_compile")

_argument_names = ["arg0", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6"]

_SDSC_BUNDLE = _get_env_bool("SPYRE_SUPERDSC_BUNDLE")


def _collect_sizevar_symbols_by_name() -> dict[str, sympy.Symbol]:
    """
    Collect the canonical sizevar symbols currently known to inductor, indexed by name.
    """
    from torch._inductor.virtualized import V
    
    symbols: dict[str, sympy.Symbol] = {}
    sizevars = V.graph.sizevars
    
    def _add_from_mapping(mapping):
        if isinstance(mapping, dict):
            for key in mapping.keys():
                if isinstance(key, sympy.Symbol) and key.name not in symbols:
                    symbols[key.name] = key
    
    _add_from_mapping(getattr(sizevars, "var_to_val", None))
    _add_from_mapping(getattr(sizevars, "precomputed_replacements", None))
    
    shape_env = getattr(sizevars, "shape_env", None)
    if shape_env is not None:
        _add_from_mapping(getattr(shape_env, "var_to_val", None))
        _add_from_mapping(getattr(shape_env, "replacements", None))
    
    return symbols


def _concretize_dimension(dim: Expr | int) -> int:
    """
    Convert a symbolic or concrete dimension to a concrete integer.
    
    This function implements a robust 4-step conversion process:
    1. Collect canonical symbols and replace
    2. Try size_hint() lookup
    3. Direct evaluation if no free symbols
    4. Symbol substitution with hints
    
    Args:
        dim: Either a concrete int or a sympy.Expr (potentially symbolic)
    
    Returns:
        Concrete integer dimension value
    
    Raises:
        TypeError: If the dimension cannot be concretized
    """
    from torch._inductor.virtualized import V
    
    if not isinstance(dim, Expr):
        return int(dim)
    
    sizevars = V.graph.sizevars
    expr: Expr = dim
    
    # Step 1: Collect canonical symbols and replace
    if expr.free_symbols:
        known_symbols = _collect_sizevar_symbols_by_name()
        replacements = {
            sym: known_symbols[sym.name]
            for sym in expr.free_symbols
            if sym.name in known_symbols
        }
        if replacements:
            expr = expr.xreplace(replacements)
    
    # Step 2: Try size_hint()
    try:
        return int(sizevars.size_hint(expr))
    except Exception:
        pass
    
    # Step 3: Direct evaluation if no free symbols
    if not expr.free_symbols:
        return int(expr)
    
    # Step 4: Symbol substitution with hints
    symbol_hints: dict[sympy.Symbol, int] = {}
    for sym in expr.free_symbols:
        try:
            symbol_hints[sym] = int(sizevars.size_hint(sym))
        except Exception:
            continue
    
    if symbol_hints:
        reduced = expr.subs(symbol_hints)
        if not reduced.free_symbols:
            return int(reduced)
    
    raise TypeError(f"Cannot concretize symbolic iteration dimension: {expr}")


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    kernel_output_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
    return kernel_output_dir


class SpyreAsyncCompile:
    def __init__(self) -> None:
        pass

    def sdsc(self, kernel_name: str, specs: list[Union[OpSpec | UnimplementedOp]]):
        # 1. Generate SDSC.json for each OpSpec
        sdscs = []
        arg_mappings = []
        for ks in specs:
            if isinstance(ks, UnimplementedOp):
                print(f"WARNING: Compiling unimplemented {ks.op} to runtime exception")
                return SpyreUnimplementedRunner(kernel_name, ks.op)

            inputs = []
            outputs = []
            arg_map = []
            for index, ts in enumerate(ks.args):
                # use node seq (idx in nodes) to verify whether to reuse lx for this buffer,
                # in case same Op used twice in sequence and only want pin 1 of them
                lx_addr = None
                for k, addr in getattr(ts, "allocation", {}).items():
                    if kernel_name.split("_")[-1] == k.replace("lx:", ""):
                        lx_addr = addr

                if ts.is_input:
                    inputs.append(
                        {
                            "name": _argument_names[index],
                            "it_dim_map": ts.it_dim_map,
                            "device_layout": ts.device_layout,
                            "lx_addr": lx_addr,
                        }
                    )
                    arg_map.append(ts.arg_index)
                else:
                    outputs.append(
                        {
                            "name": _argument_names[index],
                            "it_dim_map": ts.it_dim_map,
                            "device_layout": ts.device_layout,
                            "lx_addr": lx_addr,
                        }
                    )
                    arg_map.append(ts.arg_index)
            
            # Convert symbolic dimensions to concrete integers for JSON serialization
            concrete_dimensions = []
            for dim in ks.iteration_space:
                concrete_dimensions.append(_concretize_dimension(dim))
            
            kernel_descriptor = {
                "name": kernel_name,
                "reduction": ks.is_reduction,
                "op": ks.op,
                "dimensions": concrete_dimensions,
                "inputs": inputs,
                "outputs": outputs,
            }
            if ks.op_info is not None:
                kernel_descriptor["op_info"] = ks.op_info
            pointers = dict(zip(_argument_names, SEGMENT_OFFSETS))
            dt_sdsc = generate_sdsc(pointers, **kernel_descriptor)
            sdscs.append(dt_sdsc)
            arg_mappings.append(arg_map)

        # Write SDSCs to file system, invoke backend compiler, and return KernelRunner
        kernel_output_dir = get_output_dir(kernel_name)
        if _SDSC_BUNDLE:
            for idx, sdsc in enumerate(sdscs):
                with open(
                    os.path.join(kernel_output_dir, f"sdsc_{idx}.json"), "w"
                ) as file:
                    logger.info(f"Generating {file.name}")
                    json.dump(sdsc, file, indent=2)
            with open(os.path.join(kernel_output_dir, "bundle.mlir"), "w") as file:
                logger.info(f"Generating {file.name}")
                file.write("module {\n")
                file.write("\tfunc.func @sdsc_bundle() {\n")
                for i in range(len(sdscs)):
                    file.write(
                        '\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_'
                        + f"{i}"
                        + '.json"}\n'
                    )
                file.write("\t\treturn\n")
                file.write("\t}\n")
                file.write("}\n")

            subprocess.run(
                ["dxp_standalone", "--bundle", "-d", kernel_output_dir], check=True
            )
            convert_artifacts(kernel_output_dir)

            return SpyreSDSCKernelRunner(kernel_name, [kernel_output_dir], arg_mappings)
        else:
            # Process each SuperDSC separately
            sdsc_dirs = []
            for sdsc in enumerate(sdscs):
                kernel_output_dir = get_output_dir(kernel_name)
                subdir = os.path.join(kernel_output_dir, "execute", kernel_name)
                os.makedirs(subdir, exist_ok=True)
                with open(os.path.join(subdir, "sdsc.json"), "w") as file:
                    logger.info(f"Generating {file.name}")
                    json.dump(dt_sdsc, file, indent=2)
                sdsc_dirs.append(kernel_output_dir)
                arg_mappings.append(arg_map)

            for dir in sdsc_dirs:
                subprocess.run(["dxp_standalone", "-d", dir], check=True)
                convert_artifacts(dir)

            return SpyreSDSCKernelRunner(kernel_name, sdsc_dirs, arg_mappings)

    def wait(self, scope: dict[str, Any]) -> None:
        pass
