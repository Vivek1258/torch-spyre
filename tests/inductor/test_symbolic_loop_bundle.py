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

"""Bundle emission for a SYMBOLIC loop bound.

The property under test: one binary serves a declared range. The varying
dimension arrives as an ``!sdscbundle.input_arg``, the device derives the trip
count from it with ``ceildivsi``, and nothing is specialized per count.

``compile_op_spec`` is mocked, so no Spyre hardware is needed. The specs are
hand-built here, which is deliberate: this file tests the EMISSION, and building
the specs by hand is what keeps that separable from the lowering that produces
them in a real compile.

Every assertion dumps the full bundle on failure. These tests are developed on a
machine that cannot build torch-spyre (s390x only), so a pod log has to be
diagnosable on its own without a debugger.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import sympy
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.utils._sympy.functions import FloorDiv

from torch_spyre._inductor.codegen.bundle import (
    _decompose_symbolic_count,
    _mlir_count_lines,
    generate_bundle,
)
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec

# Matches the worked example in the HLD: dim 0 varies between 64 and 512 with a
# granularity of 64, so the tile is 64 rows and the trip count is S // 64.
SYM_NAME = "s0"
GRANULARITY = 64
MAX_VALUE = 512


def _sdsc_json(sdsc_idx: int = 0, num_cores: int = 1) -> dict:
    """Minimal SDSC JSON with NO dimension symbol.

    The absence is the point. On this route the loop is explicit and the SDSC
    describes one static tile, so the tile has no symbolic dim and
    ``dimToSymbolMapping_`` stays empty. The varying dimension lives only at the
    bundle level.
    """
    return {
        f"{sdsc_idx}_fused_test": {
            "numCoresUsed_": num_cores,
            "dscs_": [
                {"op": {"dimToSymbolMapping_": {}, "scheduleTree_": []}},
            ],
        }
    }


def _op_spec() -> OpSpec:
    """Stub OpSpec; its content is irrelevant because compile_op_spec is mocked."""
    return OpSpec(
        op="gelu",
        is_reduction=False,
        iteration_space={},
        args=[],
        op_info={},
    )


def _symbolic_loop(count=None, bounds=None) -> LoopSpec:
    """A LoopSpec whose trip count is S // GRANULARITY."""
    sym = sympy.Symbol(SYM_NAME, positive=True, integer=True)
    return LoopSpec(
        count=FloorDiv(sym, GRANULARITY) if count is None else count,
        body=[_op_spec()],
        count_symbol_bounds=(
            {SYM_NAME: (MAX_VALUE, GRANULARITY)} if bounds is None else bounds
        ),
    )


class TestDecomposeSymbolicCount(unittest.TestCase):
    """_decompose_symbolic_count must never guess."""

    def test_floordiv_symbol_by_integer(self):
        sym = sympy.Symbol(SYM_NAME, positive=True, integer=True)
        self.assertEqual(
            _decompose_symbolic_count(FloorDiv(sym, 64)), (SYM_NAME, 64)
        )

    def test_bare_symbol_is_divisor_one(self):
        sym = sympy.Symbol(SYM_NAME, positive=True, integer=True)
        self.assertEqual(_decompose_symbolic_count(sym), (SYM_NAME, 1))

    def test_unrecognized_shape_returns_none(self):
        """Two symbols is not a shape we can emit, so it must decline."""
        a = sympy.Symbol("s0", positive=True, integer=True)
        b = sympy.Symbol("s1", positive=True, integer=True)
        self.assertIsNone(_decompose_symbolic_count(a * b))


class TestMlirCountLines(unittest.TestCase):
    """The lines that define %loop_bound_N."""

    def test_concrete_count_is_a_constant(self):
        self.assertEqual(
            _mlir_count_lines(sympy.Integer(8), 0, {}),
            ["%loop_bound_0 = arith.constant 8 : index"],
        )

    def test_symbolic_count_uses_ceildivsi(self):
        sym = sympy.Symbol(SYM_NAME, positive=True, integer=True)
        lines = _mlir_count_lines(
            FloorDiv(sym, GRANULARITY), 0, {SYM_NAME: "%dim_s0"}
        )
        self.assertEqual(
            lines,
            [
                f"%tile_0 = arith.constant {GRANULARITY} : index",
                "%loop_bound_0 = arith.ceildivsi %dim_s0, %tile_0 : index",
            ],
            msg=f"emitted lines were: {lines}",
        )

    def test_missing_input_arg_names_the_symbol(self):
        """An unbound symbol must fail loudly and say which one, for the pod log."""
        sym = sympy.Symbol("s9", positive=True, integer=True)
        with self.assertRaises(NotImplementedError) as ctx:
            _mlir_count_lines(FloorDiv(sym, 64), 0, {SYM_NAME: "%dim_s0"})
        self.assertIn("s9", str(ctx.exception))
        self.assertIn("count_symbol_bounds", str(ctx.exception))


class TestSymbolicLoopBundle(InductorTestCase):
    """End-to-end bundle text for a loop with a symbolic trip count."""

    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()
        super().tearDown()

    def _run(self, op_specs, compiled_entries):
        # compile_op_spec is called twice per OpSpec: a probe for the cache key,
        # then the real one. Mirrors test_symbolic_dim_bundle.py.
        side_effects = [e for entry in compiled_entries for e in (entry, entry)]
        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=side_effects,
        ):
            generate_bundle("test", self.output_dir, op_specs, pool_size=0)
        with open(os.path.join(self.output_dir, "bundle.mlir")) as f:
            return f.read()

    def _bundle_with_symbolic_loop(self) -> str:
        entry = (_sdsc_json(), [0], [], [])
        return self._run([_symbolic_loop()], [entry])

    def assert_in_bundle(self, needle: str, bundle: str):
        self.assertIn(
            needle,
            bundle,
            msg=f"\nexpected to find:\n  {needle}\nin bundle.mlir:\n{bundle}",
        )

    def test_dimension_is_a_function_parameter(self):
        """The varying dimension enters as an input_arg carrying its bounds."""
        bundle = self._bundle_with_symbolic_loop()
        self.assert_in_bundle(
            f"%dim_{SYM_NAME}_base: !sdscbundle.input_arg"
            f"<index, granularity={GRANULARITY}, max_value={MAX_VALUE}>",
            bundle,
        )

    def test_dimension_is_extracted(self):
        bundle = self._bundle_with_symbolic_loop()
        self.assert_in_bundle(
            f"%dim_{SYM_NAME} = sdscbundle.input_arg_extract value from"
            f" %dim_{SYM_NAME}_base",
            bundle,
        )

    def test_bound_is_derived_not_baked(self):
        """The heart of it: the count comes from the argument, not a constant."""
        bundle = self._bundle_with_symbolic_loop()
        self.assert_in_bundle(
            f"%loop_bound_0 = arith.ceildivsi %dim_{SYM_NAME}, %tile_0 : index",
            bundle,
        )
        self.assertNotIn(
            "%loop_bound_0 = arith.constant",
            bundle,
            msg=f"loop bound was baked to a constant:\n{bundle}",
        )

    def test_loop_uses_the_derived_bound(self):
        bundle = self._bundle_with_symbolic_loop()
        self.assert_in_bundle(
            "scf.for %i_0 = %c0 to %loop_bound_0 step %c1", bundle
        )

    def test_execute_node_carries_no_symbols(self):
        """symbol_ids must stay empty or every dispatch pays program correction.

        This is the invariant the whole design exists to protect, and the one
        the earlier symbolic-address POC violated with 57 symbol ids.
        """
        bundle = self._bundle_with_symbolic_loop()
        self.assert_in_bundle('"symbol_ids"=[]', bundle)

    def test_concrete_loop_is_unchanged(self):
        """A non-symbolic loop must emit exactly as it did before."""
        entry = (_sdsc_json(), [0], [], [])
        loop = LoopSpec(count=sympy.Integer(4), body=[_op_spec()])
        bundle = self._run([loop], [entry])
        self.assert_in_bundle("%loop_bound_0 = arith.constant 4 : index", bundle)
        self.assertNotIn("ceildivsi", bundle)
        self.assertNotIn("input_arg<index, granularity=", bundle)


if __name__ == "__main__":
    unittest.main()
