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
count from it as (ub - lb) / step, and nothing is specialized per count.

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
    _loop_level,
    _scaled_level_strides,
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


# Where the runtime reads the dimension from: args[0].size(0).
SOURCE = (0, 0)


def _symbolic_loop(count=None, bounds=None, sources=None) -> LoopSpec:
    """A LoopSpec whose trip count is S // GRANULARITY."""
    sym = sympy.Symbol(SYM_NAME, positive=True, integer=True)
    return LoopSpec(
        count=FloorDiv(sym, GRANULARITY) if count is None else count,
        body=[_op_spec()],
        count_symbol_bounds=(
            {SYM_NAME: (MAX_VALUE, GRANULARITY)} if bounds is None else bounds
        ),
        count_symbol_sources=({SYM_NAME: SOURCE} if sources is None else sources),
    )


class TestDecomposeSymbolicCount(unittest.TestCase):
    """_decompose_symbolic_count must never guess."""

    def test_floordiv_symbol_by_integer(self):
        sym = sympy.Symbol(SYM_NAME, positive=True, integer=True)
        self.assertEqual(_decompose_symbolic_count(FloorDiv(sym, 64)), (SYM_NAME, 64))

    def test_round_tripped_floor_form_decomposes_the_same(self):
        """The generated source IS the reload path, and it loses the FloorDiv type.

        Op specs are serialized as ``sympify('<str>')``. ``str(FloorDiv(s, 64))``
        is ``(s//64)``, and sympy parses ``//`` back into ``sympy.floor(s/64)``
        rather than torch's FloorDiv. Both spellings mean the same thing for a
        positive integer, and both must decompose identically, or a kernel dies
        in the bundle with "not a recognized trip-count shape" the moment its
        specs come back through the generated source. This is the fourth place
        that reload path has bitten this feature.
        """
        sym = sympy.Symbol(SYM_NAME, positive=True, integer=True)
        direct = FloorDiv(sym, GRANULARITY)
        round_tripped = sympy.sympify(str(direct))
        self.assertEqual(
            _decompose_symbolic_count(round_tripped),
            (SYM_NAME, GRANULARITY),
            msg=f"round-tripped {round_tripped!r} (type "
            f"{type(round_tripped).__name__}) did not decompose",
        )
        self.assertEqual(
            _decompose_symbolic_count(round_tripped),
            _decompose_symbolic_count(direct),
        )

    def test_bare_symbol_is_divisor_one(self):
        sym = sympy.Symbol(SYM_NAME, positive=True, integer=True)
        self.assertEqual(_decompose_symbolic_count(sym), (SYM_NAME, 1))

    def test_unrecognized_shape_returns_none(self):
        """Two symbols is not a shape we can emit, so it must decline."""
        a = sympy.Symbol("s0", positive=True, integer=True)
        b = sympy.Symbol("s1", positive=True, integer=True)
        self.assertIsNone(_decompose_symbolic_count(a * b))


class TestLoopLevelPlan(unittest.TestCase):
    """How one scf.for level is emitted, and what it does to that level's strides."""

    def test_concrete_count_is_a_constant_bound_stepping_by_one(self):
        level = _loop_level(sympy.Integer(8), 0, {})
        self.assertEqual(level.setup, ("%loop_bound_0 = arith.constant 8 : index",))
        self.assertEqual(level.bound, "%loop_bound_0")
        self.assertEqual(level.step, "%c1")
        self.assertEqual(level.stride_scale, 1)

    def test_symbolic_count_steps_by_granularity_over_the_dimension(self):
        """The accepted shape. An authored ceildivsi is on the backend reject list."""
        sym = sympy.Symbol(SYM_NAME, positive=True, integer=True)
        level = _loop_level(FloorDiv(sym, GRANULARITY), 0, {SYM_NAME: "%dim_s0"})
        self.assertEqual(level.setup, (f"%step_0 = arith.constant {GRANULARITY} : index",))
        self.assertEqual(level.bound, "%dim_s0", msg="bound must BE the dimension")
        self.assertEqual(level.step, "%step_0")
        self.assertEqual(
            level.stride_scale,
            GRANULARITY,
            msg="the loop var now counts elements, so strides must shrink by G",
        )
        self.assertNotIn("ceildivsi", " ".join(level.setup))

    def test_tile_size_one_needs_no_step_constant(self):
        sym = sympy.Symbol(SYM_NAME, positive=True, integer=True)
        level = _loop_level(sym, 0, {SYM_NAME: "%dim_s0"})
        self.assertEqual(level.setup, ())
        self.assertEqual(level.bound, "%dim_s0")
        self.assertEqual(level.step, "%c1")
        self.assertEqual(level.stride_scale, 1)

    def test_missing_input_arg_names_the_symbol(self):
        """An unbound symbol must fail loudly and say which one, for the pod log."""
        sym = sympy.Symbol("s9", positive=True, integer=True)
        with self.assertRaises(NotImplementedError) as ctx:
            _loop_level(FloorDiv(sym, 64), 0, {SYM_NAME: "%dim_s0"})
        self.assertIn("s9", str(ctx.exception))
        self.assertIn("count_symbol_bounds", str(ctx.exception))


class TestScaledLevelStrides(unittest.TestCase):
    """The affine stride scaling, which TWO call sites depend on agreeing.

    _collect_affine_maps builds the map index from this and _emit_specs looks it
    up. When only the building side scaled, emission died with a bare
    `KeyError: (8192,)` against a table keyed `(128,)`. Both now call this one
    function, so the divergence is structurally impossible, and these pin its
    behaviour.
    """

    def test_symbolic_level_shrinks_by_its_step(self):
        # 8192 elements per tile, loop steps by 64, so 128 per unit of loop var.
        self.assertEqual(
            list(_scaled_level_strides([{"s": 8192}], [GRANULARITY])), [(0, 128)]
        )

    def test_concrete_level_is_untouched(self):
        self.assertEqual(list(_scaled_level_strides([{"s": 8192}], [1])), [(0, 8192)])

    def test_missing_scale_defaults_to_one(self):
        """A level with no recorded scale must behave exactly as before."""
        self.assertEqual(list(_scaled_level_strides([{"s": 8192}], [])), [(0, 8192)])

    def test_empty_levels_are_skipped_but_keep_their_index(self):
        """The index is what aligns a stride to its loop variable."""
        self.assertEqual(
            list(_scaled_level_strides([{}, {"s": 8192}], [1, GRANULARITY])),
            [(1, 128)],
        )

    def test_non_divisible_stride_raises(self):
        """Silently mis-addressing is the worst outcome here, so refuse."""
        with self.assertRaises(AssertionError) as ctx:
            list(_scaled_level_strides([{"s": 100}], [GRANULARITY]))
        self.assertIn("not divisible", str(ctx.exception))


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
            self.symbol_kinds = generate_bundle(
                "test", self.output_dir, op_specs, pool_size=0
            )
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
        """The heart of it: the bound IS the argument, not a constant."""
        bundle = self._bundle_with_symbolic_loop()
        self.assert_in_bundle(f"%step_0 = arith.constant {GRANULARITY} : index", bundle)
        self.assertNotIn(
            "%loop_bound_0 = arith.constant",
            bundle,
            msg=f"loop bound was baked to a constant:\n{bundle}",
        )
        self.assertNotIn(
            "ceildivsi",
            bundle,
            msg=f"an authored divide is on the backend reject list:\n{bundle}",
        )

    def test_loop_steps_over_the_dimension(self):
        """bound=S step=G, so the device derives (ub - lb) / step itself."""
        bundle = self._bundle_with_symbolic_loop()
        self.assert_in_bundle(
            f"scf.for %i_0 = %c0 to %dim_{SYM_NAME} step %step_0", bundle
        )

    def test_execute_node_carries_no_symbols(self):
        """symbol_ids must stay empty or every dispatch pays program correction.

        This is the invariant the whole design exists to protect, and the one
        the earlier symbolic-address POC violated with 57 symbol ids.
        """
        bundle = self._bundle_with_symbolic_loop()
        self.assert_in_bundle('"symbol_ids"=[]', bundle)

    def test_symbol_kinds_carry_the_dispatch_binding(self):
        """The runtime builds its payload from this list, so the dimension
        must be in it, last, and must say which tensor dim to read."""
        self._bundle_with_symbolic_loop()
        loop_dims = [sk for sk in self.symbol_kinds if sk.is_loop_dimension]
        self.assertEqual(len(loop_dims), 1, msg=f"kinds were {self.symbol_kinds}")
        self.assertIs(
            self.symbol_kinds[-1],
            loop_dims[0],
            msg="loop dimensions are emitted last, so they must be appended last",
        )
        sk = loop_dims[0]
        self.assertEqual((sk.arg_index, sk.dim_index), SOURCE)
        self.assertEqual(sk.granularity, GRANULARITY)
        self.assertEqual(sk.max_value, MAX_VALUE)
        self.assertEqual(sk.pytorch_sym, SYM_NAME)
        self.assertFalse(
            sk.is_dimension,
            msg="must not look like an SDSC dimension symbol, async_compile "
            "refuses those",
        )

    def test_missing_source_refuses_to_emit(self):
        """A parameter nobody can bind is worse than a loud failure."""
        entry = (_sdsc_json(), [0], [], [])
        with self.assertRaises(NotImplementedError) as ctx:
            self._run([_symbolic_loop(sources={})], [entry])
        self.assertIn(SYM_NAME, str(ctx.exception))

    def test_concrete_loop_is_unchanged(self):
        """A non-symbolic loop must emit exactly as it did before."""
        entry = (_sdsc_json(), [0], [], [])
        loop = LoopSpec(count=sympy.Integer(4), body=[_op_spec()])
        bundle = self._run([loop], [entry])
        self.assert_in_bundle("%loop_bound_0 = arith.constant 4 : index", bundle)
        self.assertNotIn("ceildivsi", bundle)
        self.assertNotIn("input_arg<index, granularity=", bundle)


class TestLoopSpecPlumbing(unittest.TestCase):
    """The three places a new LoopSpec field has to be taught about.

    Every one of these is a real failure this POC shipped with. The bundle
    tests above all passed while a real compile could not get past the first
    of them, because they call generate_bundle directly and never go through
    provenance or the reload path. That is the exact false-green shape the
    design doc warns about, so these assert the plumbing rather than the
    output.
    """

    def test_schema_validation_accepts_the_new_field(self):
        """_validate_finalized_schema rejects any LoopSpec field it does not know.

        Without the matching _EXPECTED_LOOP_SPEC_SCHEMA entry this raises
        TypeError on EVERY compile, symbolic or not, because provenance is
        built for every kernel.
        """
        from torch_spyre._inductor.kernel_provenance import (
            build_kernel_provenance_descriptor,
        )

        descriptor = build_kernel_provenance_descriptor([_symbolic_loop()])
        self.assertTrue(descriptor.key)

    def test_concrete_loop_key_is_unchanged_by_the_new_field(self):
        """A concrete loop must hash exactly as before, or the cache is invalidated."""
        from torch_spyre._inductor.kernel_provenance import _canonical_spec

        canonical = _canonical_spec(LoopSpec(count=sympy.Integer(4), body=[_op_spec()]))
        self.assertNotIn(
            "count_symbol_bounds",
            canonical,
            msg=f"concrete loop canonical form grew a key: {canonical}",
        )

    def test_different_bounds_give_different_cache_keys(self):
        """Same count, different declared max, different bundle. Must not share a key.

        The bounds are not metadata: they become granularity= and max_value=
        on the input_arg, so a shared key would hand back a bundle built for
        the wrong maximum.
        """
        from torch_spyre._inductor.kernel_provenance import (
            build_kernel_provenance_descriptor,
        )

        key_512 = build_kernel_provenance_descriptor(
            [_symbolic_loop(bounds={SYM_NAME: (512, 64)})]
        ).key
        key_1024 = build_kernel_provenance_descriptor(
            [_symbolic_loop(bounds={SYM_NAME: (1024, 64)})]
        ).key
        self.assertNotEqual(key_512, key_1024)

    def test_generated_source_round_trips_the_bounds(self):
        """The generated wrapper IS the reload path.

        ShapeEnv is gone by then, so bounds that are not written into the
        source cannot be recovered, and the bundle fails with "no input_arg
        parameter" for a symbol it can see in the count.
        """
        from torch._inductor.utils import IndentedBuffer
        from torch_spyre._inductor.spyre_kernel import _codegen_op_spec_list

        buf = IndentedBuffer()
        _codegen_op_spec_list([_symbolic_loop()], buf, str)
        src = buf.getvalue()

        self.assertIn("count_symbol_bounds=", src, msg=src)
        self.assertIn(repr(SYM_NAME), src, msg=src)
        self.assertIn(str(MAX_VALUE), src, msg=src)
        self.assertIn(str(GRANULARITY), src, msg=src)

    def test_generated_source_omits_the_field_for_a_concrete_loop(self):
        """Keeps generated output byte-identical for every existing kernel."""
        from torch._inductor.utils import IndentedBuffer
        from torch_spyre._inductor.spyre_kernel import _codegen_op_spec_list

        buf = IndentedBuffer()
        _codegen_op_spec_list(
            [LoopSpec(count=sympy.Integer(4), body=[_op_spec()])], buf, str
        )
        self.assertNotIn("count_symbol_bounds", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
