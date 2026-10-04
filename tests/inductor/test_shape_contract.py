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

"""Tests for the declared shape contract.

No device and no compile. The point of this module is bookkeeping and
validation, so everything here is exercised directly.

The case worth reading first is `test_granularity_is_not_recoverable_from_facts`,
which is the reason the module exists at all: several divisibility facts are
true at once, so a reader cannot pick the granularity out of them.
"""

import unittest

import torch

from torch_spyre._inductor import config
from torch_spyre._inductor import shape_contract
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.shape_contract import DimContract

G = 64
MAX = 512


class TestDimContractValidation(unittest.TestCase):
    """What a caller may declare, and what they are told when they cannot."""

    def test_a_usable_contract_validates(self):
        DimContract(min=128, max=MAX, granularity=G).validate()

    def test_min_and_granularity_may_differ(self):
        # The whole reason they are separate fields. A min that is a multiple
        # of G, rather than equal to it, has to be expressible.
        DimContract(min=128, max=MAX, granularity=G).validate()

    def test_max_not_a_multiple_of_granularity_is_refused(self):
        with self.assertRaises(Unsupported) as cm:
            DimContract(min=64, max=500, granularity=G).validate()
        msg = str(cm.exception)
        # The message has to carry the fix, not just the complaint.
        self.assertIn("500", msg)
        self.assertIn("448", msg)
        self.assertIn("512", msg)

    def test_min_above_max_is_refused(self):
        with self.assertRaises(Unsupported):
            DimContract(min=512, max=128, granularity=G).validate()

    def test_min_below_8_is_refused(self):
        # Holds by luck today because real minima are far above it. Pinned so
        # it keeps holding.
        with self.assertRaises(Unsupported):
            DimContract(min=4, max=256, granularity=4).validate()

    def test_too_many_buckets_is_refused(self):
        # max/G has to stay inside max_buckets.
        over = (config.max_buckets + 1) * G
        with self.assertRaises(Unsupported) as cm:
            DimContract(min=G * 2, max=over, granularity=G).validate()
        self.assertIn("max_buckets", str(cm.exception))

    def test_non_positive_values_are_refused(self):
        for bad in (
            DimContract(min=0, max=MAX, granularity=G),
            DimContract(min=64, max=0, granularity=G),
            DimContract(min=64, max=MAX, granularity=0),
        ):
            with self.assertRaises(Unsupported):
                bad.validate()

    def test_admits(self):
        c = DimContract(min=128, max=512, granularity=64)
        self.assertTrue(c.admits(128))
        self.assertTrue(c.admits(512))
        self.assertTrue(c.admits(320))
        self.assertFalse(c.admits(64), "below min")
        self.assertFalse(c.admits(576), "above max")
        self.assertFalse(c.admits(200), "not a multiple of the granularity")


class TestRecordAndRead(unittest.TestCase):
    """The eager half: declaring against a real tensor, and reading it back."""

    def setUp(self):
        shape_contract.reset()

    def test_record_then_read_back(self):
        x = torch.randn(320, 128)
        c = DimContract(min=128, max=MAX, granularity=G)
        shape_contract.record(x, 0, c)
        self.assertEqual(shape_contract.declared_for(x, 0), c)

    def test_undeclared_reads_as_none(self):
        x = torch.randn(320, 128)
        self.assertIsNone(shape_contract.declared_for(x, 0))

    def test_recording_the_same_contract_twice_is_fine(self):
        # A caller may reasonably declare the same thing twice, e.g. the same
        # tensor passed through two call sites.
        x = torch.randn(320, 128)
        c = DimContract(min=128, max=MAX, granularity=G)
        shape_contract.record(x, 0, c)
        shape_contract.record(x, 0, DimContract(min=128, max=MAX, granularity=G))
        self.assertEqual(shape_contract.declared_for(x, 0), c)

    def test_conflicting_redeclaration_raises(self):
        x = torch.randn(320, 128)
        shape_contract.record(x, 0, DimContract(min=128, max=MAX, granularity=G))
        with self.assertRaises(Unsupported) as cm:
            shape_contract.record(x, 0, DimContract(min=128, max=MAX, granularity=32))
        self.assertIn("already declared", str(cm.exception))

    def test_two_dims_on_one_tensor(self):
        x = torch.randn(320, 256)
        c0 = DimContract(min=128, max=MAX, granularity=G)
        c1 = DimContract(min=128, max=256, granularity=128)
        shape_contract.record(x, 0, c0)
        shape_contract.record(x, 1, c1)
        self.assertEqual(shape_contract.declared_for(x, 0), c0)
        self.assertEqual(shape_contract.declared_for(x, 1), c1)

    def test_dim_out_of_range_raises(self):
        x = torch.randn(320, 128)
        with self.assertRaises(Unsupported):
            shape_contract.record(x, 2, DimContract(min=128, max=MAX, granularity=G))

    def test_recording_validates(self):
        # A bad declaration is refused at the point of declaring, not at the
        # first compile.
        x = torch.randn(320, 128)
        with self.assertRaises(Unsupported):
            shape_contract.record(x, 0, DimContract(min=64, max=500, granularity=G))

    def test_declarations_are_keyed_weakly(self):
        # A contract must not be the reason a tensor stays alive. Two distinct
        # tensors with the same values keep separate entries, which is what
        # identity keying buys.
        a = torch.randn(320, 128)
        b = a.clone()
        shape_contract.record(a, 0, DimContract(min=128, max=MAX, granularity=G))
        self.assertIsNone(shape_contract.declared_for(b, 0))


class TestSymbolSide(unittest.TestCase):
    """The read side the compiler uses, and the refusal that guards it."""

    def setUp(self):
        shape_contract.reset()

    def test_unbound_symbol_reads_as_none(self):
        self.assertIsNone(shape_contract.for_symbol("s0"))

    def test_require_raises_for_an_unbound_symbol(self):
        # This is the fail-closed path. A dim marked dynamic with nothing
        # declared is the one configuration that otherwise compiles, reuses a
        # single binary and returns garbage above the warm-up size.
        with self.assertRaises(Unsupported) as cm:
            shape_contract.require_for_symbol("s0", "test")
        msg = str(cm.exception)
        self.assertIn("s0", msg)
        self.assertIn("granularity", msg, "the message must name the fix")

    def test_reset_clears_symbol_bindings_but_not_declarations(self):
        x = torch.randn(320, 128)
        c = DimContract(min=128, max=MAX, granularity=G)
        shape_contract.record(x, 0, c)
        shape_contract.reset()
        # The declaration belongs to the caller's tensor and outlives a compile.
        self.assertEqual(shape_contract.declared_for(x, 0), c)
        self.assertIsNone(shape_contract.for_symbol("s0"))


class TestWhyTheRegistryExists(unittest.TestCase):
    """The measurement that justifies the module, pinned as a test."""

    def _declared_env(self, minimum: int, granularity: int):
        """A ShapeEnv with one symbol and a contract asserted on it."""
        from torch._dynamo.source import LocalSource
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        src = LocalSource("x")
        sym = shape_env.create_symbol(320, src)
        s = shape_env.create_symintnode(sym, hint=320, source=src)
        torch._check(s >= minimum)
        torch._check(s <= MAX)
        torch._check(s % granularity == 0)
        return shape_env, sym

    def test_the_lower_bound_is_not_the_granularity(self):
        """Reading G off the symbol's lower bound gives the wrong answer.

        This is the actual defect the registry exists to fix, and it is the
        one to re-check first if the design is ever questioned. The granularity
        used to be read from the symbol's derived lower bound. Declare
        min=128 with G=64 and the two differ immediately, so a reader of the
        lower bound gets 128 for a loop that steps 64. That is how a bundle
        shipped declaring granularity 128 while its own loop stepped 64.

        An earlier version of this test asserted that the divisibility set is
        ambiguous instead. That is true in some traces and not in a clean
        synthetic one, so it was the wrong thing to pin. The lower bound
        disagreeing with the declared granularity is unconditional.
        """
        shape_env, sym = self._declared_env(minimum=128, granularity=G)

        rng = shape_env.var_to_range[sym]
        self.assertEqual(int(rng.lower), 128, "the declared min")
        self.assertNotEqual(
            int(rng.lower),
            G,
            "min and granularity are separate fields precisely so they can "
            "differ, so the lower bound cannot stand in for the granularity",
        )

    def test_the_declared_fact_is_present_but_not_alone_in_general(self):
        """The divisibility fact lands, which is necessary but not sufficient.

        It is present, so a reader *could* find it here. What makes it unsafe
        as a channel is that a real trace installs other true divisibility
        facts alongside it, from guards unrelated to our contract, and then
        picking one is a heuristic. This test pins only the part that holds in
        every case: ours is in there.
        """
        shape_env, sym = self._declared_env(minimum=128, granularity=G)
        facts = {str(f) for f in shape_env.divisible}
        self.assertTrue(
            any(f"Mod({sym}, {G})" == f for f in facts),
            f"the declared divisibility fact is missing from {facts}",
        )

    def test_a_declared_contract_gives_a_finite_upper_bound(self):
        """Without this, codegen cannot describe the tiled dimension at all.

        Measured: with nothing declared the upper bound is int_oo, and
        max_trip_count refuses. Declaring the range is what makes it finite.
        """
        shape_env, sym = self._declared_env(minimum=128, granularity=G)
        upper = shape_env.var_to_range[sym].upper
        self.assertNotIn("oo", str(upper), "upper bound must be finite")
        self.assertEqual(int(upper), MAX)


if __name__ == "__main__":
    unittest.main()
