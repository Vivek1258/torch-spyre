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

    def test_granularity_is_not_recoverable_from_facts(self):
        """Several divisibility facts are true at once, so G cannot be inferred.

        With min=128 and G=64 declared, the ShapeEnv ends up holding
        Mod(s,64), Mod(s,2) and Mod(s, s//2). All three are true. A reader
        wanting "the" granularity has to choose, and choosing is a heuristic
        rather than a declaration. That is the whole argument for declaring it.
        """
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch._dynamo.source import LocalSource

        shape_env = ShapeEnv()
        src = LocalSource("x")
        sym = shape_env.create_symbol(320, src)
        s = shape_env.create_symintnode(sym, hint=320, source=src)

        torch._check(s >= 128)
        torch._check(s % 64 == 0)

        facts = {str(f) for f in shape_env.divisible}
        self.assertIn("Mod(s0, 64)", facts, f"declared fact missing: {facts}")
        # More than one true fact means a reader cannot pick unambiguously.
        self.assertGreater(
            len(facts),
            1,
            "if only one fact were ever present the registry would be "
            "unnecessary, so this test is the thing to re-check first if the "
            "design is ever questioned",
        )


if __name__ == "__main__":
    unittest.main()
