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

"""A symbolic loop count through the whole Spyre pipeline, on device.

Every other test of this feature stops somewhere: at a helper, at a synthetic
cond graph, at ``generate_bundle`` with a mocked SDSC. This one compiles a
``for_each_tile`` region over a marked-dynamic dimension and runs it.

WHAT THIS CANNOT YET ASSERT, and the reason is a missing dependency rather
than a gap in the tests. The headline claim is one binary serving a range of
sizes. That needs the varying tensor's HBM buffer reserved for the declared
maximum, which is PR #4326 and is not in this tree: there is no
``spyre_empty_reserved`` and ``.to()`` takes no ``max=``. Without it a larger
size is a fresh, differently sized allocation, so the layout guard sees a new
layout and recompiles. That is safe, just not yet useful.

So these assert what is assertable alone: the region compiles through the real
pipeline, the symbolic count survives into the kernel, the emitted structure is
the loop form the design specifies, and the numbers are right at the size it
was compiled for. The multi-size no-recompile proof is the combined milestone
with PR #4326 and belongs in that test.

The range is declared with ``torch._check`` inside the traced function. That is
also temporary: once the contract reader lands the transfer call carries it and
these three lines go away.
"""

import unittest
from functools import wraps

import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch._inductor.utils import run_and_get_code
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME

ROWS = 256
COLS = 64
TILE = 64
MIN_ROWS = 64
MAX_ROWS = 512

ATOL = 1e-2
RTOL = 1e-2


def _with_dynamo_reset(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        torch._dynamo.reset()
        return fn(*args, **kwargs)

    return wrapper


def tiled_gelu(x):
    """One tiled pointwise region over a dimension declared dynamic.

    The three ``torch._check`` calls are the contract: a range so the ShapeEnv
    has a finite ceiling for the geometry, and the divisibility so the tile
    count is exact. ``for_each_tile`` re-states the divisibility itself, but
    the range has to come from here because nothing else puts it in scope.
    """
    rows = x.shape[0]
    torch._check(rows >= MIN_ROWS)
    torch._check(rows <= MAX_ROWS)
    torch._check(rows % TILE == 0)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, torch.nn.functional.gelu(tile)

    _carry, out = for_each_tile(
        body, (x,), dims=(0,), tile_size=TILE, out_dim=0
    )
    return out


class TestSymbolicLoopOnDevice(unittest.TestCase):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, value in list(vars(cls).items()):
            if name.startswith("test") and callable(value):
                setattr(cls, name, _with_dynamo_reset(value))

    @staticmethod
    def _compile_and_run(rows=ROWS):
        x = torch.randn(rows, COLS, dtype=torch.float16)
        reference = torch.nn.functional.gelu(x.float())

        on_device = x.to(DEVICE_NAME)
        torch._dynamo.mark_dynamic(on_device, 0)

        compiled = torch.compile(tiled_gelu, backend="inductor", fullgraph=True)
        out, code = run_and_get_code(compiled, on_device)
        return out, reference, code

    def test_the_numbers_are_right(self):
        out, reference, _code = self._compile_and_run()

        torch.testing.assert_close(
            out.cpu().float(), reference, atol=ATOL, rtol=RTOL
        )

    def test_the_count_reaches_the_kernel_as_a_symbol(self):
        """Not specialised to 256 somewhere along the way.

        If the count arrived concrete, every assertion about the loop form
        below would still pass while the feature was entirely absent, which is
        the false green this whole area is prone to.
        """
        _out, _reference, code = self._compile_and_run()
        source = "\n".join(code)

        self.assertIn("LoopSpec(", source)
        self.assertIn("count_symbol_bounds", source)
        self.assertNotIn(
            f"count=sympify('{ROWS // TILE}')",
            source,
            "the trip count was specialised to this call's size, so the kernel "
            "is size-specific and nothing here is measuring a symbolic loop",
        )

    def test_the_declared_maximum_is_what_was_carried(self):
        """Not the warm-up size. This is the one that catches hint-sized
        geometry, which is correct at the compiled size and wrong above it."""
        _out, _reference, code = self._compile_and_run()
        source = "\n".join(code)

        self.assertIn(str(MAX_ROWS), source)
        self.assertIn(f"{TILE})", source)


class TestItIsStillCorrectAtAnotherSize(unittest.TestCase):
    """A second size, compiled separately.

    Deliberately NOT asserting one binary: without the reservation of PR #4326
    this recompiles, and asserting otherwise would be asserting a bug. What it
    does prove is that the geometry built from the declared maximum is correct
    at more than one actual size, which is the half of the claim that does not
    need the reservation.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, value in list(vars(cls).items()):
            if name.startswith("test") and callable(value):
                setattr(cls, name, _with_dynamo_reset(value))

    def test_a_smaller_size(self):
        out, reference, _code = TestSymbolicLoopOnDevice._compile_and_run(rows=128)

        torch.testing.assert_close(
            out.cpu().float(), reference, atol=ATOL, rtol=RTOL
        )

    def test_the_minimum_size(self):
        """One tile exactly, which is the known rough edge."""
        out, reference, _code = TestSymbolicLoopOnDevice._compile_and_run(
            rows=MIN_ROWS
        )

        torch.testing.assert_close(
            out.cpu().float(), reference, atol=ATOL, rtol=RTOL
        )


if __name__ == "__main__":
    unittest.main()
