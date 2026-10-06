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

"""ONE BINARY ACROSS FIVE SIZES. The combined milestone, and nothing else proves it.

Needs both halves in the tree. The symbolic loop gives the trip count somewhere
to vary. The max-strided reservation of PR #4326 gives every size the same
allocation and the same layout, so the recompile guard does not fire. Neither
half produces this result alone: without the reservation ``test_symbolic_loop_e2e``
recompiles per size, which is safe and useless, and without the loop there is
nothing for a reused binary to do differently.

HOW NO-RECOMPILE IS MEASURED, because this is the project's headline claim and a
bundle counter that could fabricate it has already shipped once. Two independent
instruments, which have to agree:

1. ``kernel_cache.get_kernel_registry()``. One cache key with one miss and four
   hits is the claim. A second key means a second binary, whatever the timings
   say.
2. ``kernel_cache.get_cache_stats()["total_cached_kernels"]``, the committed
   kernel directories on disk. Must not grow after the warm-up.

If those two disagree, the harness is wrong and not the system. Say so rather
than picking the friendlier one.

THE WARM-UP SIZE IS 320 ON PURPOSE. Geometry built from a 320-row warm-up
instead of the declared 512 was measured correct at 128 and 256 and wrong at 448
and 512, relative error 3.96 and 4.37. Warming up at 320 and then running 448
and 512 is exactly that failure, so this ordering is the regression test for it.

DO NOT ``rm -rf`` THE CACHE ROOT WHEN RUNNING THIS. ``TORCHINDUCTOR_CACHE_DIR``
is the pinned root both instruments read, and deleting it between sizes reports
a real recompile as a reuse. ``mkdir -p`` it and leave it alone.
"""

import unittest
from functools import wraps

import regex as re
import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
import torch_spyre._C
from torch._inductor.utils import run_and_get_code
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME
from torch_spyre.execution.kernel_cache import get_cache_stats, get_kernel_registry

COLS = 64
TILE = 64
MIN_ROWS = 64
MAX_ROWS = 512

WARMUP_ROWS = 320
LATER_ROWS = (128, 256, 448, 512)

ATOL = 1e-2
RTOL = 1e-2

CONTRACT = {0: {"min": MIN_ROWS, "max": MAX_ROWS, "granularity": TILE}}


def tiled_gelu(x):
    """The same region as the standalone e2e test, declared the same way.

    The three checks stay even though ``.to(dynamic=...)`` now carries the
    contract. PR #4326 marks the dim bare on purpose: ``min=``/``max=`` on
    ``mark_dynamic`` installs a ``StrictMinMaxConstraint``, and the
    ``size % granularity`` check added later then reads as breaking that
    promise and raises ``ConstraintViolationError`` instead of an ordinary
    guard miss. So the range is declared from in here.
    """
    rows = x.shape[0]
    torch._check(rows >= MIN_ROWS)
    torch._check(rows <= MAX_ROWS)
    torch._check(rows % TILE == 0)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, torch.nn.functional.gelu(tile)

    _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=TILE, out_dim=0)
    return out


def _to_device(rows):
    x = torch.randn(rows, COLS, dtype=torch.float16)
    return x, x.to(DEVICE_NAME, dynamic=CONTRACT)


def _device_sizes(source):
    return re.findall(r"device_size=(\[[0-9, ]*\])", source)


class _Sweep:
    """What one multi-size run measured. Built once, asserted many times."""

    def __init__(self):
        self.source = ""
        self.keys_after_warmup = set()
        self.keys_at_end = set()
        self.cached_after_warmup = 0
        self.cached_at_end = 0
        self.entries_at_end = {}
        self.per_size = {}


def _run_sweep():
    sweep = _Sweep()
    registry = get_kernel_registry()

    # One reset, then every size goes through the SAME compiled callable. A
    # reset between sizes would re-trace and make reuse unmeasurable.
    torch._dynamo.reset()
    compiled = torch.compile(tiled_gelu, backend="inductor", fullgraph=True)

    host, dev = _to_device(WARMUP_ROWS)
    out, code = run_and_get_code(compiled, dev)
    sweep.source = "\n".join(code)
    sweep.per_size[WARMUP_ROWS] = _record(host, out)
    sweep.keys_after_warmup = set(registry.all_entries())
    sweep.cached_after_warmup = get_cache_stats()["total_cached_kernels"]

    for rows in LATER_ROWS:
        host, dev = _to_device(rows)
        sweep.per_size[rows] = _record(host, compiled(dev))

    sweep.keys_at_end = set(registry.all_entries())
    sweep.cached_at_end = get_cache_stats()["total_cached_kernels"]
    sweep.entries_at_end = registry.all_entries()
    return sweep


def _record(host, out):
    # get_device_size_in_bytes takes a layout or a (device_size, dtype) pair,
    # never a tensor, so the layout has to be fetched first.
    layout = torch_spyre._C.get_spyre_tensor_layout(out)
    return {
        "shape": tuple(out.shape),
        "bytes": torch_spyre._C.get_device_size_in_bytes(layout),
        "out": out.cpu().float(),
        "reference": torch.nn.functional.gelu(host.float()),
    }


def _with_dynamo_reset(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        torch._dynamo.reset()
        return fn(*args, **kwargs)

    return wrapper


class TestOneBinaryAcrossSizes(unittest.TestCase):
    """The sweep runs once in setUpClass. Each test names one property of it."""

    sweep: _Sweep

    @classmethod
    def setUpClass(cls):
        cls.sweep = _run_sweep()

    def test_one_cache_key_served_every_size(self):
        """The headline. Measured against the registry, not timed."""
        sweep = self.sweep
        self.assertEqual(
            sweep.keys_at_end,
            sweep.keys_after_warmup,
            "a new kernel cache key appeared after the warm-up, so a later size "
            f"compiled its own binary. Added: "
            f"{sorted(sweep.keys_at_end - sweep.keys_after_warmup)}\n"
            f"{get_kernel_registry().summary()}",
        )

    def test_the_kernel_directory_count_did_not_grow(self):
        """The second, independent instrument. Must agree with the first."""
        sweep = self.sweep
        self.assertEqual(
            sweep.cached_at_end,
            sweep.cached_after_warmup,
            "committed kernel directories grew from "
            f"{sweep.cached_after_warmup} to {sweep.cached_at_end}, so something "
            "compiled. If the registry test passed and this one failed, the "
            "harness is wrong, not the system",
        )

    def test_the_reused_binary_recorded_hits(self):
        """A reused binary that is never hit is a binary nothing ran.

        Without this, a key count that stays flat because the second size
        failed before reaching the cache would read as reuse.
        """
        totals = [
            (key[:12], meta["hit_count"], meta["miss_count"])
            for key, meta in self.sweep.entries_at_end.items()
        ]
        hits = sum(h for _k, h, _m in totals)
        self.assertGreaterEqual(
            hits,
            len(LATER_ROWS),
            f"expected at least {len(LATER_ROWS)} cache hits, one per size after "
            f"the warm-up, got {hits}. Per key: {totals}",
        )

    def test_the_numbers_are_right_at_every_size(self):
        """A reused binary that is wrong is the failure mode this guards."""
        for rows, rec in sorted(self.sweep.per_size.items()):
            with self.subTest(rows=rows):
                torch.testing.assert_close(
                    rec["out"], rec["reference"], atol=ATOL, rtol=RTOL
                )

    def test_the_logical_shape_tracks_the_runtime_size(self):
        for rows, rec in sorted(self.sweep.per_size.items()):
            with self.subTest(rows=rows):
                self.assertEqual(rec["shape"], (rows, COLS))

    def test_device_size_does_not_track_the_runtime_size(self):
        """CHECK THIS FIRST IF ANYTHING ELSE LOOKS WRONG.

        A device_size that moves with the runtime size means the geometry came
        from the warm-up hint rather than the declared maximum, which is correct
        at the compiled size and wrong above it. That exact confusion has
        happened three times on this work, which is why it gets its own test
        with its own name.
        """
        sizes = _device_sizes(self.sweep.source)
        self.assertTrue(
            sizes,
            "no device_size appeared in the generated source, so this test is "
            "measuring nothing. The serializer's field name has probably moved",
        )
        self.assertNotIn(
            f"[{WARMUP_ROWS}, {COLS}]",
            sizes,
            "a device_size equals the warm-up shape exactly, so geometry was "
            f"built from the hint and not from max={MAX_ROWS}. device_sizes "
            f"found: {sizes}",
        )

    def test_the_output_allocation_is_the_same_at_every_size(self):
        """Pins whether outputs are reserved at max or allocated at S.

        We do not currently know which, and either answer is informative: the
        same byte count everywhere means the output was reserved too, and a
        byte count that tracks the size means only inputs were. It is asserted
        rather than printed so the answer cannot drift unnoticed.
        """
        by_size = {
            rows: rec["bytes"] for rows, rec in sorted(self.sweep.per_size.items())
        }
        self.assertEqual(
            len(set(by_size.values())),
            1,
            "the output allocation changes with the runtime size, so outputs "
            "are not reserved at the declared maximum. That is a finding and "
            f"not necessarily a bug, but record it before changing this test: "
            f"{by_size}",
        )

    def test_the_bundle_still_has_the_symbolic_loop_form(self):
        """The reuse claim is only interesting if the loop is still symbolic."""
        source = self.sweep.source
        self.assertIn("count_symbol_bounds", source)
        carried = re.search(
            r"count_symbol_bounds=\{'(s\d+)': \((\d+), (\d+)\)\}", source
        )
        self.assertIsNotNone(carried, f"nothing carried the range:\n{source[:2000]}")
        self.assertEqual(
            (int(carried.group(2)), int(carried.group(3))), (MAX_ROWS, TILE)
        )


class TestTheEdgesAreStillRefusedOrHandled(unittest.TestCase):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, value in list(vars(cls).items()):
            if name.startswith("test") and callable(value):
                setattr(cls, name, _with_dynamo_reset(value))

    def test_a_size_that_is_not_a_multiple_of_the_granularity_is_refused(self):
        """200 rows against granularity 64.

        Either end may refuse it and both are correct: the transfer checks
        today's real size against the granularity, and the compiled region
        re-states the divisibility with torch._check. The test asserts a loud
        failure and names which end produced it, because a silent wrong answer
        here is the one outcome that matters.
        """
        with self.assertRaises(Exception) as caught:
            _host, dev = _to_device(200)
            compiled = torch.compile(tiled_gelu, backend="inductor", fullgraph=True)
            compiled(dev).cpu()
        self.assertNotIsInstance(
            caught.exception,
            AssertionError,
            f"refused by a bare assert rather than a diagnosable error: "
            f"{caught.exception}",
        )

    def test_one_tile_exactly(self):
        """The known rough edge, at the declared minimum."""
        host, dev = _to_device(MIN_ROWS)
        compiled = torch.compile(tiled_gelu, backend="inductor", fullgraph=True)
        out = compiled(dev)
        torch.testing.assert_close(
            out.cpu().float(),
            torch.nn.functional.gelu(host.float()),
            atol=ATOL,
            rtol=RTOL,
        )

    def test_the_contract_is_readable_back_off_the_tensor(self):
        """N5's accessor, which A.1 does not use but the contract depends on.

        A.2's bridge reads the range from here instead of from torch._check, so
        a mismatch between what was declared and what comes back would surface
        as a wrong loop bound much later.
        """
        _host, dev = _to_device(WARMUP_ROWS)
        self.assertEqual(torch_spyre._C.get_reserved_dims(dev), CONTRACT)


if __name__ == "__main__":
    unittest.main()
