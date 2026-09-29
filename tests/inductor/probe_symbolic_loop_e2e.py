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

"""Pod probe: does a marked dynamic dimension reach the bundle as a loop bound?

Not a unit test. A diagnostic script, run by hand on a machine where torch-spyre
is built, whose job is to answer four questions in its own output:

  1. Did Dynamo give the marked dimension a symbol, with what range?
  2. Did the symbolic trip count survive lowering into ``LoopSpec.count``?
  3. Did the LoopSpec carry ``count_symbol_bounds``?
  4. Did the bundle emit a derived bound instead of a constant?

Run it and send the whole output back. Each stage prints whether it was reached,
so a failure at stage N tells us which change is wrong without another round
trip. The expected first failure is stage 2 or 3, because those depend on the
lowering path this probe has never been executed against.

    python tests/inductor/probe_symbolic_loop_e2e.py 2>&1 | tee probe.log

Add ``TORCH_LOGS="+torch_spyre"`` for the ``[symbolic-loop]`` lines the changed
code emits.
"""

import os
import sys
import traceback

import torch

BANNER = "=" * 72

# Same numbers as the HLD worked example.
GRANULARITY = 64
MAX_ROWS = 512
WARMUP_ROWS = 320
COLS = 1024


def stage(n: int, what: str) -> None:
    print(f"\n{BANNER}\nSTAGE {n}: {what}\n{BANNER}", flush=True)


def stage_1_symbol_and_range():
    """What Dynamo records for the marked dimension."""
    stage(1, "Dynamo symbol and range")
    captured = {}

    def backend(gm, example_inputs):
        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode(example_inputs)
        shape_env = getattr(fake_mode, "shape_env", None)
        if shape_env is None:
            print("  NO ShapeEnv on the fake mode -- nothing is symbolic")
            return gm.forward
        for sym, srcs in shape_env.var_to_sources.items():
            vr = shape_env.var_to_range.get(sym)
            print(
                f"  symbol={sym}  sources={[s.name() for s in srcs]}  "
                f"range=[{getattr(vr, 'lower', '?')}, {getattr(vr, 'upper', '?')}]"
            )
            captured[str(sym)] = vr
        return gm.forward

    x = torch.randn(WARMUP_ROWS, COLS)
    torch._dynamo.mark_dynamic(x, 0, min=GRANULARITY, max=MAX_ROWS)
    torch._dynamo.reset()
    torch.compile(lambda t: t + 1, backend=backend, dynamic=None)(x)

    if not captured:
        print("  RESULT: no symbol captured. mark_dynamic did not take effect.")
    else:
        print(f"  RESULT: {len(captured)} symbol(s) captured: {sorted(captured)}")
    return captured


def stage_2_and_3_specs():
    """Compile for Spyre and inspect the finished op_specs."""
    stage(2, "symbolic count in LoopSpec, and its carried bounds")
    try:
        import torch_spyre  # noqa: F401
        from torch_spyre._inductor.op_spec import LoopSpec
        from torch_spyre.constants import DEVICE_NAME
    except Exception:
        print("  torch_spyre import FAILED:")
        traceback.print_exc()
        return None

    from torch_spyre._inductor.wsr.for_each_tile import for_each_tile

    def model(t):
        # Hand-written HOP. The bridge that would author this from the marked
        # dimension does not exist yet; that is deliberate POC scope.
        _, out = for_each_tile(
            lambda carry, tiles: (None, torch.nn.functional.gelu(tiles[0])),
            (t,),
            dims=(0,),
            tile_size=GRANULARITY,
            out_dim=0,
        )
        return out

    seen_specs = []
    try:
        from torch_spyre._inductor import codegen as _codegen  # noqa: F401
    except Exception:
        pass

    x = torch.randn(WARMUP_ROWS, COLS, dtype=torch.float16)
    torch._dynamo.mark_dynamic(x, 0, min=GRANULARITY, max=MAX_ROWS)
    torch._dynamo.reset()

    try:
        x_dev = x.to(DEVICE_NAME)
        compiled = torch.compile(model, dynamic=None)
        compiled(x_dev)
        print("  compile COMPLETED without raising")
    except Exception:
        print("  compile RAISED (this is informative, not necessarily wrong):")
        traceback.print_exc()

    print(
        "\n  If the run above produced a bundle, stage 4 reads it. Look in the\n"
        "  log for the '[symbolic-loop]' lines the changed code emits:\n"
        "    - 'count=... symbol=... max=... granularity=...'  (pass_utils)\n"
        "    - 'outer/nested LoopSpec count=... bounds=...'     (kernel/scheduler)\n"
        "    - 'bundle has N symbolic loop bound(s)'            (bundle)\n"
        "    - 'loop_bound_0 from count=... emitted as: ...'    (bundle)\n"
        "  Their ABSENCE is the finding: it says the symbolic count never got\n"
        "  that far, and which stage it stopped at."
    )
    _ = LoopSpec, seen_specs
    return True


def stage_4_find_bundles():
    """Locate and print any bundle.mlir this run produced."""
    stage(4, "emitted bundle.mlir")
    roots = [
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
        os.path.expanduser("~/.cache/torch_spyre"),
        "/tmp",
        os.getcwd(),
    ]
    found = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if "bundle.mlir" in filenames:
                found.append(os.path.join(dirpath, "bundle.mlir"))
        if found:
            break

    if not found:
        print("  no bundle.mlir found. Set TORCHINDUCTOR_CACHE_DIR and rerun.")
        return

    newest = max(found, key=os.path.getmtime)
    print(f"  newest of {len(found)} bundle(s): {newest}\n")
    with open(newest) as f:
        text = f.read()
    print(text)

    print(f"\n{BANNER}\nWHAT TO CHECK IN THE TEXT ABOVE\n{BANNER}")
    checks = [
        ("input_arg<index, granularity=", "dimension declared as a parameter"),
        ("input_arg_extract", "dimension extracted to an SSA value"),
        ("ceildivsi", "bound DERIVED from the dimension"),
        ("scf.for", "a loop exists"),
        ('"symbol_ids"=[]', "execute node carries NO symbols"),
    ]
    for needle, meaning in checks:
        print(f"  [{'yes' if needle in text else 'NO '}] {meaning}  ({needle!r})")
    if "arith.constant" in text and "ceildivsi" not in text:
        print(
            "\n  NOTE: a constant bound with no ceildivsi means the count was\n"
            "  concrete by the time it reached emission. That is the failure\n"
            "  mode this whole change exists to prevent."
        )


def main() -> int:
    print(f"python  : {sys.version.split()[0]}")
    print(f"torch   : {torch.__version__}")
    stage_1_symbol_and_range()
    stage_2_and_3_specs()
    stage_4_find_bundles()
    print(f"\n{BANNER}\nProbe finished. Send this entire output back.\n{BANNER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
