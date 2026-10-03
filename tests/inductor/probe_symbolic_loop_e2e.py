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

    SPYRE_INDUCTOR_LOG=1 SPYRE_INDUCTOR_LOG_LEVEL=INFO \
        python tests/inductor/probe_symbolic_loop_e2e.py 2>&1 | tee probe.log

Do NOT use ``TORCH_LOGS="+spyre.inductor"``. It breaks ``import torch`` itself
with ``ModuleNotFoundError: No module named 'spyre'``, because torch's parser
accepts only a registered log, an artifact, or an importable module. Measured.
torch_spyre's own deprecation message recommending it is wrong. The default
spyre log level is WARNING, so without SPYRE_INDUCTOR_LOG_LEVEL=INFO every
``[symbolic-loop]`` line is silently dropped.
"""

import os
import sys
import traceback

import torch

BANNER = "=" * 72

# Same numbers as the HLD worked example.
GRANULARITY = 64
MAX_ROWS = 512
# 320 rows is 5 tiles of 64 and sits inside [64, 512]. 128 cols is 2 sticks per
# row at fp16, matching the STICK_COLS the passing fixtures use.
ROWS = 320
COLS = 128
WARMUP_ROWS = ROWS


def stage(n: int, what: str) -> None:
    print(f"\n{BANNER}\nSTAGE {n}: {what}\n{BANNER}", flush=True)


def _source_name(src) -> str:
    """Name of a ShapeEnv source, whichever shape ``.name`` takes.

    It is a method on older PyTorch and a plain str property on 2.13, and
    calling the str is a TypeError that aborts the whole probe. A diagnostic
    script must never be the thing that fails.
    """
    name = getattr(src, "name", None)
    if callable(name):
        try:
            return str(name())
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            return f"<name() raised {exc!r}>"
    if name is not None:
        return str(name)
    return repr(src)


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
                f"  symbol={sym}  sources={[_source_name(s) for s in srcs]}  "
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


def _abs_tiled_fn(a, tile_size):
    """Byte-for-byte the shape of for_each_tile_fixtures.abs_tiled_fn.

    Deliberately identical to a fixture that already passes on device, so the
    only difference between the control and the experiment below is whether
    dim 0 is marked dynamic. Anything else that breaks is then the harness,
    not the feature.

    Single operand on purpose. Marking two operands gives two independent
    symbols and therefore two trip counts, which this design refuses until the
    bridge emits a torch._check equality. That is a real constraint, but it is
    not the one this probe is measuring, so it is kept out of the way.
    """
    from torch_spyre._inductor.wsr.for_each_tile import for_each_tile

    # NOTE: a torch._check here does NOT help, which was measured, not assumed.
    # The bad tile extent is created by the view inside for_each_tile's own
    # _xs_leaf, during scan's re-trace, which a guard added in this frame does
    # not reach. The assertion has to live at the view itself, so it is in
    # _xs_leaf now. Left as a comment so nobody re-tries it from here.

    def body(_, ops):
        (a_tile,) = ops
        return None, a_tile.abs()

    _, out = for_each_tile(body, (a,), dims=(0,), tile_size=tile_size, out_dim=0)
    return out


def _compile_and_run(a_dev, label):
    """Compile and run one case, reporting the outcome rather than raising.

    ``fullgraph=True`` is NOT optional. Every for_each_tile test in this repo
    uses it, and for_each_tile's own docstring says a direct user HOP relies on
    fullgraph capture to permit the scalar read that Inductor's
    scan-to-while-loop pass performs. Without it the compile dies far away in
    post_grad with DataDependentOutputException on aten._local_scalar_dense,
    which looks like a backend bug and is not one.
    """
    torch._dynamo.reset()
    compiled = torch.compile(_abs_tiled_fn, backend="inductor", fullgraph=True)
    try:
        out = compiled(a_dev, GRANULARITY)
        print(f"  {label}: compile COMPLETED, output shape {tuple(out.shape)}")
        return out
    except Exception:
        print(f"  {label}: compile RAISED")
        traceback.print_exc()
        return None


def stage_2_and_3_specs():
    """Compile for Spyre: first a concrete control, then the symbolic case."""
    stage(2, "symbolic count in LoopSpec, and its carried bounds")
    try:
        import torch_spyre  # noqa: F401
        from torch_spyre.constants import DEVICE_NAME
    except Exception:
        print("  torch_spyre import FAILED:")
        traceback.print_exc()
        return None

    a = torch.randn(ROWS, COLS, dtype=torch.float16)
    ref = a.float().abs()

    # --- control: same kernel, concrete shape -------------------------------
    # If this fails, the symbolic result below means nothing.
    print("\n  [control] concrete shape, no mark_dynamic")
    out = _compile_and_run(a.to(DEVICE_NAME), "control")
    if out is not None:
        err = (out.cpu().float() - ref).abs().max().item()
        print(f"  control: max abs error vs CPU = {err:.4f}")

    # --- experiment: identical, but dim 0 is symbolic -----------------------
    # Marked AFTER .to(), because .to() returns a NEW tensor and mark_dynamic
    # records on the object. Marking the CPU tensor would silently lose it,
    # which is what the first version of this probe did.
    print("\n  [experiment] dim 0 marked dynamic")
    # Reserve the HBM buffer at MAX along dim 0 (torch-spyre#4326). That builds
    # the SpyreTensorLayout from the PADDED shape while dma_sizes stay at the
    # real shape, so the device geometry the SDSC describes covers the largest
    # trip count the bundle can reach. Without it the layout is hint-sized: the
    # loop count is max-based but the geometry is sized for this one call, and
    # phase zero can go green while being specialised to a single size.
    #
    # Falls back so this probe still runs on a tree without #4326, and says
    # which path it took, because the two give different device_size and that
    # changes how the rest of the output should be read.
    try:
        a_dev = a.to(DEVICE_NAME, max=MAX_ROWS)
        print(f"  HBM reserved at max={MAX_ROWS} (#4326 present)")
    except (TypeError, ValueError) as exc:
        a_dev = a.to(DEVICE_NAME)
        print(
            f"  NO max reservation, falling back to a plain .to() "
            f"({type(exc).__name__}: {exc}). #4326 is not applied, so "
            f"device_size will be hint-sized at {ROWS}, not {MAX_ROWS}."
        )
    try:
        print(f"  input device layout: {a_dev.device_tensor_layout()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  input device layout unavailable: {exc}")
    torch._dynamo.mark_dynamic(a_dev, 0, min=GRANULARITY, max=MAX_ROWS)
    out = _compile_and_run(a_dev, "experiment")
    if out is not None:
        err = (out.cpu().float() - ref).abs().max().item()
        print(f"  experiment: max abs error vs CPU = {err:.4f}")

    print(
        "\n  If the run above produced a bundle, stage 4 reads it. Look in the\n"
        "  log for the '[symbolic-loop]' lines the changed code emits:\n"
        "    - 'count=... symbol=... max=... granularity=...'  (pass_utils)\n"
        "    - 'outer/nested LoopSpec count=... bounds=...'     (kernel/scheduler)\n"
        "    - 'bundle has N symbolic loop bound(s)'            (bundle)\n"
        "    - 'loop_bound_0 from count=... emitted as: ...'    (bundle)\n"
        "  Their ABSENCE is the finding: it says the symbolic count never got\n"
        "  that far, and which stage it stopped at.\n"
        "\n"
        "  Read the control line first. If the control also failed, the harness\n"
        "  is wrong and the experiment tells you nothing."
    )
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
    # The accepted shape is `scf.for %i = %c0 to %dim_<sym> step %step_N`, NOT
    # an authored divide: arith.ceildivsi is on the backend's reject list, while
    # a runtime-valued bound is accepted, so the device derives the trip count
    # as (ub - lb) / step itself.
    checks = [
        ("input_arg<index, granularity=", "dimension declared as a parameter"),
        ("input_arg_extract", "dimension extracted to an SSA value"),
        ("to %dim_", "loop bound IS the dimension, not a constant"),
        ("step %step_", "loop steps by the granularity, so the device divides"),
        ("scf.for", "a loop exists"),
    ]
    for needle, meaning in checks:
        print(f"  [{'yes' if needle in text else 'NO '}] {meaning}  ({needle!r})")
    if "ceildivsi" in text:
        print(
            "\n  WRONG: an authored arith.ceildivsi is on the backend reject\n"
            "  list (dxp.cpp). The bound must be the dimension with step=G."
        )
    if "to %loop_bound_" in text and "to %dim_" not in text:
        print(
            "\n  NOTE: a constant bound means the count was concrete by the\n"
            "  time it reached emission. That is the failure mode this whole\n"
            "  change exists to prevent."
        )

    # symbol_ids is NOT expected to be empty. deeptools confirmed it still
    # carries symbolic ADDRESSES, and our own bundles show "symbol_ids"=[-1,-2].
    # An earlier version of this probe checked for an empty list, which always
    # reads as a failure and tells us nothing. The property that actually
    # matters is narrower: the VARYING DIMENSION must never reach an execute
    # node. Addresses are patched when an allocation changes, which is rare. A
    # value derived from the varying size would be patched on every call, which
    # is the ~795us per dispatch this whole design exists to avoid.
    execute_lines = [ln for ln in text.splitlines() if "sdsc_execute" in ln]
    leaked = [ln.strip() for ln in execute_lines if "%dim_" in ln]
    print(
        f"  [{'NO ' if leaked else 'yes'}] varying dimension stays OUT of every "
        f"sdsc_execute  ({len(execute_lines)} execute node(s))"
    )
    for ln in leaked:
        print(f"        LEAKED: {ln}")


def main() -> int:
    print(f"python  : {sys.version.split()[0]}")
    print(f"torch   : {torch.__version__}")
    # Each stage is isolated: an early stage blowing up must not hide the
    # later ones, because the later ones are the interesting part. A probe
    # that stops at the first traceback costs a whole round trip to the pod.
    for fn in (stage_1_symbol_and_range, stage_2_and_3_specs, stage_4_find_bundles):
        try:
            fn()
        except Exception:
            print(f"\n  STAGE FUNCTION {fn.__name__} RAISED, continuing anyway:")
            traceback.print_exc()
    print(f"\n{BANNER}\nProbe finished. Send this entire output back.\n{BANNER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
