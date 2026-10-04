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

"""E1: a real nn.Module with WEIGHTS and a symbolic leading dim.

    python tests/inductor/e1_mlp_symbolic.py
    python tests/inductor/e1_mlp_symbolic.py linear_tiled   # one stage

Writes `e1_mlp_symbolic_results.json` next to itself, after every stage.

Everything measured so far has been a hand-written kernel that already calls
for_each_tile. This is the first rung with a real module, real nn.Linear
weights and torch.compile on an nn.Module rather than a function. So it asks
three things nothing before it could:

  1. Do the WEIGHTS stay static while the row dim is symbolic? A weight that
     gets marked dynamic would be a disaster: the geometry of a parameter must
     not depend on the batch.
  2. Does `dynamic=None` avoid marking the stick dim? The HLD says True would
     mark every dim including the stick, and a symbol on the stick breaks the
     static-address property.
  3. Do the ops a real module produces (addmm with a bias, relu, layernorm,
     a residual add) behave inside a tiled body, where previously only
     hand-written fixtures have been tried?

TWO TRACKS, and the difference between them IS the bridge.

  TRACK A, "plain": the module as anyone would write it, with NO tiling. The
  bridge that would wrap a region in for_each_tile DOES NOT EXIST yet, so
  nothing tiles the symbolic dim and the symbol lands directly in an op's
  iteration space. This is EXPECTED TO FAIL. It is run anyway, because where
  it fails first is precisely the list of work the bridge has to do, and that
  list is currently guesswork.

  TRACK B, "tiled": the same arithmetic hand-wrapped in for_each_tile with the
  weights as INVARIANT operands (`dims=(0, None, ...)`). This emulates what the
  bridge would generate, so it measures the END STATE. This is the track that
  must work.

Read the two together. Track A failing at op X and track B passing says "the
bridge must tile X", which is a design statement. Track A passing would be a
surprise worth more than track B.
"""

import argparse
import faulthandler
import json
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable

import torch

BANNER = "=" * 78
RESULTS_PATH = "e1_mlp_symbolic_results.json"

# Set False (or pass --no-contract) to reproduce F062 deliberately: the
# reservation max does not reach the compiler, so every tiled stage dies on
# "no finite upper bound". Keep the reproduction available, because that gap is
# the main argument for the contract registry and it should stay measurable.
DECLARE_CONTRACT = True

FP = torch.float16
G = 64                       # G_internal, the tile
MAX_SIZE = 512
WARM = 320                   # 5 tiles: more than one, not the max, not a power of two
SIZES = (128, 256, 320, 448, 512)
D_IN = 128                   # 2 sticks at fp16
D_HID = 256
D_OUT = 64


# ---------------------------------------------------------------------------
# Reuse the matrix's measurement helpers rather than copying them.
#
# `find_bundles` there was fixed after it FABRICATED binary reuse: it re-chose
# its cache root per call and returned a count, so a real recompile reported as
# "same binary". Copying the helper here would mean copying the bug back, or
# forgetting to copy the fix. Importing it means E1 inherits both the fix and
# any future one.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import symbolic_experiment_matrix as mx  # noqa: E402


# ---------------------------------------------------------------------------
# the model, both tracks
# ---------------------------------------------------------------------------


class Mlp(torch.nn.Module):
    """The plain module. Track A compiles this directly.

    Deliberately ordinary: nn.Linear with a bias, so the graph carries addmm
    rather than mm, which is what a real model produces and what the fixtures
    have never exercised.
    """

    def __init__(self, with_norm=False, with_residual=False):
        super().__init__()
        self.fc1 = torch.nn.Linear(D_IN, D_HID, dtype=FP)
        self.fc2 = torch.nn.Linear(D_HID, D_OUT, dtype=FP)
        self.norm = torch.nn.LayerNorm(D_HID, dtype=FP) if with_norm else None
        self.with_residual = with_residual
        self.skip = torch.nn.Linear(D_IN, D_OUT, dtype=FP) if with_residual else None

    def forward(self, x):
        h = self.fc1(x)
        if self.norm is not None:
            h = self.norm(h)
        h = torch.relu(h)
        out = self.fc2(h)
        if self.with_residual:
            out = out + self.skip(x)
        return out


def declare_contract(x):
    """Assert the shape contract INSIDE the traced region.

    This was missing from the first version of E1, and the omission measured
    something important (F062): `.to("spyre", max=512)` reserves the buffer at
    512, but the compiler learns NOTHING from it. Every tiled stage failed with
    "symbolic loop count has no finite upper bound" while the audit in the same
    run showed device_size=[2, 512, 64], i.e. the buffer WAS max-reserved. Two
    views of one fact with no channel between them.

    So the range has to be re-stated here, inside the trace, where the symbol
    exists. That is exactly the job the contract registry should do for the
    user instead of making every caller remember it.

    torch._check and not mark_dynamic(min=, max=): the strict-constraint form
    refuses the divisibility narrowing, measured both on device and in E0 4b.
    """
    n = x.size(0)
    torch._check(n >= G)
    torch._check(n <= MAX_SIZE)
    torch._check(n % G == 0)
    return x


def _tiled(fn_body, operands, dims):
    from torch_spyre._inductor.wsr import for_each_tile  # noqa: PLC0415

    _, out = for_each_tile(fn_body, operands, dims=dims, tile_size=G, out_dim=0)
    return out


def tiled_linear(x, w1, b1):
    """Track B stage 1: one addmm, weight and bias INVARIANT."""

    def body(_, ops):
        x_t, w, b = ops
        return None, torch.addmm(b, x_t, w)

    return _tiled(body, (x, w1, b1), (0, None, None))


def tiled_linear_relu(x, w1, b1):
    def body(_, ops):
        x_t, w, b = ops
        return None, torch.relu(torch.addmm(b, x_t, w))

    return _tiled(body, (x, w1, b1), (0, None, None))


def tiled_mlp(x, w1, b1, w2, b2):
    """Two matmuls and a pointwise, all inside ONE tiled body.

    The intermediate h never leaves the body, so it is tile-sized and the
    symbol stays in the trip count. That is the shape the bridge should
    produce for a whole region, as opposed to one loop per op.
    """

    def body(_, ops):
        x_t, wa, ba, wb, bb = ops
        h = torch.relu(torch.addmm(ba, x_t, wa))
        return None, torch.addmm(bb, h, wb)

    return _tiled(body, (x, w1, b1, w2, b2), (0, None, None, None, None))


def tiled_mlp_norm(x, w1, b1, w2, b2, nw, nb):
    """Adds a LayerNorm, i.e. a REDUCTION inside the body over the static
    feature dim. softmax_row proved the shape in a fixture; this is the same
    thing produced by a real nn.LayerNorm."""

    def body(_, ops):
        x_t, wa, ba, wb, bb, g, beta = ops
        h = torch.addmm(ba, x_t, wa)
        h = torch.nn.functional.layer_norm(h, (D_HID,), g, beta, 1e-5)
        return None, torch.addmm(bb, torch.relu(h), wb)

    return _tiled(
        body, (x, w1, b1, w2, b2, nw, nb), (0, None, None, None, None, None, None)
    )


def tiled_mlp_norm_noaffine(x, w1, b1, w2, b2):
    """LayerNorm with NO affine parameters: the reduction chain alone.

    mlp_tiled passes and mlp_norm_tiled does not, and they differ by exactly
    two things: the mean/var/rsqrt chain layer_norm decomposes to, and the two
    1-D affine operands (weight and bias of length D_HID) broadcast against a
    2-D tile. This stage removes the second, so if it PASSES the problem is the
    1-D broadcast, and if it FAILS the problem is the decomposition itself.
    """

    def body(_, ops):
        x_t, wa, ba, wb, bb = ops
        h = torch.addmm(ba, x_t, wa)
        h = torch.nn.functional.layer_norm(h, (D_HID,), None, None, 1e-5)
        return None, torch.addmm(bb, torch.relu(h), wb)

    return _tiled(body, (x, w1, b1, w2, b2), (0, None, None, None, None))


def tiled_mlp_norm_manual(x, w1, b1, w2, b2, nw, nb):
    """The same normalisation written out by hand, affine included.

    If the hand-written form passes where nn.functional.layer_norm fails, the
    problem is in the DECOMPOSITION rather than in the mathematics or the 1-D
    operands, which would make it a lowering fix rather than a design limit.
    """

    def body(_, ops):
        x_t, wa, ba, wb, bb, g, beta = ops
        h = torch.addmm(ba, x_t, wa)
        mean = h.mean(dim=-1, keepdim=True)
        var = (h - mean).pow(2).mean(dim=-1, keepdim=True)
        h = (h - mean) * torch.rsqrt(var + 1e-5) * g + beta
        return None, torch.addmm(bb, torch.relu(h), wb)

    return _tiled(
        body, (x, w1, b1, w2, b2, nw, nb), (0, None, None, None, None, None, None)
    )


def tiled_mlp_residual(x, w1, b1, w2, b2, sw, sb):
    """A skip connection. Both branches read the SAME tile, so the symbolic dim
    appears twice in the body without a second symbol. Tests that a diamond
    inside the body is fine, which every residual block needs."""

    def body(_, ops):
        x_t, wa, ba, wb, bb, ws, bs = ops
        h = torch.relu(torch.addmm(ba, x_t, wa))
        return None, torch.addmm(bb, h, wb) + torch.addmm(bs, x_t, ws)

    return _tiled(
        body, (x, w1, b1, w2, b2, sw, sb), (0, None, None, None, None, None, None)
    )


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    id: str
    track: str            # plain | tiled
    title: str
    question: str
    predict: str
    build: Callable       # (size) -> (args, cpu_ref)
    call: Callable        # (*args) -> out      (traced)
    tol: float = 5e-2
    notes: str = ""


def _rand(*shape):
    return torch.randn(*shape, dtype=FP)


def _weights(with_norm=False, with_residual=False):
    torch.manual_seed(0)
    return Mlp(with_norm=with_norm, with_residual=with_residual)


def _f32(model):
    """An fp32 CPU copy carrying the SAME weights."""
    m32 = _weights(model.norm is not None, model.with_residual).float()
    m32.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    return m32


# Each stage needs a reference for what IT computes, not for the whole module.
# Measured the hard way: comparing linear_tiled's [S, 256] against the full
# model's [S, 64] gave a shape mismatch, which the runner stored as err=None
# and the summary showed as a plain "fails". Two stages looked like backend
# failures when the harness was simply asking the wrong question.
def ref_linear(model, x):
    with torch.no_grad():
        return _f32(model).fc1(x.float())


def ref_linear_relu(model, x):
    with torch.no_grad():
        return torch.relu(_f32(model).fc1(x.float()))


def ref_norm_noaffine(model, x):
    """fp32 reference for the no-affine norm variant."""
    m = _f32(model)
    with torch.no_grad():
        h = m.fc1(x.float())
        h = torch.nn.functional.layer_norm(h, (D_HID,), None, None, 1e-5)
        return m.fc2(torch.relu(h))


def ref_full(model, x):
    with torch.no_grad():
        return _f32(model)(x.float())


_ref = ref_full  # the plain track runs the whole module


def build_plain(size, with_norm=False, with_residual=False):
    model = _weights(with_norm, with_residual)
    x = _rand(size, D_IN)
    return (model, x), _ref(model, x)


def build_tiled(size, parts, ref_fn=None, with_norm=False, with_residual=False):
    """`parts` names which parameters the tiled callable wants, in order."""
    model = _weights(with_norm, with_residual)
    x = _rand(size, D_IN)
    sd = model.state_dict()
    # nn.Linear stores [out, in]; a bare addmm wants [in, out].
    table = {
        "w1": sd["fc1.weight"].t().contiguous(),
        "b1": sd["fc1.bias"],
        "w2": sd["fc2.weight"].t().contiguous(),
        "b2": sd["fc2.bias"],
        "nw": sd.get("norm.weight"),
        "nb": sd.get("norm.bias"),
        "sw": sd["skip.weight"].t().contiguous() if with_residual else None,
        "sb": sd["skip.bias"] if with_residual else None,
    }
    args = (x, *[table[p] for p in parts])
    return args, (ref_fn or ref_full)(model, x)


def build_matrix_e1():
    return [
        # ---- TRACK A: no tiling. Expected to fail; WHERE it fails is the point.
        Stage(
            id="linear_plain",
            track="plain",
            title="nn.Linear only, symbolic rows, NO tiling",
            question="With no bridge to wrap this in a loop, where does a "
            "symbolic row count fail first? The answer is the bridge's "
            "requirement list.",
            predict="MEASURED 2026-10-04: it does NOT fail. It compiles, reuses "
            "ONE binary across every size, and returns GARBAGE above the "
            "warm-up size (err 0.89 at 448 and 512 against 0.003 at and below "
            "320). First silent wrong answer in the project. Re-run to confirm "
            "the in-trace contract does not change it: the absence of tiling, "
            "not the absence of a range, is the cause.",
            build=lambda sz: build_plain(sz),
            call=lambda model, x: model(x),
        ),
        Stage(
            id="mlp_plain",
            track="plain",
            title="full MLP (2 Linear + ReLU), symbolic rows, NO tiling",
            question="Same question for a whole module rather than one op. Does "
            "it fail in the same place, or does a second op fail differently?",
            predict="FAILS, same class. Listed separately because the FIRST "
            "failing op may differ once there is more than one.",
            build=lambda sz: build_plain(sz),
            call=lambda model, x: model(x),
        ),
        # ---- TRACK B: hand-tiled, emulating what the bridge would emit.
        Stage(
            id="linear_tiled",
            track="tiled",
            title="one addmm inside for_each_tile, weight+bias invariant",
            question="Does a REAL weight survive as an invariant operand, and "
            "does addmm-with-bias work inside a tiled body? Every fixture so "
            "far used bare mm with no bias.",
            predict="works, one binary. The weight is invariant so its geometry "
            "never depends on the row count.",
            build=lambda sz: build_tiled(sz, ("w1", "b1"), ref_linear),
            call=tiled_linear,
            notes="The first test of nn.Linear's bias, i.e. addmm, under a "
            "symbolic count.",
        ),
        Stage(
            id="linear_relu_tiled",
            track="tiled",
            title="addmm + relu inside the body",
            question="Does a pointwise op chained after the matmul stay inside "
            "the same tile?",
            predict="works, one binary",
            build=lambda sz: build_tiled(sz, ("w1", "b1"), ref_linear_relu),
            call=tiled_linear_relu,
        ),
        Stage(
            id="mlp_tiled",
            track="tiled",
            title="TWO addmm plus relu in ONE tiled body, one intermediate",
            question="The real question for the bridge: can a whole REGION live "
            "in one loop, with the intermediate never leaving the body? One "
            "loop per op would mean N sequential device loops each re-reading "
            "its tile from HBM.",
            predict="works, one binary, and the intermediate h is tile-sized "
            "rather than full-sized",
            build=lambda sz: build_tiled(sz, ("w1", "b1", "w2", "b2"), ref_full),
            call=tiled_mlp,
            notes="If this works it is the strongest argument that region-sized "
            "tiling is the right bridge shape.",
        ),
        Stage(
            id="mlp_norm_tiled",
            track="tiled",
            title="+ LayerNorm, a reduction inside the body from a real module",
            question="softmax_row proved a reduction-in-body with a hand-written "
            "fixture. Does nn.functional.layer_norm, which decomposes to mean, "
            "var and rsqrt, behave the same?",
            predict="works, one binary. The reduction is over the static feature "
            "dim so the body sees a concrete tile.",
            build=lambda sz: build_tiled(
                sz, ("w1", "b1", "w2", "b2", "nw", "nb"), ref_full, with_norm=True
            ),
            call=tiled_mlp_norm,
        ),
        Stage(
            id="mlp_norm_noaffine_tiled",
            track="tiled",
            title="LayerNorm with NO affine params: the reduction chain alone",
            question="mlp_tiled passes and mlp_norm_tiled fails on an "
            "ElementArrangement mismatch. They differ by the mean/var/rsqrt "
            "chain AND by two 1-D affine operands. Which one is it?",
            predict="PASSES. softmax_row already proved a reduction in a tiled "
            "body, so the suspect is the 1-D broadcast rather than the "
            "reduction. If this fails instead, the decomposition is the "
            "problem and that is the more awkward answer.",
            build=lambda sz: build_tiled(
                sz, ("w1", "b1", "w2", "b2"), ref_norm_noaffine
            ),
            call=tiled_mlp_norm_noaffine,
            notes="Half of a two-case bisection with mlp_norm_manual_tiled.",
        ),
        Stage(
            id="mlp_norm_manual_tiled",
            track="tiled",
            title="the same normalisation written by hand, affine included",
            question="If the hand-written mean/var/rsqrt/affine passes where "
            "nn.functional.layer_norm fails, the problem is the DECOMPOSITION, "
            "not the mathematics or the 1-D operands.",
            predict="uncertain, and that is why it is worth running. A pass "
            "makes it a lowering fix; a failure points at the 1-D affine "
            "broadcast, which would be consistent with the no-affine stage "
            "passing.",
            build=lambda sz: build_tiled(
                sz, ("w1", "b1", "w2", "b2", "nw", "nb"), ref_full, with_norm=True
            ),
            call=tiled_mlp_norm_manual,
            notes="Other half of the bisection. Read the three norm stages "
            "together: noaffine, manual, and the nn.functional one.",
        ),
        Stage(
            id="mlp_residual_tiled",
            track="tiled",
            title="+ a residual branch reading the same tile twice",
            question="A diamond inside the body: two branches read one tile and "
            "their results are added. Every residual block needs this.",
            predict="works, one binary. Both branches read the SAME tile, so "
            "there is no second symbol and no equality to assert.",
            build=lambda sz: build_tiled(
                sz, ("w1", "b1", "w2", "b2", "sw", "sb"), ref_full, with_residual=True
            ),
            call=tiled_mlp_residual,
        ),
    ]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def which_inputs_are_symbolic(logs):
    """Pull the audit's verdict on which GRAPH INPUTS carry a symbol.

    This is how question 1 gets answered. A weight showing up here would mean a
    parameter's geometry depends on the batch, which must never happen.
    """
    out = []
    for line in logs:
        if "[audit] input=" in line:
            out.append(line.split("[audit] ", 1)[1][:160])
    return sorted(set(out))


def run_one(stage, compiled, size, device_name, collector, first):
    args_cpu, ref = stage.build(size)
    dev = []
    for i, a in enumerate(args_cpu):
        if isinstance(a, torch.nn.Module):
            dev.append(a.to(device_name))
        elif isinstance(a, torch.Tensor):
            if i == 1 and stage.track == "plain":
                # track A: the input is arg 1 (after the module)
                d, _how = mx.to_device(a, device_name, MAX_SIZE, 0)
                torch._dynamo.mark_dynamic(d, 0)
                dev.append(d)
            elif i == 0 and stage.track == "tiled":
                d, _how = mx.to_device(a, device_name, MAX_SIZE, 0)
                torch._dynamo.mark_dynamic(d, 0)
                dev.append(d)
            else:
                # WEIGHTS: moved plainly and NEVER marked.
                dev.append(a.to(device_name))
        else:
            dev.append(a)

    before = mx.find_bundles()
    t0 = time.perf_counter()
    try:
        out = compiled(*dev)
        wall = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        return {
            "size": size,
            "ok": False,
            "stage": "launch" if "kernel_runner.py" in tb else "compile",
            "error": f"{type(exc).__name__}: {str(exc)[:4000]}",
            "where": mx._blame(tb),
            "bundles_added": len(mx.find_bundles() - before),
            "logs": collector.take(),
        }

    new = mx.find_bundles() - before
    added = len(new)
    r32 = ref.float()
    scale = max(r32.abs().max().item(), 1e-3)
    shapes = [list(out.shape), list(ref.shape)]
    mismatch = tuple(out.shape) != tuple(ref.shape)
    if mismatch:
        err = float("inf")
    else:
        err = (out.cpu().float() - r32).abs().max().item() / scale
    logs = collector.take()
    return {
        "size": size,
        "ok": err <= stage.tol and (first or added == 0),
        "stage": "ran",
        "err": None if err == float("inf") else round(err, 6),
        "shape_mismatch": mismatch,
        "shapes": shapes,
        "recompiled": added > 0,
        "bundles_added": added,
        "new_bundles": sorted(os.path.basename(os.path.dirname(b)) for b in new),
        "symbolic_inputs": which_inputs_are_symbolic(logs),
        "wall_s": round(wall, 4),
        "logs": logs,
    }


def run_stage(stage, device_name, collector):
    print(f"\n{BANNER}\n{stage.id}  [track {stage.track}]: {stage.title}\n{BANNER}")
    print(f"  question : {stage.question}")
    print(f"  predict  : {stage.predict}")
    rec = {
        "id": stage.id,
        "track": stage.track,
        "title": stage.title,
        "question": stage.question,
        "predict": stage.predict,
        "notes": stage.notes,
        "sizes": [],
    }
    torch._dynamo.reset()
    collector.take()

    call = stage.call
    if DECLARE_CONTRACT:
        def traced(*args):
            # The dynamic tensor is arg 1 for the plain track (the module is
            # arg 0) and arg 0 for the tiled track.
            idx = 1 if stage.track == "plain" else 0
            declare_contract(args[idx])
            return call(*args)

        traced.__name__ = f"traced_{stage.id}"
    else:
        traced = call
    compiled = torch.compile(traced, backend="inductor", fullgraph=True, dynamic=None)

    warm = run_one(stage, compiled, WARM, device_name, collector, first=True)
    rec["sizes"].append(warm)
    _print(warm, WARM)
    if warm["ok"]:
        for sz in SIZES:
            if sz == WARM:
                continue
            r = run_one(stage, compiled, sz, device_name, collector, first=False)
            rec["sizes"].append(r)
            _print(r, sz)
    else:
        print("  warm-up failed, skipping the other sizes (they would say nothing)")

    good = [r for r in rec["sizes"] if r.get("ok")]
    rec["verdict"] = (
        "works" if len(good) == len(rec["sizes"]) else "partial" if good else "fails"
    )
    print(f"  VERDICT  : {rec['verdict']}  ({len(good)}/{len(rec['sizes'])} sizes)")
    sym = warm.get("symbolic_inputs") or []
    if sym:
        print("  symbolic graph inputs reported by the audit:")
        for s in sym:
            print(f"    {s}")
    return rec


def _print(r, size):
    flag = "yes" if r.get("ok") else "NO "
    if r["stage"] == "ran":
        extra = "RECOMPILED" if r.get("recompiled") else "same binary"
        if r.get("shape_mismatch"):
            print(f"  [{flag}] {size:>5}  SHAPE MISMATCH out={r['shapes'][0]} "
                  f"ref={r['shapes'][1]}  <- the HARNESS is asking the wrong "
                  f"question, not a backend failure")
            return
        print(f"  [{flag}] {size:>5} ({size // G} tiles)  err={r.get('err')}  "
              f"{extra}  {r.get('wall_s')}s")
    else:
        print(f"  [{flag}] {size:>5}  {r['stage'].upper()}: {r.get('error')}")
        if r.get("where"):
            print(f"         at {r['where']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("only", nargs="*", help="stage ids (default: all)")
    ap.add_argument("--out", default=RESULTS_PATH)
    ap.add_argument("--track", choices=("plain", "tiled"), help="only one track")
    ap.add_argument("--no-contract", action="store_true",
                    help="omit the in-trace range declaration, to reproduce F062")
    args = ap.parse_args()
    if args.no_contract:
        global DECLARE_CONTRACT
        DECLARE_CONTRACT = False
        print("contract: NOT declared in-trace (reproducing F062 deliberately)")

    if not os.path.isabs(args.out):
        args.out = os.path.join(_HERE, args.out)

    # Same reasoning as the matrix: these loggers set their own level at
    # creation time, so raise it BEFORE torch_spyre is imported or the trace
    # is silently incomplete.
    os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
    os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "INFO")

    faulthandler.enable()
    for nm in ("SIGTERM", "SIGABRT"):
        sig = getattr(signal, nm, None)
        if sig is not None:
            try:
                faulthandler.register(sig, chain=True)
            except (RuntimeError, ValueError, OSError):
                pass

    print(f"python  : {sys.version.split()[0]}")
    print(f"torch   : {torch.__version__}")
    print(f"results : {args.out}  (rewritten after every stage)")
    print(f"bundles : {mx.bundle_root()}  (pinned)")
    print(f"contract: declared in-trace = {DECLARE_CONTRACT}  "
          f"(G={G}, min={G}, max={MAX_SIZE})")

    try:
        import torch_spyre  # noqa: F401
        from torch_spyre.constants import DEVICE_NAME
    except Exception:
        print("torch_spyre import FAILED:")
        traceback.print_exc()
        return 1

    collector = mx.SymbolicLogCollector()
    logging.getLogger("spyre").addHandler(collector)
    forced = 0
    for name, obj in list(logging.Logger.manager.loggerDict.items()):
        if name.startswith("spyre") and isinstance(obj, logging.Logger):
            obj.setLevel(logging.INFO)
            forced += 1
    print(f"logging : {forced} spyre.* logger(s) forced to INFO")

    stages = build_matrix_e1()
    if args.track:
        stages = [s for s in stages if s.track == args.track]
    if args.only:
        want = set(args.only)
        unknown = want - {s.id for s in stages}
        if unknown:
            print(f"unknown stage id(s): {sorted(unknown)}")
            print(f"available: {[s.id for s in stages]}")
            return 2
        stages = [s for s in stages if s.id in want]

    out = {
        "schema": 1,
        "torch": torch.__version__,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "g_internal": G,
        "max_size": MAX_SIZE,
        "planned": [s.id for s in stages],
        "stages": [],
    }

    def flush():
        tmp = args.out + ".part"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, args.out)

    for st in stages:
        try:
            out["stages"].append(run_stage(st, DEVICE_NAME, collector))
        except Exception:
            print(f"\n  STAGE {st.id} BLEW UP IN THE HARNESS:")
            traceback.print_exc()
            out["stages"].append(
                {"id": st.id, "track": st.track, "verdict": "harness_error",
                 "error": traceback.format_exc()[-1500:]}
            )
        except BaseException as exc:
            print(f"\n  STAGE {st.id} INTERRUPTED BY {type(exc).__name__}")
            out["stages"].append(
                {"id": st.id, "verdict": "interrupted", "error": type(exc).__name__}
            )
            out["interrupted_at"] = st.id
            flush()
            raise
        flush()
        sys.stdout.flush()

    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    flush()

    print(f"\n{BANNER}\nE1 SUMMARY\n{BANNER}")
    for rec in out["stages"]:
        v = rec.get("verdict", "?")
        sizes = rec.get("sizes", [])
        ok = sum(1 for r in sizes if r.get("ok"))
        mark = {"works": "yes", "partial": "..", "fails": "NO "}.get(v, "?? ")
        print(f"  [{mark}] {rec.get('track','?'):<6} {rec['id']:<22} {v:<8} "
              f"{ok}/{len(sizes)} sizes")
    print("\n  READ THE TWO TRACKS TOGETHER. 'plain' failing at op X with")
    print("  'tiled' passing says the bridge must tile X. That is a design")
    print("  statement, and it is the main output of this experiment.")
    n_logs = sum(len(r.get("logs") or []) for s in out["stages"] for r in s.get("sizes", []))
    print(f"\n  wrote {args.out}  ({n_logs} captured [symbolic-loop] log lines)")
    if n_logs < 5 * max(1, len(out["stages"])):
        print("  WARNING: suspiciously few log lines; the trace may be INCOMPLETE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
