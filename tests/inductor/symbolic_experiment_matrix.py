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

"""Symbolic-shape experiment matrix: one binary, many kernels, many sizes.

Not a unit test. The exploration harness. It drives the for_each_tile fixtures
that are already proven ON DEVICE, marks a dimension dynamic, and then asks the
same two questions of every one of them:

  1. Does it compile, launch and give the right numbers at the warm-up size?
  2. Does the SAME binary stay correct at the other sizes in the declared range,
     with no recompile?

Every scenario carries the question it exists to answer and an explicit
PREDICTION, so a surprise is visible rather than something to rationalise after
the fact. Results go to a machine-readable JSON so the findings accumulate
instead of living in a terminal scrollback.

    SPYRE_INDUCTOR_LOG=1 SPYRE_INDUCTOR_LOG_LEVEL=INFO \
    TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
      python tests/inductor/symbolic_experiment_matrix.py 2>&1 | tee matrix.log

    # one scenario at a time while iterating
    python tests/inductor/symbolic_experiment_matrix.py matmul_split_m

Do NOT use TORCH_LOGS="+spyre.inductor", it breaks `import torch` itself. Add
TORCH_LOGS="recompiles" when a recompile needs explaining: that artifact is
registered and safe.
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
from dataclasses import dataclass, field
from typing import Callable

import torch

BANNER = "=" * 78
RESULTS_PATH = "symbolic_experiments_results.json"

# Granularity and ceiling used unless a scenario overrides them. 64 is the
# fixtures' own tile_size and one stick at fp16.
G = 64
MAX_SIZE = 512
WARM = 320  # 5 tiles: more than one, not the max, not a power of two


# ---------------------------------------------------------------------------
# Scenario description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One kernel plus the dimension we make vary.

    ``build(size)`` returns ``(args, cpu_ref)`` with CPU tensors. ``marks`` are
    the ``(arg_position, dim)`` pairs to mark dynamic and to declare a range
    for. ``equalities`` are ``((i, di), (j, dj))`` pairs the traced function
    asserts equal BEFORE calling the kernel, which is how the design proposes to
    collapse two independently marked operands onto one symbol.
    """

    id: str
    title: str
    question: str
    predict: str
    mode: str  # map | reduction | nested | attention | indirect
    fixture: Callable
    build: Callable
    marks: tuple
    granularity: int = G
    max_size: int = MAX_SIZE
    warm: int = WARM
    sizes: tuple = ()
    equalities: tuple = ()
    tol: float = 2e-2
    skip: str = ""
    notes: str = ""
    extra: tuple = field(default_factory=tuple)


def traced_factory(sc: Scenario):
    """Wrap the fixture so the traced graph declares the admissible range.

    The range is declared with torch._check rather than mark_dynamic(min=, max=)
    because min/max is a strict CONSTRAINT: Dynamo then promises every value in
    the range works and refuses any guard that narrows it, and for_each_tile
    necessarily narrows it (a tile_size that does not divide the length is
    refused). Measured: the constraint form dies in produce_guards with
    ConstraintViolationError on (S % G) == 0. torch._check refines the ShapeEnv
    range without making that promise, so compute_symbolic_bounds still reads
    max and granularity for the bundle's input_arg.
    """

    def traced(*args):
        for pos, dim in sc.marks:
            torch._check(args[pos].size(dim) >= sc.granularity)
            torch._check(args[pos].size(dim) <= sc.max_size)
        for (i, di), (j, dj) in sc.equalities:
            torch._check(args[i].size(di) == args[j].size(dj))
        return sc.fixture(*args, *sc.extra)

    traced.__name__ = f"traced_{sc.id}"
    return traced


# ---------------------------------------------------------------------------
# Log capture: the trace, kept per scenario rather than printed and lost
# ---------------------------------------------------------------------------


class SymbolicLogCollector(logging.Handler):
    """Collects the [symbolic-loop] records so each result carries its trace."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record):  # noqa: D102
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if "[symbolic-loop]" in msg:
            self.records.append(f"{record.name}| {msg}")

    def take(self) -> list:
        out = self.records
        self.records = []
        return out


# ---------------------------------------------------------------------------
# Bundle inspection: what actually reached the backend
# ---------------------------------------------------------------------------


def find_bundles() -> list:
    """Every bundle.mlir under the cache root, oldest first."""
    roots = [
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
        os.path.expanduser("~/.cache/torch_spyre"),
        "/tmp",
    ]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        found = []
        for dirpath, _, filenames in os.walk(root):
            if "bundle.mlir" in filenames:
                found.append(os.path.join(dirpath, "bundle.mlir"))
        if found:
            return sorted(found, key=os.path.getmtime)
    return []


def bundle_facts(path: str) -> dict:
    """The properties of an emitted bundle that decide whether this works.

    Deliberately includes the per-core offsets. Those are ADDRESSES derived from
    device_size, and a hint-sized device_size made them disagree between the
    input and the output, which silently broke every size above the warm-up one
    until concretize_expr was changed to use the ShapeEnv max.
    """
    try:
        with open(path) as f:
            text = f.read()
    except OSError as exc:
        return {"error": str(exc)}
    import re

    return {
        "path": path,
        "dim_is_a_parameter": "input_arg<index, granularity=" in text,
        "bound_is_the_dimension": "to %dim_" in text,
        "steps_by_granularity": "step %step_" in text,
        "authored_divide": "ceildivsi" in text,
        "bound_is_a_constant": bool(re.search(r"to %loop_bound_\d+", text)),
        "core_offsets": sorted(set(re.findall(r"_core_offset_(\d+)", text))),
        "affine_maps": re.findall(r"#map_\d+ = affine_map<[^>]*>", text),
        "symbol_ids": re.findall(r'"symbol_ids"=\[([^\]]*)\]', text),
        "dim_leaked_into_execute": any(
            "%dim_" in ln for ln in text.splitlines() if "sdsc_execute" in ln
        ),
        "n_execute": text.count("sdsc_execute"),
        "n_loops": text.count("scf.for"),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def to_device(t, device_name, max_size, marked_dim):
    """Move to device, reserving at max along the marked dim when supported.

    The reservation (torch-spyre#4326) builds the SpyreTensorLayout from the
    PADDED shape while the DMA stays at the real shape, so the device geometry
    stays identical across runtime sizes and the recompile guard never fires on
    it. Without it the input geometry is warm-up-sized.
    """
    if marked_dim is None:
        return t.to(device_name), "plain"
    try:
        return t.to(device_name, max=max_size), "reserved"
    except (TypeError, ValueError):
        return t.to(device_name), "plain-fallback"


def run_one_size(sc, compiled, size, device_name, collector, first):
    """One (scenario, size) measurement."""
    args_cpu, ref = sc.build(size)
    marked = {pos: dim for pos, dim in sc.marks}
    dev_args = []
    reservation = set()
    for i, a in enumerate(args_cpu):
        if isinstance(a, torch.Tensor):
            d, how = to_device(a, device_name, sc.max_size, marked.get(i))
            reservation.add(how)
            if i in marked:
                torch._dynamo.mark_dynamic(d, marked[i])
            dev_args.append(d)
        else:
            dev_args.append(a)

    before = len(find_bundles())
    t0 = time.perf_counter()
    try:
        out = compiled(*dev_args)
        wall = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        at_launch = "launch_jobplan" in tb or "kernel_runner.py" in tb
        return {
            "size": size,
            "ok": False,
            "stage": "launch" if at_launch else "compile",
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "where": _blame(tb),
            "bundles_added": len(find_bundles()) - before,
            "logs": collector.take(),
        }

    added = len(find_bundles()) - before
    outs = out if isinstance(out, (tuple, list)) else (out,)
    refs = ref if isinstance(ref, (tuple, list)) else (ref,)
    errs, shapes = [], []
    for o, r in zip(outs, refs):
        shapes.append([list(o.shape), list(r.shape)])
        if tuple(o.shape) != tuple(r.shape):
            errs.append(float("inf"))
            continue
        rf = r.float()
        # RELATIVE, scaled by the reference's own magnitude. An absolute
        # tolerance that suits a pointwise abs is meaningless for a matmul
        # accumulating 256 fp16 products, and one that suits the matmul would
        # not catch a wrong pointwise result.
        scale = max(rf.abs().max().item(), 1e-3)
        errs.append((o.cpu().float() - rf).abs().max().item() / scale)
    worst = max(errs) if errs else float("inf")
    return {
        "size": size,
        "ok": worst <= sc.tol and (first or added == 0),
        "stage": "ran",
        "err": None if worst == float("inf") else round(worst, 6),
        "shapes": shapes,
        "recompiled": added > 0,
        "bundles_added": added,
        "reservation": sorted(reservation),
        "wall_s": round(wall, 4),
        "logs": collector.take(),
    }


def _blame(tb: str) -> str:
    """The deepest torch-spyre or torch frame, which is where to start reading."""
    lines = [ln.strip() for ln in tb.splitlines() if ln.strip().startswith("File ")]
    ours = [ln for ln in lines if "torch-spyre" in ln or "torch_spyre" in ln]
    return (ours or lines or ["?"])[-1][:220]


def run_scenario(sc: Scenario, device_name: str, collector) -> dict:
    print(f"\n{BANNER}\n{sc.id}: {sc.title}\n{BANNER}")
    print(f"  question : {sc.question}")
    print(f"  predict  : {sc.predict}")
    print(f"  mode     : {sc.mode}   G={sc.granularity}  max={sc.max_size}")
    rec = {
        "id": sc.id,
        "title": sc.title,
        "question": sc.question,
        "predict": sc.predict,
        "mode": sc.mode,
        "granularity": sc.granularity,
        "max_size": sc.max_size,
        "warm": sc.warm,
        "marks": [list(m) for m in sc.marks],
        "equalities": [[list(a), list(b)] for a, b in sc.equalities],
        "notes": sc.notes,
        "sizes": [],
    }
    if sc.skip:
        print(f"  SKIPPED: {sc.skip}")
        rec["skipped"] = sc.skip
        return rec

    torch._dynamo.reset()
    collector.take()
    compiled = torch.compile(traced_factory(sc), backend="inductor", fullgraph=True)

    bundles_before = len(find_bundles())
    warm = run_one_size(sc, compiled, sc.warm, device_name, collector, first=True)
    rec["sizes"].append(warm)
    _print_size(warm, sc.warm, sc.granularity)

    new_bundles = find_bundles()[bundles_before:]
    rec["bundle"] = bundle_facts(new_bundles[-1]) if new_bundles else None

    if warm["ok"]:
        for size in sc.sizes:
            if size == sc.warm:
                continue
            r = run_one_size(sc, compiled, size, device_name, collector, first=False)
            rec["sizes"].append(r)
            _print_size(r, size, sc.granularity)
    else:
        print("  warm-up failed, not trying the other sizes (they would say nothing)")

    good = [r for r in rec["sizes"] if r.get("ok")]
    rec["verdict"] = (
        "works" if len(good) == len(rec["sizes"]) else
        "partial" if good else "fails"
    )
    print(f"  VERDICT  : {rec['verdict']}  ({len(good)}/{len(rec['sizes'])} sizes)")
    return rec


def _print_size(r, size, g):
    tiles = f"{size / g:.2f}" if size % g else str(size // g)
    flag = "yes" if r.get("ok") else "NO "
    if r["stage"] == "ran":
        extra = "RECOMPILED" if r.get("recompiled") else "same binary"
        print(f"  [{flag}] {size:>5} ({tiles} tiles)  err={r.get('err')}  {extra}"
              f"  {r.get('wall_s')}s")
    else:
        print(f"  [{flag}] {size:>5} ({tiles} tiles)  {r['stage'].upper()}: "
              f"{r.get('error')}")
        if r.get("where"):
            print(f"         at {r['where']}")


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------
#
# Every kernel here is one the device e2e suite already exercises CONCRETELY
# (tests/inductor/test_for_each_tile_e2e.py imports all of them), so a failure
# is about the symbolic dimension and not about device support for the op. That
# is the whole reason for reusing them rather than writing fresh kernels.
#
# Inputs are fp16 throughout, matching the pointwise case already proven on
# device, with references computed in fp32. Errors are RELATIVE, so one
# tolerance is meaningful across a pointwise abs and a 256-deep matmul.

FP = torch.float16
SIZES_G64 = (64, 128, 256, 320, 448, 512)
SIZES_G64_RAGGED = (100, 511)
SIZES_G128 = (128, 256, 384, 512)
N_COLS = 128  # 2 sticks at fp16
MM_K = 256
MM_N = 64


def _rand(*shape):
    return torch.randn(*shape, dtype=FP)


def build_pointwise_1(size):
    a = _rand(size, N_COLS)
    return (a, G), a.float().abs()


def build_pointwise_2(size):
    a, b = _rand(size, N_COLS), _rand(size, N_COLS)
    return (a, b, G), (a.float() + b.float())


def build_softmax_row(size):
    x = _rand(size, N_COLS)
    return (x, G), torch.softmax(x.float(), dim=-1)


def build_mm_split_m(size):
    x, y = _rand(size, MM_K), _rand(MM_K, MM_N)
    return (x, y), x.float() @ y.float()


def build_mm_split_k(size):
    x, y = _rand(MM_K, size), _rand(size, MM_N)
    acc0 = torch.zeros(MM_K, MM_N, dtype=FP)
    return (x, y, acc0), acc0.float() + x.float() @ y.float()


def build_nested_m_then_k(size):
    x, y = _rand(size, MM_K), _rand(MM_K, MM_N)
    return (x, y), x.float() @ y.float()


def build_attention(size):
    q = _rand(128, 128)
    k, v = _rand(size, 128), _rand(size, 128)
    qf, kf, vf = q.float(), k.float(), v.float()
    probs = torch.softmax(qf @ kf.transpose(-1, -2), dim=-1)
    return (q, k, v), probs @ vf


def _fixtures():
    """Import the fixtures by directory, not as a package.

    `tests` has no __init__.py, so `from tests.inductor import ...` relies on
    namespace-package resolution and on the repo root being on sys.path. Putting
    this file's own directory on the path and importing the module flat works
    whichever way the script is invoked.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import for_each_tile_fixtures as fx  # noqa: PLC0415

    return fx


def build_matrix():
    fx = _fixtures()
    return [
        Scenario(
            id="pointwise_abs",
            title="map, pointwise abs, one operand tiled on dim 0",
            question="The baseline. Does the simplest possible body hold across "
            "the whole declared range on one binary?",
            predict="works at every size except exactly one tile, which "
            "recompiles on an upstream contiguity guard",
            mode="map",
            fixture=fx.abs_tiled_fn,
            build=build_pointwise_1,
            marks=((0, 0),),
            sizes=SIZES_G64,
        ),
        Scenario(
            id="pointwise_abs_ragged",
            title="adversarial: sizes that are NOT multiples of the granularity",
            question="Invariant 4. A non-multiple must be REFUSED, never run one "
            "tile short and return a correctly shaped tensor with a stale tail.",
            predict="both sizes refused, because _normalize_in_specs raises and "
            "the divisibility test is a Dynamo guard",
            mode="map",
            fixture=fx.abs_tiled_fn,
            build=build_pointwise_1,
            marks=((0, 0),),
            sizes=SIZES_G64_RAGGED,
            warm=320,
            notes="A pass here means every size in this row FAILED to run, "
            "which is the desired outcome. Read the per-size lines, not the "
            "verdict.",
        ),
        Scenario(
            id="pointwise_add_two_symbols",
            title="two operands co-indexed, BOTH marked: two independent symbols",
            question="The two-symbol problem. Two marked operands give two "
            "symbols and therefore two trip counts, which cannot be launched. "
            "What actually happens?",
            predict="refused, or the num_tiles mismatch in _normalize_in_specs "
            "fires. Should NOT silently produce one of the two counts.",
            mode="map",
            fixture=fx.add_tiled_fn,
            build=build_pointwise_2,
            marks=((0, 0), (1, 0)),
            sizes=(320,),
        ),
        Scenario(
            id="pointwise_add_two_symbols_checked",
            title="same, but the traced graph asserts the two dims are equal first",
            question="Does a torch._check equality collapse two marked operands "
            "onto one symbol? This is the design's proposed answer and it has "
            "never been run.",
            predict="unifies and works, which would validate the bridge's plan "
            "to emit one equality per shared axis",
            mode="map",
            fixture=fx.add_tiled_fn,
            build=build_pointwise_2,
            marks=((0, 0), (1, 0)),
            equalities=(((0, 0), (1, 0)),),
            sizes=SIZES_G64,
        ),
        Scenario(
            id="softmax_row",
            title="map, a REDUCTION inside the tile body (row softmax)",
            question="The body reduces over the untiled last dim. Does a "
            "reduction inside a symbolic-count loop body behave?",
            predict="works. The body sees a static tile, so its reduction is "
            "ordinary.",
            mode="map",
            fixture=fx.softmax_row_tiled_fn,
            build=build_softmax_row,
            marks=((0, 0),),
            sizes=SIZES_G64,
        ),
        Scenario(
            id="matmul_split_m",
            title="MATMUL, map mode, M tiled, Y invariant",
            question="GEMM is the case that decides the real performance "
            "argument and it is unmeasured. Does a matmul body work with a "
            "symbolic tile count?",
            predict="works. M-tiling is embarrassingly parallel and the "
            "reduction dim K is untouched.",
            mode="map",
            fixture=fx.split_m_fn,
            build=build_mm_split_m,
            marks=((0, 0),),
            sizes=SIZES_G64,
            tol=5e-2,
            notes="fp16 accumulation over K=256, hence the looser tolerance.",
        ),
        Scenario(
            id="matmul_split_k",
            title="MATMUL, REDUCTION mode, K tiled, caller-supplied accumulator",
            question="Olivier's adversarial case. Reduction mode carries an "
            "accumulator across iterations, so the loop has a real dependency "
            "between steps. And K is shared by both operands, so this is also "
            "the two-symbol case.",
            predict="two symbols, so refused or mismatched. The accumulator "
            "itself should be fine once the counts agree.",
            mode="reduction",
            fixture=fx.split_k_caller_init_fn,
            build=build_mm_split_k,
            marks=((0, 1), (1, 0)),
            sizes=(320,),
            tol=5e-2,
        ),
        Scenario(
            id="matmul_split_k_checked",
            title="same reduction, with the shared K asserted equal first",
            question="Does reduction mode work end to end once the two symbols "
            "are collapsed? This is the single most important row in the matrix: "
            "a cross-iteration dependency plus a symbolic count.",
            predict="works. Nothing in the carry depends on the trip count, and "
            "the body is a static tile.",
            mode="reduction",
            fixture=fx.split_k_caller_init_fn,
            build=build_mm_split_k,
            marks=((0, 1), (1, 0)),
            equalities=(((0, 1), (1, 0)),),
            sizes=SIZES_G64,
            tol=5e-2,
        ),
        Scenario(
            id="nested_m_then_k",
            title="NESTED, outer M map with a symbolic count, inner K reduction",
            question="Two loop levels where only the OUTER count varies. Does "
            "the nesting survive, and does loop_count carry per level?",
            predict="works, and LoopSpec nests with a symbolic outer count and a "
            "concrete inner one",
            mode="nested",
            fixture=fx.nested_split_m_then_k_fn,
            build=build_nested_m_then_k,
            marks=((0, 0),),
            sizes=SIZES_G64,
            tol=5e-2,
        ),
        Scenario(
            id="attention_online_softmax",
            title="ATTENTION, online softmax, 3-leaf carry, KV length symbolic",
            question="The real target shape. A three-value carry (running max, "
            "running denominator, accumulator) over a symbolic number of KV "
            "tiles. K and V are both tiled on the same axis, so two symbols.",
            predict="two symbols, so refused until the equality is asserted",
            mode="attention",
            fixture=fx.online_softmax_fn,
            build=build_attention,
            marks=((1, 0), (2, 0)),
            granularity=128,
            sizes=(384,),
            warm=384,
            tol=8e-2,
            notes="fp16 online-softmax recurrence, loose tolerance on purpose.",
        ),
        Scenario(
            id="attention_online_softmax_checked",
            title="same attention, with K and V asserted the same length",
            question="Does flash-attention's inner loop run over a symbolic KV "
            "length on one binary? This is the shape the whole feature exists "
            "to serve.",
            predict="works, and is the strongest result the matrix can produce",
            mode="attention",
            fixture=fx.online_softmax_fn,
            build=build_attention,
            marks=((1, 0), (2, 0)),
            equalities=(((1, 0), (2, 0)),),
            granularity=128,
            sizes=SIZES_G128,
            warm=384,
            tol=8e-2,
        ),
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("only", nargs="*", help="scenario ids to run (default: all)")
    ap.add_argument("--out", default=RESULTS_PATH)
    args = ap.parse_args()

    # A bare relative default lands in whatever directory the run started
    # from, which is how the last run's output went missing. Anchor it next to
    # this script and print it up front.
    if not os.path.isabs(args.out):
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)

    # A device-level abort kills the process without unwinding python, so the
    # only way to learn where it died is to have the fault handler installed.
    faulthandler.enable()
    for signame in ("SIGTERM", "SIGABRT"):
        sig = getattr(signal, signame, None)
        if sig is not None:
            try:
                faulthandler.register(sig, chain=True)
            except (RuntimeError, ValueError, OSError):
                pass

    print(f"python  : {sys.version.split()[0]}")
    print(f"torch   : {torch.__version__}")
    print(f"results : {args.out}  (rewritten after every scenario)")

    # tests/ is not a package on sys.path when run as a script.
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    if repo not in sys.path:
        sys.path.insert(0, repo)

    try:
        import torch_spyre  # noqa: F401
        from torch_spyre.constants import DEVICE_NAME
    except Exception:
        print("torch_spyre import FAILED:")
        traceback.print_exc()
        return 1

    collector = SymbolicLogCollector()
    spyre_log = logging.getLogger("spyre")
    spyre_log.addHandler(collector)
    spyre_log.setLevel(logging.INFO)

    scenarios = build_matrix()
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {s.id for s in scenarios}
        if unknown:
            print(f"unknown scenario id(s): {sorted(unknown)}")
            print(f"available: {[s.id for s in scenarios]}")
            return 2
        scenarios = [s for s in scenarios if s.id in wanted]

    out = {
        "schema": 1,
        "torch": torch.__version__,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "planned": [s.id for s in scenarios],
        "scenarios": [],
    }

    def flush_results():
        """Write what we have so far.

        The whole run used to be written once, after the loop. A scenario that
        took the process down hard -- a device abort, an OOM kill, a Ctrl-C --
        is not an `Exception`, so it skipped the write and eleven scenarios'
        worth of results went with it. Writing after every scenario costs
        nothing and means a crash loses one row, not all of them.
        """
        tmp = args.out + ".part"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, args.out)

    for sc in scenarios:
        try:
            out["scenarios"].append(run_scenario(sc, DEVICE_NAME, collector))
        except Exception:
            print(f"\n  SCENARIO {sc.id} BLEW UP IN THE HARNESS:")
            traceback.print_exc()
            out["scenarios"].append(
                {"id": sc.id, "verdict": "harness_error",
                 "error": traceback.format_exc()[-1500:]}
            )
        except BaseException as exc:
            # Ctrl-C and SystemExit land here. Save, then let it through.
            print(f"\n  SCENARIO {sc.id} INTERRUPTED BY {type(exc).__name__}")
            out["scenarios"].append(
                {"id": sc.id, "verdict": "interrupted",
                 "error": type(exc).__name__}
            )
            out["interrupted_at"] = sc.id
            flush_results()
            print(f"  partial results saved to {args.out}")
            raise
        flush_results()
        # tee plus a scrolled-off terminal means the log file is the record,
        # so make sure the record is on disk before the next scenario starts.
        sys.stdout.flush()

    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    flush_results()

    print(f"\n{BANNER}\nMATRIX SUMMARY\n{BANNER}")
    for rec in out["scenarios"]:
        v = rec.get("verdict", "?")
        sizes = rec.get("sizes", [])
        ok = sum(1 for r in sizes if r.get("ok"))
        mark = {"works": "yes", "partial": "..", "fails": "NO "}.get(v, "?? ")
        print(f"  [{mark}] {rec['id']:<38} {v:<14} {ok}/{len(sizes)} sizes")
        if rec.get("predict") and v != "works":
            print(f"         predicted: {rec['predict'][:100]}")
    print(f"\n  wrote {args.out}")
    print("  Send that file back. It carries the per-size numbers, the bundle")
    print("  facts and the captured [symbolic-loop] trace for every scenario.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
