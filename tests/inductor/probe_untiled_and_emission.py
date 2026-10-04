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

"""Two open questions, one run. Not a test, a probe: run it and read it.

Q1. Does a pointwise model with NO for_each_tile actually serve a range from
    one binary? We have been claiming it does, in the HLD and in a PR
    description, on the strength of a refusal message saying symbolic work
    division "covers pointwise (and non-matmul reduction) ops only". That says
    what work division ACCEPTS. It does not say the result is one binary, and
    every one of the twelve matrix scenarios is tiled, so nothing has ever
    measured it. Also worth knowing: with no loop the SDSC is emitted at the
    MAX, so the kernel may be computing max rows on every call. Correct under
    invariant 5, but a different sentence from "it works".

Q2. Where can the contract be asserted from? The range has to reach the
    ShapeEnv for max_trip_count to find a finite upper bound, and `.to()` runs
    eagerly, before the symbol exists. Part B asks which of three spellings
    actually lands the facts.

Run:
    export TORCHINDUCTOR_CACHE_DIR=/tmp/sym_cache
    rm -rf /tmp/sym_cache && mkdir -p /tmp/sym_cache
    unset TORCH_LOGS
    python tests/inductor/probe_untiled_and_emission.py
"""

import argparse
import json
import os
import sys
import traceback

# Spyre's own loggers set their level when they are created, so raise it before
# torch_spyre is imported or the trace comes back empty.
os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "INFO")

import torch  # noqa: E402

G = 64
MIN_SIZE = 128
MAX_SIZE = 512
SIZES = [320, 128, 256, 448, 512]  # 320 first: warm up away from an endpoint
WIDTH = 128

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "probe_untiled_results.json"
)


def bundle_root() -> str:
    """The pinned cache root, resolved once.

    Resolving it per call is how an earlier harness fabricated binary reuse:
    the root moved under it and a real recompile came out as "same binary".
    """
    return os.environ.get("TORCHINDUCTOR_CACHE_DIR") or "/tmp/sym_cache"


_ROOT = bundle_root()


def find_bundles() -> set:
    """Every compiled bundle under the pinned root, as a set of paths.

    A set, not a count, so the caller can diff two observations and name what
    is new. A count can go down and hide a recompile.
    """
    found = set()
    for dirpath, _dirnames, filenames in os.walk(_ROOT):
        for fn in filenames:
            if fn.endswith(".mlir") or "sdsc" in fn:
                found.add(os.path.join(dirpath, fn))
    return found


# --------------------------------------------------------------------------
# Part A: does an UNTILED pointwise model serve a range from one binary?
# --------------------------------------------------------------------------


def pointwise_only(x):
    """No matmul, no for_each_tile. The case we have been claiming works."""
    return torch.relu(x * 2.0) + 1.0


def declare(x):
    """State the contract where the symbol exists, i.e. inside the trace."""
    n = x.size(0)
    torch._check(n >= MIN_SIZE)
    torch._check(n <= MAX_SIZE)
    torch._check(n % G == 0)
    return x


def part_a(device: str, out: dict) -> None:
    print("\n" + "=" * 74)
    print("PART A: untiled pointwise, does one binary serve the range?")
    print("=" * 74)

    def traced(x):
        declare(x)
        return pointwise_only(x)

    compiled = torch.compile(traced, backend="inductor", fullgraph=True, dynamic=None)
    rows = []

    for size in SIZES:
        torch.manual_seed(0xA11)
        cpu = torch.randn(size, WIDTH, dtype=torch.float16)
        ref = pointwise_only(cpu.float())

        dev = cpu.to(device)
        torch._dynamo.mark_dynamic(dev, 0)

        before = find_bundles()
        row = {"size": size}
        try:
            got = compiled(dev).cpu().float()
            new = find_bundles() - before
            scale = max(ref.abs().max().item(), 1e-3)
            row.update(
                ok=True,
                bundles_added=len(new),
                err=round((got - ref).abs().max().item() / scale, 6),
                shape=list(got.shape),
            )
        except BaseException as exc:  # noqa: BLE001
            row.update(
                ok=False,
                error=f"{type(exc).__name__}: {str(exc)[:4000]}",
                where=_blame(traceback.format_exc()),
                bundles_added=len(find_bundles() - before),
            )
        rows.append(row)
        flag = "OK " if row["ok"] else "NO "
        rest = {k: v for k, v in row.items() if k != "size"}
        print(f"  [{flag}] {size:>4}  {json.dumps(rest)[:150]}")

    out["part_a"] = rows

    added = [r["bundles_added"] for r in rows]
    if all(r["ok"] for r in rows):
        one_binary = sum(added[1:]) == 0
        print(f"\n  VERDICT: {'ONE BINARY' if one_binary else 'A BINARY PER SIZE'}")
        print(f"  bundles added per size: {added}")
        if not one_binary:
            print("  So the 'untiled pointwise works' claim must come out of")
            print("  the HLD and the PR description.")
    else:
        print("\n  VERDICT: untiled pointwise does NOT compile. The claim is wrong")
        print("  and the first failing message above is the real capability.")


# --------------------------------------------------------------------------
# Part B: which spelling lands the contract in the ShapeEnv?
# --------------------------------------------------------------------------


def part_b(out: dict) -> None:
    """Pure torch, no device. Which spelling gives a finite upper bound?

    max_trip_count needs a finite ShapeEnv upper bound or it refuses. `.to()`
    cannot provide one because it runs before the symbol exists. So: can the
    bound be installed from outside the traced function at all, or does it have
    to be asserted inside?
    """
    print("\n" + "=" * 74)
    print("PART B: where can the range be installed so the ShapeEnv holds it?")
    print("=" * 74)

    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch._dynamo.source import LocalSource

    rows = []

    def observe(label, fn):
        env = ShapeEnv()
        src = LocalSource("x")
        sym = env.create_symbol(320, src)
        s = env.create_symintnode(sym, hint=320, source=src)
        row = {"spelling": label}
        try:
            fn(s)
            rng = env.var_to_range.get(sym)
            row.update(
                lower=str(getattr(rng, "lower", None)),
                upper=str(getattr(rng, "upper", None)),
                upper_is_finite="oo" not in str(getattr(rng, "upper", "oo")),
                divisible=sorted(str(f) for f in env.divisible),
            )
        except BaseException as exc:  # noqa: BLE001
            row.update(raised=f"{type(exc).__name__}: {str(exc)[:200]}")
        rows.append(row)
        print(f"  {label}")
        for k, v in row.items():
            if k != "spelling":
                print(f"      {k} = {v}")

    observe("nothing declared", lambda s: None)
    observe(
        "torch._check range only",
        lambda s: (torch._check(s >= MIN_SIZE), torch._check(s <= MAX_SIZE)),
    )
    observe(
        "torch._check range + divisibility",
        lambda s: (
            torch._check(s >= MIN_SIZE),
            torch._check(s <= MAX_SIZE),
            torch._check(s % G == 0),
        ),
    )
    observe("divisibility only", lambda s: torch._check(s % G == 0))

    out["part_b"] = rows
    print("\n  Read the upper_is_finite column. Only a spelling that gives a")
    print("  finite upper bound lets max_trip_count describe the dimension.")


def _blame(tb: str) -> str:
    """The last torch_spyre frame, which is usually the interesting one."""
    for line in reversed(tb.splitlines()):
        if "torch_spyre" in line and line.strip().startswith("File "):
            return line.strip()
    return tb.strip().splitlines()[-1] if tb.strip() else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=("a", "b"), help="run only one part")
    args = ap.parse_args()

    print(f"python  : {sys.version.split()[0]}")
    print(f"torch   : {torch.__version__}")
    print(f"bundles : {_ROOT}  (pinned)")
    print(f"results : {RESULTS_PATH}")
    print(f"contract: G={G}, min={MIN_SIZE}, max={MAX_SIZE}")

    out = {"torch": torch.__version__, "g": G, "min": MIN_SIZE, "max": MAX_SIZE}

    if args.part in (None, "b"):
        # Part B first: pure torch, so it still answers something if the device
        # part dies.
        try:
            part_b(out)
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
        _flush(out)

    if args.part in (None, "a"):
        try:
            import torch_spyre  # noqa: F401
            from torch_spyre.constants import DEVICE_NAME
        except Exception:
            print("\ntorch_spyre import FAILED, skipping part A:")
            traceback.print_exc()
            _flush(out)
            return 1
        try:
            part_a(DEVICE_NAME, out)
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
        _flush(out)

    print(f"\nwrote {RESULTS_PATH}")
    return 0


def _flush(out: dict) -> None:
    """Write after every part, so a later crash cannot lose an earlier result."""
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, RESULTS_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
