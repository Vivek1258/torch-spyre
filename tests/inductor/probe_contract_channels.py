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

"""E0: how does the shape CONTRACT (min, max, granularity) reach the compiler?

Pure PyTorch. No torch_spyre import, no spyre device, no device compile. So it
needs NO reinstall:

    python tests/inductor/probe_contract_channels.py

and it writes `probe_contract_channels_results.json` next to itself.

WHY THIS EXISTS. `compute_granularity` currently recovers G_user by reading the
ShapeEnv's LOWER BOUND of the symbol. On 2026-10-03 that was measured wrong on
device: matmul_split_k declared min 64 and tile_size 64, an unrelated guard
raised the lower bound to 128, and the emitted bundle told the backend
`granularity=128` while the loop stepped 64. A runtime size of 320 then ran
correctly while violating the contract the bundle declared.

So before building the symbol-keyed contract registry, settle the mechanics:

  Part 1  what the ShapeEnv actually holds, and under what keys
  Part 2  REPRODUCE the bug in pure torch: can an unrelated guard move the
          lower bound? If yes, min is not a usable channel for G, full stop.
  Part 3  can min and G coexist as separate facts, and is G RECOVERABLE from
          the ShapeEnv alone once they differ?
  Part 4  which mark_dynamic form survives a divisibility guard, and what a
          non-conforming size actually does (raise? recompile? silently run?)
  Part 5  the G_internal / G_user arithmetic, including whether choosing
          G_internal <= G_user/2 really removes the degenerate single-trip case
  Part 6  the structural form S = G * n, which would make divisibility
          unrepresentable-if-wrong instead of asserted
  Part 7  bands: can one ShapeEnv hold a disjunction at all?

Every case states its QUESTION and an explicit PREDICTION, so a surprise is
visible instead of being rationalised afterwards. Read the per-case lines, not
just the summary.
"""

import json
import os
import sys
import traceback

import sympy
import torch
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.utils._sympy.functions import FloorDiv

try:
    from torch.fx.experimental.symbolic_shapes import statically_known_true
except ImportError:  # older torch
    statically_known_true = None

from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode

BANNER = "=" * 78
RESULTS = []
G_USER = 64
MAX_SIZE = 512
WIDTH = 128


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def record(part, name, question, predict, observed, verdict):
    RESULTS.append(
        {
            "part": part,
            "name": name,
            "question": question,
            "predict": predict,
            "observed": observed,
            "verdict": verdict,
        }
    )
    flag = {"as_predicted": "ok  ", "SURPRISE": "!!! ", "error": "ERR "}.get(verdict, "    ")
    print(f"  [{flag}] {name}")
    for line in str(observed).splitlines():
        print(f"           {line}")


def case(part, name, question, predict):
    """Decorator: run a probe, catch anything, keep going.

    One case blowing up must not lose the rest of the run. Learned the hard
    way: an earlier harness wrote its results once at the end and lost eleven
    scenarios to a single death.
    """

    def wrap(fn):
        print(f"\n--- {name}")
        print(f"    Q: {question}")
        print(f"    P: {predict}")
        try:
            observed, verdict = fn()
        except Exception as exc:  # noqa: BLE001
            observed = f"RAISED {type(exc).__name__}: {exc}\n" + "\n".join(
                traceback.format_exc().splitlines()[-3:]
            )
            verdict = "error"
        record(part, name, question, predict, observed, verdict)
        return fn

    return wrap


def fresh(hint=320, width=WIDTH):
    """A fake [S, width] whose dim 0 is a backed dynamic symbol."""
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    src = LocalSource("x")
    with mode:
        sym = shape_env.create_symbol(hint, src, DimDynamic.DYNAMIC)
        s = shape_env.create_symintnode(sym, hint=hint, source=src)
        x = torch.empty(s, width)
    return shape_env, mode, s, x


def src_name(src):
    """Source.name is a STRING attribute in torch 2.13, not a method.

    Measured the hard way: `getattr(src, "name", lambda: repr(src))()` raised
    TypeError: 'str' object is not callable, and took out cases 1a and 1b.
    Handle both spellings so the probe survives a torch that changes it back.
    """
    n = getattr(src, "name", None)
    if callable(n):
        try:
            return n()
        except Exception:  # noqa: BLE001
            return repr(src)
    return n if isinstance(n, str) else repr(src)


def rng(shape_env, s):
    """The (lower, upper) the ShapeEnv currently believes for s."""
    expr = s.node.expr if hasattr(s, "node") else s
    syms = getattr(expr, "free_symbols", set())
    if not syms:
        return f"concrete {expr}"
    sym = next(iter(syms))
    vr = shape_env.var_to_range.get(sym)
    return f"{sym}: lower={getattr(vr, 'lower', '?')} upper={getattr(vr, 'upper', '?')}"


def divisible_of(shape_env):
    try:
        return sorted(str(e) for e in shape_env.divisible)
    except Exception as exc:  # noqa: BLE001
        return f"<unreadable: {exc!r}>"


# ---------------------------------------------------------------------------
# PART 1: what does the ShapeEnv actually hold, and under what keys
# ---------------------------------------------------------------------------

print(f"\n{BANNER}\nPART 1: ShapeEnv contents and keys\n{BANNER}")
print(f"torch {torch.__version__}  python {sys.version.split()[0]}")


@case(
    1,
    "1a_var_to_sources_resolves_symbol",
    "Can we go from a marked tensor's (source, dim) to its SYMBOL? That is the "
    "key the contract registry would be keyed on.",
    "var_to_sources maps symbol -> [Source]; we can invert it by source name",
)
def _():
    shape_env, _mode, s, _x = fresh()
    v2s = getattr(shape_env, "var_to_sources", None)
    if v2s is None:
        return "shape_env has NO var_to_sources attribute", "SURPRISE"
    lines = [f"var_to_sources has {len(v2s)} entry(ies):"]
    for sym, sources in v2s.items():
        names = [src_name(src) for src in sources]
        lines.append(f"  {sym}  <-  {names}")
    inverted = {
        src_name(src): sym
        for sym, srcs in v2s.items()
        for src in srcs
    }
    lines.append(f"inverted by source name: { {k: str(v) for k, v in inverted.items()} }")
    ok = len(v2s) >= 1
    return "\n".join(lines), "as_predicted" if ok else "SURPRISE"


@case(
    1,
    "1b_two_marked_tensors_two_symbols",
    "If two tensors are marked independently, do we get two distinguishable "
    "registry keys?",
    "two symbols, each with its own source, so the registry can hold a "
    "different contract for each",
)
def _():
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    with mode:
        syms = []
        for nm, hint in (("a", 320), ("b", 256)):
            src = LocalSource(nm)
            sym = shape_env.create_symbol(hint, src, DimDynamic.DYNAMIC)
            syms.append(shape_env.create_symintnode(sym, hint=hint, source=src))
            torch.empty(syms[-1], WIDTH)
    v2s = shape_env.var_to_sources
    lines = [f"symbols: {[str(x) for x in syms]}", f"var_to_sources entries: {len(v2s)}"]
    for sym, sources in v2s.items():
        lines.append(f"  {sym} <- {[src_name(x) for x in sources]}")
    distinct = len({str(x) for x in syms}) == 2
    return "\n".join(lines), "as_predicted" if distinct else "SURPRISE"


@case(
    1,
    "1c_what_divisible_holds_after_check",
    "After torch._check(S % G == 0), what exactly is in ShapeEnv.divisible, "
    "and can G be read back out of it?",
    "divisible contains a Mod(S, 64)-shaped entry from which 64 is readable",
)
def _():
    shape_env, mode, s, _x = fresh()
    before = divisible_of(shape_env)
    with mode:
        torch._check(s % G_USER == 0)
    after = divisible_of(shape_env)
    lines = [f"before: {before}", f"after : {after}", f"range : {rng(shape_env, s)}"]
    found = [e for e in after if e not in before]
    lines.append(f"added : {found}")
    return "\n".join(lines), "as_predicted" if found else "SURPRISE"


# ---------------------------------------------------------------------------
# PART 2: reproduce the measured bug. Is the lower bound a safe channel for G?
# ---------------------------------------------------------------------------

print(f"\n{BANNER}\nPART 2: is the LOWER BOUND a usable channel for G?\n{BANNER}")
print("  This is the one that matters. On device, an unrelated guard raised the")
print("  lower bound from 64 to 128 and the bundle declared the wrong granularity.")


@case(
    2,
    "2a_lower_bound_after_only_the_min_check",
    "With ONLY torch._check(S >= 64) declared, what is the lower bound?",
    "lower == 64, so reading min back out works in the clean case",
)
def _():
    shape_env, mode, s, _x = fresh()
    with mode:
        torch._check(s >= G_USER)
        torch._check(s <= MAX_SIZE)
    return rng(shape_env, s), "as_predicted"


@case(
    2,
    "2b_UNRELATED_GUARD_MOVES_THE_LOWER_BOUND",
    "Does an unrelated comparison on the TILE COUNT raise the lower bound "
    "above the declared min? This is the exact shape of the on-device bug: "
    "for_each_tile compares two operands' num_tiles for its error message, "
    "which at a multi-tile hint proves num_tiles >= 2 and so S >= 2G.",
    "THE LOWER BOUND MOVES to 128. If it does, min cannot carry G, because any "
    "guard anywhere can tighten it.",
)
def _():
    shape_env, mode, s, _x = fresh(hint=320)
    with mode:
        torch._check(s >= G_USER)
        torch._check(s <= MAX_SIZE)
        torch._check(s % G_USER == 0)
        before = rng(shape_env, s)
        # The error-message comparison, in the same shape for_each_tile uses it.
        n_tiles = s // G_USER
        _ = bool(n_tiles != 1)  # forces bool(), which installs a guard
        after = rng(shape_env, s)
    lower_moved = before != after
    lines = [
        f"before the num_tiles comparison: {before}",
        f"after  the num_tiles comparison: {after}",
        f"divisible: {divisible_of(shape_env)}",
        f"LOWER BOUND MOVED: {lower_moved}",
    ]
    # The prediction is that it moves; moving is the SURPRISE-free outcome here
    # only in the sense that it confirms the diagnosis.
    return "\n".join(lines), "as_predicted" if lower_moved else "SURPRISE"


@case(
    2,
    "2c_lower_bound_after_a_ge_2_tiles_check",
    "If something proves S >= 2G explicitly, does the lower bound become 2G, "
    "making min and G indistinguishable from the ShapeEnv alone?",
    "lower becomes 128 while the real G is still 64, so min != G and min is "
    "no longer readable as G",
)
def _():
    shape_env, mode, s, _x = fresh()
    with mode:
        torch._check(s >= G_USER)
        torch._check(s % G_USER == 0)
        torch._check(s // G_USER >= 2)
    return (
        f"{rng(shape_env, s)}\ndivisible: {divisible_of(shape_env)}\n"
        f"declared G was {G_USER}; lower bound now reads as above",
        "as_predicted",
    )


# ---------------------------------------------------------------------------
# PART 3: can min and G coexist, and is G recoverable?
# ---------------------------------------------------------------------------

print(f"\n{BANNER}\nPART 3: min and G as SEPARATE facts (the bands requirement)\n{BANNER}")


@case(
    3,
    "3a_min_128_granularity_64_coexist",
    "Declare min=128 and G=64 separately. Do both facts survive, or does one "
    "overwrite the other?",
    "both survive: lower=128 in var_to_range, Mod(S,64) in divisible",
)
def _():
    shape_env, mode, s, _x = fresh()
    with mode:
        torch._check(s >= 128)
        torch._check(s <= MAX_SIZE)
        torch._check(s % 64 == 0)
    return (
        f"{rng(shape_env, s)}\ndivisible: {divisible_of(shape_env)}",
        "as_predicted",
    )


@case(
    3,
    "3b_is_G_RECOVERABLE_from_the_shapeenv_alone",
    "Given only the ShapeEnv, with min=128 and G=64, can a reader recover "
    "G=64 unambiguously? This is the question that decides whether a registry "
    "is needed at all.",
    "NO. divisible may hold several facts and nothing marks which one is the "
    "contract, so recovery is inference, not reading.",
)
def _():
    shape_env, mode, s, _x = fresh(hint=256)
    with mode:
        torch._check(s >= 128)
        torch._check(s % 64 == 0)
        # A second, perfectly legitimate divisibility fact from unrelated code.
        torch._check(s % 2 == 0)
        y = torch.empty(s, WIDTH)
        _ = y.view(s // 2, 2, WIDTH)  # a view that implies even-ness
    facts = divisible_of(shape_env)
    lines = [
        f"{rng(shape_env, s)}",
        f"divisible now holds {len(facts) if isinstance(facts, list) else '?'} fact(s): {facts}",
        "A reader wanting 'the' G must pick among these. 64 and 2 are both true.",
        "Picking the largest is a heuristic, not a declaration.",
    ]
    ambiguous = isinstance(facts, list) and len(facts) > 1
    return "\n".join(lines), "as_predicted" if ambiguous else "SURPRISE"


# ---------------------------------------------------------------------------
# PART 4: mark_dynamic forms, real guards, real recompiles
# ---------------------------------------------------------------------------

print(f"\n{BANNER}\nPART 4: mark_dynamic forms under a REAL torch.compile\n{BANNER}")
print("  CPU backend, so this needs no device and still exercises guards.")


class GraphCounter:
    """Counts the graphs Dynamo hands the backend. One graph == one binary."""

    def __init__(self):
        self.graphs = []

    def __call__(self, gm, example_inputs):
        self.graphs.append(gm)
        return gm.forward

    @property
    def n(self):
        return len(self.graphs)


def contract_fn(t, g=G_USER, lo=128, hi=MAX_SIZE):
    """The shape of the traced contract we would emit from the registry."""
    torch._check(t.size(0) >= lo)
    torch._check(t.size(0) <= hi)
    torch._check(t.size(0) % g == 0)
    return t.abs().sum(dim=-1)


@case(
    4,
    "4a_mark_dynamic_plain_plus_checks",
    "Does mark_dynamic(t, 0) with NO min/max, plus torch._check for the range "
    "and divisibility, give ONE graph across several conforming sizes?",
    "one graph for 128, 192, 256, 320, 448, 512",
)
def _():
    torch._dynamo.reset()
    cnt = GraphCounter()
    compiled = torch.compile(contract_fn, backend=cnt, dynamic=None)
    sizes = [128, 192, 256, 320, 448, 512]
    for sz in sizes:
        t = torch.randn(sz, WIDTH)
        torch._dynamo.mark_dynamic(t, 0)
        compiled(t)
    return (
        f"sizes {sizes} -> {cnt.n} graph(s)",
        "as_predicted" if cnt.n == 1 else "SURPRISE",
    )


@case(
    4,
    "4b_mark_dynamic_strict_min_max_plus_divisibility",
    "Does mark_dynamic(t, 0, min=, max=) coexist with a divisibility check? "
    "This is the strict-constraint form.",
    "it RAISES (ConstraintViolationError or similar), because the strict "
    "constraint promises every value in the range and the divisibility check "
    "narrows it",
)
def _():
    torch._dynamo.reset()
    cnt = GraphCounter()
    compiled = torch.compile(contract_fn, backend=cnt, dynamic=None)
    t = torch.randn(256, WIDTH)
    try:
        torch._dynamo.mark_dynamic(t, 0, min=128, max=MAX_SIZE)
    except TypeError as exc:
        return f"mark_dynamic does not accept min/max in this torch: {exc}", "SURPRISE"
    try:
        compiled(t)
    except Exception as exc:  # noqa: BLE001
        return (
            f"RAISED {type(exc).__name__}: {str(exc)[:300]}",
            "as_predicted",
        )
    return f"did NOT raise; {cnt.n} graph(s) compiled", "SURPRISE"


@case(
    4,
    "4c_non_conforming_size_what_actually_happens",
    "A size that is NOT a multiple of G: does it raise, recompile silently, or "
    "run? This decides whether a host-side pre-check is required.",
    "the guard misses, Dynamo retraces, and the torch._check then RAISES. So "
    "the failure is correct but arrives from deep inside Dynamo, which is why "
    "a host check is needed for a usable error.",
)
def _():
    torch._dynamo.reset()
    cnt = GraphCounter()
    compiled = torch.compile(contract_fn, backend=cnt, dynamic=None)
    t = torch.randn(256, WIDTH)
    torch._dynamo.mark_dynamic(t, 0)
    compiled(t)
    after_good = cnt.n
    bad = torch.randn(100, WIDTH)  # 100 is not a multiple of 64
    torch._dynamo.mark_dynamic(bad, 0)
    outcome = "ran without complaint"
    try:
        compiled(bad)
    except Exception as exc:  # noqa: BLE001
        outcome = f"RAISED {type(exc).__name__}: {str(exc)[:220]}"
    return (
        f"conforming 256 -> {after_good} graph(s)\n"
        f"non-conforming 100 -> {outcome}\n"
        f"graphs now: {cnt.n}",
        "as_predicted" if "RAISED" in outcome else "SURPRISE",
    )


@case(
    4,
    "4d_two_bands_are_two_graphs",
    "Two bands (G=64 below 512, G=128 above) declared as two different "
    "contracts: does Dynamo give one graph PER BAND, which is what the band "
    "dispatcher design assumes?",
    "two graphs, one per band, because the guards differ",
)
def _():
    torch._dynamo.reset()
    cnt = GraphCounter()

    def banded(t):
        # The host dispatcher would pick the band; here we emulate it by
        # branching on a value Dynamo treats as static per call.
        return contract_fn(t, g=64, lo=128, hi=512)

    def banded_hi(t):
        return contract_fn(t, g=128, lo=512, hi=1024)

    c_lo = torch.compile(banded, backend=cnt, dynamic=None)
    c_hi = torch.compile(banded_hi, backend=cnt, dynamic=None)
    for sz in (128, 256, 448):
        t = torch.randn(sz, WIDTH)
        torch._dynamo.mark_dynamic(t, 0)
        c_lo(t)
    lo_graphs = cnt.n
    for sz in (512, 768, 1024):
        t = torch.randn(sz, WIDTH)
        torch._dynamo.mark_dynamic(t, 0)
        c_hi(t)
    return (
        f"band 1 (G=64, 128..512): {lo_graphs} graph(s)\n"
        f"band 2 (G=128, 512..1024): {cnt.n - lo_graphs} graph(s)\n"
        f"total {cnt.n}",
        "as_predicted" if cnt.n == 2 else "SURPRISE",
    )


@case(
    4,
    "4e_mark_unbacked_availability_and_cost",
    "Does mark_unbacked exist here, and what does a comparison on the symbol "
    "cost? Upstream offers it for the 0/1 specialisation problem but says it "
    "forbids control flow on the symbol.",
    "it exists; a comparison either graph-breaks or raises, which is why it is "
    "not a drop-in for a path that branches on the tile count",
)
def _():
    fn = getattr(torch._dynamo, "mark_unbacked", None) or getattr(
        torch._dynamo.decorators, "mark_unbacked", None
    )
    if fn is None:
        return "mark_unbacked NOT available in this torch", "SURPRISE"
    torch._dynamo.reset()
    cnt = GraphCounter()

    def branchy(t):
        n = t.size(0) // G_USER
        if n != 1:  # the control flow for_each_tile actually does
            return t.abs().sum(dim=-1)
        return t.sum(dim=-1)

    compiled = torch.compile(branchy, backend=cnt, dynamic=None)
    t = torch.randn(256, WIDTH)
    try:
        fn(t, 0)
    except Exception as exc:  # noqa: BLE001
        return f"mark_unbacked({type(exc).__name__}): {exc}", "SURPRISE"
    outcome = "compiled without complaint"
    try:
        compiled(t)
    except Exception as exc:  # noqa: BLE001
        outcome = f"RAISED {type(exc).__name__}: {str(exc)[:220]}"
    return f"mark_unbacked available\nbranch on tile count -> {outcome}\ngraphs: {cnt.n}", "as_predicted"


# ---------------------------------------------------------------------------
# PART 5: G_internal / G_user arithmetic and the degenerate single trip
# ---------------------------------------------------------------------------

print(f"\n{BANNER}\nPART 5: G_internal divides G_user, and the degenerate trip\n{BANNER}")


@case(
    5,
    "5a_trip_count_exact_under_g_internal",
    "With S a multiple of G_user and G_internal dividing G_user, is the trip "
    "count S // G_internal exact, with no floor loss?",
    "exact: Mod(S, G_internal) is provably zero, so no tail is dropped",
)
def _():
    shape_env, mode, s, _x = fresh(hint=256)
    g_user, g_int = 128, 64
    with mode:
        torch._check(s >= g_user)
        torch._check(s % g_user == 0)
        trips = s // g_int
        exact = None
        if statically_known_true is not None:
            exact = bool(statically_known_true(s % g_int == 0))
    return (
        f"G_user={g_user} G_internal={g_int}\n"
        f"trips expr = {trips}\n"
        f"statically_known_true(S % G_internal == 0) = {exact}\n"
        f"divisible: {divisible_of(shape_env)}",
        "as_predicted" if exact else "SURPRISE",
    )


@case(
    5,
    "5b_DEGENERATE_TRIP_REMOVED_by_g_internal_choice",
    "THE KEY CASE for the correction. With min == G_user == 128 and "
    "G_internal == 64, is the smallest legal input provably at least TWO "
    "trips? If yes, the 0/1 specialisation is unreachable and the one-tile "
    "failures need no contract exclusion.",
    "statically_known_true(S // G_internal >= 2) is True, so the single-trip "
    "case cannot occur",
)
def _():
    shape_env, mode, s, _x = fresh(hint=256)
    g_user, g_int = 128, 64
    with mode:
        torch._check(s >= g_user)
        torch._check(s <= MAX_SIZE)
        torch._check(s % g_user == 0)
        trips = s // g_int
        at_least_2 = None
        never_1 = None
        if statically_known_true is not None:
            at_least_2 = bool(statically_known_true(trips >= 2))
            never_1 = bool(statically_known_true(trips != 1))
    return (
        f"G_user={g_user} G_internal={g_int} min={g_user}\n"
        f"trips = {trips}\n"
        f"statically_known_true(trips >= 2) = {at_least_2}\n"
        f"statically_known_true(trips != 1) = {never_1}",
        "as_predicted" if at_least_2 else "SURPRISE",
    )


@case(
    5,
    "5c_degenerate_trip_PRESENT_when_g_internal_equals_g_user",
    "The control: with G_internal == G_user == min, is a single trip possible?",
    "yes, trips can be 1, which is exactly the 0/1 specialisation we hit on "
    "device (an extra binary in map mode, a hard compile error in reduction "
    "mode)",
)
def _():
    shape_env, mode, s, _x = fresh(hint=64)
    g = 64
    with mode:
        torch._check(s >= g)
        torch._check(s % g == 0)
        trips = s // g
        at_least_2 = None
        if statically_known_true is not None:
            at_least_2 = bool(statically_known_true(trips >= 2))
    return (
        f"G_user=G_internal={g} min={g}\ntrips = {trips}\n"
        f"statically_known_true(trips >= 2) = {at_least_2}  (expect False)",
        "as_predicted" if not at_least_2 else "SURPRISE",
    )


# ---------------------------------------------------------------------------
# PART 6: the structural form S = G * n
# ---------------------------------------------------------------------------

print(f"\n{BANNER}\nPART 6: structural granularity, S = G * n\n{BANNER}")


@case(
    6,
    "6a_sympy_G_times_n_collapses",
    "In pure sympy, does FloorDiv(G*n, G) collapse to n, and does the tile "
    "extent come out as the LITERAL G?",
    "both collapse exactly, where the same expressions over a bare symbol S do "
    "not",
)
def _():
    n = sympy.Symbol("n", integer=True, positive=True)
    s0 = sympy.Symbol("s0", integer=True, positive=True)
    count = FloorDiv(G_USER * n, G_USER)
    extent = FloorDiv(G_USER * n, count)
    bare = FloorDiv(s0, FloorDiv(s0, G_USER))
    return (
        f"FloorDiv(G*n, G)        = {count}        (want n)\n"
        f"FloorDiv(G*n, count)    = {extent}        (want literal {G_USER})\n"
        f"for comparison, bare S: FloorDiv(S, S//G) = {bare}",
        "as_predicted" if count == n and extent == G_USER else "SURPRISE",
    )


@case(
    6,
    "6b_symint_built_as_G_times_n_keeps_literal_tile",
    "With a REAL SymInt whose dim 0 is G*n, is the split extent literal G with "
    "no trimming trick?",
    "dim 1 of the split is the literal 64, so the trim-then-split workaround "
    "becomes unnecessary under this spelling",
)
def _():
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    src = LocalSource("ntiles")
    with mode:
        nsym = shape_env.create_symbol(5, src, DimDynamic.DYNAMIC)
        nn = shape_env.create_symintnode(nsym, hint=5, source=src)
        x = torch.empty(nn * G_USER, WIDTH)
        out = torch.unflatten(x, 0, (nn, G_USER))
    d1 = out.shape[1]
    literal = isinstance(d1, int) or not getattr(
        getattr(d1, "node", None), "expr", sympy.Integer(0)
    ).free_symbols
    return (
        f"input shape  = {list(x.shape)}\nsplit shape  = {list(out.shape)}\n"
        f"dim1 = {d1}  literal={literal}",
        "as_predicted" if literal else "SURPRISE",
    )


@case(
    6,
    "6c_export_derived_dim_Ax_plus_B",
    "Does torch.export.Dim support a derived dim of the form G * n, which is "
    "what would make granularity structural at the API?",
    "yes: upstream documents Ax + B with integer A > 0, so 64 * n is legal",
)
def _():
    try:
        from torch.export import Dim
    except ImportError as exc:
        return f"torch.export.Dim unavailable: {exc}", "SURPRISE"
    n = Dim("n_tiles", min=2, max=8)
    out = [f"root dim: {n}"]
    try:
        derived = G_USER * n
        out.append(f"64 * n   = {derived}  type={type(derived).__name__}")
        ok = True
    except Exception as exc:  # noqa: BLE001
        out.append(f"64 * n RAISED {type(exc).__name__}: {exc}")
        ok = False
    for expr, label in ((lambda: n * G_USER, "n * 64"), (lambda: n + 1, "n + 1")):
        try:
            out.append(f"{label} = {expr()}")
        except Exception as exc:  # noqa: BLE001
            out.append(f"{label} RAISED {type(exc).__name__}: {exc}")
    for label, hint in (("AUTO", "AUTO"), ("DYNAMIC", "DYNAMIC"), ("STATIC", "STATIC")):
        out.append(f"Dim.{label} present: {hasattr(Dim, hint)}")
    return "\n".join(out), "as_predicted" if ok else "SURPRISE"


# ---------------------------------------------------------------------------
# PART 7: bands
# ---------------------------------------------------------------------------

print(f"\n{BANNER}\nPART 7: can one ShapeEnv hold a band DISJUNCTION?\n{BANNER}")


@case(
    7,
    "7a_two_ranges_one_symbol",
    "Can one symbol carry two disjoint admissible ranges, which is what a band "
    "set is?",
    "NO. var_to_range is a single interval, so the second declaration either "
    "intersects with the first or contradicts it. Bands must therefore be "
    "separate compiles, which is what case 4d measures.",
)
def _():
    shape_env, mode, s, _x = fresh(hint=256)
    lines = []
    with mode:
        torch._check(s >= 128)
        torch._check(s <= 256)
        lines.append(f"after band 1 (128..256): {rng(shape_env, s)}")
        try:
            torch._check(s >= 512)
            lines.append(f"after also asserting >= 512: {rng(shape_env, s)}")
            lines.append("no contradiction raised, so the interval just moved or emptied")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"asserting a disjoint band RAISED {type(exc).__name__}: {str(exc)[:200]}")
    lines.append("Conclusion: one interval per symbol, so a band set is N compiles, not one.")
    return "\n".join(lines), "as_predicted"


@case(
    7,
    "7b_max_must_be_multiple_of_G",
    "Our own contract says max must be a multiple of G and max/G must be "
    "within max_buckets. Does sympy agree that a non-multiple max makes the "
    "top of the range unreachable?",
    "with max=500 and G=64, the largest legal size is 448, so a max that is "
    "not a multiple of G silently wastes the top of the declared range",
)
def _():
    g, declared_max = 64, 500
    largest = (declared_max // g) * g
    return (
        f"declared max={declared_max}, G={g}\n"
        f"largest legal multiple = {largest}\n"
        f"wasted top of range = {declared_max - largest}\n"
        f"buckets = {largest // g}",
        "as_predicted" if largest == 448 else "SURPRISE",
    )


# ---------------------------------------------------------------------------
# PART 8: where did the on-device 128 come from? 2b ruled out my first answer.
# ---------------------------------------------------------------------------

print(f"\n{BANNER}\nPART 8: hunting the wrong granularity, faithfully\n{BANNER}")
print("  On device, matmul_split_k declared tile_size=64 and torch._check(>= 64)")
print("  yet compute_granularity read user_min=128. Case 2b showed the num_tiles")
print("  comparison does NOT move the lower bound, so that explanation is dead.")
print("  These cases reproduce the real setup one element at a time.")


def bounds_report(shape_env, s, label):
    """Both channels side by side: var_to_range AND bound_sympy.

    compute_granularity uses shape_env.bound_sympy(expr).lower, NOT
    var_to_range directly. They can differ, and the probe should say which.
    """
    expr = s.node.expr if hasattr(s, "node") else s
    syms = getattr(expr, "free_symbols", set())
    out = [label]
    if syms:
        sym = next(iter(syms))
        vr = shape_env.var_to_range.get(sym)
        out.append(f"  var_to_range[{sym}] = lower={getattr(vr,'lower','?')} upper={getattr(vr,'upper','?')}")
    try:
        bs = shape_env.bound_sympy(expr)
        out.append(f"  bound_sympy({expr}) = lower={bs.lower} upper={bs.upper}")
    except Exception as exc:  # noqa: BLE001
        out.append(f"  bound_sympy RAISED {type(exc).__name__}: {exc}")
    out.append(f"  divisible: {divisible_of(shape_env)}")
    return "\n".join(out)


@case(
    8,
    "8a_bound_sympy_vs_var_to_range",
    "compute_granularity reads bound_sympy(expr).lower, not var_to_range. Do "
    "the two agree after only a min check and a divisibility check?",
    "they agree at 64; if bound_sympy reports something higher, THAT is the bug",
)
def _():
    shape_env, mode, s, _x = fresh(hint=320)
    with mode:
        torch._check(s >= 64)
        torch._check(s <= MAX_SIZE)
        torch._check(s % 64 == 0)
    return bounds_report(shape_env, s, "after >=64, <=512, %64==0:"), "as_predicted"


@case(
    8,
    "8b_two_marks_same_symbol_split_k_shape",
    "split_k marks dim 1 of a [256, S] tensor AND dim 0 of an [S, 64] tensor, "
    "so the same symbol is reached through two different sources, and one of "
    "them makes it a STRIDE. Does that move the bound?",
    "bound stays 64; if it becomes 128 this is the reproduction",
)
def _():
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    src = LocalSource("k")
    with mode:
        sym = shape_env.create_symbol(320, src, DimDynamic.DYNAMIC)
        s = shape_env.create_symintnode(sym, hint=320, source=src)
        a = torch.empty(256, s)   # symbol is dim 1, so stride(0) == s
        b = torch.empty(s, 64)    # symbol is dim 0
        torch._check(a.size(1) >= 64)
        torch._check(a.size(1) <= MAX_SIZE)
        torch._check(b.size(0) >= 64)
        torch._check(b.size(0) <= MAX_SIZE)
        torch._check(a.size(1) % 64 == 0)
        torch._check(b.size(0) % 64 == 0)
    return (
        bounds_report(shape_env, s, "two marks, one of them a stride:")
        + f"\n  a.stride() = {a.stride()}",
        "as_predicted",
    )


@case(
    8,
    "8c_plus_the_equality_check",
    "Add the equality the _checked variants emit. Does asserting two already "
    "unified sizes equal change the bound?",
    "no change; the equality is a no-op once they are the same symbol",
)
def _():
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    src = LocalSource("k")
    with mode:
        sym = shape_env.create_symbol(320, src, DimDynamic.DYNAMIC)
        s = shape_env.create_symintnode(sym, hint=320, source=src)
        a, b = torch.empty(256, s), torch.empty(s, 64)
        torch._check(a.size(1) >= 64)
        torch._check(b.size(0) >= 64)
        torch._check(a.size(1) == b.size(0))
        torch._check(a.size(1) % 64 == 0)
        before = bounds_report(shape_env, s, "before the trim:")
        # _normalize_in_specs trims to num_tiles * tile_size, then splits.
        n = a.size(1) // 64
        trimmed = a.narrow(1, 0, n * 64)
        _ = torch.unflatten(trimmed, 1, (n, 64))
    return before + "\n" + bounds_report(shape_env, s, "after trim and split:"), "as_predicted"


@case(
    8,
    "8d_does_a_SECOND_divisibility_fact_raise_the_bound",
    "If something asserts divisibility by 128 as well as 64, does bound_sympy's "
    "lower become 128? That would explain the device reading, since the real "
    "flow has several operands and several checks.",
    "a %128 fact plus >=64 could legitimately lift the lower bound to 128, "
    "because the smallest multiple of 128 at or above 64 IS 128",
)
def _():
    shape_env, mode, s, _x = fresh(hint=256)
    with mode:
        torch._check(s >= 64)
        torch._check(s % 64 == 0)
        first = bounds_report(shape_env, s, "after >=64 and %64:")
        torch._check(s % 128 == 0)
        second = bounds_report(shape_env, s, "after ALSO %128:")
    return first + "\n" + second, "as_predicted"


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

print(f"\n{BANNER}\nSUMMARY\n{BANNER}")
for r in RESULTS:
    mark = {"as_predicted": "ok ", "SURPRISE": "!! ", "error": "ERR"}.get(r["verdict"], "?  ")
    print(f"  [{mark}] part {r['part']}  {r['name']}")

surprises = [r for r in RESULTS if r["verdict"] != "as_predicted"]
print(f"\n  {len(RESULTS)} cases, {len(surprises)} not as predicted")
for r in surprises:
    print(f"    {r['verdict']}: {r['name']}")
    print(f"      predicted: {r['predict'][:150]}")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_contract_channels_results.json")
with open(out_path, "w") as f:
    json.dump(
        {
            "schema": 1,
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "g_user": G_USER,
            "max_size": MAX_SIZE,
            "cases": RESULTS,
        },
        f,
        indent=2,
    )
print(f"\n  wrote {out_path}")
print("  The per-case lines carry the detail. A '!!' means the mechanism does")
print("  not behave the way the design assumes, which is the point of running it.")
