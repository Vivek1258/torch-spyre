"""Which guards does the output-side flatten add, and can a spelling avoid them?

Pure PyTorch. No torch_spyre import, no device, no compile, so it needs NO
reinstall -- just `python tests/inductor/probe_output_flatten_guards.py`.

Background. for_each_tile's `_stacked_to_full` folds scan's stacked output
[count, extent, W] down to [count*extent, W] with `flatten(0, 1)`. Under a
symbolic count that flatten installs guards, and one of them showed up in a real
ConstraintViolationError as:

    (L['a'].size()[0] // 64) != 1

which makes the compiled artifact valid only for count != 1, so a 1-tile call
recompiles. Measured: 64 rows is CORRECT but on a second binary.

Two more appeared alongside it, both always-true-but-unprovable:

    0 <= S - 64*(S//64)                        <- the narrow's own bounds check
    ((S//64) % (64*(S//64))) != 0

This asks torch directly which spelling of the fold adds which guards, so a fix
is chosen from evidence rather than from a guess about the reshape helper.
"""

import traceback

import torch
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

G = 64
S_HINT = 320
MAX_ROWS = 512
WIDTH = 128

print(f"torch {torch.__version__}")

RESULTS = []


def fresh():
    """A fake [S, 128] with dim 0 symbolic and the range we actually declare."""
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    src = LocalSource("a")
    with mode:
        sym = shape_env.create_symbol(S_HINT, src, DimDynamic.DYNAMIC)
        s = shape_env.create_symintnode(sym, hint=S_HINT, source=src)
        x = torch.empty(s, WIDTH)
        # Same way the probe declares it, so the guard set is comparable.
        torch._check(s >= G)
        torch._check(s <= MAX_ROWS)
    return shape_env, mode, s, x


def guard_strs(shape_env) -> list:
    out = []
    for g in getattr(shape_env, "guards", []) or []:
        expr = getattr(g, "expr", g)
        out.append(str(expr))
    return out


def case(name, fold):
    """Build the stacked ys, fold it, and report only the NEW guards."""
    print("\n" + "=" * 78)
    print(f"CASE {name}")
    shape_env, mode, s, x = fresh()
    try:
        with mode:
            count = s // G
            # What scan hands back: [count, extent, W], stacked on dim 0.
            ys = torch.empty(count, G, WIDTH)
            before = set(guard_strs(shape_env))
            out = fold(ys, count)
            new = [g for g in guard_strs(shape_env) if g not in before]
        print(f"  result shape = {list(out.shape)}")
        print(f"  new guards ({len(new)}):")
        for g in new:
            print(f"    {g}")
        suspicious = [g for g in new if "!= 1" in g or "Ne" in g]
        RESULTS.append((name, list(out.shape), len(new), suspicious))
    except Exception as exc:  # noqa: BLE001
        print(f"  RAISED {type(exc).__name__}: {exc}")
        print("  " + "\n  ".join(traceback.format_exc().splitlines()[-3:]))
        RESULTS.append((name, "RAISED", -1, [str(exc)]))


# What for_each_tile does today.
case("A flatten(0, 1)  (today)", lambda ys, n: ys.flatten(0, 1))

# Spellings that might avoid the per-dim walk in the reshape helper.
case("B reshape(-1, W)", lambda ys, n: ys.reshape(-1, WIDTH))
case("C view(n*G, W)", lambda ys, n: ys.view(n * G, WIDTH))
case("D as_strided((n*G, W), (W, 1))  -- sizes taken verbatim",
     lambda ys, n: ys.as_strided((n * G, WIDTH), (WIDTH, 1)))

# Did the as_strided branch in _stacked_to_full even FIRE? It is gated on the
# leading axes being PROVABLY contiguous, and if that is not provable we fall
# back to flatten and nothing changed. Measured, not assumed.
print("\n" + "=" * 78)
print("CASE E  is the contiguity predicate provable, i.e. did the fix fire?")
shape_env, mode, s, x = fresh()
with mode:
    count = s // G
    ys = torch.empty(count, G, WIDTH)
    pred = ys.stride(0) == ys.size(1) * ys.stride(1)
    print(f"  ys shape  = {list(ys.shape)}")
    print(f"  ys stride = {list(ys.stride())}")
    print(f"  stride(0)={ys.stride(0)}  size(1)*stride(1)={ys.size(1) * ys.stride(1)}")
    print(f"  raw predicate = {pred!r} (type {type(pred).__name__})")
    try:
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        provable = bool(statically_known_true(pred))
    except Exception as exc:  # noqa: BLE001
        provable = f"raised {exc!r}"
    print(f"  statically_known_true -> {provable}")
RESULTS.append(("E contiguity provable (did the fix fire)", "-", 0 if provable is True else -1,
                [] if provable is True else [f"NOT provable: {provable}"]))

# The OTHER side. _xs_leaf trims then splits [G*n, W] into [n, G, W], which also
# walks the dims and may install the same Ne(count, 1). If it does, fixing only
# the output fold cannot remove the one-tile recompile, which is what we saw.
def xs_case(name, split):
    print("\n" + "=" * 78)
    print(f"CASE {name}")
    shape_env, mode, s, x = fresh()
    try:
        with mode:
            count = s // G
            trimmed = x.narrow(0, 0, count * G)
            before = set(guard_strs(shape_env))
            out = split(trimmed, count)
            new = [g for g in guard_strs(shape_env) if g not in before]
        print(f"  result shape = {list(out.shape)}")
        print(f"  new guards ({len(new)}):")
        for g in new:
            print(f"    {g}")
        suspicious = [g for g in new if "!= 1" in g or "Ne" in g]
        RESULTS.append((name, list(out.shape), len(new), suspicious))
    except Exception as exc:  # noqa: BLE001
        print(f"  RAISED {type(exc).__name__}: {exc}")
        RESULTS.append((name, "RAISED", -1, [str(exc)]))


xs_case("X1 narrow + unflatten(0, (n, G))  (today)",
        lambda tr, n: torch.unflatten(tr, 0, (n, G)))
xs_case("X2 narrow + as_strided((n, G, W), (G*W, W, 1))",
        lambda tr, n: tr.as_strided((n, G, WIDTH), (G * WIDTH, WIDTH, 1)))

# WHO calls check_contiguous_sizes_strides? The real recompile names it:
#   (a.size()[0] // 64) != 1  # _prims_common/__init__.py:285 in
#                               check_contiguous_sizes_strides
# as_strided added nothing in isolation (case D), so the likely caller is the
# empty_strided that upstream's decompose_scan_to_while_loop uses to
# pre-allocate scan's output buffer. If that is where it comes from, the guard
# is not ours to remove at all.
def alloc_case(name, build):
    print("\n" + "=" * 78)
    print(f"CASE {name}")
    shape_env, mode, s, x = fresh()
    try:
        with mode:
            count = s // G
            before = set(guard_strs(shape_env))
            out = build(count)
            new = [g for g in guard_strs(shape_env) if g not in before]
        print(f"  result shape = {list(out.shape)}")
        print(f"  new guards ({len(new)}):")
        for g in new:
            print(f"    {g}")
        suspicious = [g for g in new if "Ne" in g]
        RESULTS.append((name, list(out.shape), len(new), suspicious))
    except Exception as exc:  # noqa: BLE001
        print(f"  RAISED {type(exc).__name__}: {exc}")
        RESULTS.append((name, "RAISED", -1, [str(exc)]))


alloc_case("P1 empty_strided((n, G, W), (G*W, W, 1))  <- what scan pre-allocates",
           lambda n: torch.empty_strided((n, G, WIDTH), (G * WIDTH, WIDTH, 1)))
alloc_case("P2 empty((n, G, W))  <- for comparison",
           lambda n: torch.empty(n, G, WIDTH))

print("\n" + "=" * 78)
print("SUMMARY  (want: correct shape, and NO '!= 1' guard)")
for name, shape, n_new, suspicious in RESULTS:
    flag = "OK  " if n_new >= 0 and not suspicious else "    "
    print(f"  {flag}{name}")
    print(f"        shape={shape}  new_guards={n_new}")
    for g in suspicious:
        print(f"        SUSPICIOUS: {g}")
print(
    "\n  A spelling with no '!= 1' guard and the right shape is the fix for the\n"
    "  1-tile recompile. If every spelling adds it, the guard is inherent to\n"
    "  folding a symbolic leading dim and the recompile is the cost of it."
)
