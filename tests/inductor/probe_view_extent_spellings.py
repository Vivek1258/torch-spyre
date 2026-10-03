"""Which spelling of the xs split keeps the tile extent a literal under a symbolic length?

Pure PyTorch. No torch_spyre import, no spyre device, no compile. So it needs NO
reinstall -- just `python tests/inductor/probe_view_extent_spellings.py`.

Background: for_each_tile's _xs_leaf splits dim 0 into (count, extent). Under a
symbolic length S the extent comes out as S//(S//64) instead of the literal 64,
and every pass below the splice needs a concrete tile. This asks torch directly
which spelling, if any, keeps it literal, and whether stating divisibility helps.
"""

import traceback

import sympy
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.utils._sympy.functions import FloorDiv

try:
    from torch.fx.experimental.symbolic_shapes import statically_known_true
except ImportError:  # older torch
    statically_known_true = None

from torch._dynamo.source import LocalSource

G = 64          # tile extent
S_HINT = 320    # 5 tiles, same as the e2e probe
WIDTH = 128

print(f"torch {torch.__version__}")

RESULTS = []


def _fresh_via_create_symbol():
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    src = LocalSource("x")
    with mode:
        sym = shape_env.create_symbol(S_HINT, src, DimDynamic.DYNAMIC)
        s = shape_env.create_symintnode(sym, hint=S_HINT, source=src)
        x = torch.empty(s, WIDTH)
    return shape_env, mode, s, x


def _fresh_via_from_tensor():
    from torch.fx.experimental.symbolic_shapes import StatelessSymbolicContext

    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    real = torch.empty(S_HINT, WIDTH)
    ctx_kwargs = {"dynamic_sizes": [DimDynamic.DYNAMIC, DimDynamic.STATIC]}
    try:
        ctx = StatelessSymbolicContext(**ctx_kwargs)
    except TypeError:
        ctx_kwargs["dynamic_strides"] = [DimDynamic.INFER_STRIDE] * 2
        ctx = StatelessSymbolicContext(**ctx_kwargs)
    with mode:
        x = mode.from_tensor(real, source=LocalSource("x"), symbolic_context=ctx)
    return shape_env, mode, x.shape[0], x


_STRATEGY = None


def fresh():
    """A fake [S, 128] with dim 0 a backed dynamic symbol, hint 320."""
    global _STRATEGY
    if _STRATEGY is not None:
        return _STRATEGY()
    errors = []
    for fn in (_fresh_via_create_symbol, _fresh_via_from_tensor):
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(out[2], int):
            _STRATEGY = fn
            print(f"symbolic tensor built via {fn.__name__}, dim0 = {out[2]}")
            return out
        errors.append(f"{fn.__name__}: dim0 came back concrete ({out[2]})")
    raise RuntimeError("could not build a symbolic fake tensor:\n  " + "\n  ".join(errors))


def verdict(shape_env, out):
    """Is dim 1 the literal G?"""
    if out is None:
        return "n/a"
    d1 = out.shape[1]
    if isinstance(d1, int):
        return f"LITERAL {d1}" if d1 == G else f"int {d1}"
    expr = d1.node.expr
    if expr == G:
        return f"LITERAL {expr}"
    simp = None
    try:
        simp = shape_env.simplify(expr)
    except Exception as exc:  # noqa: BLE001
        simp = f"<simplify raised {exc!r}>"
    skt = None
    if statically_known_true is not None:
        try:
            skt = bool(statically_known_true(d1 == G))
        except Exception as exc:  # noqa: BLE001
            skt = f"<raised {exc!r}>"
    return f"SYMBOLIC {expr}  | simplify -> {simp} | statically_known_true(==G) -> {skt}"


def case(name, body, *, check_divisible=False):
    print("\n" + "=" * 78)
    print(f"CASE {name}   (torch._check divisibility first = {check_divisible})")
    out = None
    shape_env = None
    try:
        shape_env, mode, s, x = fresh()
        with mode:
            if check_divisible:
                torch._check(s % G == 0)
                print(f"  after _check: divisible={shape_env.divisible}")
            out = body(s, x)
        print(f"  shape  = {list(out.shape)}")
        print(f"  stride = {list(out.stride())}")
    except Exception as exc:  # noqa: BLE001
        print(f"  RAISED {type(exc).__name__}: {exc}")
        print("  " + "\n  ".join(traceback.format_exc().splitlines()[-4:]))
    v = verdict(shape_env, out) if shape_env is not None else "setup failed"
    print(f"  dim1 -> {v}")
    if shape_env is not None:
        print(f"  divisible={shape_env.divisible}")
    RESULTS.append((name, check_divisible, v))


# The two spellings the POC has actually shipped.
case("A upstream  unflatten(x, 0, (S//G, G))", lambda s, x: torch.unflatten(x, 0, (s // G, G)))
case("B HEAD      unflatten(x, 0, (-1, G))", lambda s, x: torch.unflatten(x, 0, (-1, G)))
case("C upstream + divisibility", lambda s, x: torch.unflatten(x, 0, (s // G, G)), check_divisible=True)
case("D HEAD + divisibility", lambda s, x: torch.unflatten(x, 0, (-1, G)), check_divisible=True)

# Other ways to say the same thing.
case("E view(S//G, G, W)", lambda s, x: x.view(s // G, G, WIDTH))
case("F reshape(-1, G, W)", lambda s, x: x.reshape(-1, G, WIDTH))
case("G as_strided -- takes sizes verbatim, no numel reconciliation",
     lambda s, x: x.as_strided((s // G, G, WIDTH), (G * WIDTH, WIDTH, 1)))
case("H as_strided + divisibility",
     lambda s, x: x.as_strided((s // G, G, WIDTH), (G * WIDTH, WIDTH, 1)), check_divisible=True)
case("I view with an explicitly multiplied count  view(S//G, G, W) after _check((S//G)*G==S)",
     lambda s, x: (torch._check((s // G) * G == s), x.view(s // G, G, WIDTH))[1])

# The narrow trick. Trim to n*G FIRST, so the length the view is handed is
# structurally a multiple of G. Then the trailing split factor is
# FloorDiv(G*n, n), which collapses by gcd without any prover involved.
# narrow is a view, so no copy, and the tail it drops is the ragged tail
# _normalize_in_specs already refuses to accept.
def _narrow(s, x):
    n = s // G
    trimmed = x.narrow(0, 0, n * G)
    print(f"  trimmed dim0 = {trimmed.shape[0]}  (want 64*(S//64), NOT S)")
    return n, trimmed


case("L narrow(0,0,(S//G)*G) then unflatten(0, (S//G, G))",
     lambda s, x: (lambda n, tr: torch.unflatten(tr, 0, (n, G)))(*_narrow(s, x)))
case("M narrow then unflatten(0, (-1, G))",
     lambda s, x: (lambda n, tr: torch.unflatten(tr, 0, (-1, G)))(*_narrow(s, x)))
case("N narrow then view(S//G, G, W)",
     lambda s, x: (lambda n, tr: tr.view(n, G, WIDTH))(*_narrow(s, x)))
case("O narrow + divisibility then unflatten(0, (S//G, G))",
     lambda s, x: (lambda n, tr: torch.unflatten(tr, 0, (n, G)))(*_narrow(s, x)),
     check_divisible=True)

# Does a count-shaped symbol dissolve the problem? This is the `granularity as a
# derived dim` idea: introduce n and let S be G*n rather than marking S itself.
print("\n" + "=" * 78)
print("CASE J  pure sympy: is FloorDiv(G*n, FloorDiv(G*n, G)) provably G?")
n = sympy.Symbol("n", integer=True, positive=True)
inner = FloorDiv(G * n, G)
outer = FloorDiv(G * n, inner)
print(f"  FloorDiv(G*n, G)       = {inner}")
print(f"  FloorDiv(G*n, inner)   = {outer}")
print(f"  equals G? {outer == G}")
s0 = sympy.Symbol("s0", integer=True, positive=True)
print(f"  for comparison, FloorDiv(s0, FloorDiv(s0, G)) = {FloorDiv(s0, FloorDiv(s0, G))}")
RESULTS.append(("J sympy FloorDiv(G*n, G*n//G)", False, f"equals G: {outer == G}"))

# Same question but on the expression the narrow trick actually produces, where
# the count is itself a FloorDiv and not a clean symbol. This is the one that
# decides whether narrow works.
print("\n" + "=" * 78)
print("CASE J2  the real expression: is FloorDiv(G*(s0//G), s0//G) provably G?")
cnt = FloorDiv(s0, G)
collapsed = FloorDiv(G * cnt, cnt)
print(f"  count            = {cnt}")
print(f"  FloorDiv(G*cnt, cnt) = {collapsed}")
print(f"  equals G? {collapsed == G}")
print(f"  sympy.gcd(G*cnt, cnt) = {sympy.gcd(G * cnt, cnt)}")
RESULTS.append(("J2 sympy FloorDiv(G*(s0//G), s0//G)", False, f"equals G: {collapsed == G}"))

# Learn the mechanism from the installed torch, not from guesswork.
print("\n" + "=" * 78)
print("MECHANISM  where the trailing split factor is computed")
import inspect
try:
    src = inspect.getsource(FloorDiv.eval)
    keep = [ln for ln in src.splitlines() if "gcd" in ln or "return" in ln]
    print("  FloorDiv.eval, gcd and return lines:")
    for ln in keep[:14]:
        print("   ", ln.strip()[:150])
except Exception as exc:  # noqa: BLE001
    print(f"  could not read FloorDiv.eval: {exc}")
try:
    import torch._refs as refs

    src = inspect.getsource(refs._reshape_view_helper)
    hits = [(i, ln) for i, ln in enumerate(src.splitlines(), 1) if "//" in ln and ("length" in ln or "accum" in ln)]
    print(f"  _reshape_view_helper, lines doing a floor-divide ({len(hits)} found):")
    for i, ln in hits[:12]:
        print(f"    {i:>4} {ln.strip()[:140]}")
except Exception as exc:  # noqa: BLE001
    print(f"  could not read _reshape_view_helper: {exc}")

# And the same thing end to end: a tensor whose dim 0 IS G*n.
print("\n" + "=" * 78)
print("CASE K  tensor built as [G*n, W], then unflatten(0, (n, G))")
try:
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    src = LocalSource("nsym")
    with mode:
        nsym = shape_env.create_symbol(S_HINT // G, src, DimDynamic.DYNAMIC)
        nn = shape_env.create_symintnode(nsym, hint=S_HINT // G, source=src)
        x = torch.empty(nn * G, WIDTH)
        print(f"  input shape = {list(x.shape)}")
        out = torch.unflatten(x, 0, (nn, G))
    print(f"  shape  = {list(out.shape)}")
    print(f"  dim1 -> {verdict(shape_env, out)}")
    RESULTS.append(("K dim0 = G*n, unflatten(n, G)", False, verdict(shape_env, out)))
except Exception as exc:  # noqa: BLE001
    print(f"  RAISED {type(exc).__name__}: {exc}")
    RESULTS.append(("K dim0 = G*n", False, f"RAISED {exc}"))

print("\n" + "=" * 78)
print("SUMMARY  (want: dim1 LITERAL 64)")
for name, chk, v in RESULTS:
    flag = "OK  " if str(v).startswith("LITERAL") or "equals G: True" in str(v) else "    "
    print(f"  {flag}{name}{'  [+divisible]' if chk else ''}\n        {v}")
