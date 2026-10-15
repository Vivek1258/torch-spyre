# A symbol's journey: gelu over a declared dimension

One real function, followed through every stage to the device. Every graph, error and bundle listing
below was produced by running the code, not written by hand.

## The function

```python
MIN_ROWS, MAX_ROWS, TILE, COLS = 128, 512, 64, 64

def tiled_gelu(x):
    rows = x.shape[0]
    torch._check(rows >= MIN_ROWS)
    torch._check(rows <= MAX_ROWS)
    torch._check(rows % TILE == 0)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, torch.nn.functional.gelu(tile)

    _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=TILE, out_dim=0)
    return out
```

```python
x = torch.randn(rows, COLS, dtype=torch.float16)
on_device = x.to("spyre", dynamic={0: {"min": 128, "max": 512, "granularity": 64}})
out = torch.compile(tiled_gelu, dynamic=None)(on_device)
```

The claim: this compiles once and then runs correctly at 128, 256, 384 and 512 rows with no further
compiles.

---

## Stage 1. The transfer, and the birth of the symbol

`.to("spyre", dynamic={...})` does three separate things.

**Validate.** `_parse_dynamic_spec` checks exactly one dim is declared, that `min` and `max` are both
present and positive, that `min <= max`, that `max` is a multiple of the granularity, that `min` is
at or above the floor of 8, that `max / granularity` does not exceed 32 buckets, that the dim is in
range, and that the tensor's current size at that dim is itself inside the range and a multiple of
the granularity. Anything wrong raises a `ValueError` here, before any device work.

**Allocate at the ceiling.** `spyre_empty_reserved` allocates for `max` rows, not for the rows
passed, and allocates **max strided**, so every stride above the declared dim is computed as if the
dim were at its maximum. `copy_tensor` then copies the real data in.

**Mark the dimension dynamic.**

```python
if mark_dynamic:
    torch._dynamo.mark_dynamic(dst, spec.dim)
```

### Where the range actually comes from

The mark is **deliberately bare**, and the code says why:

> Passing `min=`/`max=` installs a `StrictMinMaxConstraint`, which promises every value in that range
> is valid. The granularity check the compiler adds later (`size % granularity == 0`) then looks like
> it is violating that promise for a non-multiple size, raising `ConstraintViolationError` instead of
> a normal guard miss.

So the symbol is created with PyTorch's default range: lower bound 2, **no finite upper bound**.

The declared range lives on the tensor, where `get_reserved_dims` can read it back, and the compiler
never sees it. The range the compiler uses comes from the three lines inside the function:

```python
torch._check(rows >= MIN_ROWS)
torch._check(rows <= MAX_ROWS)   # <- the load-bearing one
torch._check(rows % TILE == 0)
```

Without a finite upper bound there is no maximum to build geometry against, and the kernel silently
specialises to one size instead.

**This is the biggest honest gap in the current design.** Today the caller states the range twice:
once in the `dynamic=` dict for the allocator, once in `torch._check` for the compiler. The follow-up
work is a reader that takes the declaration off the graph input and injects the range, so the three
lines disappear. The same gap is why a second region further down a chain refuses: the symbol crosses
the region boundary and the range does not.

---

## Stage 2. Tracing, and the real graph

Dynamo runs the function once with a frame-evaluation hook and records the operations. Python control
flow does not survive: an `if` is executed during the trace and only the taken branch appears.
Whatever the trace relied on is recorded as a guard.

**This is the actual graph Dynamo produced**, trimmed to the relevant lines:

```
floordiv: "Sym((s77//64))" = s77 // 64
mul:      "Sym(64*((s77//64)))" = floordiv * 64
narrow:   "f32[64*((s77//64)), 64]" = l_x_.narrow(0, 0, mul)
xs:       "f32[(s77//64), 64, 64]"  = narrow.unflatten(0, (floordiv, 64))

init:     "i64[]" = torch.zeros((), dtype=torch.int64)
movedim:  "f32[(s77//64), 64, 64]" = torch.movedim(xs, 0, 0)

scan_combine_fn_0 = self.scan_combine_fn_0
scan = torch.ops.higher_order.scan(scan_combine_fn_0, [init], [movedim], [])
out:      "f32[(s77//64), 64, 64]" = scan[1]
flatten:  "f32[64*((s77//64)), 64]" = out.flatten(0, 1)
return (flatten,)
```

Four things to read off it.

`x.shape[0]` is `s77`, not 512. The trip count is `s77 // 64`, carried in the type as
`Sym((s77//64))`.

The **view spelling** is visible: `narrow` to `64*(s77//64)` first, then `unflatten` into
`[(s77//64), 64, 64]`. Trim before splitting. Split first and the extent comes out as `S // (S // G)`
and never simplifies back to 64.

`scan` is a `torch.ops.higher_order` node. The body is not inlined, it is a **subgraph reference**,
`scan_combine_fn_0`.

And the body itself:

```
class scan_combine_fn_0(torch.nn.Module):
    def forward(self, child: "i64[]", child_1: "f32[64, 64]"):
        child_2: "i64[]"     = child.clone()
        child_3: "f32[64, 64]" = torch._C._nn.gelu(child_1)
        return [child_2, child_3]
```

**The tile is `f32[64, 64]`. Fully static. No symbol anywhere inside the body.** That single fact
causes the upstream bug three stages from now, so it is worth noticing here.

---

## Stage 3. Why there has to be a higher-order operator

If the tiling were a Python `for` loop, tracing would unroll it: eight tiles would become eight copies
of the gelu body and the graph would be specialised to eight. That is what we are trying to escape.

A higher-order operator takes **another graph** as an argument. The body stays a subgraph and the
iteration count stays a **value** in the graph instead of being baked into its structure. A loop that
survives tracing as a loop is the precondition for a symbolic trip count.

**`for_each_tile` is not itself a higher-order operator.** It is an ordinary Python function in
`wsr/for_each_tile.py`, and its docstring puts it exactly: it is `scan` with the tiling made explicit.
It builds a `TileSpec` per operand, reshapes each into an `xs` leaf of shape `[num_tiles, ...]`, and
calls upstream's `scan`:

```python
final_carry, ys = scan(combine_fn, scan_init, xs, dim=0, reverse=reverse)
```

### Two real restrictions, both hit while producing the listings above

**No aliasing.** Returning the carry unchanged gives:

```
UncapturedHigherOrderOpError: scan must be captured completely with torch.compile.
  Encountered aliasing during higher order op tracing
  Hint: Replace `return input` with `return input.clone()` to avoid aliasing.
```

**A Python float in the body is refused.** Anything the body closes over is lifted into the subgraph
as an extra input, and a lifted Python float becomes a `SymFloat`, which the operator's validation
does not accept. A multiply by `1.0` folds away and hides it, `0.5` does not. This is why every
compiled block in the encoder adapters is blocked: they all pass an attention scale explicitly.

---

## Stage 4. Post-grad, and an upstream defect

The post-grad graph, again produced by running it:

```
%arg0_1   : placeholder                      <- s77 arrives as a graph input
%arg1_1   : placeholder                      <- the tensor
%floordiv : operator.floordiv(%arg0_1, 64)   <- the trip count
%mul      : operator.mul(%floordiv, 64)
%slice_1  : aten.slice.Tensor(%arg1_1, 0, 0, %mul)
%view     : aten.reshape.default(%slice_1, [%floordiv, 64, 64])
%permute  : aten.permute.default(%view, [0, 1, 2])
%scan     : ...
```

and the body, now decomposed to aten, with gelu expanded:

```
%clone : aten.clone.default(%arg0_1)
%mul   : aten.mul.Tensor(%arg1_1, 0.5)
%mul_1 : aten.mul.Tensor(%arg1_1, 0.7071067811865476)
%erf   : aten.erf.default(%mul_1)
%add   : aten.add.Tensor(%erf, 1)
%mul_2 : aten.mul.Tensor(%mul, %add)
return [clone, mul_2]
```

Still no symbol in the body. Every shape there is concrete.

### The upstream defect, reproduced in pure PyTorch

Inductor then lowers `scan` to a `while_loop`. That step rebuilds the scan output's shape, and here
is what it does:

```python
bound_symbols = {
    arg.node.expr: arg
    for arg in pytree.tree_leaves((args, scan_length))
    if isinstance(arg, torch.SymInt)
}
ys_outs = [torch.empty_strided(resolve_shape_to_proxy(ys_out.size(), bound_symbols), ...)]
```

The key is `arg.node.expr`. For the scan length that expression **is** `FloorDiv(s77, 64)`, so the
dictionary's one key is a compound expression.

`resolve_shape_to_proxy` hands that dictionary to `sympy_interp`, whose signature types it as
`dict[sympy.Symbol, Any]` and which consults it in exactly one place:

```python
elif isinstance(expr, sympy.Symbol):
    if (r := env.get(expr, _nil)) is not _nil:
        return r
    elif missing_handler:
        return missing_handler(expr)
    else:
        raise KeyError(expr)
```

**The lookup only ever happens at a bare `Symbol` leaf.** So for the output shape entry
`FloorDiv(s77, 64)`:

1. not a `Symbol`, so no lookup. Recurse into the arguments.
2. reach `s77`, which is a `Symbol`, so look that up.
3. `s77` is not a key. The only key is `FloorDiv(s77, 64)`.
4. `raise KeyError(s77)`.

The dictionary held the exact answer and was never asked for it.

**Why `s77` is not also a key** is what makes this ours to find. Upstream's own comment says free
symbols of tensor shapes are lifted as subgraph inputs by Dynamo. True when the body needs them. Our
body is fully static, as Stage 2 showed, so `s77` is never lifted, and the only SymInt in the pytree
is the scan length whose expression is the compound. The defect needs exactly a static body plus a
trip count derived from a dynamic dim, which is the shape of a symbolic loop bound.

Running it gives, verbatim:

```
File ".../_inductor/fx_passes/post_grad.py", line 567, in lower_to_while_loop
File ".../_inductor/fx_passes/post_grad.py", line 569, in <listcomp>
    resolve_shape_to_proxy(ys_out.size(), bound_symbols),
File ".../_inductor/fx_passes/post_grad.py", line 451, in resolve_shape_to_proxy
    sympy_interp(
File ".../utils/_sympy/interp.py", line 214, in <listcomp>
File ".../utils/_sympy/interp.py", line 208, in sympy_interp
    raise KeyError(expr)
KeyError: s77
```

**This reproduces in about twenty lines of pure PyTorch with no Spyre in it**, which is what makes it
fileable upstream:

```python
import torch
from torch._higher_order_ops.scan import scan

def tiled_gelu(x):
    rows = x.shape[0]
    torch._check(rows >= 128); torch._check(rows <= 512); torch._check(rows % 64 == 0)
    n = rows // 64
    xs = x.narrow(0, 0, n * 64).unflatten(0, (n, 64))
    def combine(carry, tile):
        return carry.clone(), torch.nn.functional.gelu(tile)
    _, ys = scan(combine, torch.zeros((), dtype=torch.int64), xs, dim=0)
    return ys.flatten(0, 1)

x = torch.randn(256, 64)
torch._dynamo.mark_dynamic(x, 0)
torch.compile(tiled_gelu, fullgraph=True, dynamic=None)(x)   # KeyError: s77
```

**Our patch** matches the whole expression before upstream's symbol walk, one element at a time so
anything else keeps upstream's own type checking and messages:

```python
expr = getattr(getattr(size, "node", None), "expr", None)
if expr is not None and expr in bound_symbols:
    resolved.append(bound_symbols[expr])
else:
    resolved.append(original([size], bound_symbols)[0])
```

With the patch applied the same script gets past it, and then stops on the next upstream gate, the
`.item()` the lowering calls on its scalar induction variable, which `for_each_tile` handles by
permitting scalar outputs on that mode.

---

## Stage 5. `splice_while_loops`, and why it has to exist

torch-spyre hooks into Inductor at four places, wired in `patches.py`:

| hook | when |
|---|---|
| `pre_grad_custom_pass` | before autograd |
| `post_grad_custom_pre_pass` | after autograd, before upstream's post-grad passes |
| `post_grad_custom_post_pass` | after them |
| pre-scheduling | ours, injected by wrapping `GraphLowering._update_scheduler` |

**`splice_while_loops` is the first pass in the pre-scheduling list.** It removes the `while_loop`
node entirely, inlines its body as ordinary Inductor IR, and stamps `loop_info` on each op so later
passes know it sits inside a loop level.

What happens if it does not run is not theoretical. Push the same script through the CPU path, where
that pass does not exist, and the while_loop survives to subgraph wrapper codegen:

```
AssertionError: For (s77//64), expected [s77] to have been codegen-ed.
  torch/_inductor/codegen/wrapper.py  _verify_input_symbol_assignment
```

That is the pass earning its place. It is also why one of our early tests was wrong: it ran on the
CPU fixture path and hit this, which looks like a backend failure and is actually a path we never
take.

### IR, then OpSpec, then SDSC

The splice works at the **IR and scheduler** level. OpSpecs do not exist yet. They are built later,
during kernel codegen in `spyre_kernel.py`. The full chain:

```
while_loop node
  -> splice_while_loops      -> inlined IR + loop_info stamps
  -> scheduling and codegen  -> OpSpecs, wrapped in a LoopSpec
  -> one SDSC JSON per OpSpec, plus bundle.mlir carrying the loop
```

---

## Stage 6. The LoopSpec

```python
@dataclass
class LoopSpec:
    count                  # trip count, may be a symbolic expression
    body                   # OpSpec, UnimplementedOp, or a nested LoopSpec
    count_symbol_bounds    # {name: (max_value, tile_size)}
    count_symbol_sources   # {name: (arg_index, dim_index)}
```

For our gelu region it is, in the form the tests construct it:

```python
LoopSpec(
    count=FloorDiv(s77, 64),
    body=[...],                              # the OpSpecs for the tile's work
    count_symbol_bounds={"s77": (512, 64)},  # max_value, tile_size
    count_symbol_sources={"s77": (0, 0)},    # read arg 0's dim 0 at launch
)
```

A nested same-dimension loop, measured on device, comes out as:

```python
inner = LoopSpec(count=sympy.Integer(2), body=[])          # literal
outer = LoopSpec(count=FloorDiv(s77, 64), body=[inner])    # symbolic
```

Exactly one symbolic count, and the inner level is a constant, because the inner level is handed the
enclosing level's tile. That is the whole Phase 2 answer in two lines.

The three fields carry three different jobs:

**`count`** is the expression. **`count_symbol_bounds`** is resolved by the scheduler while the
ShapeEnv exists, and carried, because codegen also runs in a reload phase where it is gone. It is
keyed by the symbol's **name**, never the object, because a reload builds a fresh `Symbol('s77')`
without the original's assumptions and sympy counts assumptions as part of identity.
**`count_symbol_sources`** is resolved during kernel codegen while the launch argument ordering is
still live, and **it is what becomes the kDimension payload**. That field is the join between the
compiler and the launch slot.

---

## Stage 7. Recovery

`decompose_tiled_count` reads `FloorDiv(s77, 64)` back apart into `(s77, 64)`. This is how the
granularity is **recovered** rather than inferred. Because `for_each_tile(tile_size=64)` put the 64
into the count in the first place, the granularity the bundle declares and the step the loop takes
come from one expression and cannot disagree.

It accepts both spellings of the division at this one point, because the kernel serializer writes
expressions as `sympify('<str>')`, `str(FloorDiv(s, 64))` prints as `(s//64)`, and sympy re-parses
that into `floor(s/64)`, a different type.

`symbolic_count_bounds` then asks the ShapeEnv for the finite upper bound, gets 512 from the
`torch._check`, and stores it. No finite bound means an empty map, a warning, and a kernel that
specialises: a worse binary, not a wrong one.

---

## Stage 8. Geometry at the declared maximum

This is not one place, and neither "the SDSC" nor "the bundle loop count" is the answer alone.

The rule: **everything that describes capacity is built at the declared maximum, and exactly one
number is runtime.**

The SDSC describes one tile's compute, and separately it has to describe how to walk the operand
across tiles. That walk needs two numbers, set in `superdsc.py`:

```python
tile_size  = int(coeff) * arg_elem_bytes
trip_count = op_spec.tiled_symbol_trip_counts.get(sym, 1)
sdsc_dim_advance[sdsc_sym] = (tile_size, trip_count)
```

`trip_count` comes from `max_trip_count`, which takes the ShapeEnv **upper bound** and never the
hint. For our example that is 8, not 2.

> **The SDSC describes how far the walk can reach. The bundle decides how many steps actually run.**

The SDSC's 8 describes the reserved buffer. The bundle's loop bound, fed by the kDimension slot, says
run 2 this time. Confusing the two is exactly the bug `max_trip_count` warns about: baking one call's
size into the SDSC specialises it while the binary still claims to serve the whole range, and that is
invisible at the warm-up size.

Everywhere the maximum lands:

| Place | What is at the maximum |
|---|---|
| HBM allocation | the buffer, 512 rows, max strided |
| the tensor's device layout | `device_size` reflects the reservation |
| SDSC, per tiled symbol | `(tile_size_bytes, max_trip_count)`, the dimension's full extent |
| bundle parameter | literally `max_value=512` |
| codegen-allocated intermediates | `std::max(device_size_bytes, cpu_size_bytes)` |
| everything `concretize_expr` touches | work division, coordinates, alignment. 142 call sites |

This was measured the wrong way round first, which is why it is a result and not a preference. A
binary built from a 320-row warm-up was correct at 128 and 256 and **wrong** at 448 and 512, relative
error 3.96 and 4.37. Over-declaring is harmless, under-declaring is a silent wrong answer.

**It is only safe because of the reservation.** Geometry for 512 rows addresses memory that exists
only because the transfer allocated 512 rows. That is where the two PRs join.

---

## Stage 9. The emitted bundle

Verified on device, for this exact region at 64 columns fp16:

```mlir
#map_0 = affine_map<(d0)[s0] -> (s0 + 128*d0)>
func.func @sdsc_bundle(
    %arg_0_base_addr: !sdscbundle.input_arg<index>,
    %arg_1_base_addr: !sdscbundle.input_arg<index>,
    %dim_s77_base: !sdscbundle.input_arg<index, granularity=64, max_value=512>) {
  %dim_s77 = sdscbundle.input_arg_extract value from %dim_s77_base : ... -> index
  %c0 = arith.constant 0 : index
  %step_0 = arith.constant 64 : index
  scf.for %i_0 = %c0 to %dim_s77 step %step_0 {
    %addr_0 = affine.apply #map_0(%i_0)[%arg_0]
    sdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_0.json", ...}
  }
}
```

Points to make from it. The dimension parameter is last and is the only one carrying `granularity`
and `max_value`. Its extract precedes the loop constants. The bound is the dimension, the step is a
constant 64, the lower bound is zero, so the device's `(ub - lb) / step` is free and **no division is
authored anywhere**. The affine stride is 128, the per-row stride, because 64 columns of fp16 is 128
bytes and the loop variable counts rows rather than tiles.

The SDSC itself is tile-sized and fully static. Nothing symbolic reaches it.

---

## Stage 10. kDimension

### What a correction slot is

A Spyre program is prepared once and dispatched many times. Values that differ per dispatch cannot be
baked in, the obvious one being the HBM addresses of the actual tensors. So the job plan has a
HostCompute step that patches those values in just before the run. That is program correction, and
each patched value is a slot.

This is not new and not ours. Base addresses are corrected every dispatch already, which is where the
roughly 795 microseconds of per-dispatch cost comes from. It is also why the design insists no address
is **derived from** the varying size: such an address would be recomputed every call, and we would
have traded a compile per shape for a correction per call.

### The payload

```cpp
struct SymbolicArg {
  SymbolicArgKind kind;   // kAddress or kDimension
  int64_t tensor_id;
  int64_t dim_index = -1;
  int64_t value = -1;
};
```

`kAddress` is pre-existing: the slot carries a tensor's composite device address. `kDimension` is what
this work added: the slot carries the one integer that differs between a 128-row call and a 512-row
call.

### Why the value is not in the descriptor

`value` exists and is unused by both kinds. The descriptor is built **once**, at prepare time, and is
invariant across launches. The size it names changes on **every** call. So the descriptor carries only
where to read from, and the read happens at launch:

```cpp
case SymbolicArgKind::kDimension: {
  const at::Tensor& tensor = tensors[arg.tensor_id];
  TORCH_CHECK(arg.dim_index >= 0 && arg.dim_index < tensor.dim(), ...);
  resolved[i] = static_cast<int64_t>(tensor.size(arg.dim_index));
  break;
}
```

`tensor.size(dim)` is the **logical** size, deliberately. Under a max-strided reservation the buffer
holds 512 rows and the tensor says 128. Reading the allocated extent instead would run the maximum
number of trips on every call, give correct numbers, and silently deliver none of the benefit.

### Where the payload is built

`kernel_runner.py`, once, from the canonical symbol order, with an offset because `call_kernel`
prepends the pool tensor when there is one:

```python
offset = 1 if self.symbol_kinds[0].is_pool else 0
for sk in self.symbol_kinds:
    if sk.is_pool:
        payload.append(SymbolicArg(kind=kAddress, tensor_id=0))
    elif sk.is_loop_dimension:
        payload.append(SymbolicArg(kind=kDimension,
                                   tensor_id=sk.arg_index + offset,
                                   dim_index=sk.dim_index))
    else:
        payload.append(SymbolicArg(kind=kAddress, tensor_id=sk.arg_index + offset))
```

The payload is invariant across launches. Only what it resolves to changes.

---

## What actually varies at runtime

| Size | Trip count | Buffer | SDSC | Geometry | Binary |
|---|---|---|---|---|---|
| 128 | 2 | 512 rows | walk of 8 | built for 512 | the same one |
| 256 | 4 | 512 rows | walk of 8 | built for 512 | the same one |
| 384 | 6 | 512 rows | walk of 8 | built for 512 | the same one |
| 512 | 8 | 512 rows | walk of 8 | built for 512 | the same one |

One integer moves.

## Summary

The symbol is born at the transfer when the dimension is marked, gets its range from three
`torch._check` calls because nothing injects it automatically yet, survives tracing inside a
higher-order operator because a Python loop would have been unrolled, is taken back apart into a
symbol and a granularity at the splice, is carried to codegen by name because the ShapeEnv does not
survive a reload, becomes one bundle parameter, and is bound as one integer through a kDimension
correction slot at launch.

Everything else was built at the declared maximum and does not move.
