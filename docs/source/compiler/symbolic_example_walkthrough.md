# One call, traced all the way through, with and without our changes

A single concrete example followed end to end with real numbers, then every change
justified by the exact thing that breaks without it.

Written to defend PR #5186 in review. Companion to `symbolic_lowering_flow.md`,
which is the diagram version, and `symbolic_shapes_brief.html`, which is the readout.

---

## The example

```python
# the model
W = torch.randn(64, 64, dtype=torch.float16).to("spyre")     # static
b = torch.randn(64, dtype=torch.float16).to("spyre")         # static

# the varying input, declared once
x = torch.randn(128, 64, dtype=torch.float16)
x = x.to("spyre", dynamic={0: dict(min=128, max=512, granularity=64)})

def ffn(x, weight, bias):
    rows = x.shape[0]
    torch._check(rows >= 128)
    torch._check(rows <= 512)
    torch._check(rows % 64 == 0)

    def body(_carry, tiles):
        t, w, bb = tiles
        return None, torch.nn.functional.gelu(t @ w + bb)

    _carry, out = for_each_tile(
        body, (x, weight, bias), dims=(0, None, None), tile_size=64, out_dim=0
    )
    return out

compiled = torch.compile(ffn, backend="inductor", fullgraph=True)
y = compiled(x, W, b)       # 128 rows
```

Numbers that stay fixed for the rest of this document:

| | |
|---|---|
| columns | 64, fp16, so **128 bytes per row** |
| granularity G | **64 rows**, so **8192 bytes per tile** |
| declared max | **512 rows** |
| this call | **128 rows**, so **2 trips** |
| declared min | 128 rows |

---

## Part 1. The walkthrough

### Step 1. `.to()` reserves at the maximum (PR #5179)

The allocator is called with the strides of **512** rows while the logical shape stays
`(128, 64)`.

```
logical shape      (128, 64)
allocated extent   512 rows worth of HBM
reserved_dims      {0: {min: 128, max: 512, granularity: 64}}
```

The point of this step: a 128-row tensor and a 512-row tensor now produce the
**identical Spyre tensor layout**, because the layout is computed from the strides and
the strides came from 512 in both cases.

Then `mark_dynamic(dst, 0)` is called **bare**, with no `min=` or `max=`.

### Step 2. Dynamo traces, and a symbol appears

```
x.shape          (s77, 64)
s77 range        [2, oo)        <- no ceiling yet, because the mark was bare
```

### Step 3. The three in-region checks give the symbol its range

```
after torch._check(rows >= 128)      128 <= s77
after torch._check(rows <= 512)      128 <= s77 <= 512     <- the load-bearing one
after torch._check(rows % 64 == 0)   Mod(s77, 64) == 0
```

The ceiling of 512 is what every later sizing decision reads. Without it the kernel
compiles, works at this size, and silently specialises.

### Step 4. `for_each_tile` builds the tiling, and G enters the system

`_normalize_in_specs` runs over the three operands.

```
operand 0, x        dims=0     -> TileSpec(SLICE, axis=0, extent=64)
                               -> num_tiles = s77 // 64
operand 1, weight   dims=None  -> passed whole every trip, stays static
operand 2, bias     dims=None  -> passed whole every trip, stays static
```

**The granularity enters exactly here and nowhere else.** The 64 passed as `tile_size`
lands inside the expression `s77 // 64`, so there is one object carrying both the
symbol and the granularity. Nothing downstream has to be told G separately, and
nothing can disagree about it.

### Step 5. The view that cuts x into tiles

```python
moved   = x                              # dim already 0, movedim is a no-op
trimmed = moved.narrow(0, 0, (s77 // 64) * 64)
xs      = torch.unflatten(trimmed, 0, (s77 // 64, 64))
```

```
xs.shape     (s77 // 64, 64, 64)
             tiles       rows  cols
extent       64          <- a literal, which is the whole point of the narrow
```

Both operations are views, so nothing copies. At this call `s77 // 64` is 2, so `xs`
is `(2, 64, 64)`.

### Step 6. The loop is built, and its condition carries the count

`for_each_tile` produces a scan, which upstream lowers to a `while_loop`. The loop
condition reduces to one comparison:

```
load(iteration_var, 0) < FloorDiv(s77, 64)
                         ^^^^^^^^^^^^^^^^^
                         arrives as ops.index_expr, not ops.constant
```

### Step 7. Splice, then recover

`splice_while_loops` runs first in the pass list and inlines the body. The varying
dimension is among the operands as a **SymInt**, which is why the operand list has to
be realigned here.

`_extract_trip_count` replays the condition's `inner_fn` with a recording handler and
pulls out:

```
count = FloorDiv(s77, 64)
```

### Step 8. The middle of the pipeline sees a static tile

From here until codegen, nothing knows a symbol exists. The body op's iteration space
is a 64 by 64 tile. Layouts, stickification, work division, scratchpad allocation and
the cost model all run exactly as they do for a static kernel.

```
sdsc_0.json      a 64-row by 64-column gelu over a matmul, fully concrete
                 byte-identical whether the call is 128 rows or 512 rows
```

### Step 9. The scheduler records the bounds

```python
symbolic_count_bounds(FloorDiv(s77, 64))
# decompose_tiled_count   -> (s77, 64)
# finite_upper_or_none    -> 512
# returns
{"s77": (512, 64)}
```

Stored on the `LoopSpec`:

```
LoopSpec.count                = FloorDiv(s77, 64)
LoopSpec.count_symbol_bounds  = {"s77": (512, 64)}
```

Resolved here because this is the **last point the ShapeEnv exists**. Codegen also runs
in a reload phase where it is gone.

### Step 10. Kernel codegen records where the number comes from

`_resolve_loop_dimension_sources` pairs each symbol with a launch argument.

```
live call args    ["arg0_1", "arg1_1", "arg2_1"]   # x, weight, bias
arg0_1 sizes      [s77, 64]
s77 found at      arg_index 0, dim_index 0
```

```
LoopSpec.count_symbol_sources = {"s77": (0, 0)}
```

### Step 11. Geometry is sized from the maximum

```python
trip_count = max_trip_count(FloorDiv(s77, 64))     # 512 // 64 = 8
```

```
tiled_symbol_trip_counts[sym]   = 8
SDSC pre-tiling extent          = 8 tiles * 64 rows = 512 rows
```

And every structural parameter that touches `s77` goes through `concretize_expr`,
which returns **512**, not 128.

**The two numbers people confuse.** `max_trip_count` is **8**, the largest trip count.
`count_symbol_bounds` carries **512**, the dimension's own ceiling. One is the other
divided by 64.

### Step 12. The kernel is written to generated Python

```python
# in the generated wrapper file
LoopSpec(count=sympify('(s77//64)'), count_symbol_bounds={'s77': (512, 64)}, ...)
```

The cache key hashes the count **after normalising it through one `sympify`**, because
on reload `(s77//64)` re-parses as `floor(s77/64)` and the raw strings differ.

### Step 13. The bundle is emitted

```python
loop_dim_bounds  = _merge_loop_symbol_maps(specs, "count_symbol_bounds")   # {"s77": (512, 64)}
loop_dim_sources = _merge_loop_symbol_maps(specs, "count_symbol_sources")  # {"s77": (0, 0)}

SymbolKind.loop_dimension(granularity=64, max_value=512,
                          pytorch_sym="s77", arg_index=0, dim_index=0)
```

```
_count_scale(FloorDiv(s77, 64))  -> 64        # the stride divisor
_loop_level(...)                 -> bound = %dim_s77, step = %step_0, stride_scale = 64
_scaled_strides(...)             -> 8192 / 64 = 128
```

Final bundle:

```mlir
#map_0 = affine_map<(d0)[s0] -> (s0 + 128*d0)>

func.func @sdsc_bundle(
    %arg_0_base_addr: !sdscbundle.input_arg<index>,
    %arg_1_base_addr: !sdscbundle.input_arg<index>,
    %dim_s77_base:    !sdscbundle.input_arg<index, granularity=64, max_value=512>) {

  %dim_s77 = sdscbundle.input_arg_extract value from %dim_s77_base : ... -> index
  %c0      = arith.constant 0  : index
  %step_0  = arith.constant 64 : index

  scf.for %i_0 = %c0 to %dim_s77 step %step_0 {
    %addr_0 = affine.apply #map_0(%i_0)[%arg_0]
    sdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_0.json", ...}
  }
}
```

### Step 14. The gate, then the backend

`async_compile` checks for SDSC dimension symbols. Our kind reports `is_dimension`
False, so it passes. DeepTools builds **one binary** covering the declared range.
Measured: 10.9 s for this compile.

### Step 15. Launch, and the only number that varies

```
payload (built once, invariant)
  SymbolicArg(kind=kAddress,    tensor_id=0)
  SymbolicArg(kind=kAddress,    tensor_id=1)
  SymbolicArg(kind=kDimension,  tensor_id=0, dim_index=0)

this call
  tensor.size(0) = 128                 <- the LOGICAL size, not the allocated one
  device derives (128 - 0) / 64 = 2 trips

trip 0: addr = arg_0 + 128*0   =  arg_0          rows   0..63
trip 1: addr = arg_0 + 128*64  =  arg_0 + 8192   rows  64..127
```

The second call at 512 rows runs the **same binary**, binds 512, and takes 8 trips.
Nothing else changes.

---

## Part 2. With and without each change

Each row is one change, what breaks without it, and how we know. The "how we know"
column matters: some of these were measured on hardware, some fail by construction,
and one is design reasoning.

| # | Change | Without it | How we know |
|---|---|---|---|
| 1 | state divisibility as a fact | the trimmed length stays `64*(s77//64)` instead of simplifying back to `s77` | design reasoning |
| 2 | trim before splitting | tile extent becomes `s77 // (s77 // 64)`, an expression, so **every stride below it is an expression** | measured |
| 3 | fold with `as_strided` | `flatten` installs `Ne(s77//64, 1)`, splitting the binary at one tile | measured |
| 4 | prover accepts `index_expr` | prover declines, loop falls back, **one binary per size** | by construction |
| 5 | realign SymInt operands | placeholders map to the **wrong buffers** after the missing operand | by construction |
| 6 | two `LoopSpec` fields | codegen has no bounds at reload, so no bundle parameter can be declared | by construction |
| 7 | `max_trip_count` | `int()` on a symbolic count **raises** | by construction |
| 8 | `concretize_expr` to the max | geometry sized from the warm-up hint, **wrong at larger sizes** | measured |
| 9 | computed loop bound and step | bound stays a hardcoded constant, so **one binary per size** | by construction |
| 10 | stride scaling | address advances **64 tiles per trip** instead of one | by construction |
| 11 | `scale_stack` in both passes | read side registers an unscaled stride, **two affine maps for one address** | caught by test |
| 12 | a separate symbol kind | the `async_compile` gate **refuses our own bundle** | by construction |
| 13 | all four reload requirements | two kernels sharing a cache entry, or a reloaded kernel missing its own | by construction |
| 14 | the `kDimension` case | launch raises **"kDimension is not yet implemented"** | by construction |
| 15 | the payload branch | the dimension slot carries an **address where a row count belongs** | by construction |
| 16 | the `patches.py` shim | `KeyError(s77)` during scan lowering, hard crash | by construction |

Below, the five worth explaining properly.

---

### Change 2. Trim before splitting. The one that decides whether the feature works

**Without it**, `_xs_leaf` asks `unflatten` for sizes `(s77 // 64, 64)` against a tensor
whose dim 0 is `s77`. `unflatten` has to reconcile the two. On a concrete 512 it checks
`8 * 64 == 512`, agrees, and hands back the 64 you asked for. On a symbolic `s77` it
cannot prove `(s77 // 64) * 64 == s77`, so it re-derives the extent by dividing:

```
extent = s77 // (s77 // 64)
```

That is an expression, not the number 64. Sanity-check it by substituting a value: at
`s77 = 100` it evaluates to `100 // 1`, which is 100, nowhere near 64.

**Why that is fatal, and this is the sentence to use.** Every stride below this point is
computed from the extent. An extent that is an expression makes every stride an
expression, and the SDSC stops being a static tile program. The entire design rests on
the tile being static, so this single view spelling decides whether anything works.

**With it**, the narrow makes dim 0 literally `64 * (s77 // 64)`, so the division
cancels by common factor:

```
extent = (64 * (s77 // 64)) // (s77 // 64) = 64
```

```
                             WITHOUT                      WITH
tensor handed to unflatten   dim0 = s77                   dim0 = 64 * (s77 // 64)
extent returned              s77 // (s77 // 64)           64
tile byte stride             an expression                8192
SDSC                         not static                   static
```

**For a concrete length both spellings are identical.** The trim drops zero elements.
So this is not a special case bolted on for symbols, it is a spelling that is correct
either way.

---

### Change 8. Geometry from the declared maximum. The one that was measured wrong first

**Without it**, `concretize_expr` falls back to `optimization_hint`, which is the size
of the call that happened to trigger the compile. Build a binary during a 320-row
warm-up and every structural parameter is sized for 320 rows. The binary then claims to
serve the declared range and does not.

**Measured, and this is the table to show:**

| binary built from | 128 rows | 256 | 448 | 512 |
|---|---|---|---|---|
| a 320-row warm-up hint | ok | ok | **err 3.96** | **err 4.37** |
| the declared max of 512 | ok | ok | ok | ok |

The failure is invisible at and below the warm-up size, which is exactly why it has to
be a rule and not a judgement call.

**With it**, a symbol with a finite ShapeEnv bound resolves to that bound. The asymmetry
is what makes the choice obvious: over-declaring the geometry wastes a little
addressable space and is harmless, under-declaring it returns wrong numbers silently.

**The dependency to state out loud.** Geometry sized from 512 is addressable **only
because step 1 reserved the buffer for 512**. Two of our functions say so in their
docstrings and cite #5179. There is no fallback here: the backend refuses an address
correction for a call inside a loop and fails the pass, so there is no slow-but-working
path where a symbolic address gets corrected per trip.

**The new branch fires only for a symbol with a finite bound**, so a concrete expression
and an accidentally-dynamic symbol both behave exactly as before.

---

### Change 10. Stride scaling. The arithmetic worth doing on a whiteboard

The loop form and the stride scale are **one decision**, not two. Here is why.

A concrete loop keeps the form this emitter always produced: step 1, loop variable
counts **tiles**.

```
WITHOUT a symbolic bound (unchanged behaviour)
  scf.for %i = 0 to 8 step 1          i = 0, 1, 2, ... 7      i counts TILES
  addr = base + 8192*i                                        8192 = tile stride
```

A symbolic loop takes `to <dim> step <G>`, so the loop variable counts **elements**.

```
WITH a symbolic bound
  scf.for %i = 0 to %dim step 64      i = 0, 64, 128, ...     i counts ROWS
  addr = base + 128*i                                         128 = ROW stride
```

**If you emitted the symbolic loop form and kept the tile stride**, which is the bug
this change prevents:

```
  scf.for %i = 0 to %dim step 64      i = 0, 64, 128, ...
  addr = base + 8192*i
  trip 0: base + 0
  trip 1: base + 8192*64  = base + 524288       <- 64 tiles too far
```

Every trip after the first reads and writes outside its tile. So `stride_scale` is not
an optimisation, it is the other half of changing the loop form.

The divisor is exact by construction rather than fitted: a level's stride is its
per-element step times the tile size, so dividing by the tile size leaves the
per-element step. `_scaled_strides` asserts divisibility and raises if it ever fails.

---

### Change 4. The prover accepting `index_expr`. The silent one

**Without it** nothing crashes, nothing warns, and the numbers are right. The recorder
collected only `ops.constant`, a symbolic count arrives as `ops.index_expr`, so the
recorder saw zero candidate bounds, the prover returned `None`, and the loop quietly
took the non-symbolic path. You get a correct binary **per size**, which is exactly
what the feature exists to avoid.

**This is the most dangerous class of failure in the whole PR**, because the test that
catches it has to assert the absence of a recompile rather than the presence of a
result.

**With it**, both spellings land in one list:

```python
if name in ("constant", "index_expr"):
    self.bounds.append(args[0])
```

**The second half of the change is the diagnostic.** `_extract_trip_count` used to
return bare `None` from **nine** different conditions. It now returns a `ProverResult`
carrying a reason:

```
cond_subgraph offered 0 candidate bounds ([]), expected 1
cond_subgraph issued comparisons ['le'], expected ['lt']
cond_subgraph loads 'buf3', not the first placeholder 'arg0_1'
```

Declining is not an error, the kernel just specialises. So without a reason channel, a
loop that **failed to be recognised** is indistinguishable from one that was **never
asked for**. Those are completely different bugs with the same observable outcome. This
half is a WSR debuggability improvement independent of symbolic shapes.

---

### Change 5. Realigning the operands. Why the obvious fix was wrong twice

`splice_while_loop` pairs the body graph's placeholders with real operands **by
position**.

**Without the change:**

```
body_graph_input_names   4 placeholders
while_op.inputs          3 entries        <- the SymInt dropped out
```

`ExternKernel.__init__` repacks `[*carried, *additional]` through `_split_by_sym_type`
into a tensor-only `.inputs`. A varying dimension arrives as a SymInt operand, so it
disappears from `.inputs` while the body graph still lists its placeholder. Every
position after the missing one then resolves to the wrong buffer, so a load that should
read the bias reads the weight.

**With the change**, fall back to the unpacked concatenation, but **only when the
lengths say an operand actually went missing**:

```python
operands = while_op.inputs
if len(operands) < len(body_graph_input_names):
    operands = [*(while_op.carried_inputs or ()), *(while_op.additional_inputs or ())]
```

The condition matters. `.inputs` and `.carried_inputs` hold **different objects**, so
swapping unconditionally would change which buffer a concrete operand resolves to, on
every existing kernel.

Then the SymInt operand itself has no buffer to point a read at, and finding the right
test for that took three attempts:

| attempt | why it was wrong |
|---|---|
| `isinstance(x, ShapeAsConstantBuffer)` | missed a wrapped one, and needed a new import |
| `_storage_name(x) is None` | **stricter** than "has a name", because it insists on reaching an `ir.Buffer`. Skipped an operand the splice needs and broke a passing test |
| `try: x.get_name() except NotImplementedError` | keys on the actual condition, so anything that produced a name before still does |

**The general lesson**, and it is a good one to offer: when the condition is "this
operation fails", test that, not a predicate that approximates it. Attempt two looked
like good reuse and was a behaviour change in disguise.

---

## Part 3. The three ways this could have gone wrong quietly

Worth having ready, because the question behind most review questions is "how do you
know it is right".

**A binary that specialises while looking like it works.** Changes 4 and 9. Correct
numbers, one binary per size, no error anywhere. The only test that catches it asserts
an absence: one call into the compiler across four sizes, one cache key, one Dynamo
graph. We also hit the inverse of this, a test that passed as `set() == set()` because
the instrument it read was switched off, so the suite now has a test whose only job is
to prove the instruments are live.

**A binary that is correct at the size you tested and wrong above it.** Change 8.
Invisible at and below the warm-up size. Caught by sweeping sizes **above** the one that
triggered the compile, which is now how every size sweep in the suite is written.

**A field lost on reload.** Change 13. Inductor writes a Python file and re-executes it
later, so a new value needs the field, the provenance schema entry, the cache-key entry
and the serializer line. Missing the cache-key entry means two kernels differing only in
declared max share one entry, which serves the wrong binary. Every test for these
round-trips through the **real** serializer rather than a hand-built object.

---

## Part 4. What we did not change, which is the headline for a WSR review

| | |
|---|---|
| `TileSpec` | **no field added.** `num_tiles` was already there, from the WSR refactor in #3293 |
| `OpSpec` | **no field added.** `tiled_symbols` and `tiled_symbol_trip_counts` were already there |
| every pre-scheduling pass | unchanged |
| stickification and layout propagation | unchanged |
| work division | unchanged on this route, which returns empty, and callers already treat empty as normal |
| scratchpad allocation | unchanged |
| the cost model | unchanged, and this is an honest gap, see below |
| every SDSC | unchanged. `sdsc_0.json` is byte-identical at 128 and 512 rows |
| the layout guard | unchanged. Nothing varies once the layout is held at the max, so there is nothing to skip |

`for_each_tile.py` is **+32 lines, -2**, across three changes, all of them view
spellings and all of them no-ops for a concrete length.

`LoopSpec` gained two fields, both defaulting to empty, so every concrete loop behaves
exactly as before.

---

## Part 5. The gaps to volunteer before anyone finds them

**The cost model knows nothing about this.** It prices device time. It already drops the
underfill derate when the per-core tile height is symbolic, and because geometry now
resolves to the declared maximum across 145 `concretize_expr` call sites, it may be
pricing a 512-row binary for a 128-row call. **We have not measured that.** One probe,
assigned.

**Op coverage on this branch is two shapes**, the pointwise gelu in the end-to-end test
and `gelu(tile @ w + b)` in the demo. The longer list people quote is from the POC tree,
which is a different and older main. Building the real matrix is assigned work.

**The single-tile split is not fixed**, only our contribution to it removed. The
remaining source is upstream, measured in both trace orders, traced to an exact line,
assigned. Numerics are correct at one tile. Cost is one extra binary at the minimum size.

**`concretize_expr`'s 145 call sites are not audited.** We shipped the broad version and
said so in the commit message. The sites are not all the same kind of thing: most are
structural extents where the max is right, a few are predicates, and a couple become
dictionary keys. That audit is scoped work.
