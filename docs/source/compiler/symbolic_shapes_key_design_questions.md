# Symbolic shapes: the key design questions

Each section is a question the design had to answer, what was measured, and what the answer commits
us to. Where something is unmeasured it says so.

---

## 0. The one rule

A Spyre binary is compiled for one exact shape, so serving at several sizes costs a full compile
each. The design removes that, and the whole of it reduces to one sentence:

> **The varying size reaches only the loop trip count, and nothing else.**

The "nothing else" is not tidiness. Addresses are corrected on every dispatch, around 795
microseconds on the slow path. An address or stride **derived from** the varying size has to be
recomputed every call, so you would have traded a compile per shape for a correction per call and won
nothing.

One precision, because it is the thing people misread. The rule is about **derivation**, not about
whether symbols appear. Base addresses are symbolic on every Spyre bundle including ours, and that is
free. Our verified bundle carries `symbol_ids=[-1]`.

Everything below is a consequence of that one rule, or a test of it.

---

## 1. Does the contract survive a kernel boundary

The chain raised in review was `host -> HBM -> kernel 1 -> (op) -> HBM -> kernel 2 -> fails`. The
concern was structural rather than speculative: the contract has exactly one writer, inside the
host-to-device transfer, so a tensor born on the device never passes it. Both consumers compile one
block at a time, so a chain of kernels handing tensors to each other is the normal shape of a model,
not an edge case.

Measured with a four-stage chain:

| What | Result |
|---|---|
| intermediate's declaration | **None**, as predicted. It is device born |
| intermediate's size | 65536 bytes at every size, the declared maximum |
| compiles | one kernel per stage, four traces for four stages |
| numbers | worst relative error 0.0040 |
| ATen fallback in the chain (`cumsum`, confirmed firing) | correct, one kernel for the whole range |
| unaligned width, 96 elements, 192 host bytes against 256 device | allocation is the device footprint at the maximum |

**The `(not correct layout)` step does not occur on any constructible path and the `-> fails` step did
not happen.**

Two reasons, and the second is the load-bearing one.

Intermediates come out at the maximum because generated code emits the device layout literal at the
maximum, and the allocator takes `std::max(device_size_bytes, cpu_size_bytes)`. Note what this is
**not**: the SDSC is tile sized and static, and plays no part in sizing the output. It is the wrapper
code that allocates.

And the real answer to why a device-born tensor having no contract is harmless:

> **Binary reuse follows from layout constancy, not from the declaration.**

Dynamo guards every device tensor on its device layout. A buffer reserved at the maximum presents the
**same layout at every size**, so the guard passes and the binary is reused. The intermediate does not
need to carry the contract for the next kernel to be reused. It needs to have the same shape of
memory, which it does, because our codegen sized it at the maximum.

**Design consequence.** Max sizing is a property of buffers our own codegen allocates. Anything
allocated by an ATen fallback, an eager clone or a plain transfer is sized to that call. That is a
boundary worth knowing: it is why a clone does not inherit the reservation, and why anything that
later attaches a contract to a tensor it did not allocate must verify the storage first.

---

## 2. Does it survive a compile boundary

Different question, opposite answer, and the most important result in the set.

> **The symbol crosses a region boundary. The range does not.**

A second compiled region with no in-region range **refuses**:

```
Unsupported: symbolic loop count (s77//64) has no finite upper bound, so SDSC codegen
cannot describe the tiled dimension's extent. Declare the range for ['s77'] in the
traced region
```

The refusal is itself the proof the symbol arrived. A region that had seen a concrete size would have
compiled happily.

### The mechanism, measured

The declaring transfer and a compiled region's output carry **different** marks:

```
marked input : _dynamo_dynamic_indices      = {0}      strict, set by the transfer
compiled out : _dynamo_weak_dynamic_indices = {0}      weak, set by the compiled region
```

So the next compiled region treats that dim as dynamic from its very first trace. An eager op does
not propagate it. Isolated with two distinct code objects, so the result is not Dynamo's
per-code-object automatic dynamic:

```python
def stage1(x): return torch.nn.functional.gelu(x) + 1.0
def stage2(x): return torch.nn.functional.relu(x) * 2.0   # different code object

x = torch.randn(256, 64); torch._dynamo.mark_dynamic(x, 0)

mid_compiled = torch.compile(stage1, dynamic=None)(x)   # stage2 sees ['s77']
mid_eager    = stage1(x)                                # stage2 sees []
```

Each `torch.compile` has its own ShapeEnv, so region 2 mints a **fresh** symbol. It prints as `s77`
too, which is numbering, not identity.

### Why the range cannot ride along

The range was never on the mark. The mark is bare on purpose:

```python
# Deliberately bare: no min=/max= here. Passing them installs a StrictMinMaxConstraint,
# which promises every value in that range is valid. The granularity check the compiler
# adds later (size % granularity == 0) then looks like it's violating that promise for a
# non-multiple size, raising ConstraintViolationError instead of a normal guard miss.
torch._dynamo.mark_dynamic(dst, spec.dim)
```

The range lives as **facts in that region's ShapeEnv**, installed by `torch._check`. A ShapeEnv does
not leave its compile. So region 2's fresh symbol starts with PyTorch's default range, lower bound 2
and no upper bound, and refuses for want of a ceiling.

### It does not depend on how regions are written

The first result used three separately compiled regions, and neither consumer builds layers that way.
Both create per-layer closures in a loop, so every layer shares one code object. Four arms were run:
separate definitions, one factory instantiated three times, one compiled function called three times,
and the same with stage 0 taking a different branch.

| Round | A | B | C | D |
|---|---|---|---|---|
| no range anywhere | refuse at stage 1 | refuse at stage 1 | refuse at stage 1 | refuse at stage 1 |
| range on stage 0 only | refuse at stage 2 | refuse at stage 2 | refuse at stage 2 | refuse at stage 2 |

Identical behaviour, byte-identical message. Region identity is irrelevant.

### The split, stated as a design property

| Property | Crosses a compile boundary | Why |
|---|---|---|
| allocation and layout | **yes** | the buffer is physically max strided and codegen sized the intermediate at the maximum |
| dynamism, that this dim varies | **yes** | the weak mark rides on the output tensor |
| the range, min, max, granularity | **no** | it is ShapeEnv state, and a ShapeEnv does not leave its compile |

**Design consequence, and this is the single clearest piece of forward work.** Today the caller states
the range twice, once in the `dynamic=` dict for the allocator and once in `torch._check` for the
compiler. The follow-up reads the declaration off the graph input with `get_reserved_dims` and
re-installs those three facts into each region's own ShapeEnv. That is all it has to do. Nothing more
exotic, and the interface does not change.

---

## 3. Which axis can carry a declaration

Ten shape and dimension combinations, on device.

| case | result |
|---|---|
| dim 0, any rank | works |
| dim d > 0 with every dimension above it of size 1 | works |
| dim d > 0 with any dimension above it greater than 1 | refused at the transfer |
| the innermost dim | refused outright |

Batch is the only thing varying across these three rows, which is what makes it clean:

```
rank3 [B=1,S,H] dim1   accepted   65536 bytes   geom (512, 1, 1, 64)
rank3 [B=2,S,H] dim1   REFUSED    Invalid device sizes and stride map
rank3 [B=3,S,H] dim1   REFUSED    Invalid device sizes and stride map
```

So the rule is **effectively outermost**: every dimension above the declared one must be size 1.

Two refusals, two different causes, worth separating because they fail differently. The innermost
axis is refused by an explicit check, since padding a stick dimension changes the stride map for every
dimension above it. An axis with a real extent above it is refused by the host-to-device layout
mapping, which cannot describe a padded axis sitting underneath one.

Both happen **at the transfer, before anything is traced**, so this is a scope limit rather than a
hazard.

**The distinction that matters for planning, and that was conflated for a day:**

| use case | varying axis | status |
|---|---|---|
| packed token count, decoder | dim 0 | supported |
| **encoder dynamic batching** | batch, dim 0 | **supported** |
| **encoder variable sequence length** | sequence, dim 1 of rank 2, innermost | **refused** |

Dynamic batching works. Only a variable sequence length needs the missing capability, and that
capability is runtime work on the reservation, independent of this design.

Two more things held on every accepted axis. A declaration moves no data: a declared transfer is
byte-identical to an undeclared one. And byte counts matched the max-strided prediction exactly,
65536 / 131072 / 65536.

---

## 4. What if the declared granularity and the executed tile disagree

The contract granularity and a loop level's tile are **different kinds of thing**, and the API calling
both of them "granularity" is where the confusion gets manufactured. The contract granularity says
which sizes are admissible. A loop level's tile says how the region cuts the tensor it was handed.

Five cells of declared against executed were swept. **Every mismatch refused.** Nothing accepted an
untileable size, nothing computed a wrong answer.

So the safe direction is confirmed: the execution tile must divide the contract granularity. A region
tiled at 32 under a contract of 64 serves the whole range with one binary. In the other direction a
size can be legal under the contract and not tileable by the region, and the region refuses.

**A disagreement costs a failed compile, not a dropped tail. No correctness cross-check is needed.**

What the refusal does **not** do is name the contract granularity, because the region cannot see it.
A caller who declared 32 and met a region tiled at 64 sees only "64". That is a message quality gap
and it is the one real item left in the guard work.

Omitting the granularity entirely is already caught well. It defaults to 1, and the transfer refuses
with a message that explains the cost:

```
max=512 / granularity=1 = 512 buckets, exceeding the cap of 32
(each bucket costs roughly 5us of host correction on every dispatch)
```

---

## 5. Is the interface right

```python
x = x.to("spyre", dynamic={0: dict(min=128, max=512, granularity=64)})
compiled = torch.compile(model, dynamic=None)
```

Nothing measured requires changing this call. Every alternative was measured and rejected for a
specific reason, not on taste.

**Compiling with dynamic shapes on** marks every dimension including the innermost, which puts a
symbol into a stride rather than into a trip count. That breaks rule 0 directly.

**The stock dynamic-marking interface** installs a strict range constraint, so adding a divisibility
requirement on top raises a constraint violation. A range and a granularity cannot both be expressed
through it. This is also why our own mark is bare.

**Declaring per region** means annotating every block in the model, which both consumers would have to
do at every layer.

One result worth keeping because it changes what is *required* versus what is *worth doing*. Two
tensors declared separately and tied with an equality check compile to one binary, so a shared
declaration object is **not required for correctness**. It is still worth building, for two different
reasons: a decomposition cannot call `.to()`, so it needs to resolve a declaration by identity, and it
is the only way a refusal can name both the declared granularity and the executed tile.

---

## 6. Does this repeat the hint work

The closed PR tried to move WSR hints from declaring a trip **count** to declaring a tile **size**.
Same direction this design needs, because the varying quantity has to sit on the count side. It took
a month and did not merge.

The review's root cause, in the reviewer's words:

> Spyre hints tile **abstract dimensions** not tensors. The mapping of loop dimensions to operation
> dimensions to tensor dimensions is **indirect** via named dimensions.

And the author's own summary after deleting every special case he had added, net minus 41 lines:
*"every special case I was adding was an artifact of measuring the wrong thing."*

The sentence that transfers:

> **A count is invariant to which loop var a name lands on. A size is not.**

`[B,H]` with `B=1` folds onto one loop var of extent 8, so inspecting the op recovers 8 for `B` when
the answer is 1. Hardware caught the end state: `declared tile_size 1 vs derived extent 64`.

### Why the mechanism cannot recur here

Three reasons, each verified in the tree rather than argued.

**There is no name, so nothing to resolve.** `for_each_tile` is handed its operands and divides the
tensor in hand. Zero references to `named_dim`, `spyre_hint`, `declare_tensor_dim`,
`num_tiles_per_dim` or `tile_size_per_dim` in the construct or its lowering.

**A nested level is handed the enclosing level's tile**, so its units are correct by data flow rather
than by a lookup that could resolve wrongly.

**The compiler never reads the declaration.** `reserved_dims` appears zero times in the whole tree.
The granularity is recovered out of `FloorDiv(s, G)`, so the bundle's declared granularity and the
loop's step come from one expression and cannot disagree.

Worth saying plainly: hints were being retired for higher-order operators anyway, and `for_each_tile`
is the construct that replaced them. This is built on the replacement, not on the thing that failed.

### Where it can still reach us

Four places, and they should be said before anyone finds them.

**The attention tile-size search is the live instance.** The chooser searches upward from
`ceil(seq/max_extent)` for an exact divisor. Correct today and better than the old form, but a search
over the divisors of an extent is exactly what cannot survive a symbolic extent. The fix is the closed
PR's own thesis applied at the call sites: declare the size, derive the count. Phase 1 does not route
through the chooser. Phase 2 depends on it.

**The follow-up region builder is a decomposition, so it cannot call `.to()`.** Identical shape to the
withdrawn SDPA half of that PR. The declaration object has to land before the builder is written.

**Phase 1 is safe by narrowness, not by construction.** With dim 0, outermost, one level, the tensor
axis and the loop variable coincide. For nesting they do not.

**One sentence in the design doc is currently the bug.** It specifies the region builder as taking its
tile "from the same object" as the declaration. At an inner level that is precisely the defect above.
It was tested on device and refuses loudly, naming both numbers, so a violation costs a failed compile
rather than a wrong answer. The sentence still needs tightening before the builder is written.

### The part worth taking seriously

That PR was technically correct at the end, hardware verified across six suites, net negative lines,
and it still did not merge. **What cost the month was implementing before the semantics were agreed.**
What turned it round was one page of semantics.

So before the follow-up: one page saying which quantity is declared, which derived, in what units, and
a single policy for a non-dividing extent. Agreed first, built second.

---

## 7. Is Phase 2 an extension or a redesign

The fear is reasonable and specific: that Phase 1 ships on a design Phase 2 has to throw away, which
is what the WSR 1 to WSR 2 transition was.

Phase 2's actual shape was run on device. Five arms:

| arm | verdict | trip counts per level |
|---|---|---|
| one level | REUSE, err 0.0017 | `('floor(s77/64)',)` |
| two symbolic dimensions | REFUSE at launch | `('floor(s77/64)', 'floor(s17/64)')` |
| same dim, 64 then 32 | REUSE, err 0.0017 | **`('floor(s77/64)', '2')`** |
| same dim, inner 48 does not divide | REFUSE, names both numbers | - |
| the builder hazard, inner level handed the contract granularity | REFUSE, same message | - |

Two results carry the answer.

**A nest over one dimension carries exactly one symbol.** The inner count is the literal 2, because
the inner level is handed the enclosing tile, so its extent is a literal. One binary for the whole
range. This let the invariant strengthen to **at most one symbolic count per dimension, inner levels
over it are static**.

**A nest over two different dimensions lowers correctly**, reaching codegen with both trip counts
emitted, and is then refused at the launch boundary by a check in our own C++ that hard-codes a
two-step job plan and asserts step 1 is a Compute.

So the position is not that Phase 2 should extend. It is that Phase 2's shape was run, the lowering
produced the right thing, and what stopped it was an assertion we wrote ourselves.

### Why this is structurally unlike the WSR transition

WSR 1 needed a redesign because the **mechanism** was wrong: an indirect, lossy map from names to
tensor axes. The fix was not a capability, it was a different addressing model.

Here the addressing model is "a level divides the tensor it was handed", and nesting is the same
operation applied to the result. **There is no map to get wrong**, so there is nothing for a later
phase to discover is unsound.

| would force a redesign | state |
|---|---|
| a nested level in the wrong units | impossible by construction, the level is handed the tensor |
| the inner count turning out symbolic | **measured, it is a literal.** The main open risk, now closed |
| the builder passing the contract granularity to an inner level | a rule for code not yet written. Tested, refuses loudly |
| reduction along the varying axis | Phase 2 by design, an existing invariant covers it |
| the declared axis not being outermost | real, but runtime capability rather than design |

### What Phase 2 waits on

Three things, none of them design.

The reservation accepting a non-outermost axis, which is what a variable sequence length needs.

Relaxing that job-plan step check. One honest caveat: we know `steps[1]` is not a Compute and we have
**not** captured what it actually is. Dump the step kinds before writing the fix.

Reversing the derivation in the attention decomposition, one change in three call sites.

### The rule Phase 2 must not break

> A declaration describes a **dimension's** admissible values: min, max, and the step between them. It
> never supplies a tile to a loop level. A loop level's tile is chosen in the units of the tensor that
> level receives. The only constraint linking the declaration to the nest is at the **outermost** level
> over that dimension, where its tile must divide the contract granularity. Every inner level
> constrains against its **enclosing tile**, never against the declaration.

---

## 8. What happens with two or three symbolic dimensions

The two cases get conflated and behave differently.

**Same dimension, two levels.** One symbol. The inner count is a literal. Nothing new.

**Different dimensions, both symbolic.** Two symbols, both emitted correctly. The data flow argument
still holds: level 2 receives a tensor whose dim 0 has already been reduced to the outer tile, a
literal, while dim 1 is untouched and still carries its symbol. So level 2's count is symbolic and
correctly so. Each level still divides only what it was handed.

Addresses stay safe too. A two-level nest's affine map is `base + i*stride_i + j*stride_j`, and both
strides are static because the buffer is max strided. The address is derived from the loop variables,
never from the sizes.

### When two symbols are actually needed

Less often than it first appears.

| case | symbolic dims | nest |
|---|---|---|
| decoder, packed token count | 1 | one level |
| encoder dynamic batching | 1 | one level |
| variable sequence length | 1 | one level, needs the non-outermost axis |
| paged attention decode | 1, the block count. Query length is 1 | one level |
| full prefill, query and KV both varying | 2 **unless tied** | two levels |

Attention wants a nest over query tiles and KV blocks, each level seeing the full extent of its own
axis. But in decode the query length is 1, so there is nothing to tile. And in a full prefill the two
lengths are equal, so an equality check collapses them into one symbol, which is a mechanism already
measured working across two separately declared tensors.

The genuinely two-symbol case is chunked prefill with an independently varying chunk size and KV
length. Real, but narrower than "any attention model".

### What is open

**Three levels is unmeasured.** One and two were run. The invariants permit more, since the backend
asserts one symbol per symbolic **loop** rather than per kernel. But it should not be claimed until it
is run, and no real case has been found that needs it.

**The dispatch count grows multiplicatively.** Two symbolic levels means `count1 x count2` iterations
and every dispatch pays correction cost. Whether that crosses a threshold where the saved compiles
stop paying is not measured.

**How many of our own guards assume one level.** The step-ordering check is one we have found. Expect
one or two more of that kind, each a hard-coded shape in a validator rather than anything in the
lowering, and each found by running the case rather than by reasoning about it.

That is the real difference. A redesign is when the mechanism cannot express the thing. This is a
mechanism that already expresses it, with a handful of our own guards that assume less than the
mechanism supports.

---

## 9. Can this reach a real serving workload

Checked against the plugin's tree, not assumed.

**What is already in our favour.** The decode path **already runs on `for_each_tile`** by default,
head major, with batched decode enabled in production. Paged attention is genuinely a custom op, so
the attention decomposition blocker does not touch decode at all. And the tiled axis in all three
decode kernels is already dim 0 of the block or chunk index and its mask, so the varying size already
reaches only the trip count on their existing code.

**What is achievable:** the KV block count symbolic, on one binary.

**What is not, with the current kernel:** a symbolic batch. The batch is structural there, feeding
`entries = num_seqs * blocks_per_chunk` and then a chain of reshapes inside the body. Under rule 0 it
would resolve to the declared maximum, so every decode step would pay full `max_num_seqs` cost, which
is worse than their existing power-of-two ladder for small batches.

So the honest scope is **block count symbolic, batch pinned at `max_num_seqs` with the padded lanes
they already support.** Promising both is how a demo slips.

**What it removes.** Everything there is compiled `dynamic=False`, with three bucket ladders
pre-compiled at warmup, padding up and masking out. Their own measured number: Granite 3.3 8B compile
time 583 seconds without `for_each_tile` and 206 with it. The first saving was the unrolled page loop,
already taken. The residual 206 seconds is the bucket cross product, and that is what this removes.
Their documentation asks for it directly:

> Every graph is compiled `dynamic=False` because torch-spyre's Inductor backend rejects `SymInt`
> shapes. This is the root reason the plugin buckets shapes everywhere.

**What is needed on their side**, roughly a week for someone who knows that tree: five `dynamic=False`
flags flipped, the attention scale passed as a zero-dimensional tensor at five sites, the
`[:num_blocks]` slices deleted so the host builds tables at the declared extent, the block-count
padding changed to ceil-to-granularity, the chunk size pinned so only the chunk count varies, the
declaration added at two or three call sites, and the warmup recorder collapsed from a cross product.

**Two unknowns**, both cheap to settle. Whether `.to(device_layout=..., dynamic={...})` composes on
one call, which the head-major path needs and which is a question for us. And whether a Python branch
on a soft-cap float inside a tiled body actually bites, which should only happen when a model enables
a soft cap, since a Python branch on a Python float resolves at trace time.

---

## 10. What the design commits to

The invariants that are load bearing, as opposed to the full list:

**No address or stride is computed from the varying dimension.** Everything else follows from this.

**The SDSC describes one tile and is fully static.** The maximum belongs in the bundle argument and in
the HBM layout, never in the per-op description.

**At most one symbolic count per dimension.** Inner levels over that dimension are static.

**A tensor carrying a declaration has storage at the declared maximum.** This holds by construction
today because the only writer of the declaration also performs the allocation. Any mechanism that
attaches a declaration to a tensor it did not allocate must verify this, because the backend asserts
an address exists and never that it fits.

**The declared axis is effectively outermost.**

**The runtime size is within range and a multiple of the granularity.** Enforced at the transfer. The
device evaluates the trip count as a plain truncating divide with no remainder check anywhere, so the
host-side check is the only thing between a non-conforming size and silent data loss.

## What is deliberately deferred

Carrying the range across a region boundary, which is the follow-up PR.

A non-outermost declared axis, which is runtime capability work.

More than one symbolic dimension in a kernel, which lowers today and is blocked by one of our own
validators.

The attention derivation, which is one change in three call sites.

## What is still open

Whether a disk cache key is stable across processes for any kernel. The static control's key moved
between two processes sharing one root, so an ordinary kernel never hits its own disk cache. The
symbolic path is the better behaved of the two.

What `steps[1]` contains for a two-symbol nest.

Whether the device cost model prices the declared maximum rather than the call.

Whether three nested symbolic levels work, and whether anything needs them.

---

## The summary that fits on one slide

Eight experiments on device. **Every failure mode found is a refusal or a retrace. No path was found
that compiles, runs and returns a wrong answer.** That covers the chain with every form of
intermediate, a bounded symbol over an undersized buffer, every granularity mismatch, a declaration on
an unsupported axis, both nesting violations, and a symbolic extent through attention.

The interface holds. Phase 2 is an extension, and its shape has already been run. What it waits on is
two capability gaps and one defect, none of which is a design question.
