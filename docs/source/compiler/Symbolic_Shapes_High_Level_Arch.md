# Symbolic Shapes: High-Level Architecture

Running example: a `(560, 1024)` tensor with dim 0 marked dynamic
between 56 and 616.

```mermaid
flowchart TB
    U["User Code<br/>x = torch.rand((560, 1024))<br/>torch._dynamo.mark_dynamic(x, dim=0, min=56, max=616)<br/>compiled = torch.compile(fn)"]:::entry
    D["Dynamo<br/>trace to FX graph<br/>allocate sympy symbol s97 for the marked dim<br/>ShapeEnv holds s97 in (56, 616)"]:::stage
    L["Layout Propagation<br/>assign tile / stick layouts<br/>keep s97 alive in tensor sizes<br/>tensor shape becomes (s97, 1024)"]:::stage
    W["Core Division<br/>shard each op across cores<br/>split count n must divide granularity<br/>granularity=56, cores=32 gives n=28"]:::update
    S["Snapshot<br/>record (max, granularity) per symbol into op spec<br/>symbolic_dim_bounds = {s97: (616, 56)}<br/>ShapeEnv exits after this stage"]:::update
    C["SDSC + bundle.mlir Generation<br/>SDSC: dim symbol IDs (s97 = -1) + per-core symbolic addresses<br/>bundle.mlir: parametric input_arg + symbol operands<br/>one binary covers the whole (56, 616) range"]:::update
    DT["DeepTools + Spyre Runtime<br/>compile SDSC to Spyre binary<br/>at dispatch runtime resolves s97 (e.g. 168 = 3 * 56)<br/>patches per-core addresses"]:::entry

    U --> D
    D --> L
    L --> W
    W --> S
    S --> C
    C --> DT

    subgraph BOUNDS["Symbolic Bounds Track"]
      direction TB
      B["max = 616 (from user upper)<br/>granularity = 56 (from user lower)<br/>feeds Core Division and Snapshot"]:::update
    end

    D -.-> B
    B -.-> W
    B -.-> S

    classDef entry fill:#e1f5fe,stroke:#0277bd,color:#01579b
    classDef stage fill:#f5f5f5,stroke:#616161,color:#212121
    classDef update fill:#fff3e0,stroke:#f57c00,color:#e65100
```

## PR references

- Max: [#2003](https://github.com/torch-spyre/torch-spyre/pull/2003)
- Granularity: [#2379](https://github.com/torch-spyre/torch-spyre/pull/2379)
- Symbolic Core Division for pointwise ops:
  [#2499](https://github.com/torch-spyre/torch-spyre/pull/2499)
- Symbolic SDSC generation:
  [#2673](https://github.com/torch-spyre/torch-spyre/pull/2673)
- bundle.mlir + per-core start addresses: in flight (this PR)

