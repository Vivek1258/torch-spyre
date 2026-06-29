# Symbolic Shapes Roadmap

A single-glance view of where the work is, what is in flight, and what
is queued. Status colours: green = landed, yellow = in flight,
blue = next to open, grey = future.

```mermaid
flowchart TB
    %% ============ FOUNDATION (LANDED) ============
    subgraph FOUNDATION ["Foundation (landed)"]
        direction TB
        F218[#218<br/>OpSpec carries sympy]
        F219[#219<br/>Constraints from ShapeEnv]
        F1371[#1371<br/>Compile symbolic /<br/>runtime concrete]
        F1372[#1372<br/>Bounded bucketing<br/>n divides granularity]
        F1373[#1373<br/>views.py opt-in]
        F2284[#2284<br/>mark_dynamic extraction]
        F2287[#2287<br/>Work division]
        F2288[#2288<br/>SDSC JSON emission]
    end

    %% ============ END-TO-END UNBLOCK (IN FLIGHT) ============
    subgraph INFLIGHT ["End-to-end unblock (in flight)"]
        direction TB
        I2289[#2289<br/>Per-core symbolic<br/>start addresses]
        I2500[#2500<br/>bundle.mlir input_args<br/>plus auto-enable]
        I2434[#2434<br/>Runtime HBM at max<br/>plus stride derivation]
    end

    %% ============ ARCHITECTURAL FOUNDATION (NEXT) ============
    subgraph ARCH ["Architectural foundation (open next)"]
        direction TB
        A5[Multi-SDSC symbol<br/>ID uniqueness]
    end

    %% ============ PHASE 1.B (NEXT) ============
    subgraph P1B ["Phase 1.B (open after Phase 0 lands)"]
        direction TB
        P1B1[Reductions<br/>Core Division]
        P1B2[Reductions<br/>SDSC and bundle.mlir]
        P1B3[Matmul / BMM<br/>Core Division]
        P1B4[Matmul / BMM<br/>SDSC and bundle.mlir]
    end

    %% ============ PHASE 2 (FUTURE) ============
    subgraph P2 ["Phase 2 (future)"]
        direction TB
        P2A[Phase 2.A<br/>Symbolic stick dim]
        P2B[Phase 2.B<br/>Middle and multi-dim<br/>symbolic]
    end

    %% ============ PHASE 3 (FUTURE) ============
    subgraph P3 ["Phase 3 (future)"]
        direction TB
        P3R[Reduction along<br/>symbolic axis]
        P3L[Symbolic LX<br/>scratchpad]
        P3P[Recompile policy<br/>plus perf tuning]
    end

    %% ============ WORKLOAD MILESTONES ============
    V1[vLLM packed-token<br/>non-attention path<br/>single binary]:::value
    V2[FMS batched serving<br/>single binary<br/>requires FMS-side mark_dynamic]:::value
    V3[FMS seq-axis activates<br/>vision / speech encoders<br/>single binary]:::value
    V4[FMS decode attention<br/>against non-paged cache]:::value

    %% ============ DEPENDENCIES ============
    FOUNDATION --> INFLIGHT
    INFLIGHT --> ARCH
    ARCH --> P1B
    P1B --> P1B1
    P1B --> P1B3
    P1B1 --> P1B2
    P1B3 --> P1B4

    P1B2 --> V1
    P1B4 --> V1
    P1B2 --> V2
    P1B4 --> V2

    P1B --> P2A
    P2A --> P2B
    P2B --> V3

    P2B --> P3
    P3 --> P3R
    P3 --> P3L
    P3 --> P3P
    P3R --> V4

    %% ============ STYLING ============
    classDef done fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20
    classDef inflight fill:#fff9c4,stroke:#f57f17,color:#e65100
    classDef next fill:#bbdefb,stroke:#1565c0,color:#0d47a1
    classDef future fill:#eeeeee,stroke:#616161,color:#212121
    classDef value fill:#f8bbd0,stroke:#ad1457,color:#880e4f,stroke-width:2px

    class F218,F219,F1371,F1372,F1373,F2284,F2287,F2288 done
    class I2289,I2500,I2434 inflight
    class A5,P1B1,P1B2,P1B3,P1B4 next
    class P2A,P2B,P3R,P3L,P3P future
```

## Legend

- **Green (Foundation)**: landed via PRs #2003, #2379, #2499, #2673. Closed issues.
- **Yellow (In flight)**: open issues on the critical path for end-to-end dispatch.
- **Blue (Next)**: queued to open immediately after the in-flight three land.
- **Grey (Future)**: scoped for later phases; no concrete issue yet.
- **Pink (Workload milestones)**: where real users see value.

## Reading the diagram

The vertical layering shows the dependency order: foundation enables the
in-flight unblock, which enables Phase 1.B, which delivers the first
workload milestones. Phases 2 and 3 follow.

The architectural foundation (Multi-SDSC ID uniqueness) lands between
the in-flight three and Phase 1.B because Phase 1.B's SDSC work assumes
bundle-global symbol IDs from day one. Landing it after Phase 1.B would
require ID rework inside #2/#4 above.

Phase 2.A is sequenced before Phase 2.B because it isolates stick-dim
mechanics from the broader middle-dim work; it does not unlock a
workload on its own.

Phase 3 is the only path to FMS-style decode attention against a
non-paged variable-length cache, because that workload reduces along
a symbolic axis.

## Cross-team dependencies (not shown on the dependency graph)

- **DeepTools**: symbolic SDSC consumer, JIT Program Correction at
  dispatch.
- **Runtime team**: #2434 implementation.
- **FMS team**: `mark_dynamic` calls on `input_ids` and KV-cache batch
  dim once Phase 1.B lands.
- **vLLM (spyre-inference)**: padding to `granularity` multiples at
  runtime; no code change for Phase 1.A.
