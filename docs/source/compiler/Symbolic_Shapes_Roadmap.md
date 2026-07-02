# Symbolic Shapes Roadmap

**Last updated:** 2026-07-02
**Target:** August deadline. Two-week sprints. Development and testing
run in parallel; every dev PR gets a test-plan slice in the same
sprint, with a feedback arrow when bugs surface.

## Quick navigation

- [Already landed](#already-landed)

## Already landed

Foundation is closed. Eight issues merged via PRs #2003, #2379, #2499,
#2673: `mark_dynamic` extraction, symbolic OpSpec, bucketing
invariant, work division, SDSC JSON emission with dim-symbol metadata.
See the [GitHub Issues Compendium](Symbolic_Shapes_GitHub_Issues.md)
for the verbatim record.

## Sprint roadmap

Left to right is time. Vertical rows inside each sprint are the four
parallel tracks. Dotted arrows carry the test-fix loop back into dev.

```mermaid
flowchart LR
    %% =============== SPRINT 1 ===============
    subgraph SPR1 ["Sprint 1: Jul 2 - Jul 16"]
        direction TB
        S1_DEV["<b>Dev</b><br/>#2289 lands<br/>#2500 in review<br/>#221 late-bind explor."]:::inflight
        S1_CT["<b>Cross-team</b><br/>0.2 DeepTools interface kickoff<br/>0.3 HBM API decision<br/>#2408 DT-side sync"]:::inflight
        S1_VLLM["<b>vLLM + Granite</b><br/>0.1 Granite readiness memo<br/>Plugin blocker triage"]:::inflight
        S1_TEST["<b>Testing</b><br/>#2279 pointwise coverage<br/>#2289 pod tests<br/>#3005 constraint bug"]:::inflight
    end

    %% =============== SPRINT 2 ===============
    subgraph SPR2 ["Sprint 2: Jul 16 - Jul 30"]
        direction TB
        S2_DEV["<b>Dev</b><br/>#2500 lands<br/>Ticket 1 Multi-SDSC IDs<br/>Ticket 2 Reductions core div<br/>Ticket 4 BMM core div"]:::next
        S2_CT["<b>Cross-team</b><br/>#2434 runtime work<br/>#2408 consumer sync<br/>0.3 API decision closed"]:::atrisk
        S2_VLLM["<b>vLLM + Granite</b><br/>Plugin coord: three blockers<br/>Granite demo scope"]:::next
        S2_TEST["<b>Testing</b><br/>#2500 pod tests<br/>Multi-SDSC unit tests<br/>#220 SDSC path tests"]:::next
    end

    %% =============== SPRINT 3 ===============
    subgraph SPR3 ["Sprint 3 buffer: Jul 30 - August"]
        direction TB
        S3_DEV["<b>Dev</b><br/>Ticket 3 Reductions SDSC<br/>Ticket 5 BMM SDSC<br/>Bug-fix window"]:::next
        S3_CT["<b>Cross-team</b><br/>#2434 lands<br/>DT consumer end-to-end"]:::atrisk
        S3_VLLM["<b>vLLM + Granite</b><br/>Granite demo prep<br/>Plugin PRs open"]:::next
        S3_TEST["<b>Testing</b><br/>End-to-end smoke<br/>Regression sweep<br/>Granite validation"]:::next
    end

    %% =============== SPRINT FLOW ===============
    SPR1 --> SPR2 --> SPR3

    %% =============== TEST-FIX LOOP ===============
    S1_TEST -.->|test bugs| S1_DEV
    S2_TEST -.->|test bugs| S2_DEV
    S3_TEST -.->|test bugs| S3_DEV
    S1_TEST -.->|carried bugs| S2_DEV
    S2_TEST -.->|carried bugs| S3_DEV

    %% =============== STYLING ===============
    classDef inflight fill:#fff9c4,stroke:#f57f17,color:#e65100
    classDef next fill:#bbdefb,stroke:#1565c0,color:#0d47a1
    classDef atrisk fill:#ffcdd2,stroke:#c62828,color:#b71c1c,stroke-width:2px
    classDef done fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20
```
md](Symbolic_Shapes_Phasewise_Usecases.md)
  for the workload-to-scope mapping.
