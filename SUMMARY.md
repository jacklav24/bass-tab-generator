# Project Summary

This project is a deterministic, inspectable audio analysis core focused on per-frame pitch estimation with explicit uncertainty and abstention. It is designed as a stable foundation for downstream analysis and experimentation, rather than a one-off application or end-user tool.

The emphasis is on **clear data contracts**, **defensive design**, and **empirical characterization** of estimator behavior before introducing machine learning or musical interpretation.

## What Makes This Project Distinct

- Deterministic, frame-based signal processing with exact time semantics  
- Explicit failure handling (`f0_hz = None` is a valid and meaningful outcome)  
- Strict separation between estimation, optional refinement, and analysis  
- Conservative temporal smoothing that may legitimately make no changes  
- Diagnostic metrics that expose estimator behavior instead of hiding errors  

Rather than optimizing pitch output, the project prioritizes **inspectability and honesty** about what the system knows—and what it does not.

## What This Project Intentionally Avoids

This project deliberately does **not**:
- Apply musical priors or interpretation at low levels  
- Enforce continuity or “fix” pitch estimates  
- Make accuracy claims without ground truth  
- Introduce machine learning prematurely  

These design decisions are documented in more detail in `NON_GOALS.md`.

## What Exists Today

The current implementation includes:
- A validated `AudioBuffer` abstraction with exact sample-to-time semantics  
- Deterministic framing into fixed-size, overlapping frames  
- Frame-local pitch estimation with confidence and explicit abstention  
- An optional, confidence-aware temporal refinement layer  
- A diagnostic layer for analyzing voicing behavior, pitch stability, and confidence reliability  

All refinement and analysis layers operate strictly downstream of core abstractions and never mutate upstream state.

## How to Read the Repository

- See `README.md` for full architectural contracts and design philosophy  
- See `NON_GOALS.md` for explicit non-goals and scope boundaries  

This project is intended to grow by **layering**, not by correction.
