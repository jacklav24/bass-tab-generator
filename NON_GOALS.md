# NON_GOALS

This document defines what **audio-pitch-core** intentionally does *not* attempt to do.

These are not temporary gaps or future roadmap items. They are explicit design boundaries that protect the architectural integrity, interpretability, and long-term usefulness of the project.

The presence of a non-goal does not imply lack of capability; it reflects deliberate restraint.

---

## 1. This Project Does Not Optimize for “Best Guess” Outputs

The core does not attempt to always produce a pitch estimate.

Abstention is a valid, meaningful result. A frame may yield `None` when the signal does not support a defensible fundamental frequency estimate. Downstream consumers are expected to treat failure explicitly, not as an error to be hidden or corrected.

Silence, noise, ambiguity, and instability are first-class signal states — not problems to be smoothed away.

---

## 2. This Project Does Not Invent Continuity

The core makes no assumptions about temporal continuity, musical phrasing, or pitch persistence across frames.

There is no implicit smoothing, interpolation, or continuity enforcement that would “fill in” missing or unstable estimates. Any process that introduces continuity is a *separate, optional refinement layer* operating on already-estimated data.

Low-level estimation does not speculate.

---

## 3. This Project Does Not Perform Musical Interpretation

Audio-pitch-core is an analytical signal-processing core, not a music analysis engine.

It does not:
- Infer notes, scales, keys, or tuning systems
- Snap frequencies to musical grids
- Encode stylistic or genre-based assumptions
- Treat pitch as inherently musical rather than physical

Any musical meaning is an interpretation applied *after* estimation, outside the core.

---

## 4. This Project Does Not Hide Uncertainty

Uncertainty is not collapsed into a single “best” value.

Confidence is explicit, inspectable, and preserved alongside estimates. Downstream layers are not permitted to retroactively “fix” or overwrite upstream uncertainty.

Ambiguity is surfaced, not resolved implicitly.

---

## 5. This Project Does Not Use Machine Learning by Default

There is no embedded machine learning, learned heuristics, or data-driven optimization in the core estimation path.

This is intentional:
- Determinism is prioritized over adaptive behavior
- Inspectability is prioritized over opaque performance gains
- Failure modes must be explainable without reference to training data

Machine learning may exist *outside* this core, but it is not a dependency of correctness.

---

## 6. This Project Does Not Mutate or Repair Prior State

Estimation stages produce new, immutable objects.

Downstream layers do not correct, reinterpret, or modify upstream outputs. If a later stage disagrees with an earlier one, it produces a new artifact — it does not rewrite history.

The analytical chain remains auditable end-to-end.

---

## 7. This Project Does Not Conflate Estimation, Refinement, and Interpretation

The system explicitly separates:
- **Estimation**: What can be defensibly inferred from raw signal
- **Refinement**: Optional post-processing of estimates
- **Interpretation**: Domain-specific meaning assigned to results

Crossing these boundaries is considered a design error.

---

## 8. This Project Does Not Optimize for Convenience or Real-Time UX

The core is not designed to:
- Always return an answer
- Mask invalid inputs
- Prefer speed over correctness
- Behave like a production audio plugin or DAW component

Its primary responsibility is analytical correctness and transparency.

---

## 9. This Project Does Not Compete With Full-Stack Audio Frameworks

Audio-pitch-core is intentionally narrow in scope.

It does not attempt to replace:
- End-to-end music information retrieval systems
- Real-time pitch correction tools
- Feature-rich audio analysis libraries
- High-level creative or interactive tooling

Its role is to serve as a trustworthy analytical substrate, not a complete solution.

---

## Closing Note

The absence of a feature should never be interpreted as an oversight unless explicitly stated elsewhere.

Boundaries are how this project maintains clarity, trustworthiness, and long-term value.
