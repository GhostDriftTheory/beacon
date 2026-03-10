# Beacon ( MG-OS + GD-Attention Demo )

<p align="center">
  <img src="protect_then_select_architecture.png" width="720">
</p>

**Related resources**

Paper (preprint)  
MG-OS: https://zenodo.org/records/17712891  
GD-Attention: https://zenodo.org/records/16757311  

Project page  
https://www.ghostdriftresearch.com/

Repository  
https://github.com/GhostDriftTheory/semantic-selection-demo


**Minimal architecture demo of:**

- **Transformer-style softmax attention**
- **MG-OS conditional barrier layer**
- **GD-Attention-style single-candidate selection**

This repository visualizes a compact architectural contrast:

**Softmax mixes.**  
**MG-OS + GD-Attention protects at-risk minority-important signals and then selects.**

---

## Core architecture path

```text
Transformer-like attention logits
↓
MG-OS conditional barrier
↓
GD-style single selection
```

This demo is designed to make that path visible in a small, reproducible form.

---

## What this updated demo shows

The app compares two routes on the same toy task.

### 1. Baseline: softmax attention

- compute attention logits
- apply softmax
- mix candidate values
- produce an output distribution

### 2. Proposed: MG-OS + GD-Attention

- compute attention logits
- detect whether a minority-important candidate is at risk of premature collapse
- activate an **MG-OS-style barrier** only in those risky cases
- preserve the candidate above a protective floor or near-contention region
- apply **GD-style single-candidate selection**
- produce an output from the selected candidate

---

## Main visible outputs

This app focuses on the following architecture-level differences:

- **minority survival**
- **minority recall**
- **minority top-2 retention**
- **collapse avoidance**
- **rescue rate**
- **false rescue rate**
- **barrier activation rate**
- **Gamma proxy shift**

The point is not only whether the final prediction changes, but **how the internal competition changes before selection**.

---

## Representative case view

The app shows one representative case in detail.

Rather than displaying an arbitrary minority example, it preferentially displays a **representative rescue case**:

- a case where softmax fails
- but MG-OS + GD-Attention succeeds

If no such case exists under the current setting, the app falls back to:

- the largest gamma-gain minority case, or
- the hardest raw-minority case

For that case, the app visualizes:

- query and candidate-key geometry
- raw attention logits vs barriered logits
- softmax weights vs protected weights
- softmax mixed output vs GD-style selected output
- minority rank change
- Gamma change
- barrier boost size

This is meant to make the **protect-then-select** mechanism directly observable.

---

## Batch evaluation view

The app also evaluates the architecture over many samples.

It reports:

- accuracy
- balanced accuracy
- minority survival
- minority recall
- minority top-2 retention
- collapse rate on minority cases
- rescue rate
- false rescue rate
- barrier activation rate
- mean Gamma delta

These metrics are intended to show not only whether the proposed route helps minority-important cases, but also whether it does so **without blindly biasing every sample**.

---

## Why the conditional barrier matters

This demo does **not** implement the barrier as a permanent minority bias.

Instead, the barrier is intended to activate only when the minority-important candidate is:

- near contention,
- under a protective floor,
- or in an ambiguous competitive state.

This makes the demo closer to an **architecture intervention** than to a fixed class preference.

---

## Presets included

The app includes three initial presets:

- **Rescue-first**
- **Balanced**
- **Mild stress**

These presets are there so the architectural contrast is visible quickly, especially in the rescue-oriented regime.

---

## Repository structure

- `app.py` — Streamlit app
- `requirements.txt` — minimal dependencies

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Interpretation

This is a **compact architecture demo**, not a production model.

Its purpose is to make the following structural claim visible:

- standard softmax attention primarily **mixes**
- MG-OS introduces a **conditional protective barrier**
- GD-Attention then **selects** instead of blending everything away

The intended contribution of the demo is therefore not large-scale benchmark performance, but a visible contrast between:

- **mix-first architectures**
- and
- **protect-then-select architectures**

---

## Scope

This repository does **not** include:

- a full transformer implementation
- a full MoE implementation
- LLM training
- real-world deployment claims
- large-scale benchmark claims

It is a compact visualization of an architecture idea.

