import math
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="MG-OS + GD-Attention Demo",
    page_icon="🧭",
    layout="wide",
)


# ============================================================
# Helpers
# ============================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def floor3(x: float) -> float:
    return math.floor(float(x) * 1000.0) / 1000.0


def normalized_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1.0)
    h = -np.sum(q * np.log(q), axis=1)
    return h / math.log(p.shape[1])


def class_recall(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> float:
    mask = y_true == cls
    if not np.any(mask):
        return 0.0
    return float(np.mean(y_pred[mask] == cls))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    recalls = [class_recall(y_true, y_pred, cls) for cls in range(num_classes)]
    return float(np.mean(recalls))


def apply_preset(name: str) -> None:
    presets = {
        "Rescue-first": {
            "batch_size": 900,
            "minority_fraction": 0.10,
            "noise_std": 0.42,
            "deletion_strength": 0.70,
            "score_noise": 0.08,
            "sharpness": 2.8,
            "majority_bias": 1.65,
            "barrier_floor": 0.18,
            "rescue_margin": 1.45,
            "barrier_strength": 2.8,
            "entropy_trigger": 0.63,
            "trigger_slack": 0.45,
            "detect_floor": 0.18,
        },
        "Balanced": {
            "batch_size": 900,
            "minority_fraction": 0.10,
            "noise_std": 0.38,
            "deletion_strength": 0.58,
            "score_noise": 0.07,
            "sharpness": 2.7,
            "majority_bias": 1.35,
            "barrier_floor": 0.16,
            "rescue_margin": 1.20,
            "barrier_strength": 2.1,
            "entropy_trigger": 0.61,
            "trigger_slack": 0.35,
            "detect_floor": 0.16,
        },
        "Mild stress": {
            "batch_size": 900,
            "minority_fraction": 0.08,
            "noise_std": 0.34,
            "deletion_strength": 0.45,
            "score_noise": 0.05,
            "sharpness": 2.5,
            "majority_bias": 1.10,
            "barrier_floor": 0.14,
            "rescue_margin": 1.00,
            "barrier_strength": 1.8,
            "entropy_trigger": 0.59,
            "trigger_slack": 0.30,
            "detect_floor": 0.14,
        },
    }
    for k, v in presets[name].items():
        st.session_state[k] = v


# ============================================================
# Toy architecture ingredients
# ============================================================

def make_candidate_table() -> Tuple[np.ndarray, list[str], np.ndarray]:
    keys = np.array(
        [
            [-2.20, -0.75],
            [ 2.05, -0.70],
            [ 0.05,  1.72],
            [ 0.78,  2.58],
            [ 0.62,  1.28],
        ],
        dtype=float,
    )
    names = [
        "majority-0",
        "majority-1",
        "majority-2",
        "minority-important",
        "distractor",
    ]
    value_map = np.array(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.00, 1.00],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=float,
    )
    return keys, names, value_map


def sample_queries(
    batch_size: int,
    minority_fraction: float,
    noise_std: float,
    deletion_strength: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys, _, _ = make_candidate_table()

    p_major = max(0.0, 1.0 - minority_fraction) / 3.0
    probs = np.array([p_major, p_major, p_major, minority_fraction], dtype=float)
    probs = probs / probs.sum()
    y = rng.choice(4, size=batch_size, p=probs)

    X = np.zeros((batch_size, 2), dtype=float)
    minority_case = np.zeros(batch_size, dtype=bool)

    majority_2 = keys[2]
    minority = keys[3]
    distractor = keys[4]

    for i in range(batch_size):
        q = keys[y[i]].copy() + rng.normal(0.0, noise_std, size=2)

        if y[i] == 3:
            minority_case[i] = True
            alpha = deletion_strength
            pull = (0.58 * majority_2) + (0.30 * distractor) + (0.12 * minority)
            q = (1.0 - alpha) * q + alpha * pull
            q += rng.normal(0.0, 0.10 + 0.15 * deletion_strength, size=2)

        X[i] = q

    return X, y, minority_case


def compute_attention_logits(
    X: np.ndarray,
    sharpness: float,
    majority_bias: float,
    seed: int,
    score_noise: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keys, _, _ = make_candidate_table()
    scale = math.sqrt(X.shape[1])
    attention_logits = sharpness * ((X @ keys.T) / scale)

    bias = np.array(
        [
            majority_bias,
            majority_bias,
            majority_bias,
            -majority_bias,
            0.35 * majority_bias,
        ],
        dtype=float,
    )
    attention_logits = attention_logits + bias

    if score_noise > 0:
        attention_logits = attention_logits + rng.normal(0.0, score_noise, size=attention_logits.shape)

    return attention_logits


@dataclass
class ForwardResult:
    raw_logits: np.ndarray
    raw_weights: np.ndarray
    protected_logits: np.ndarray
    protected_weights: np.ndarray
    baseline_probs: np.ndarray
    proposed_probs: np.ndarray
    baseline_pred: np.ndarray
    proposed_pred: np.ndarray
    selected_candidate: np.ndarray
    gamma_before: np.ndarray
    gamma_after: np.ndarray
    gamma_delta: np.ndarray
    barrier_boost: np.ndarray
    barrier_activated: np.ndarray
    raw_entropy: np.ndarray
    minority_rank_before: np.ndarray
    minority_rank_after: np.ndarray


def barrier_layer(
    raw_logits: np.ndarray,
    barrier_floor: float,
    rescue_margin: float,
    barrier_strength: float,
    entropy_trigger: float,
    trigger_slack: float,
    minority_idx: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    protected = raw_logits.copy()
    raw_weights = softmax(raw_logits, axis=1)
    raw_entropy = normalized_entropy(raw_weights)

    batch = raw_logits.shape[0]
    boost = np.zeros(batch, dtype=float)
    activated = np.zeros(batch, dtype=bool)
    gamma_before = np.zeros(batch, dtype=float)
    gamma_after = np.zeros(batch, dtype=float)
    rank_before = np.zeros(batch, dtype=int)
    rank_after = np.zeros(batch, dtype=int)

    for i in range(batch):
        others = [j for j in range(raw_logits.shape[1]) if j != minority_idx]
        best_other_before = float(np.max(raw_logits[i, others]))
        minority_before = float(raw_logits[i, minority_idx])
        gamma_before[i] = minority_before - best_other_before
        rank_before[i] = 1 + int(np.sum(raw_logits[i] > minority_before))

        minority_weight = float(raw_weights[i, minority_idx])
        gap = best_other_before - minority_before
        near_contention = gap <= rescue_margin
        under_floor = minority_weight < barrier_floor
        ambiguous = raw_entropy[i] >= entropy_trigger
        activate = (under_floor and gap <= rescue_margin + trigger_slack) or (near_contention and ambiguous)

        if activate:
            activated[i] = True
            margin_boost = barrier_strength * max(0.0, rescue_margin - gap)
            if margin_boost > 0:
                protected[i, minority_idx] += margin_boost
                boost[i] += margin_boost

            weights_now = softmax(protected[i:i + 1], axis=1)[0]
            w_min = float(weights_now[minority_idx])
            if w_min < barrier_floor:
                other_logits = protected[i, others]
                max_other = np.max(other_logits)
                logsumexp_others = float(np.log(np.sum(np.exp(other_logits - max_other))) + max_other)
                target_logit = math.log(barrier_floor / max(1e-8, 1.0 - barrier_floor)) + logsumexp_others
                needed = max(0.0, target_logit - protected[i, minority_idx])
                protected[i, minority_idx] += needed
                boost[i] += needed

        best_other_after = float(np.max(protected[i, others]))
        minority_after = float(protected[i, minority_idx])
        gamma_after[i] = minority_after - best_other_after
        rank_after[i] = 1 + int(np.sum(protected[i] > minority_after))

    protected_weights = softmax(protected, axis=1)
    gamma_delta = gamma_after - gamma_before
    return (
        protected,
        protected_weights,
        gamma_before,
        gamma_after,
        gamma_delta,
        boost,
        activated,
        raw_entropy,
        rank_before,
        rank_after,
    )


def forward_architecture(
    X: np.ndarray,
    sharpness: float,
    majority_bias: float,
    barrier_floor: float,
    rescue_margin: float,
    barrier_strength: float,
    entropy_trigger: float,
    trigger_slack: float,
    seed: int,
    score_noise: float,
) -> ForwardResult:
    _, _, value_map = make_candidate_table()

    raw_logits = compute_attention_logits(
        X=X,
        sharpness=sharpness,
        majority_bias=majority_bias,
        seed=seed,
        score_noise=score_noise,
    )
    raw_weights = softmax(raw_logits, axis=1)

    (
        protected_logits,
        protected_weights,
        gamma_before,
        gamma_after,
        gamma_delta,
        boost,
        activated,
        raw_entropy,
        rank_before,
        rank_after,
    ) = barrier_layer(
        raw_logits=raw_logits,
        barrier_floor=barrier_floor,
        rescue_margin=rescue_margin,
        barrier_strength=barrier_strength,
        entropy_trigger=entropy_trigger,
        trigger_slack=trigger_slack,
        minority_idx=3,
    )

    baseline_probs = raw_weights @ value_map
    baseline_pred = np.argmax(baseline_probs, axis=1)

    selected_candidate = np.argmax(protected_logits, axis=1)
    proposed_probs = value_map[selected_candidate]
    proposed_pred = np.argmax(proposed_probs, axis=1)

    return ForwardResult(
        raw_logits=raw_logits,
        raw_weights=raw_weights,
        protected_logits=protected_logits,
        protected_weights=protected_weights,
        baseline_probs=baseline_probs,
        proposed_probs=proposed_probs,
        baseline_pred=baseline_pred,
        proposed_pred=proposed_pred,
        selected_candidate=selected_candidate,
        gamma_before=gamma_before,
        gamma_after=gamma_after,
        gamma_delta=gamma_delta,
        barrier_boost=boost,
        barrier_activated=activated,
        raw_entropy=raw_entropy,
        minority_rank_before=rank_before,
        minority_rank_after=rank_after,
    )


# ============================================================
# Metrics and representative case
# ============================================================

def architecture_metrics(
    y: np.ndarray,
    minority_case: np.ndarray,
    result: ForwardResult,
    detect_floor: float,
    minority_idx: int = 3,
) -> Dict[str, float]:
    base_acc = float(np.mean(result.baseline_pred == y))
    prop_acc = float(np.mean(result.proposed_pred == y))
    base_bal_acc = balanced_accuracy(y, result.baseline_pred)
    prop_bal_acc = balanced_accuracy(y, result.proposed_pred)

    if np.any(minority_case):
        raw_minority_weights = result.raw_weights[minority_case, minority_idx]
        protected_minority_weights = result.protected_weights[minority_case, minority_idx]

        base_survival = float(np.mean(raw_minority_weights >= detect_floor))
        prop_survival = float(np.mean(protected_minority_weights >= detect_floor))
        base_top2 = float(np.mean(result.minority_rank_before[minority_case] <= 2))
        prop_top2 = float(np.mean(result.minority_rank_after[minority_case] <= 2))

        base_minority_acc = float(np.mean(result.baseline_pred[minority_case] == 3))
        prop_minority_acc = float(np.mean(result.proposed_pred[minority_case] == 3))

        base_collapse = float(np.mean(result.baseline_pred[minority_case] != 3))
        prop_collapse = float(np.mean(result.proposed_pred[minority_case] != 3))

        rescue_mask = minority_case & (result.baseline_pred != y) & (result.proposed_pred == y)
        false_rescue_mask = minority_case & (result.baseline_pred == y) & (result.proposed_pred != y)
        rescue_rate = float(np.mean(rescue_mask[minority_case]))
        false_rescue_rate = float(np.mean(false_rescue_mask[minority_case]))

        barrier_on_minority = float(np.mean(result.barrier_activated[minority_case]))
        gamma_gain_minority = float(np.mean(result.gamma_delta[minority_case]))
    else:
        base_survival = prop_survival = 0.0
        base_top2 = prop_top2 = 0.0
        base_minority_acc = prop_minority_acc = 0.0
        base_collapse = prop_collapse = 0.0
        rescue_rate = false_rescue_rate = 0.0
        barrier_on_minority = gamma_gain_minority = 0.0

    return {
        "baseline_acc": base_acc,
        "proposed_acc": prop_acc,
        "delta_acc": prop_acc - base_acc,
        "baseline_bal_acc": base_bal_acc,
        "proposed_bal_acc": prop_bal_acc,
        "delta_bal_acc": prop_bal_acc - base_bal_acc,
        "baseline_minority_survival": base_survival,
        "proposed_minority_survival": prop_survival,
        "delta_minority_survival": prop_survival - base_survival,
        "baseline_minority_top2": base_top2,
        "proposed_minority_top2": prop_top2,
        "delta_minority_top2": prop_top2 - base_top2,
        "baseline_minority_acc": base_minority_acc,
        "proposed_minority_acc": prop_minority_acc,
        "delta_minority_acc": prop_minority_acc - base_minority_acc,
        "baseline_collapse": base_collapse,
        "proposed_collapse": prop_collapse,
        "delta_collapse": prop_collapse - base_collapse,
        "rescue_rate": rescue_rate,
        "false_rescue_rate": false_rescue_rate,
        "barrier_activation_rate": float(np.mean(result.barrier_activated)),
        "barrier_activation_on_minority": barrier_on_minority,
        "gamma_before_mean": float(np.mean(result.gamma_before)),
        "gamma_after_mean": float(np.mean(result.gamma_after)),
        "gamma_delta_mean": float(np.mean(result.gamma_delta)),
        "gamma_delta_minority": gamma_gain_minority,
    }


def choose_representative_case(
    y: np.ndarray,
    minority_case: np.ndarray,
    result: ForwardResult,
) -> Tuple[int, str]:
    rescue = np.where(minority_case & (result.baseline_pred != y) & (result.proposed_pred == y))[0]
    if rescue.size > 0:
        best = rescue[np.argmax(result.gamma_delta[rescue])]
        return int(best), "Representative rescue case"

    minority_idxs = np.where(minority_case)[0]
    if minority_idxs.size > 0:
        best = minority_idxs[np.argmax(result.gamma_delta[minority_idxs])]
        return int(best), "Largest gamma-gain minority case"

    best = int(np.argmin(result.raw_weights[:, 3]))
    return best, "Hardest raw-minority case"


# ============================================================
# Plotting
# ============================================================

def plot_single_case(
    query: np.ndarray,
    y_true: int,
    candidate_names: list[str],
    keys: np.ndarray,
    result: ForwardResult,
    idx: int,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.8))

    # Query / keys geometry
    ax = axes[0]
    ax.scatter(keys[:, 0], keys[:, 1], s=120)
    for i, name in enumerate(candidate_names):
        ax.text(keys[i, 0] + 0.04, keys[i, 1] + 0.04, name, fontsize=9)
    ax.scatter(query[0], query[1], s=170, marker="x", linewidths=3)
    ax.set_title(f"Query vs keys | true class={y_true}")
    ax.grid(alpha=0.2)

    # Logits before / after barrier
    ax = axes[1]
    x = np.arange(len(candidate_names))
    width = 0.36
    ax.bar(x - width / 2, result.raw_logits[idx], width=width, label="raw logits")
    ax.bar(x + width / 2, result.protected_logits[idx], width=width, label="barriered logits")
    ax.set_xticks(x)
    ax.set_xticklabels(candidate_names, rotation=25, ha="right")
    ax.set_title("Attention logits")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    # Weights before / after barrier
    ax = axes[2]
    ax.bar(x - width / 2, result.raw_weights[idx], width=width, label="softmax weights")
    ax.bar(x + width / 2, result.protected_weights[idx], width=width, label="protected weights")
    ax.set_xticks(x)
    ax.set_xticklabels(candidate_names, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Attention weights")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    # Output comparison
    ax = axes[3]
    cls = np.arange(4)
    ax.bar(cls - width / 2, result.baseline_probs[idx], width=width, label="softmax mix")
    ax.bar(cls + width / 2, result.proposed_probs[idx], width=width, label="MG-OS + GD select")
    ax.set_xticks(cls)
    ax.set_xticklabels(["0", "1", "2", "3"])
    ax.set_ylim(0, 1)
    ax.set_title(
        "Output probabilities\n"
        f"base={result.baseline_pred[idx]} | proposed={result.proposed_pred[idx]} | cand={result.selected_candidate[idx]}"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def plot_delta_bars(metrics: Dict[str, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = [
        "Δ acc",
        "Δ balanced acc",
        "Δ minority survival",
        "Δ minority recall",
        "Δ minority top-2",
        "Δ collapse",
    ]
    values = [
        metrics["delta_acc"],
        metrics["delta_bal_acc"],
        metrics["delta_minority_survival"],
        metrics["delta_minority_acc"],
        metrics["delta_minority_top2"],
        metrics["delta_collapse"],
    ]
    ax.bar(labels, values)
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Architecture deltas (proposed - softmax)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_batch_bars(metrics: Dict[str, float]) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.2))

    axes[0].bar(["softmax", "MG-OS+GD"], [metrics["baseline_acc"], metrics["proposed_acc"]])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Accuracy")
    axes[0].grid(alpha=0.2)

    axes[1].bar(["softmax", "MG-OS+GD"], [metrics["baseline_bal_acc"], metrics["proposed_bal_acc"]])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Balanced accuracy")
    axes[1].grid(alpha=0.2)

    axes[2].bar(["softmax", "MG-OS+GD"], [metrics["baseline_minority_survival"], metrics["proposed_minority_survival"]])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Minority survival")
    axes[2].grid(alpha=0.2)

    axes[3].bar(["softmax", "MG-OS+GD"], [metrics["baseline_collapse"], metrics["proposed_collapse"]])
    axes[3].set_ylim(0, 1)
    axes[3].set_title("Collapse rate on minority cases")
    axes[3].grid(alpha=0.2)

    fig.tight_layout()
    return fig


def plot_gamma_hist(gamma_before: np.ndarray, gamma_after: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(gamma_before, bins=28, alpha=0.60, label="before barrier")
    ax.hist(gamma_after, bins=28, alpha=0.60, label="after barrier")
    ax.axvline(0.0, linestyle="--")
    ax.set_title("Gamma proxy shift")
    ax.set_xlabel("minority logit - best rival logit")
    ax.set_ylabel("count")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_selection_counts(
    baseline_pred: np.ndarray,
    proposed_pred: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["0", "1", "2", "3"]

    base_counts = np.bincount(baseline_pred, minlength=4)
    prop_counts = np.bincount(proposed_pred, minlength=4)

    axes[0].bar(labels, base_counts)
    axes[0].set_title("Softmax predicted classes")
    axes[0].grid(alpha=0.2)

    axes[1].bar(labels, prop_counts)
    axes[1].set_title("MG-OS + GD predicted classes")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    return fig


# ============================================================
# UI
# ============================================================
st.title("MG-OS + GD-Attention Architecture Demo")
st.caption(
    "Transformer-like attention logits -> MG-OS barrier -> GD-style single selection."
)

with st.sidebar:
    st.header("Preset")
    preset = st.selectbox("Initial view", ["Rescue-first", "Balanced", "Mild stress"], index=0)
    if st.button("Apply preset", use_container_width=True):
        apply_preset(preset)

    st.markdown("---")
    st.header("Global")
    seed = st.slider("Seed", min_value=1, max_value=9999, value=184, step=1)
    batch_size = st.slider("Batch size", min_value=100, max_value=4000, value=st.session_state.get("batch_size", 900), step=50, key="batch_size")

    st.markdown("---")
    st.header("Task")
    minority_fraction = st.slider("Minority fraction", min_value=0.02, max_value=0.40, value=st.session_state.get("minority_fraction", 0.10), step=0.01, key="minority_fraction")
    noise_std = st.slider("Query noise", min_value=0.05, max_value=1.20, value=st.session_state.get("noise_std", 0.42), step=0.01, key="noise_std")
    deletion_strength = st.slider("Minority evidence deletion", min_value=0.0, max_value=1.0, value=st.session_state.get("deletion_strength", 0.70), step=0.01, key="deletion_strength")
    score_noise = st.slider("Score noise", min_value=0.0, max_value=1.0, value=st.session_state.get("score_noise", 0.08), step=0.01, key="score_noise")

    st.markdown("---")
    st.header("Softmax attention")
    sharpness = st.slider("Logit sharpness", min_value=0.5, max_value=6.0, value=st.session_state.get("sharpness", 2.8), step=0.05, key="sharpness")
    majority_bias = st.slider("Bias toward majority", min_value=0.0, max_value=4.0, value=st.session_state.get("majority_bias", 1.65), step=0.05, key="majority_bias")

    st.markdown("---")
    st.header("MG-OS barrier + GD selection")
    barrier_floor = st.slider("Minority floor", min_value=0.01, max_value=0.45, value=st.session_state.get("barrier_floor", 0.18), step=0.01, key="barrier_floor")
    rescue_margin = st.slider("Rescue margin", min_value=0.0, max_value=3.0, value=st.session_state.get("rescue_margin", 1.45), step=0.05, key="rescue_margin")
    barrier_strength = st.slider("Barrier strength", min_value=0.0, max_value=6.0, value=st.session_state.get("barrier_strength", 2.8), step=0.05, key="barrier_strength")
    entropy_trigger = st.slider("Entropy trigger", min_value=0.20, max_value=0.95, value=st.session_state.get("entropy_trigger", 0.63), step=0.01, key="entropy_trigger")
    trigger_slack = st.slider("Trigger slack", min_value=0.0, max_value=1.50, value=st.session_state.get("trigger_slack", 0.45), step=0.01, key="trigger_slack")
    detect_floor = st.slider("Detection floor", min_value=0.01, max_value=0.45, value=st.session_state.get("detect_floor", 0.18), step=0.01, key="detect_floor")


# ============================================================
# Run simulation
# ============================================================
keys, candidate_names, _ = make_candidate_table()
X, y, minority_case = sample_queries(
    batch_size=batch_size,
    minority_fraction=minority_fraction,
    noise_std=noise_std,
    deletion_strength=deletion_strength,
    seed=seed,
)

result = forward_architecture(
    X=X,
    sharpness=sharpness,
    majority_bias=majority_bias,
    barrier_floor=barrier_floor,
    rescue_margin=rescue_margin,
    barrier_strength=barrier_strength,
    entropy_trigger=entropy_trigger,
    trigger_slack=trigger_slack,
    seed=seed + 17,
    score_noise=score_noise,
)
metrics = architecture_metrics(
    y=y,
    minority_case=minority_case,
    result=result,
    detect_floor=detect_floor,
)

single_idx, single_reason = choose_representative_case(y=y, minority_case=minority_case, result=result)


# ============================================================
# Summary strip
# ============================================================
cols = st.columns(6)
cols[0].metric("Δ accuracy", f"{metrics['delta_acc']:+.3f}", f"softmax {metrics['baseline_acc']:.3f} -> proposed {metrics['proposed_acc']:.3f}")
cols[1].metric("Δ balanced acc", f"{metrics['delta_bal_acc']:+.3f}", f"{metrics['baseline_bal_acc']:.3f} -> {metrics['proposed_bal_acc']:.3f}")
cols[2].metric("Δ minority survival", f"{metrics['delta_minority_survival']:+.3f}", f"{metrics['baseline_minority_survival']:.3f} -> {metrics['proposed_minority_survival']:.3f}")
cols[3].metric("Rescue rate", f"{metrics['rescue_rate']:.3f}", f"false rescue {metrics['false_rescue_rate']:.3f}")
cols[4].metric("Barrier activation", f"{metrics['barrier_activation_rate']:.3f}", f"on minority {metrics['barrier_activation_on_minority']:.3f}")
cols[5].metric("Mean Δ gamma", f"{metrics['gamma_delta_mean']:+.3f}", f"minority {metrics['gamma_delta_minority']:+.3f}")

with st.expander("What this updated demo is showing", expanded=False):
    st.markdown(
        """
- **Softmax attention** mixes candidate values from raw attention weights.
- **MG-OS barrier** activates only when the minority-important candidate is near contention or at risk of premature collapse.
- **GD-style selection** then commits to one candidate after the barrier step.
- The core architectural readout is not just raw accuracy but **minority survival, rescue rate, false rescue, and gamma shift**.
        """
    )


# ============================================================
# Tabs
# ============================================================
tab1, tab2 = st.tabs(["Representative case", "Batch evaluation"])

with tab1:
    st.subheader("Representative case")
    st.caption(single_reason)

    single_fig = plot_single_case(
        query=X[single_idx],
        y_true=int(y[single_idx]),
        candidate_names=candidate_names,
        keys=keys,
        result=result,
        idx=single_idx,
    )
    st.pyplot(single_fig, use_container_width=True)

    info_cols = st.columns(6)
    info_cols[0].metric("True class", int(y[single_idx]))
    info_cols[1].metric("Softmax pred", int(result.baseline_pred[single_idx]))
    info_cols[2].metric("Proposed pred", int(result.proposed_pred[single_idx]))
    info_cols[3].metric("Selected cand", int(result.selected_candidate[single_idx]))
    info_cols[4].metric("Barrier active", "yes" if result.barrier_activated[single_idx] else "no")
    info_cols[5].metric("Entropy", f"{result.raw_entropy[single_idx]:.3f}")

    case_df = pd.DataFrame(
        {
            "candidate": candidate_names,
            "raw_logit": [round(v, 4) for v in result.raw_logits[single_idx]],
            "protected_logit": [round(v, 4) for v in result.protected_logits[single_idx]],
            "logit_delta": [round(v, 4) for v in (result.protected_logits[single_idx] - result.raw_logits[single_idx])],
            "raw_weight": [round(v, 4) for v in result.raw_weights[single_idx]],
            "protected_weight": [round(v, 4) for v in result.protected_weights[single_idx]],
            "weight_delta": [round(v, 4) for v in (result.protected_weights[single_idx] - result.raw_weights[single_idx])],
        }
    )
    st.dataframe(case_df, use_container_width=True, hide_index=True)

    st.markdown(
        f"**Minority rank:** {result.minority_rank_before[single_idx]} -> {result.minority_rank_after[single_idx]}  \n"
        f"**Gamma:** {result.gamma_before[single_idx]:.4f} -> {result.gamma_after[single_idx]:.4f}  \n"
        f"**Barrier boost:** {result.barrier_boost[single_idx]:.4f}"
    )

with tab2:
    st.subheader("Batch evaluation")

    left, right = st.columns([1.35, 1.0])
    with left:
        fig_delta = plot_delta_bars(metrics)
        st.pyplot(fig_delta, use_container_width=True)

        fig_bars = plot_batch_bars(metrics)
        st.pyplot(fig_bars, use_container_width=True)

        fig_sel = plot_selection_counts(
            baseline_pred=result.baseline_pred,
            proposed_pred=result.proposed_pred,
        )
        st.pyplot(fig_sel, use_container_width=True)

    with right:
        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "accuracy",
                    "balanced accuracy",
                    "minority survival",
                    "minority recall",
                    "minority top-2",
                    "collapse rate",
                    "rescue rate",
                    "false rescue",
                    "barrier activation",
                    "mean gamma delta",
                ],
                "softmax": [
                    floor3(metrics["baseline_acc"]),
                    floor3(metrics["baseline_bal_acc"]),
                    floor3(metrics["baseline_minority_survival"]),
                    floor3(metrics["baseline_minority_acc"]),
                    floor3(metrics["baseline_minority_top2"]),
                    floor3(metrics["baseline_collapse"]),
                    "-",
                    "-",
                    "-",
                    "-",
                ],
                "MG-OS+GD": [
                    floor3(metrics["proposed_acc"]),
                    floor3(metrics["proposed_bal_acc"]),
                    floor3(metrics["proposed_minority_survival"]),
                    floor3(metrics["proposed_minority_acc"]),
                    floor3(metrics["proposed_minority_top2"]),
                    floor3(metrics["proposed_collapse"]),
                    floor3(metrics["rescue_rate"]),
                    floor3(metrics["false_rescue_rate"]),
                    floor3(metrics["barrier_activation_rate"]),
                    floor3(metrics["gamma_delta_mean"]),
                ],
                "delta": [
                    f"{metrics['delta_acc']:+.3f}",
                    f"{metrics['delta_bal_acc']:+.3f}",
                    f"{metrics['delta_minority_survival']:+.3f}",
                    f"{metrics['delta_minority_acc']:+.3f}",
                    f"{metrics['delta_minority_top2']:+.3f}",
                    f"{metrics['delta_collapse']:+.3f}",
                    "-",
                    "-",
                    "-",
                    "-",
                ],
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        fig_gamma = plot_gamma_hist(result.gamma_before, result.gamma_after)
        st.pyplot(fig_gamma, use_container_width=True)

        st.markdown(
            """
**Interpretation**
- **Rescue rate**: cases where softmax fails but MG-OS + GD succeeds.
- **False rescue**: cases where softmax was correct but the protected selector breaks it.
- **Barrier activation**: how often the barrier actually intervenes instead of always biasing the minority token.
- **Gamma delta**: how much the minority-important candidate is moved relative to the strongest rival.
            """
        )

st.markdown("---")
st.markdown(
    "This is a compact architecture demo. It is designed to make the protect-then-select difference visible, not to claim full transformer training or large-scale benchmark performance."
)
