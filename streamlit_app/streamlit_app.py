"""
ASOS A/B Testing Explorer
Interactive Streamlit app for exploring the ASOS Digital Experiments Dataset.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "asos_digital_experiments_dataset.csv"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
SUMMARY_PATH = BASE_DIR / "outputs" / "experiment_summary.csv"
PIC_DIR = BASE_DIR / "pic"

METRIC_LABELS = {1: "Metric 1", 2: "Metric 2", 3: "Metric 3", 4: "Metric 4"}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ASOS Experimentation Analysis",
    page_icon="🧪",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_summary():
    return pd.read_csv(SUMMARY_PATH)


def z_to_p(z: float) -> float:
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Intro",
        "Background",
        "Meet the experiments",
        "Stats that stick",
        "Peeking problem",
        "Notebook throwbacks",
    ],
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = load_raw()
summary = load_summary()

# ===================================================================
# PAGE: Overview
# ===================================================================

if page == "Intro":

    st.title("ASOS Experimentation Analysis")
    st.header("Intro")
    st.code("How did ASOS use A/B tests to build a better product?")
    st.write("")
    st.write("")
    st.write("Hi, I’m Siva, MS Data Science student. I built this app so the ASOS experiments stay accessible and the curious lift stories stay easy to explore.")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("**Next page:** Use the Navigation bar in the top left corner to start")
    st.caption("Last update: Mar 4, 2026")

elif page == "Background":

    st.title("ASOS RCTs Analysis")
    st.header("Background")
    st.subheader("Dataset at a glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Experiments", df["experiment_id"].nunique())
    col2.metric("Metrics", df["metric_id"].nunique())
    col3.metric("Total rows", f"{len(df):,}")
    col4.metric("Variant IDs", ", ".join(str(v) for v in sorted(df["variant_id"].unique())))

    st.subheader("Experiment duration distribution")
    durations = df.groupby("experiment_id")["time_since_start"].max()
    duration_fig = px.histogram(
        durations,
        nbins=30,
        template="plotly_dark",
        title="How long do ASOS experiments run?",
        labels={"value": "Max time since start (hours)", "count": "Number of experiments"},
        color_discrete_sequence=["#4c78a8"],
    )
    duration_fig.add_vline(
        x=durations.mean(),
        line_dash="dot",
        line_color="white",
        annotation_text="Mean",
        annotation_position="top right",
        annotation_font_color="white",
    )
    duration_fig.update_layout(
        bargap=0.05,
        xaxis_title="Max time since start (hours)",
        yaxis_title="Number of experiments",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(duration_fig, width="stretch")

    st.subheader("Sample rows")
    st.dataframe(df.head(20), width="stretch")

# ===================================================================
# PAGE: Experiment Deep-Dive
# ===================================================================
elif page == "Meet the experiments":
    st.title("Experiment Deep-Dive")
    st.markdown(
        "Pick an experiment and a metric, then watch control and treatment "
        "means evolve over the experiment's lifetime."
    )

    experiments = sorted(df["experiment_id"].unique())
    selected_exp = st.selectbox("Experiment", experiments)

    exp_df = df[df["experiment_id"] == selected_exp]
    available_metrics = sorted(exp_df["metric_id"].unique())
    selected_metric = st.selectbox(
        "Metric",
        available_metrics,
        format_func=lambda m: METRIC_LABELS.get(m, f"Metric {m}"),
    )

    exp_metric = (
        exp_df[exp_df["metric_id"] == selected_metric]
        .sort_values("time_since_start")
    )

    st.subheader("Control vs. Treatment over time")
    line_df = (
        exp_metric.melt(
            id_vars=["time_since_start", "variant_id"],
            value_vars=["mean_c", "mean_t"],
            var_name="phase",
            value_name="metric_value",
        )
        .assign(
            phase=lambda d: d["phase"].map({"mean_c": "Control", "mean_t": "Treatment"}),
            series_label=lambda d: d.apply(
                lambda row: f"Var {row['variant_id']} {row['phase']}", axis=1
            ),
        )
    )
    time_fig = px.line(
        line_df,
        x="time_since_start",
        y="metric_value",
        color="series_label",
        line_dash="phase",
        template="plotly_dark",
        title=(
            f"Experiment {selected_exp} · "
            f"{METRIC_LABELS.get(selected_metric, f'Metric {selected_metric}')}"
        ),
        labels={
            "time_since_start": "Hours since experiment start",
            "metric_value": "Metric value",
            "series_label": "Variant & group",
        },
    )
    time_fig.update_layout(
        legend=dict(title="Variant & group", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(time_fig, width="stretch")

    st.subheader("Final snapshot")
    final = exp_metric.groupby("variant_id").last().reset_index()
    final["lift"] = final["mean_t"] - final["mean_c"]
    st.dataframe(
        final[["variant_id", "count_c", "count_t", "mean_c", "mean_t", "lift"]].rename(
            columns={
                "variant_id": "Variant",
                "count_c": "N control",
                "count_t": "N treatment",
                "mean_c": "Mean control",
                "mean_t": "Mean treatment",
                "lift": "Lift (T - C)",
            }
        ),
        width="stretch",
    )

# ===================================================================
# PAGE: Statistical Significance
# ===================================================================
elif page == "Stats that stick":
    st.title("Statistical Significance Across Experiments")
    st.markdown(
        """
        The summary below uses the **final time-point** for each
        experiment-metric-variant combination (excluding control-only rows).
        A two-sided z-test checks whether the treatment mean differs
        significantly from the control mean (alpha = 0.05).
        """
    )

    sig_count = summary["significant"].sum()
    total = len(summary)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total comparisons", total)
    col2.metric("Significant (p < 0.05)", int(sig_count))
    col3.metric("Significant %", f"{100 * sig_count / total:.1f}%")

    st.subheader("Lift distribution")
    lift_fig = px.histogram(
        summary,
        x="lift",
        nbins=40,
        title="Distribution of observed lifts",
        template="plotly_dark",
        color_discrete_sequence=["#4c78a8"],
    )
    lift_fig.add_vline(x=0, line_color="gray", line_width=1, opacity=0.8)
    lift_fig.update_layout(
        xaxis_title="Lift (treatment - control)",
        yaxis_title="Count",
        bargap=0.02,
    )
    st.plotly_chart(lift_fig, width="stretch")

    st.subheader("P-value distribution")
    pvalue_fig = px.histogram(
        summary,
        x="p_value",
        nbins=40,
        title="P-value distribution across all experiment-metric pairs",
        template="plotly_dark",
        color_discrete_sequence=["#f58518"],
    )
    pvalue_fig.add_vline(
        x=0.05,
        line_color="red",
        line_width=1.5,
        line_dash="dash",
        annotation_text="alpha = 0.05",
        annotation_position="top right",
        annotation_font_color="white",
    )
    pvalue_fig.update_layout(
        xaxis_title="P-value",
        yaxis_title="Count",
        bargap=0.02,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(pvalue_fig, width="stretch")

    st.subheader("Top lifts (absolute)")
    top = (
        summary.assign(abs_lift=lambda d: d["lift"].abs())
        .sort_values("abs_lift", ascending=False)
        .head(15)
    )
    fig_top = px.bar(
        top.assign(label=lambda d: d["experiment_id"] + " (m" + d["metric_id"].astype(str) + ")"),
        x="lift",
        y="label",
        orientation="h",
        color="significant",
        color_discrete_map={True: "#f58518", False: "#4c78a8"},
        template="plotly_dark",
        title="Top 15 absolute lifts (orange = significant)",
        labels={"lift": "Lift", "label": "Experiment"},
    )
    fig_top.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_tickformat=".4f",
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_top, width="stretch")

    st.subheader("Full summary table")
    st.dataframe(
        summary[
            [
                "experiment_id",
                "metric_id",
                "variant_id",
                "count_control",
                "count_treatment",
                "mean_control",
                "mean_treatment",
                "lift",
                "relative_lift",
                "z_score",
                "p_value",
                "significant",
            ]
        ].sort_values("p_value"),
        width="stretch",
    )

# ===================================================================
# PAGE: Sequential Testing / Peeking Problem
# ===================================================================
elif page == "Peeking problem":
    st.title("The Peeking Problem")
    st.markdown(
        "When analysts check experiment results before the planned end-date, "
        "conventional p-values lose their meaning. Each extra look inflates the "
        "false-positive rate. This page replays all 78 experiments at every recorded "
        "time snapshot and compares **naive** peeking against two alpha-spending corrections."
    )

    # --- compute sequential stats (cached) ---
    @st.cache_data
    def compute_sequential(raw_df):
        ALPHA = 0.05
        Z_CRIT = 1.959964

        sdf = raw_df.copy()
        sdf["std_error"] = np.sqrt(
            sdf["variance_c"] / sdf["count_c"] + sdf["variance_t"] / sdf["count_t"]
        )
        sdf["lift"] = sdf["mean_t"] - sdf["mean_c"]
        sdf["z_score"] = sdf["lift"] / sdf["std_error"]
        sdf["p_value"] = sdf["z_score"].apply(z_to_p)
        pooled_var = (
            ((sdf["count_c"] - 1) * sdf["variance_c"] + (sdf["count_t"] - 1) * sdf["variance_t"])
            / (sdf["count_c"] + sdf["count_t"] - 2)
        )
        sdf["std_lift"] = sdf["lift"] / np.sqrt(pooled_var).replace(0, np.nan)
        sdf = sdf.replace([np.inf, -np.inf], np.nan).dropna(subset=["std_error", "z_score"])

        sdf = sdf.sort_values(["experiment_id", "metric_id", "variant_id", "time_since_start"])
        gcols = ["experiment_id", "metric_id", "variant_id"]
        sdf["peek_num"] = sdf.groupby(gcols).cumcount() + 1
        sdf["total_peeks"] = sdf.groupby(gcols)["peek_num"].transform("max")
        sdf["info_fraction"] = sdf["peek_num"] / sdf["total_peeks"]

        sdf["pocock_threshold"] = ALPHA / sdf["total_peeks"]
        obf_z = Z_CRIT / np.sqrt(sdf["info_fraction"])
        sdf["obf_threshold"] = obf_z.apply(z_to_p)

        sdf["naive_sig"] = sdf["p_value"] < ALPHA
        sdf["pocock_sig"] = sdf["p_value"] < sdf["pocock_threshold"]
        sdf["obf_sig"] = sdf["p_value"] < sdf["obf_threshold"]

        early = sdf[sdf["peek_num"] < sdf["total_peeks"]].copy()
        early_flags = (
            early.groupby(gcols)
            .agg(naive_ever=("naive_sig", "any"), pocock_ever=("pocock_sig", "any"), obf_ever=("obf_sig", "any"))
            .reset_index()
        )

        final_look = (
            sdf[sdf["peek_num"] == sdf["total_peeks"]][gcols + ["p_value"]]
            .rename(columns={"p_value": "final_p"})
        )
        final_look["final_sig"] = final_look["final_p"] < ALPHA
        alarm_acc = early_flags.merge(final_look, on=gcols)

        def classify(row, rule_col):
            if row[rule_col] and row["final_sig"]:
                return "True Positive"
            if row[rule_col] and not row["final_sig"]:
                return "False Positive"
            if not row[rule_col] and row["final_sig"]:
                return "Missed"
            return "True Negative"

        for rule, col in [("naive", "naive_ever"), ("pocock", "pocock_ever"), ("obf", "obf_ever")]:
            alarm_acc[f"{rule}_outcome"] = alarm_acc.apply(lambda r, c=col: classify(r, c), axis=1)

        return sdf, early_flags, alarm_acc

    seq_df, early_flags, alarm_acc = compute_sequential(df)

    # --- top-level metrics ---
    gcols = ["experiment_id", "metric_id", "variant_id"]
    n_exp = seq_df["experiment_id"].nunique()
    n_series = len(early_flags)
    n_rows = len(seq_df)
    naive_early = int(early_flags["naive_ever"].sum())
    pocock_early = int(early_flags["pocock_ever"].sum())
    obf_early = int(early_flags["obf_ever"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiments", n_exp)
    c2.metric("Metric series", n_series)
    c3.metric("Total peeks", f"{n_rows:,}")
    c4.metric("Naive early flags", f"{naive_early} ({100*naive_early/n_series:.0f}%)")

    # --- rules explanation ---
    st.markdown(
        """
        | Rule | How it works |
        |---|---|
        | **Naive** | Flag significant whenever p < 0.05 at any peek |
        | **Pocock** | Constant stricter threshold at every peek (alpha / K) |
        | **O'Brien–Fleming** | Very strict early, relaxes toward 0.05 at the final look |
        """
    )

    # --- early alarm comparison ---
    st.subheader("Early stopping signals: naive vs. corrected")
    alarm_fig = go.Figure()
    rules = ["Naive", "Pocock", "O'Brien–Fleming"]
    counts = [naive_early, pocock_early, obf_early]
    colors = ["#e45756", "#f58518", "#4c78a8"]
    alarm_fig.add_trace(go.Bar(
        y=rules, x=counts, orientation="h",
        marker_color=colors, text=counts, textposition="outside",
    ))
    alarm_fig.update_layout(
        template="plotly_dark",
        title=f"Series flagged significant before final look (out of {n_series})",
        xaxis_title="Number of series",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        height=300,
    )
    st.plotly_chart(alarm_fig, use_container_width=True)

    # --- alarm accuracy confusion ---
    st.subheader("Were the early alarms right?")
    st.markdown(
        "We use the **final-peek p-value** as ground truth. An early alarm is a "
        "**True Positive** if the experiment is also significant at the end, and a "
        "**False Positive** if it isn't."
    )

    outcome_labels = ["True Positive", "False Positive", "Missed", "True Negative"]
    acc_rows = []
    for rule_name, col in [("Naive", "naive_outcome"), ("Pocock", "pocock_outcome"), ("OBF", "obf_outcome")]:
        vc = alarm_acc[col].value_counts()
        tp = vc.get("True Positive", 0)
        fp = vc.get("False Positive", 0)
        fn = vc.get("Missed", 0)
        tn = vc.get("True Negative", 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        acc_rows.append({
            "Rule": rule_name,
            "True Positive": tp, "False Positive": fp,
            "Missed (FN)": fn, "True Negative": tn,
            "Precision": f"{precision:.1%}", "Recall": f"{recall:.1%}",
        })
    acc_table = pd.DataFrame(acc_rows).set_index("Rule")
    st.dataframe(acc_table, use_container_width=True)

    outcome_colors = {"True Positive": "#4c78a8", "False Positive": "#e45756",
                      "Missed": "#f58518", "True Negative": "#72b7b2"}
    conf_fig = go.Figure()
    for label in outcome_labels:
        vals = [acc_table.loc[r, label] if label != "Missed" else acc_table.loc[r, "Missed (FN)"]
                for r in ["Naive", "Pocock", "OBF"]]
        conf_fig.add_trace(go.Bar(
            name=label, x=["Naive", "Pocock", "OBF"], y=vals,
            marker_color=outcome_colors[label], text=vals, textposition="auto",
        ))
    conf_fig.update_layout(
        barmode="group", template="plotly_dark",
        title="Alarm accuracy: confusion matrix by rule",
        yaxis_title="Number of series", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(conf_fig, use_container_width=True)

    # --- interactive monitoring chart ---
    st.subheader("Experiment monitoring chart")
    st.markdown("Pick an experiment to see its running |z-score| vs. the stopping boundaries.")
    exp_list = sorted(seq_df["experiment_id"].unique())
    sel_exp = st.selectbox("Experiment", exp_list, key="peek_exp")
    sel_sub = seq_df[(seq_df["experiment_id"] == sel_exp) & (seq_df["metric_id"] == 1)]
    if sel_sub.empty:
        sel_sub = seq_df[seq_df["experiment_id"] == sel_exp]
    sel_sub = sel_sub.sort_values("time_since_start")

    ALPHA = 0.05
    Z_CRIT = 1.959964
    K = sel_sub["total_peeks"].iloc[0]
    from scipy.stats import norm
    pocock_z = norm.ppf(1 - (ALPHA / K) / 2)
    obf_z_vals = Z_CRIT / np.sqrt(sel_sub["info_fraction"])

    mon_fig = go.Figure()
    mon_fig.add_trace(go.Scatter(
        x=sel_sub["info_fraction"], y=sel_sub["z_score"].abs(),
        mode="lines", name="|z-score|", line=dict(color="#4c78a8", width=2),
    ))
    mon_fig.add_hline(y=Z_CRIT, line_dash="dash", line_color="gray",
                      annotation_text="Naive z=1.96", annotation_font_color="white")
    mon_fig.add_hline(y=pocock_z, line_dash="dashdot", line_color="#f58518",
                      annotation_text=f"Pocock z={pocock_z:.2f}", annotation_font_color="#f58518")
    mon_fig.add_trace(go.Scatter(
        x=sel_sub["info_fraction"], y=obf_z_vals,
        mode="lines", name="OBF boundary", line=dict(color="#e45756", width=2, dash="dot"),
    ))
    mon_fig.update_layout(
        template="plotly_dark",
        title=f"Experiment {sel_exp} — sequential monitoring (K={K} peeks)",
        xaxis_title="Information fraction (0 = start, 1 = end)",
        yaxis_title="|z-score|",
        yaxis=dict(rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    st.plotly_chart(mon_fig, use_container_width=True)

    # --- per-experiment table ---
    st.subheader("Per-experiment alarm breakdown")
    per_exp = (
        alarm_acc.groupby("experiment_id")
        .agg(
            series=("final_sig", "count"),
            final_sig=("final_sig", "sum"),
            naive_alarms=("naive_ever", "sum"),
            naive_TP=("naive_outcome", lambda s: (s == "True Positive").sum()),
            naive_FP=("naive_outcome", lambda s: (s == "False Positive").sum()),
            pocock_alarms=("pocock_ever", "sum"),
            pocock_TP=("pocock_outcome", lambda s: (s == "True Positive").sum()),
            pocock_FP=("pocock_outcome", lambda s: (s == "False Positive").sum()),
            obf_alarms=("obf_ever", "sum"),
            obf_TP=("obf_outcome", lambda s: (s == "True Positive").sum()),
            obf_FP=("obf_outcome", lambda s: (s == "False Positive").sum()),
        )
        .reset_index()
        .sort_values("naive_FP", ascending=False)
    )
    per_exp = per_exp.astype({c: int for c in per_exp.columns if c != "experiment_id"})
    st.dataframe(per_exp, use_container_width=True)

# ===================================================================
# PAGE: Saved Notebook Plots
# ===================================================================
elif page == "Notebook throwbacks":
    st.title("Saved Notebook Plots")
    st.markdown(
        "These are the exact figures generated by `notebooks/ab_testing_analysis.ipynb` "
        "and saved to `outputs/plots/`. They keep the Streamlit app in sync with the "
        "notebook analysis."
    )

    lift_bar = PLOTS_DIR / "metric_lift_bar.png"
    ts_plot = PLOTS_DIR / "time_series_selected_experiment.png"

    if lift_bar.exists():
        st.subheader("Metric lift bar chart")
        st.image(str(lift_bar), width="stretch")
    else:
        st.warning(f"Plot not found: {lift_bar}")

    if ts_plot.exists():
        st.subheader("Time-series for representative experiment")
        st.image(str(ts_plot), width="stretch")
    else:
        st.warning(f"Plot not found: {ts_plot}")
