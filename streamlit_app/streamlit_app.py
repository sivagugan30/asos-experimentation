"""
ASOS A/B Testing Explorer
Interactive Streamlit app for exploring the ASOS Digital Experiments Dataset.
"""

import math
from pathlib import Path

import pandas as pd
import plotly.express as px
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
