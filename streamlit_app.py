"""
streamlit_app.py
================
Interactive Streamlit dashboard for the
"Timing the Momentum Factor Using Its Own Volatility" research stack.

Deployment entrypoint for Streamlit Community Cloud.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS = 252
RF_START = "1927-01-01"
LOOKBACKS = {"1M": 21, "6M": 126, "12M": 252}

PALETTE = {
    "raw":       "#9aa0a6",
    "binary":    "#1f77b4",
    "dyn_1m":    "#d62728",
    "dyn_6m":    "#ff7f0e",
    "dyn_12m":   "#2ca02c",
    "accent":    "#7c3aed",
}

st.set_page_config(
    page_title="Momentum x Own-Volatility - Research Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Light styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        color: #e2e8f0;
    }
    .metric-card .label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;}
    .metric-card .value { font-size: 1.6rem; font-weight: 700; margin-top: 4px;}
    .metric-card .sub   { font-size: 0.78rem; color: #cbd5e1; margin-top: 4px;}
    .small-muted { color:#64748b; font-size:0.85rem; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data + Analytics (cached)
# ---------------------------------------------------------------------------
@dataclass
class FamaFrenchData:
    momentum: pd.Series
    five_factor: pd.DataFrame


@st.cache_data(show_spinner="Fetching Fama-French data from Kenneth French Data Library...", ttl=60 * 60 * 12)
def fetch_fama_french(start: str = RF_START) -> FamaFrenchData:
    mom_raw = pdr.DataReader("F-F_Momentum_Factor_daily", "famafrench", start=start)[0]
    ff5_raw = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=start)[0]

    mom_raw.columns = [c.strip() for c in mom_raw.columns]
    mom_col = "Mom" if "Mom" in mom_raw.columns else mom_raw.columns[0]
    momentum = (mom_raw[mom_col].astype(float) / 100.0).rename("Mom")
    momentum.index = pd.to_datetime(momentum.index)
    momentum = momentum.replace([-99.99 / 100, -999 / 100], np.nan).dropna()

    ff5 = ff5_raw.astype(float) / 100.0
    ff5.index = pd.to_datetime(ff5.index)
    ff5 = ff5.dropna()

    common = momentum.index.intersection(ff5.index)
    return FamaFrenchData(momentum=momentum.loc[common], five_factor=ff5.loc[common])


def trailing_annualized_vol(returns: pd.Series, lookback: int) -> pd.Series:
    return returns.rolling(lookback, min_periods=lookback).std(ddof=1) * np.sqrt(TRADING_DAYS)


def annualized_return(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1) if len(r) else np.nan


def annualized_vol(r: pd.Series) -> float:
    return float(r.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(r: pd.Series, rf: pd.Series | None = None) -> float:
    r = r.dropna()
    if rf is not None:
        excess = r - rf.reindex(r.index).fillna(0.0)
    else:
        excess = r
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd else np.nan


def max_drawdown(r: pd.Series) -> float:
    equity = (1 + r.fillna(0)).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def drawdown_series(r: pd.Series) -> pd.Series:
    equity = (1 + r.fillna(0)).cumprod()
    return equity / equity.cummax() - 1.0


def perf_summary(r: pd.Series, rf: pd.Series | None = None) -> Dict[str, float]:
    return {
        "Ann. Return": annualized_return(r),
        "Ann. Vol": annualized_vol(r),
        "Sharpe": sharpe_ratio(r, rf=rf),
        "Max DD": max_drawdown(r),
    }


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def binary_threshold_strategy(mom: pd.Series, threshold: float, lookback: int) -> pd.Series:
    vol = trailing_annualized_vol(mom, lookback).shift(1)
    return ((vol < threshold).astype(float) * mom).rename(f"Binary_{int(threshold*100)}pct")


def binary_threshold_sweep(mom: pd.Series, thresholds: Iterable[float], lookback: int,
                           rf: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        s = binary_threshold_strategy(mom, thr, lookback).dropna()
        p = perf_summary(s, rf=rf)
        p["Threshold"] = thr
        rows.append(p)
    return pd.DataFrame(rows).set_index("Threshold")[["Ann. Return", "Ann. Vol", "Sharpe", "Max DD"]]


def dynamic_vol_scaled_strategy(mom: pd.Series, lookback: int,
                                vol_target: float | None,
                                cap: float | None = None) -> pd.Series:
    vol = trailing_annualized_vol(mom, lookback).shift(1)
    vt = float(vol.mean()) if vol_target is None else vol_target
    w = vt / vol
    if cap is not None:
        w = w.clip(upper=cap)
    return (w * mom).rename(f"VolScaled_{lookback}d")


def lookback_comparison(mom: pd.Series, vol_target: float, cap: float | None,
                        rf: pd.Series | None) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    strats: Dict[str, pd.Series] = {}
    for label, lb in LOOKBACKS.items():
        strats[f"Dyn {label}"] = dynamic_vol_scaled_strategy(mom, lb, vol_target, cap)
    strats["Raw Mom"] = mom.copy()
    rows = []
    for name, s in strats.items():
        sd = s.dropna()
        rows.append({
            "Strategy": name,
            "Mean Daily Return": sd.mean(),
            "Daily Std Dev": sd.std(ddof=1),
            "Ann. Sharpe": sharpe_ratio(sd, rf=rf),
            "Ann. Return": annualized_return(sd),
            "Max DD": max_drawdown(sd),
        })
    return pd.DataFrame(rows).set_index("Strategy"), strats


def momvol_regression(mom: pd.Series, ff5: pd.DataFrame, lookback: int):
    momvol = trailing_annualized_vol(mom, lookback).shift(1).rename("MomVol")
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    df = pd.concat([mom.rename("Mom"), ff5[factors], momvol], axis=1).dropna()
    X = sm.add_constant(df[factors + ["MomVol"]])
    return sm.OLS(df["Mom"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def metric_card(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_pct(x: float, dp: int = 2) -> str:
    return "-" if pd.isna(x) else f"{x*100:.{dp}f}%"


def fmt_num(x: float, dp: int = 2) -> str:
    return "-" if pd.isna(x) else f"{x:.{dp}f}"


def equity(r: pd.Series) -> pd.Series:
    r = r.dropna()
    return (1 + r).cumprod()


# ---------------------------------------------------------------------------
# Sidebar - Controls
# ---------------------------------------------------------------------------
st.sidebar.title("Research Controls")

with st.sidebar.expander("Sample period", expanded=True):
    start_year = st.slider("Start year", 1927, 2020, 1963, step=1,
                           help="Earlier dates use Fama-French data going back to 1927.")
    end_year = st.slider("End year", start_year + 1, 2026, 2024, step=1)

with st.sidebar.expander("Strategy parameters", expanded=True):
    primary_lookback = st.selectbox(
        "Trailing vol lookback (Module 1 + regression)",
        options=[21, 63, 126, 252, 504],
        index=3,
        format_func=lambda d: f"{d} days  ({d/21:.0f}M)" if d % 21 == 0 else f"{d} days",
    )
    vol_target = st.slider("Vol target (annualized)", 0.05, 0.40, 0.15, step=0.01,
                           format="%.2f")
    leverage_cap = st.slider("Leverage cap (gross)", 1.0, 5.0, 3.0, step=0.5,
                             help="Caps w_t = sigma_target / sigma_{t-1} to control extreme positions.")

with st.sidebar.expander("Binary sweep range", expanded=False):
    lo = st.number_input("Threshold low",  0.01, 0.50, 0.05, step=0.01)
    hi = st.number_input("Threshold high", lo + 0.01, 0.60, 0.30, step=0.01)
    step = st.number_input("Step", 0.005, 0.05, 0.01, step=0.005)
    thresholds = np.round(np.arange(lo, hi + 1e-9, step), 4)

st.sidebar.caption("Data: Kenneth R. French Data Library (daily F-F factors). "
                   "Cached for 12 hours per session.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Momentum x Own-Volatility - Research Dashboard")
st.markdown(
    "<div class='small-muted'>Live replication of "
    "<i>Timing the Momentum Factor Using Its Own Volatility</i>: binary regime filter, "
    "dynamic vol-scaling, lookback robustness, and Fama-French 5-Factor + MomVol regression.</div>",
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------------
# Load and slice data
# ---------------------------------------------------------------------------
try:
    data = fetch_fama_french()
except Exception as e:
    st.error(f"Failed to fetch Fama-French data: {e}")
    st.stop()

mask = (data.momentum.index.year >= start_year) & (data.momentum.index.year <= end_year)
mom = data.momentum.loc[mask]
ff5 = data.five_factor.loc[mask]
rf = ff5["RF"]

if len(mom) < primary_lookback + 50:
    st.warning("Sample is too short for the chosen lookback. Widen the date range.")
    st.stop()

# ---------------------------------------------------------------------------
# Pre-compute strategies
# ---------------------------------------------------------------------------
sweep = binary_threshold_sweep(mom, thresholds, primary_lookback, rf=rf)
best_threshold = float(sweep["Sharpe"].idxmax())
best_binary = binary_threshold_strategy(mom, best_threshold, primary_lookback)
dyn_12m = dynamic_vol_scaled_strategy(mom, 252, vol_target, cap=leverage_cap)
lb_table, dyn_strats = lookback_comparison(mom, vol_target, leverage_cap, rf=rf)

# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
raw_perf = perf_summary(mom, rf=rf)
best_perf = perf_summary(best_binary, rf=rf)
dyn_perf = perf_summary(dyn_12m, rf=rf)

c1, c2, c3, c4 = st.columns(4)
metric_card(c1, "Sample window", f"{mom.index.min().date()} -> {mom.index.max().date()}",
            f"{len(mom):,} trading days")
metric_card(c2, "Raw Momentum - Sharpe", fmt_num(raw_perf["Sharpe"]),
            f"Ann. Ret {fmt_pct(raw_perf['Ann. Return'])} | MaxDD {fmt_pct(raw_perf['Max DD'])}")
metric_card(c3, f"Best Binary @ {int(best_threshold*100)}% - Sharpe",
            fmt_num(best_perf["Sharpe"]),
            f"Ann. Ret {fmt_pct(best_perf['Ann. Return'])} | MaxDD {fmt_pct(best_perf['Max DD'])}")
metric_card(c4, f"Dyn 12M @ vt={fmt_pct(vol_target,0)} - Sharpe",
            fmt_num(dyn_perf["Sharpe"]),
            f"Ann. Ret {fmt_pct(dyn_perf['Ann. Return'])} | MaxDD {fmt_pct(dyn_perf['Max DD'])}")

st.write("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_binary, tab_dynamic, tab_lookback, tab_reg = st.tabs(
    ["Overview", "Binary Sweep", "Dynamic Scaling",
     "Lookback Robustness", "Factor Regression"]
)

# ---------- Overview ----------
with tab_overview:
    series_map = {
        "Raw Momentum": (mom, PALETTE["raw"]),
        f"Binary {int(best_threshold*100)}%": (best_binary, PALETTE["binary"]),
        "Dyn 1M":  (dyn_strats["Dyn 1M"],  PALETTE["dyn_1m"]),
        "Dyn 6M":  (dyn_strats["Dyn 6M"],  PALETTE["dyn_6m"]),
        "Dyn 12M": (dyn_strats["Dyn 12M"], PALETTE["dyn_12m"]),
    }

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38], vertical_spacing=0.06,
        subplot_titles=("Equity Curves (log scale)", "Drawdown Profile"),
    )
    for name, (s, color) in series_map.items():
        eq = equity(s)
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq.values, name=name,
                       line=dict(color=color, width=1.4), hovertemplate="%{y:.2f}<extra>"+name+"</extra>"),
            row=1, col=1,
        )
        dd = drawdown_series(s.dropna())
        fig.add_trace(
            go.Scatter(x=dd.index, y=dd.values, name=name, showlegend=False,
                       line=dict(color=color, width=1.0),
                       hovertemplate="%{y:.1%}<extra>"+name+"</extra>"),
            row=2, col=1,
        )
    fig.update_yaxes(type="log", row=1, col=1, title_text="Growth of $1")
    fig.update_yaxes(tickformat=".0%", row=2, col=1, title_text="Drawdown")
    fig.update_layout(
        height=720, hovermode="x unified",
        legend=dict(orientation="h", y=1.07, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance comparison")
    comp_table = pd.DataFrame({name: perf_summary(s, rf=rf) for name, (s, _) in series_map.items()}).T
    comp_table = comp_table.style.format({
        "Ann. Return": "{:.2%}", "Ann. Vol": "{:.2%}",
        "Sharpe": "{:.2f}",      "Max DD":   "{:.2%}",
    })
    st.dataframe(comp_table, use_container_width=True)

# ---------- Module 1: Binary Sweep ----------
with tab_binary:
    st.subheader("Module 1 - Binary threshold filter")
    st.markdown(
        "Long the momentum factor when **lagged** trailing volatility is below the threshold; "
        "cash otherwise."
    )

    sweep_disp = sweep.reset_index()
    sweep_disp["Threshold"] = sweep_disp["Threshold"] * 100

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.55, 0.45],
        subplot_titles=("Sharpe vs. volatility threshold",
                        "Ann. Return vs. Max Drawdown by threshold"),
    )
    fig.add_trace(
        go.Scatter(x=sweep_disp["Threshold"], y=sweep_disp["Sharpe"],
                   mode="lines+markers", line=dict(color=PALETTE["binary"], width=2),
                   marker=dict(size=7), name="Sharpe"),
        row=1, col=1,
    )
    fig.add_vline(x=best_threshold * 100, line_dash="dash", line_color=PALETTE["accent"],
                  annotation_text=f"Best @ {int(best_threshold*100)}%",
                  annotation_position="top", row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=sweep_disp["Max DD"], y=sweep_disp["Ann. Return"],
            mode="markers+text", text=[f"{int(t)}%" for t in sweep_disp["Threshold"]],
            textposition="top center",
            marker=dict(size=10, color=sweep_disp["Sharpe"],
                        colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Sharpe", x=1.02)),
            name="Threshold",
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Threshold (%)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=1, col=1)
    fig.update_xaxes(title_text="Max Drawdown", tickformat=".0%", row=1, col=2)
    fig.update_yaxes(title_text="Annualized Return", tickformat=".0%", row=1, col=2)
    fig.update_layout(height=460, template="plotly_white",
                      margin=dict(l=10, r=10, t=60, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        sweep.style.format({"Ann. Return": "{:.2%}", "Ann. Vol": "{:.2%}",
                            "Sharpe": "{:.2f}", "Max DD": "{:.2%}"}),
        use_container_width=True, height=420,
    )

# ---------- Module 2: Dynamic Scaling ----------
with tab_dynamic:
    st.subheader("Module 2 - Dynamic volatility scaling")
    st.latex(r"r^{\text{scaled}}_t \;=\; \frac{\sigma^\star}{\hat{\sigma}_{t-1}} \cdot r^{\text{mom}}_t")

    vol_series = trailing_annualized_vol(mom, primary_lookback).shift(1)
    weight = (vol_target / vol_series).clip(upper=leverage_cap)
    dyn = (weight * mom).rename("VolScaled")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.05,
        subplot_titles=("Realized trailing vol vs. target",
                        "Resulting position weight (capped)",
                        "Cumulative growth: Raw vs. Scaled"),
    )
    fig.add_trace(go.Scatter(x=vol_series.index, y=vol_series.values,
                             line=dict(color=PALETTE["raw"]), name="Trailing sigma"), 1, 1)
    fig.add_hline(y=vol_target, line_dash="dash", line_color=PALETTE["accent"],
                  annotation_text=f"vt = {vol_target:.0%}", row=1, col=1)
    fig.add_trace(go.Scatter(x=weight.index, y=weight.values,
                             line=dict(color=PALETTE["dyn_12m"]), name="Weight"), 2, 1)
    fig.add_trace(go.Scatter(x=equity(mom).index, y=equity(mom).values,
                             line=dict(color=PALETTE["raw"]), name="Raw"), 3, 1)
    fig.add_trace(go.Scatter(x=equity(dyn).index, y=equity(dyn).values,
                             line=dict(color=PALETTE["dyn_12m"]), name="Scaled"), 3, 1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(type="log", row=3, col=1)
    fig.update_layout(height=760, template="plotly_white", hovermode="x unified",
                      margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(2)
    cols[0].markdown("**Raw Momentum**")
    cols[0].dataframe(pd.DataFrame(perf_summary(mom, rf=rf), index=["value"]).T
                      .style.format("{:.4f}"), use_container_width=True)
    cols[1].markdown(f"**Vol-Scaled (lookback={primary_lookback}d, vt={vol_target:.0%}, cap={leverage_cap}x)**")
    cols[1].dataframe(pd.DataFrame(perf_summary(dyn, rf=rf), index=["value"]).T
                      .style.format("{:.4f}"), use_container_width=True)

# ---------- Module 3: Lookback Robustness ----------
with tab_lookback:
    st.subheader("Module 3 - Lookback window robustness")
    st.markdown("Same scaling rule applied with three lookbacks. Shorter windows react faster "
                "but cost more in turnover.")

    fig = go.Figure()
    color_map = {"Dyn 1M": PALETTE["dyn_1m"], "Dyn 6M": PALETTE["dyn_6m"],
                 "Dyn 12M": PALETTE["dyn_12m"], "Raw Mom": PALETTE["raw"]}
    for name, s in dyn_strats.items():
        eq = equity(s)
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=name,
                                 line=dict(color=color_map[name], width=1.4)))
    fig.update_yaxes(type="log", title_text="Growth of $1")
    fig.update_layout(height=480, template="plotly_white", hovermode="x unified",
                      title="Equity curves by lookback window",
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Comparative performance")
    st.dataframe(
        lb_table.style.format({
            "Mean Daily Return": "{:.4%}",
            "Daily Std Dev": "{:.4%}",
            "Ann. Sharpe": "{:.2f}",
            "Ann. Return": "{:.2%}",
            "Max DD": "{:.2%}",
        }),
        use_container_width=True,
    )

    # Turnover proxy (sum |delta_w|) for context on transaction-cost intuition
    st.markdown("#### Turnover proxy (annualized sum |delta w|)")
    turnover_rows = []
    for label, lb in LOOKBACKS.items():
        vol = trailing_annualized_vol(mom, lb).shift(1)
        w = (vol_target / vol).clip(upper=leverage_cap).dropna()
        ann_turnover = w.diff().abs().sum() / (len(w) / TRADING_DAYS)
        turnover_rows.append({"Lookback": label, "Approx. annual turnover": ann_turnover})
    turnover_df = pd.DataFrame(turnover_rows).set_index("Lookback")
    st.dataframe(turnover_df.style.format({"Approx. annual turnover": "{:.1f}x"}),
                 use_container_width=True)
    st.caption("Higher sum |delta w| -> higher transaction-cost drag. The 1M lookback is the most "
               "reactive but also the most expensive to trade in practice.")

# ---------- Module 4: Regression ----------
with tab_reg:
    st.subheader("Module 4 - Fama-French 5-Factor + MomVol regression")
    st.latex(
        r"r^{\text{Mom}}_t = \alpha + \beta_1\,\text{Mkt-RF}_t + \beta_2\,\text{SMB}_t + "
        r"\beta_3\,\text{HML}_t + \beta_4\,\text{RMW}_t + \beta_5\,\text{CMA}_t + "
        r"\gamma\,\text{MomVol}_{t-1} + \varepsilon_t"
    )
    model = momvol_regression(mom, ff5, primary_lookback)

    coef_df = pd.DataFrame({
        "Coefficient": model.params,
        "Std Err":     model.bse,
        "t-stat":      model.tvalues,
        "p-value":     model.pvalues,
        "CI 2.5%":     model.conf_int()[0],
        "CI 97.5%":    model.conf_int()[1],
    })

    mv_coef, mv_t, mv_p = model.params["MomVol"], model.tvalues["MomVol"], model.pvalues["MomVol"]
    sig = "Significant at 5%" if mv_p < 0.05 else "NOT significant at 5%"
    direction = "Negative (consistent with thesis)" if mv_coef < 0 else "Positive (against thesis)"

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "MomVol coefficient (gamma)", f"{mv_coef:+.5f}", direction)
    metric_card(c2, "MomVol t-statistic", f"{mv_t:+.2f}", "HAC robust (5 lags)")
    metric_card(c3, "MomVol p-value", f"{mv_p:.4g}", sig)
    metric_card(c4, "Adjusted R-sq", f"{model.rsquared_adj:.4f}",
                f"N = {int(model.nobs):,} daily obs")
    st.write("")

    bar_color = ["#dc2626" if v < 0 else "#059669" for v in coef_df["Coefficient"].drop("const")]
    fig = go.Figure(go.Bar(
        x=coef_df["Coefficient"].drop("const").index,
        y=coef_df["Coefficient"].drop("const").values,
        marker_color=bar_color,
        error_y=dict(type="data", array=1.96 * coef_df["Std Err"].drop("const").values),
        hovertemplate="<b>%{x}</b><br>beta = %{y:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title="Factor loadings with 95% HAC confidence intervals",
        template="plotly_white", height=420,
        yaxis_title="Coefficient",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Coefficient table")
    st.dataframe(coef_df.style.format("{:.5f}"), use_container_width=True)

    with st.expander("Full statsmodels summary"):
        st.code(str(model.summary()), language="text")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.write("---")
st.caption(
    "All metrics use lagged trailing volatility (information at T-1 to trade at T) to prevent "
    "lookahead bias. HAC standard errors use a 5-lag Bartlett kernel. Returns are in decimal form "
    "(0.01 = 1%). Source: Kenneth R. French Data Library."
)
