from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scripts.llm_quantization_experiment import (
    DEFAULT_BITS,
    ExperimentConfig,
    generate_weights,
    quantize_uniform,
    run_experiment,
)


st.set_page_config(
    page_title="LLM Quantization Visualizer",
    page_icon="📉",
    layout="wide",
)

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "llm_quantization_chart", "scale": 2},
}


@st.cache_data(show_spinner=False)
def cached_experiment(
    n_weights: int,
    sigma: float,
    clip_sigma: float,
    seed: int,
    bits: tuple[int, ...],
    stable_threshold: float,
    candidate_threshold: float,
    precise_threshold: float,
):
    config = ExperimentConfig(
        n_weights=n_weights,
        sigma=sigma,
        clip_sigma=clip_sigma,
        seed=seed,
        bits=bits,
        stable_nrmse_threshold=stable_threshold,
        candidate_nrmse_threshold=candidate_threshold,
        very_precise_nrmse_threshold=precise_threshold,
    )
    bit_rows, interval_rows, info, fit_bundle = run_experiment(config)
    weights, _ = generate_weights(config)
    return config, bit_rows, interval_rows, info, fit_bundle["fit_rows"], weights


def fmt(x: float, digits: int = 6) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-4 or abs(x) >= 1e4:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def metric_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows).copy()


def apply_common_layout(fig: go.Figure, *, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 50, "r": 30, "t": 70, "b": 50},
        height=440,
        template="plotly_white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


def line_chart(
    df: pd.DataFrame,
    columns: Iterable[str],
    names: dict[str, str],
    colors: dict[str, str],
    *,
    title: str,
    y_title: str,
    y_log: bool = False,
) -> go.Figure:
    fig = go.Figure()
    for col in columns:
        fig.add_trace(
            go.Scatter(
                x=df["bit"],
                y=df[col],
                mode="lines+markers",
                name=names.get(col, col),
                line={"width": 3, "color": colors.get(col)},
                marker={"size": 8},
                hovertemplate="bit=%{x}<br>%{y:.8g}<extra></extra>",
            )
        )
    apply_common_layout(fig, title=title, x_title="bit 수 b", y_title=y_title)
    fig.update_xaxes(dtick=1)
    if y_log:
        fig.update_yaxes(type="log")
    return fig


def nrmse_chart(df: pd.DataFrame, config: ExperimentConfig) -> go.Figure:
    fig = line_chart(
        df,
        ["nrmse"],
        {"nrmse": "NRMSE"},
        {"nrmse": "#2563eb"},
        title="bit 수에 따른 NRMSE와 판정 기준",
        y_title="NRMSE = RMSE / σ_W",
    )
    threshold_specs = [
        (config.candidate_nrmse_threshold, "실용 후보 기준", "#6b7280"),
        (config.stable_nrmse_threshold, "안정 기준", "#dc2626"),
        (config.very_precise_nrmse_threshold, "매우 정밀 기준", "#16a34a"),
    ]
    for y, label, color in threshold_specs:
        fig.add_hline(
            y=y,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{label} {y:g}",
            annotation_position="top right",
        )
    return fig


def memory_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["bit"],
            y=df["relative_memory_vs_16bit"],
            name="상대 메모리",
            marker_color="#0f766e",
            hovertemplate="bit=%{x}<br>16bit 대비 %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["bit"],
            y=df["relative_memory_vs_16bit"],
            mode="lines+markers",
            name="M_rel(b)=b/16",
            line={"width": 3, "color": "#14b8a6"},
        )
    )
    apply_common_layout(fig, title="bit 수에 따른 상대 메모리 사용량", x_title="bit 수 b", y_title="16bit 대비 상대 메모리")
    fig.update_yaxes(tickformat=".0%", range=[0, max(1.05, float(df["relative_memory_vs_16bit"].max()) * 1.12)])
    return fig


def interval_chart(interval_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=interval_df["interval"],
            y=interval_df["abs_avg_rate_mse_per_bit"],
            name="|평균변화율|",
            marker_color="#f97316",
            hovertemplate="구간=%{x}<br>|ΔMSE/Δb|=%{y:.8g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=interval_df["interval"],
            y=interval_df["marginal_efficiency_mse_per_relative_memory"],
            yaxis="y2",
            mode="lines+markers",
            name="한계 효율",
            line={"width": 3, "color": "#7c3aed"},
            marker={"size": 8},
            hovertemplate="구간=%{x}<br>η=%{y:.8g}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "구간별 평균변화율과 한계 효율", "x": 0.02, "xanchor": "left"},
        xaxis_title="bit 구간",
        yaxis={"title": "MSE 감소율 |ΔMSE/Δb|", "showgrid": True, "gridcolor": "rgba(0,0,0,0.08)"},
        yaxis2={"title": "한계 효율 η", "overlaying": "y", "side": "right", "showgrid": False},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hovermode="x unified",
        margin={"l": 50, "r": 55, "t": 70, "b": 50},
        height=440,
        template="plotly_white",
    )
    return fig


def quantization_step_chart(weights: np.ndarray, config: ExperimentConfig, selected_bit: int, max_points: int = 1600) -> go.Figure:
    sample = np.linspace(config.qmin, config.qmax, max_points)
    quantized, step, levels = quantize_uniform(sample, selected_bit, config.qmin, config.qmax)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample,
            y=sample,
            mode="lines",
            name="원래 값 y=x",
            line={"color": "#94a3b8", "dash": "dash", "width": 2},
            hovertemplate="원래 값=%{x:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample,
            y=quantized,
            mode="lines",
            name=f"{selected_bit}bit 양자화 후 값",
            line={"color": "#ef4444", "width": 3, "shape": "hv"},
            hovertemplate="원래 값=%{x:.4f}<br>양자화 값=%{y:.4f}<extra></extra>",
        )
    )
    apply_common_layout(
        fig,
        title=f"균등 양자화 계단 함수: {selected_bit}bit, 단계 수 {levels}, 간격 Δ≈{fmt(step, 5)}",
        x_title="원래 가중치 w",
        y_title="양자화 후 값 q",
    )
    fig.update_layout(height=460)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def histogram_chart(weights: np.ndarray, config: ExperimentConfig) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=weights,
            nbinsx=70,
            marker_color="#2563eb",
            opacity=0.82,
            name="모의 가중치",
            hovertemplate="가중치 구간=%{x}<br>개수=%{y}<extra></extra>",
        )
    )
    fig.add_vline(x=config.qmin, line_dash="dash", line_color="#dc2626", annotation_text="분석 하한")
    fig.add_vline(x=config.qmax, line_dash="dash", line_color="#dc2626", annotation_text="분석 상한")
    apply_common_layout(fig, title="모의 LLM 가중치 분포", x_title="가중치 값", y_title="개수")
    return fig


def format_bit_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "bit",
        "levels_2_power_b",
        "quant_step_delta",
        "mse",
        "mae",
        "rmse",
        "nrmse",
        "relative_memory_vs_16bit",
        "compression_ratio_vs_16bit",
        "usability_label",
    ]
    out = df[cols].copy()
    out["relative_memory_vs_16bit"] = out["relative_memory_vs_16bit"] * 100.0
    out = out.rename(
        columns={
            "bit": "bit",
            "levels_2_power_b": "단계 수 2^b",
            "quant_step_delta": "간격 Δ",
            "mse": "MSE",
            "mae": "MAE",
            "rmse": "RMSE",
            "nrmse": "NRMSE",
            "relative_memory_vs_16bit": "상대 메모리(%)",
            "compression_ratio_vs_16bit": "16bit 대비 압축률",
            "usability_label": "판정",
        }
    )
    return out


def format_interval_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "interval",
        "mse_reduction",
        "avg_rate_mse_per_bit",
        "abs_avg_rate_mse_per_bit",
        "memory_increase_relative",
        "marginal_efficiency_mse_per_relative_memory",
    ]
    out = df[cols].copy()
    return out.rename(
        columns={
            "interval": "구간",
            "mse_reduction": "MSE 감소량",
            "avg_rate_mse_per_bit": "평균변화율 ΔMSE/Δb",
            "abs_avg_rate_mse_per_bit": "감소율 절댓값",
            "memory_increase_relative": "상대 메모리 증가량",
            "marginal_efficiency_mse_per_relative_memory": "한계 효율 η",
        }
    )


def main() -> None:
    st.title("LLM 양자화 오차와 메모리 효율 변화율 시각화")
    st.caption("미적분Ⅰ 수행평가 보조 산출물 · MSE/MAE/RMSE/NRMSE · 평균변화율 · 한계 효율")

    with st.sidebar:
        st.header("실험 조건")
        n_weights = st.slider("모의 가중치 개수 N", min_value=10_000, max_value=300_000, value=100_000, step=10_000)
        sigma = st.slider("가중치 표준편차 σ", min_value=0.3, max_value=2.5, value=1.0, step=0.1)
        clip_sigma = st.slider("분석 범위: ±kσ", min_value=1.5, max_value=4.0, value=3.0, step=0.1)
        seed = st.number_input("난수 seed", min_value=0, max_value=9999, value=42, step=1)
        bits_selected = st.multiselect(
            "비교할 bit 수",
            options=[2, 3, 4, 5, 6, 8, 10, 12, 16],
            default=list(DEFAULT_BITS),
        )
        bits = tuple(sorted(set(bits_selected)))
        if len(bits) < 2:
            st.error("bit 수는 최소 2개 이상 선택해야 평균변화율을 계산할 수 있습니다.")
            st.stop()
        stable_threshold = st.slider("안정 기준 NRMSE", 0.01, 0.20, 0.05, 0.01)
        candidate_threshold = st.slider("실용 후보 기준 NRMSE", 0.05, 0.40, 0.15, 0.01)
        precise_threshold = st.slider("매우 정밀 기준 NRMSE", 0.001, 0.05, 0.01, 0.001, format="%.3f")
        selected_bit = st.selectbox("계단 함수로 볼 bit", options=bits, index=min(len(bits) - 1, 2))
        mse_log = st.toggle("MSE 그래프 로그축", value=True)

    config, bit_rows, interval_rows, info, fit_rows, weights = cached_experiment(
        n_weights=n_weights,
        sigma=float(sigma),
        clip_sigma=float(clip_sigma),
        seed=int(seed),
        bits=bits,
        stable_threshold=float(stable_threshold),
        candidate_threshold=float(candidate_threshold),
        precise_threshold=float(precise_threshold),
    )

    bit_df = metric_df(bit_rows)
    interval_df = metric_df(interval_rows)
    stable_bits = bit_df.loc[bit_df["nrmse"] <= config.stable_nrmse_threshold, "bit"].astype(int).tolist()
    min_stable = min(stable_bits) if stable_bits else None
    best_eff_row = interval_df.loc[interval_df["marginal_efficiency_mse_per_relative_memory"].idxmax()]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("안정 기준 최초 만족 bit", f"{min_stable}bit" if min_stable is not None else "없음")
    m2.metric("최고 한계 효율 구간", f"{best_eff_row['interval']}bit")
    m3.metric("가중치 표준편차", fmt(float(info["used_std"]), 5))
    m4.metric("clipping 비율", f"{float(info['clip_ratio']) * 100:.3f}%")

    st.info(
        "이 앱은 실제 LLM의 언어 생성 성능을 측정하는 도구가 아니라, 모의 가중치와 균등 양자화 조건에서 "
        "bit 수·오차·메모리 효율의 변화율을 관찰하는 교육용 시각화 도구입니다.",
        icon="ℹ️",
    )

    tab_overview, tab_rates, tab_quantization, tab_tables = st.tabs(
        ["오차/메모리 그래프", "평균변화율·한계효율", "양자화 원리", "수치표"]
    )

    with tab_overview:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                line_chart(
                    bit_df,
                    ["mse"],
                    {"mse": "MSE"},
                    {"mse": "#dc2626"},
                    title="bit 수에 따른 MSE 변화" + (" (로그축)" if mse_log else ""),
                    y_title="MSE(b)",
                    y_log=mse_log,
                ),
                width="stretch",
                config=PLOTLY_CONFIG,
            )
        with col2:
            st.plotly_chart(nrmse_chart(bit_df, config), width="stretch", config=PLOTLY_CONFIG)
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(
                line_chart(
                    bit_df,
                    ["mae", "rmse"],
                    {"mae": "MAE", "rmse": "RMSE"},
                    {"mae": "#16a34a", "rmse": "#f97316"},
                    title="bit 수에 따른 MAE와 RMSE",
                    y_title="오차",
                ),
                width="stretch",
                config=PLOTLY_CONFIG,
            )
        with col4:
            st.plotly_chart(memory_chart(bit_df), width="stretch", config=PLOTLY_CONFIG)

    with tab_rates:
        st.markdown(
            "미적분Ⅰ의 평균변화율 관점에서는 `MSE`를 성능 손실 함수 `L(b)`로 두고, "
            "인접한 bit 구간에서 `ΔMSE/Δb`를 계산합니다. 값이 음수이면 오차가 감소한 것이고, "
            "절댓값이 클수록 해당 구간의 오차 감소 효과가 큽니다."
        )
        st.plotly_chart(interval_chart(interval_df), width="stretch", config=PLOTLY_CONFIG)
        fit_df = pd.DataFrame(fit_rows)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bit_df["bit"],
                y=bit_df["mse"],
                mode="markers+lines",
                name="실험 MSE",
                marker={"size": 9, "color": "#dc2626"},
                line={"width": 2, "color": "#dc2626"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fit_df["bit"],
                y=fit_df["fitted_mse"],
                mode="lines+markers",
                name="지수감소 근사",
                line={"width": 3, "dash": "dash", "color": "#2563eb"},
            )
        )
        apply_common_layout(fig, title="MSE의 지수감소 근사", x_title="bit 수 b", y_title="MSE(b)")
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
        st.caption(
            f"근사식: MSE(b) ≈ {fmt(float(info['fit_A']))} · exp({fmt(float(info['fit_slope_exp']))}b), "
            f"로그 공간 R²={fmt(float(info['fit_r2_log_space']))}. bit 수는 실제로 이산적이므로 본문 해석은 평균변화율 중심이 안전합니다."
        )

    with tab_quantization:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(histogram_chart(weights, config), width="stretch", config=PLOTLY_CONFIG)
        with col2:
            st.plotly_chart(quantization_step_chart(weights, config, int(selected_bit)), width="stretch", config=PLOTLY_CONFIG)
        st.markdown(
            "bit 수가 커질수록 표현 단계 수 `2^b`가 늘어나고, 같은 범위를 더 촘촘하게 나누기 때문에 "
            "계단 함수의 간격 `Δ`가 작아집니다. 그래서 원래 값 `w`와 복원 값 `q`의 차이가 줄어들어 MSE, MAE, RMSE가 감소합니다."
        )

    with tab_tables:
        st.subheader("bit별 오차·메모리 지표")
        st.dataframe(
            format_bit_table(bit_df),
            width="stretch",
            hide_index=True,
            column_config={
                "간격 Δ": st.column_config.NumberColumn(format="%.6f"),
                "MSE": st.column_config.NumberColumn(format="%.8f"),
                "MAE": st.column_config.NumberColumn(format="%.8f"),
                "RMSE": st.column_config.NumberColumn(format="%.8f"),
                "NRMSE": st.column_config.NumberColumn(format="%.8f"),
                "상대 메모리(%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            },
        )
        st.subheader("구간별 평균변화율·한계 효율")
        st.dataframe(
            format_interval_table(interval_df),
            width="stretch",
            hide_index=True,
            column_config={
                "MSE 감소량": st.column_config.NumberColumn(format="%.8f"),
                "평균변화율 ΔMSE/Δb": st.column_config.NumberColumn(format="%.8f"),
                "감소율 절댓값": st.column_config.NumberColumn(format="%.8f"),
                "상대 메모리 증가량": st.column_config.NumberColumn(format="%.4f"),
                "한계 효율 η": st.column_config.NumberColumn(format="%.8f"),
            },
        )
        st.download_button(
            "bit별 지표 CSV 다운로드",
            data=bit_df.to_csv(index=False).encode("utf-8"),
            file_name="bit_metrics_interactive.csv",
            mime="text/csv",
        )
        st.download_button(
            "구간별 지표 CSV 다운로드",
            data=interval_df.to_csv(index=False).encode("utf-8"),
            file_name="interval_metrics_interactive.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown(
        "**보고서 연결:** 이 웹앱은 보고서의 수치 실험을 직접 다시 실행해 보는 보조 산출물입니다. "
        "결론을 일반적인 LLM 전체에 그대로 적용하지 말고, 모의 가중치·균등 양자화 조건에서의 변화율 해석으로 보아야 합니다."
    )


if __name__ == "__main__":
    main()
