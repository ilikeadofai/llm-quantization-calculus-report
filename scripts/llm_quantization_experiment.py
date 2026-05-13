#!/usr/bin/env python3
"""
LLM quantization simulation for a Korean high-school Calculus I report.

Report title:
MSE·MAE·RMSE 기반 LLM 양자화 오차와 메모리 효율의 변화율 분석

Purpose:
- Start from AI Basics metrics: MSE, MAE, RMSE.
- Extend them to normalized error metrics: NRMSE, relative MSE.
- Analyze bit-width b using Calculus I concepts: average rate of change and marginal efficiency.

This script intentionally uses synthetic LLM-like weights instead of a real LLM.
It does NOT claim to measure actual task accuracy/perplexity of a real model.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# -----------------------------
# Configuration
# -----------------------------

DEFAULT_BITS = [2, 3, 4, 6, 8, 16]
BASE_BITS = 16


@dataclass(frozen=True)
class ExperimentConfig:
    n_weights: int = 100_000
    sigma: float = 1.0
    clip_sigma: float = 3.0
    seed: int = 42
    bits: tuple[int, ...] = tuple(DEFAULT_BITS)
    # These thresholds are report-analysis criteria, not universal industry standards.
    # candidate 0.15: low-bit result worth considering but may need compensation methods.
    # stable    0.05: quantization error is within 5% of the original weight std.
    # precise   0.01: very close to original weights in this simplified simulation.
    stable_nrmse_threshold: float = 0.05
    candidate_nrmse_threshold: float = 0.15
    very_precise_nrmse_threshold: float = 0.01

    @property
    def qmin(self) -> float:
        return -self.clip_sigma * self.sigma

    @property
    def qmax(self) -> float:
        return self.clip_sigma * self.sigma


# -----------------------------
# Core math
# -----------------------------


def generate_weights(config: ExperimentConfig) -> tuple[np.ndarray, dict[str, float]]:
    """Generate clipped synthetic weights W ~ Normal(0, sigma^2)."""
    rng = np.random.default_rng(config.seed)
    raw = rng.normal(loc=0.0, scale=config.sigma, size=config.n_weights)
    weights = np.clip(raw, config.qmin, config.qmax)
    info = {
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "used_mean": float(np.mean(weights)),
        "used_std": float(np.std(weights)),
        "used_var": float(np.var(weights)),
        "used_mean_abs": float(np.mean(np.abs(weights))),
        "clip_count": int(np.count_nonzero((raw < config.qmin) | (raw > config.qmax))),
        "clip_ratio": float(np.mean((raw < config.qmin) | (raw > config.qmax))),
    }
    return weights, info


def quantize_uniform(weights: np.ndarray, bit: int, qmin: float, qmax: float) -> tuple[np.ndarray, float, int]:
    """Uniformly quantize weights into 2^bit levels over fixed range [qmin, qmax]."""
    levels = 2 ** bit
    step = (qmax - qmin) / (levels - 1)
    indices = np.rint((weights - qmin) / step)
    indices = np.clip(indices, 0, levels - 1)
    quantized = qmin + indices * step
    return quantized, float(step), levels


def usability_label(nrmse: float, config: ExperimentConfig) -> str:
    if nrmse <= config.very_precise_nrmse_threshold:
        return "very_precise"
    if nrmse <= config.stable_nrmse_threshold:
        return "stable"
    if nrmse <= config.candidate_nrmse_threshold:
        return "candidate"
    return "not_yet"


def run_experiment(config: ExperimentConfig) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | str]], dict[str, float], dict[str, float]]:
    weights, weight_info = generate_weights(config)
    std_w = weight_info["used_std"]
    var_w = weight_info["used_var"]
    mean_abs_w = weight_info["used_mean_abs"]

    bit_rows: list[dict[str, float | int | str]] = []

    for bit in config.bits:
        quantized, step, levels = quantize_uniform(weights, bit, config.qmin, config.qmax)
        error = weights - quantized
        mse = float(np.mean(error**2))
        mae = float(np.mean(np.abs(error)))
        rmse = float(math.sqrt(mse))
        nrmse = float(rmse / std_w)
        nmae = float(mae / mean_abs_w)
        relative_mse = float(mse / var_w)
        relative_memory = float(bit / BASE_BITS)
        compression_ratio = float(BASE_BITS / bit)

        bit_rows.append(
            {
                "bit": bit,
                "levels_2_power_b": levels,
                "quant_step_delta": step,
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "nrmse": nrmse,
                "nmae": nmae,
                "relative_mse": relative_mse,
                "memory_bits_total": config.n_weights * bit,
                "relative_memory_vs_16bit": relative_memory,
                "compression_ratio_vs_16bit": compression_ratio,
                "usable_by_nrmse_0_05": "yes" if nrmse <= config.stable_nrmse_threshold else "no",
                "usability_label": usability_label(nrmse, config),
            }
        )

    interval_rows: list[dict[str, float | str]] = []
    for left, right in zip(bit_rows, bit_rows[1:]):
        b1, b2 = int(left["bit"]), int(right["bit"])
        delta_b = b2 - b1
        memory_increase = float(right["relative_memory_vs_16bit"] - left["relative_memory_vs_16bit"])

        mse_reduction = float(left["mse"] - right["mse"])
        mae_reduction = float(left["mae"] - right["mae"])
        rmse_reduction = float(left["rmse"] - right["rmse"])
        nrmse_reduction = float(left["nrmse"] - right["nrmse"])

        interval_rows.append(
            {
                "interval": f"{b1}->{b2}",
                "b1": b1,
                "b2": b2,
                "delta_b": delta_b,
                "mse_reduction": mse_reduction,
                "mae_reduction": mae_reduction,
                "rmse_reduction": rmse_reduction,
                "nrmse_reduction": nrmse_reduction,
                "avg_rate_mse_per_bit": float((right["mse"] - left["mse"]) / delta_b),
                "abs_avg_rate_mse_per_bit": float(mse_reduction / delta_b),
                "memory_increase_relative": memory_increase,
                "marginal_efficiency_mse_per_relative_memory": float(mse_reduction / memory_increase),
                "marginal_efficiency_nrmse_per_relative_memory": float(nrmse_reduction / memory_increase),
            }
        )

    fit_info, fit_rows = fit_exponential_mse(bit_rows)
    return bit_rows, interval_rows, fit_info | weight_info, {"fit_rows": fit_rows}  # type: ignore[return-value]


def fit_exponential_mse(bit_rows: list[dict[str, float | int | str]]) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Fit MSE(b) = A * exp(slope * b), slope < 0.

    This is optional for a derivative interpretation.
    Since bit-width is actually discrete, the report should mainly use interval average rates.
    """
    bits = np.array([float(row["bit"]) for row in bit_rows], dtype=float)
    mse = np.array([float(row["mse"]) for row in bit_rows], dtype=float)

    valid = mse > 0
    slope, intercept = np.polyfit(bits[valid], np.log(mse[valid]), 1)
    predicted = np.exp(intercept + slope * bits)
    residual = np.log(mse[valid]) - (intercept + slope * bits[valid])
    total = np.log(mse[valid]) - np.mean(np.log(mse[valid]))
    r2 = 1.0 - float(np.sum(residual**2) / np.sum(total**2))
    k_base2 = -float(slope) / math.log(2)

    fit_rows = []
    for b, pred in zip(bits, predicted):
        fit_rows.append(
            {
                "bit": float(b),
                "fitted_mse": float(pred),
                "derivative_d_mse_d_bit": float(slope * pred),
                "abs_derivative": float(abs(slope * pred)),
            }
        )

    fit_info = {
        "fit_A": float(math.exp(intercept)),
        "fit_slope_exp": float(slope),
        "fit_k_base2": float(k_base2),
        "fit_r2_log_space": float(r2),
    }
    return fit_info, fit_rows


# -----------------------------
# CSV / summary output
# -----------------------------


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float, digits: int = 6) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-4 or abs(x) >= 1e4:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def write_summary(
    path: Path,
    config: ExperimentConfig,
    bit_rows: list[dict[str, float | int | str]],
    interval_rows: list[dict[str, float | str]],
    info: dict[str, float],
) -> None:
    stable_bits = [int(row["bit"]) for row in bit_rows if str(row["usable_by_nrmse_0_05"]) == "yes"]
    min_stable_bit = min(stable_bits) if stable_bits else None
    best_interval = max(interval_rows, key=lambda r: float(r["marginal_efficiency_mse_per_relative_memory"]))

    lines = []
    lines.append("# LLM 양자화 모의 실험 결과 요약")
    lines.append("")
    lines.append("## 보고서 제목")
    lines.append("")
    lines.append("**MSE·MAE·RMSE 기반 LLM 양자화 오차와 메모리 효율의 변화율 분석**")
    lines.append("")
    lines.append("## 실험 설정")
    lines.append("")
    lines.append(f"- 모의 가중치 개수: `{config.n_weights}`")
    lines.append(f"- 가중치 분포: `W ~ N(0, {config.sigma}^2)`, 분석 범위 `{config.qmin}` ~ `{config.qmax}`")
    lines.append(f"- 실험 bit: `{', '.join(map(str, config.bits))}`")
    lines.append(f"- 안정적 사용 가능 기준: `NRMSE ≤ {config.stable_nrmse_threshold}`")
    lines.append(f"- 실용 후보 기준: `NRMSE ≤ {config.candidate_nrmse_threshold}`")
    lines.append(f"- clipping 비율: `{fmt(float(info['clip_ratio']) * 100)}%`")
    lines.append("")
    lines.append("## 핵심 지표")
    lines.append("")
    lines.append("- `MSE(b) = (1/N)Σ(w_i-q_i)^2`")
    lines.append("- `MAE(b) = (1/N)Σ|w_i-q_i|`")
    lines.append("- `RMSE(b) = √MSE(b)`")
    lines.append("- `NRMSE(b) = RMSE(b)/σ_W`")
    lines.append("- `M(b)=Nb`, 상대 메모리는 `b/16`")
    lines.append("- 한계 효율: `η=(L(b1)-L(b2))/(M(b2)-M(b1))`")
    lines.append("")
    lines.append("## 자동 해석")
    lines.append("")
    if min_stable_bit is not None:
        lines.append(f"- `NRMSE ≤ {config.stable_nrmse_threshold}` 기준을 처음 만족한 bit는 **{min_stable_bit}bit**이다.")
    else:
        lines.append(f"- `NRMSE ≤ {config.stable_nrmse_threshold}` 기준을 만족한 bit가 없다.")
    lines.append(
        f"- MSE 기준 한계 효율이 가장 큰 구간은 **{best_interval['interval']}bit** 구간이다."
    )
    lines.append("- 단, 한계 효율이 크다고 해서 절대 오차가 충분히 작은 것은 아니므로, NRMSE 기준과 함께 판단해야 한다.")
    lines.append("- 따라서 보고서 결론에서는 `절대 오차 기준을 만족하는 최소 bit`와 `추가 bit의 한계 효율 감소`를 함께 해석하는 것이 좋다.")
    lines.append("")
    lines.append("## 지수감소 근사")
    lines.append("")
    lines.append(f"- `MSE(b) ≈ {fmt(float(info['fit_A']))} · exp({fmt(float(info['fit_slope_exp']))}b)`")
    lines.append(f"- 로그 공간 R²: `{fmt(float(info['fit_r2_log_space']))}`")
    lines.append("- bit 수는 실제로 이산적이므로, 본문에서는 구간별 평균변화율을 중심으로 쓰고 도함수 해석은 심화 해석으로만 사용하는 것이 안전하다.")
    lines.append("")
    lines.append("## 생성 파일")
    lines.append("")
    lines.append("- `bit_metrics.csv`: bit별 MSE/MAE/RMSE/NRMSE/메모리")
    lines.append("- `interval_metrics.csv`: 구간별 평균변화율/한계효율")
    lines.append("- `fit_metrics.csv`: 지수감소 근사와 도함수 해석용 값")
    lines.append("- `fig_*.svg`: 보고서 삽입용 그래프")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# SVG plotting without matplotlib
# -----------------------------


def _svg_text(x: float, y: float, text: str, size: int = 13, anchor: str = "middle", weight: str = "normal") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-weight="{weight}">{html.escape(text)}</text>'


def _nice_num(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < 0.001 or abs(x) >= 1000:
        return f"{x:.1e}"
    return f"{x:.3g}"


def save_line_svg(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]], str]],
    width: int = 900,
    height: int = 560,
    y_log: bool = False,
    hlines: list[tuple[float, str, str]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin_l, margin_r, margin_t, margin_b = 82, 32, 58, 82
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    all_x = [x for _, pts, _ in series for x, _ in pts]
    all_y = [y for _, pts, _ in series for _, y in pts]
    if hlines:
        all_y.extend([y for y, _, _ in hlines])
    x_min, x_max = min(all_x), max(all_x)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    if y_log:
        positive_y = [y for y in all_y if y > 0]
        y_min_t = math.log10(min(positive_y))
        y_max_t = math.log10(max(positive_y))
    else:
        y_min_t, y_max_t = min(all_y), max(all_y)
        if y_min_t > 0:
            y_min_t = 0.0
    if y_min_t == y_max_t:
        y_min_t -= 1
        y_max_t += 1
    pad = (y_max_t - y_min_t) * 0.08
    y_min_t -= pad
    y_max_t += pad

    def tx(x: float) -> float:
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def ty(y: float) -> float:
        yy = math.log10(y) if y_log else y
        return margin_t + (y_max_t - yy) / (y_max_t - y_min_t) * plot_h

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(_svg_text(width / 2, 30, title, size=20, weight="bold"))

    # Axes
    x0, y0 = margin_l, margin_t + plot_h
    svg.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{y0}" stroke="#222" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin_l}" y1="{y0}" x2="{margin_l + plot_w}" y2="{y0}" stroke="#222" stroke-width="1.5"/>')

    # X ticks use actual bit positions.
    for x in sorted(set(all_x)):
        px = tx(x)
        svg.append(f'<line x1="{px:.2f}" y1="{y0}" x2="{px:.2f}" y2="{y0 + 6}" stroke="#222"/>')
        svg.append(_svg_text(px, y0 + 24, str(int(x)) if float(x).is_integer() else _nice_num(x), size=12))

    # Y grid/ticks
    for i in range(6):
        val_t = y_min_t + (y_max_t - y_min_t) * i / 5
        if y_log:
            val = 10 ** val_t
        else:
            val = val_t
        py = margin_t + (y_max_t - val_t) / (y_max_t - y_min_t) * plot_h
        svg.append(f'<line x1="{margin_l}" y1="{py:.2f}" x2="{margin_l + plot_w}" y2="{py:.2f}" stroke="#e6e6e6"/>')
        svg.append(_svg_text(margin_l - 10, py + 4, _nice_num(val), size=11, anchor="end"))

    # Horizontal threshold lines.
    if hlines:
        for y, label, color in hlines:
            if y <= 0 and y_log:
                continue
            py = ty(y)
            svg.append(f'<line x1="{margin_l}" y1="{py:.2f}" x2="{margin_l + plot_w}" y2="{py:.2f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="6,5"/>')
            svg.append(_svg_text(margin_l + plot_w - 4, py - 6, label, size=12, anchor="end"))

    # Series
    legend_x, legend_y = margin_l + plot_w - 160, margin_t + 18
    for idx, (name, pts, color) in enumerate(series):
        points = " ".join(f"{tx(x):.2f},{ty(y):.2f}" for x, y in pts if (not y_log or y > 0))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, y in pts:
            if y_log and y <= 0:
                continue
            svg.append(f'<circle cx="{tx(x):.2f}" cy="{ty(y):.2f}" r="4.3" fill="{color}"/>')
        svg.append(f'<rect x="{legend_x}" y="{legend_y + idx * 22 - 10}" width="12" height="12" fill="{color}"/>')
        svg.append(_svg_text(legend_x + 18, legend_y + idx * 22, name, size=12, anchor="start"))

    svg.append(_svg_text(margin_l + plot_w / 2, height - 28, x_label, size=14))
    svg.append(f'<text x="22" y="{margin_t + plot_h / 2:.2f}" font-size="14" text-anchor="middle" font-family="Arial, sans-serif" transform="rotate(-90 22 {margin_t + plot_h / 2:.2f})">{html.escape(y_label)}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def save_bar_svg(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    labels: list[str],
    values: list[float],
    color: str = "#4e79a7",
    width: int = 900,
    height: int = 560,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin_l, margin_r, margin_t, margin_b = 82, 32, 58, 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    y_max = max(values) * 1.12 if values else 1
    if y_max == 0:
        y_max = 1

    def tx(i: int) -> float:
        return margin_l + (i + 0.5) / len(labels) * plot_w

    def ty(y: float) -> float:
        return margin_t + (y_max - y) / y_max * plot_h

    bar_w = plot_w / len(labels) * 0.55
    x_axis_y = margin_t + plot_h
    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(_svg_text(width / 2, 30, title, size=20, weight="bold"))
    svg.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{x_axis_y}" stroke="#222" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin_l}" y1="{x_axis_y}" x2="{margin_l + plot_w}" y2="{x_axis_y}" stroke="#222" stroke-width="1.5"/>')

    for i in range(6):
        val = y_max * i / 5
        py = ty(val)
        svg.append(f'<line x1="{margin_l}" y1="{py:.2f}" x2="{margin_l + plot_w}" y2="{py:.2f}" stroke="#e6e6e6"/>')
        svg.append(_svg_text(margin_l - 10, py + 4, _nice_num(val), size=11, anchor="end"))

    for i, (label, value) in enumerate(zip(labels, values)):
        cx = tx(i)
        top = ty(value)
        h = x_axis_y - top
        svg.append(f'<rect x="{cx - bar_w / 2:.2f}" y="{top:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')
        svg.append(_svg_text(cx, x_axis_y + 24, label, size=12))
        svg.append(_svg_text(cx, top - 6, _nice_num(value), size=11))

    svg.append(_svg_text(margin_l + plot_w / 2, height - 28, x_label, size=14))
    svg.append(f'<text x="22" y="{margin_t + plot_h / 2:.2f}" font-size="14" text-anchor="middle" font-family="Arial, sans-serif" transform="rotate(-90 22 {margin_t + plot_h / 2:.2f})">{html.escape(y_label)}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def save_hist_svg(path: Path, weights: np.ndarray, title: str = "Synthetic weight distribution") -> None:
    counts, edges = np.histogram(weights, bins=60)
    mids = (edges[:-1] + edges[1:]) / 2
    points = [(float(x), float(y)) for x, y in zip(mids, counts)]
    save_line_svg(
        path,
        title=title,
        x_label="weight value",
        y_label="count",
        series=[("weights", points, "#4e79a7")],
        y_log=False,
    )


def make_plots(out_dir: Path, config: ExperimentConfig, bit_rows: list[dict[str, float | int | str]], interval_rows: list[dict[str, float | str]]) -> None:
    weights, _ = generate_weights(config)

    bits = [float(row["bit"]) for row in bit_rows]
    mse = [float(row["mse"]) for row in bit_rows]
    mae = [float(row["mae"]) for row in bit_rows]
    rmse = [float(row["rmse"]) for row in bit_rows]
    nrmse = [float(row["nrmse"]) for row in bit_rows]
    rel_mem = [float(row["relative_memory_vs_16bit"]) for row in bit_rows]

    save_hist_svg(out_dir / "fig_weight_distribution.svg", weights)
    save_line_svg(
        out_dir / "fig_mse_by_bit_log.svg",
        title="MSE loss by quantization bit-width (log scale)",
        x_label="bit-width b",
        y_label="MSE(b), log scale",
        series=[("MSE", list(zip(bits, mse)), "#e15759")],
        y_log=True,
    )
    save_line_svg(
        out_dir / "fig_mae_rmse_by_bit.svg",
        title="MAE and RMSE by quantization bit-width",
        x_label="bit-width b",
        y_label="error",
        series=[
            ("MAE", list(zip(bits, mae)), "#59a14f"),
            ("RMSE", list(zip(bits, rmse)), "#f28e2b"),
        ],
        y_log=False,
    )
    save_line_svg(
        out_dir / "fig_nrmse_by_bit.svg",
        title="NRMSE by quantization bit-width",
        x_label="bit-width b",
        y_label="NRMSE(b) = RMSE / std(W)",
        series=[("NRMSE", list(zip(bits, nrmse)), "#4e79a7")],
        y_log=False,
        hlines=[
            (config.candidate_nrmse_threshold, f"candidate {config.candidate_nrmse_threshold:.2f}", "#999999"),
            (config.stable_nrmse_threshold, f"stable {config.stable_nrmse_threshold:.2f}", "#e15759"),
            (config.very_precise_nrmse_threshold, f"precise {config.very_precise_nrmse_threshold:.2f}", "#59a14f"),
        ],
    )
    save_line_svg(
        out_dir / "fig_memory_by_bit.svg",
        title="Relative memory usage by bit-width",
        x_label="bit-width b",
        y_label="relative memory vs 16-bit",
        series=[("M(b)=b/16", list(zip(bits, rel_mem)), "#76b7b2")],
        y_log=False,
    )

    intervals = [str(row["interval"]) for row in interval_rows]
    eff = [float(row["marginal_efficiency_mse_per_relative_memory"]) for row in interval_rows]
    eff_norm = [v / max(eff) for v in eff]
    save_bar_svg(
        out_dir / "fig_marginal_efficiency_normalized.svg",
        title="Marginal efficiency by interval (normalized)",
        x_label="bit interval",
        y_label="relative efficiency, max=1",
        labels=intervals,
        values=eff_norm,
        color="#af7aa1",
    )


# -----------------------------
# CLI
# -----------------------------


def parse_bits(text: str) -> tuple[int, ...]:
    bits = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if len(bits) < 2:
        raise argparse.ArgumentTypeError("At least two bit widths are required")
    if sorted(bits) != list(bits):
        raise argparse.ArgumentTypeError("Bits must be sorted in ascending order")
    return bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM quantization bit-width simulation")
    parser.add_argument("--n", type=int, default=100_000, help="number of synthetic weights")
    parser.add_argument("--sigma", type=float, default=1.0, help="standard deviation of synthetic weights")
    parser.add_argument("--clip-sigma", type=float, default=3.0, help="clip range as ±clip_sigma*sigma")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--bits", type=parse_bits, default=tuple(DEFAULT_BITS), help="comma-separated bit widths, e.g. 2,3,4,6,8,16")
    parser.add_argument("--stable-threshold", type=float, default=0.05, help="NRMSE threshold for stable usability")
    parser.add_argument("--candidate-threshold", type=float, default=0.15, help="NRMSE threshold for candidate usability")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        n_weights=args.n,
        sigma=args.sigma,
        clip_sigma=args.clip_sigma,
        seed=args.seed,
        bits=args.bits,
        stable_nrmse_threshold=args.stable_threshold,
        candidate_nrmse_threshold=args.candidate_threshold,
    )

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    out_dir = args.out_dir if args.out_dir is not None else project_root / "quantization_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    bit_rows, interval_rows, info, fit_bundle = run_experiment(config)
    fit_rows = fit_bundle["fit_rows"]  # type: ignore[index]

    write_csv(out_dir / "bit_metrics.csv", bit_rows)  # type: ignore[arg-type]
    write_csv(out_dir / "interval_metrics.csv", interval_rows)  # type: ignore[arg-type]
    write_csv(out_dir / "fit_metrics.csv", fit_rows)  # type: ignore[arg-type]
    write_summary(out_dir / "summary.md", config, bit_rows, interval_rows, info)
    make_plots(out_dir, config, bit_rows, interval_rows)

    stable_bits = [int(row["bit"]) for row in bit_rows if str(row["usable_by_nrmse_0_05"]) == "yes"]
    min_stable = min(stable_bits) if stable_bits else None
    best_interval = max(interval_rows, key=lambda r: float(r["marginal_efficiency_mse_per_relative_memory"]))

    print("Experiment completed.")
    print(f"Output directory: {out_dir}")
    print(f"Stable NRMSE threshold: {config.stable_nrmse_threshold}")
    print(f"Minimum stable bit: {min_stable if min_stable is not None else 'none'}")
    print(f"Highest marginal efficiency interval: {best_interval['interval']}")
    print("Generated: bit_metrics.csv, interval_metrics.csv, fit_metrics.csv, summary.md, fig_*.svg")


if __name__ == "__main__":
    main()
