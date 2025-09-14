#!/usr/bin/env python3
"""
FastAPI server to infer schema and run statistical analysis on an uploaded file.
Includes hover-only interactive visuals (Plotly) with clean static PNG fallbacks.

Isolation: each upload executes in its own temporary workspace directory
(no fixed root paths). /uploadfile/ returns a workspace_id that /demo/run uses.
Workspaces persist across multiple runs and are deleted when the client closes
the web app (client sends /workspace/close via sendBeacon).
"""
import sys
import json
import traceback
from pathlib import Path
import shutil
import tempfile
import uvicorn
import warnings
from typing import Dict, Any, Optional
import base64
import io
from uuid import uuid4
from threading import Lock

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Optional Plotly ------------------------------------------------------------
try:
    import plotly.graph_objects as go
    from plotly.offline import plot as plotly_plot
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Could not infer format",
    category=UserWarning,
)

app = FastAPI()

# Analysis constants ---------------------------------------------------------
ALLOWED_EXT = {".csv", ".xls", ".xlsx"}
HIGH_CARDINALITY_THRESHOLD = 0.5
CATEGORICAL_TOP_N = 15

BOOL_TRUE = {"true", "t", "1", "yes", "y"}
BOOL_FALSE = {"false", "f", "0", "no", "n"}

# Workspace management -------------------------------------------------------
WORKSPACES: Dict[str, Path] = {}
WS_LOCK = Lock()
WS_BASE = Path("/tmp/workspaces")
WS_BASE.mkdir(parents=True, exist_ok=True)

def _create_workspace() -> tuple[str, Path]:
    ws_id = uuid4().hex
    ws_path = WS_BASE / ws_id
    ws_path.mkdir(parents=True, exist_ok=True)
    with WS_LOCK:
        WORKSPACES[ws_id] = ws_path
    return ws_id, ws_path

def _get_workspace(ws_id: str) -> Path:
    with WS_LOCK:
        p = WORKSPACES.get(ws_id)
    if not p or not p.exists():
        raise HTTPException(status_code=400, detail="Invalid or expired workspace_id.")
    return p

def _delete_workspace(ws_id: str):
    with WS_LOCK:
        p = WORKSPACES.pop(ws_id, None)
    if p and p.exists():
        try:
            shutil.rmtree(p)
        except Exception:
            pass

@app.on_event("startup")
async def _restore_workspaces():
    """Restore existing workspace directories from disk on startup."""
    for path in WS_BASE.glob("*"):
        if path.is_dir():
            WORKSPACES[path.name] = path
# --- Helpers (types/coercion) ----------------------------------------------
def _looks_bool(series: pd.Series) -> bool:
    s_clean = series.dropna().astype(str).str.strip().str.lower()
    if s_clean.empty: return False
    return s_clean.isin(BOOL_TRUE | BOOL_FALSE).mean() > 0.95

def _to_floatable(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"[,$£€]\s*", "", regex=True).str.replace(",", "")
    return pd.to_numeric(s, errors="coerce")

def _to_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")
    if pd.api.types.is_numeric_dtype(series.dtype):
        s_num = pd.to_numeric(series, errors="coerce")
        return s_num.map(lambda v: True if v == 1 else (False if v == 0 else pd.NA)).astype("boolean")
    s = series.astype(str).str.strip().str.lower().replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    def _map(v):
        if pd.isna(v): return pd.NA
        if v in BOOL_TRUE: return True
        if v in BOOL_FALSE: return False
        return pd.NA
    return s.map(_map).astype("boolean")

def infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "BOOLEAN"
    if series.dtype == "object":
        non_null = series.dropna()
        if non_null.empty:
            return "Categorical"
        if _looks_bool(non_null):
            return "BOOLEAN"
        try:
            dt_series = pd.to_datetime(non_null, errors="coerce")
            if dt_series.notna().mean() > 0.9:
                if (dt_series.dt.hour == 0).all() and (dt_series.dt.minute == 0).all() and (dt_series.dt.second == 0).all():
                    return "DATE"
                return "TIMESTAMP"
        except Exception:
            pass
    if pd.api.types.is_numeric_dtype(series.dtype):
        nn = series.dropna()
        if not nn.empty and ((nn == 0) | (nn == 1)).mean() > 0.95:
            return "BOOLEAN"
        return "Numerical"
    if _to_floatable(series).notna().mean() > 0.9:
        return "Numerical"
    return "Categorical"

def map_to_postgres(series: pd.Series, inferred_type: str) -> str:
    if inferred_type == "BOOLEAN":
        return "BOOLEAN"
    if inferred_type in ["DATE", "TIMESTAMP"]:
        return "TIMESTAMP WITH TIME ZONE"
    if inferred_type == "Numerical":
        num_series = _to_floatable(series).dropna()
        if not num_series.empty and (num_series % 1).abs().lt(1e-9).all():
            vmin, vmax = num_series.min(), num_series.max()
            if -32768 <= vmin and vmax <= 32767:
                return "SMALLINT"
            if -2147483648 <= vmin and vmax <= 2147483647:
                return "INTEGER"
            return "BIGINT"
        return "DOUBLE PRECISION"
    return "TEXT"

def _guess_unit_symbol(raw: pd.Series) -> Optional[str]:
    if raw.dtype == object:
        s = raw.dropna().astype(str)
        if s.str.contains(r"\$", regex=True).any(): return "$"
        if s.str.contains(r"€", regex=False).any(): return "€"
        if s.str.contains(r"£", regex=False).any(): return "£"
    return None

# --- Visualization (static & interactive) ----------------------------------
def _render_static_boxplot_clean(series_numeric: pd.Series, title: str, ui: str = "light") -> str:
    data = series_numeric.dropna().to_numpy()
    if data.size == 0:
        fig, ax = plt.subplots(figsize=(6, 1.8))
        ax.text(0.5, 0.5, "No numeric data to plot", ha="center", va="center")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", transparent=True)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    try:
        if ui == "dark":
            plt.style.use("dark_background")
            grid_alpha = 0.35
            point_color_rgba = (1.0, 1.0, 1.0, 0.20)
            violin_color = "#52525B"
            whisker_color = "#CBD5E1"
        else:
            plt.style.use("seaborn-v0_8-whitegrid")
            grid_alpha = 0.55
            point_color_rgba = (0.0, 0.0, 0.0, 0.15)
            violin_color = "#E4E4E7"
            whisker_color = "#334155"
    except Exception:
        plt.style.use("default")
        grid_alpha = 0.45
        point_color_rgba = (0.0, 0.0, 0.0, 0.15)
        violin_color = "#E4E4E7"
        whisker_color = "#334155"

    q1, q3 = np.quantile(data, [0.25, 0.75])
    iqr = q3 - q1
    median_val = float(np.median(data))
    mean_val = float(np.mean(data))
    low_fence  = float(q1 - 1.5 * iqr)
    high_fence = float(q3 + 1.5 * iqr)

    plot_points = data if data.size <= 2000 else np.random.default_rng(0).choice(data, size=2000, replace=False)

    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    try:
        vp = ax.violinplot(data, vert=False, widths=0.8, showmeans=False, showmedians=False, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(violin_color); body.set_edgecolor('none'); body.set_alpha(1.0)
    except Exception:
        pass

    jitter = np.random.uniform(-0.08, 0.08, size=plot_points.size)
    ax.scatter(plot_points, jitter, color=point_color_rgba, s=8, alpha=0.6, marker='.', zorder=3)

    ax.boxplot(
        data, vert=False, notch=False, widths=0.15, patch_artist=True, manage_ticks=False,
        boxprops={"facecolor": "#38BDF8", "edgecolor": "black", "linewidth": 1.0, "alpha": 0.85},
        whiskerprops={"color": "#334155", "linewidth": 1.5},
        capprops={"color": "#334155", "linewidth": 1.5},
        medianprops={"color": "#F43F5E", "linewidth": 2.0},
        flierprops={"marker": ".", "markersize": 5, "markerfacecolor": "#64748B", "linestyle": "none", "alpha": 0.5},
        showfliers=True
    )

    ax.axvline(low_fence,  color="#64748B", linewidth=1.2, linestyle=":", alpha=0.9, zorder=2)
    ax.axvline(high_fence, color="#64748B", linewidth=1.2, linestyle=":", alpha=0.9, zorder=2)
    ax.axvline(median_val, color="#F43F5E", linewidth=1.8)
    ax.axvline(mean_val,   color="#16A34A", linewidth=1.6, linestyle="--")

    ax.set_title(f"Distribution: {title}", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Value")
    ax.set_yticks([]); ax.set_ylim(-0.5, 0.5)
    for spine in ["top", "right", "left"]: ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=grid_alpha)

    xmin, xmax = np.nanmin(data), np.nanmax(data)
    if np.isfinite(xmin) and np.isfinite(xmax):
        pad = 0.04 * (xmax - xmin if xmax > xmin else (abs(xmax) + 1))
        ax.set_xlim(xmin - pad, xmax + pad)

    legend_handles = [
        Patch(facecolor="#38BDF8", edgecolor="black", alpha=0.85, label="IQR (box)"),
        Line2D([0], [0], color="#334155", lw=1.5, label="Whiskers & caps"),
        Line2D([0], [0], color="#F43F5E", lw=2.0, label="Median"),
        Line2D([0], [0], color="#16A34A", lw=1.6, ls="--", label="Mean"),
        Line2D([0], [0], color="#64748B", lw=1.2, ls=":", label="1.5×IQR fences"),
        Line2D([0], [0], marker='.', lw=0, color=point_color_rgba, markersize=8, label="Sampled points"),
        Line2D([0], [0], marker='.', lw=0, color="#64748B", markersize=6, label="Outliers"),
        Patch(facecolor="#E4E4E7", edgecolor="none", label="Distribution (violin)"),
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Legend", borderaxespad=0.0)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=192, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig); plt.style.use("default")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

def _render_interactive_boxplot(series_numeric: pd.Series, raw_series: pd.Series, title: str, ui: str = "light") -> str:
    if not PLOTLY_AVAILABLE: return ""
    data = series_numeric.dropna().to_numpy()
    if data.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="No numeric data to plot", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(height=160, margin=dict(l=10, r=10, t=20, b=10), template=("plotly_dark" if ui=="dark" else "plotly_white"))
        return plotly_plot(fig, include_plotlyjs="cdn", output_type="div", auto_open=False)

    q1, q3 = np.quantile(data, [0.25, 0.75]); iqr = max(q3 - q1, 1e-12)
    median_val = float(np.median(data)); mean_val = float(np.mean(data))
    low_fence  = float(q1 - 1.5 * iqr); high_fence = float(q3 + 1.5 * iqr)
    unit_sym = _guess_unit_symbol(raw_series); unit_label = f" ({unit_sym})" if unit_sym else ""

    pts = data if data.size <= 5000 else np.random.default_rng(0).choice(data, size=2000, replace=False)
    name = str(title or "values")
    fig = go.Figure()
    fig.add_trace(go.Box(x=pts, name=name, orientation="h", boxpoints="suspectedoutliers", jitter=0.3, pointpos=0.0,
                         marker=dict(size=3, opacity=0.28), line=dict(width=1),
                         fillcolor="rgba(56,189,248,0.40)", hovertemplate="Value: %{x}"+unit_label+"<extra></extra>",
                         hoveron="points", showlegend=False))
    fig.add_vrect(x0=q1, x1=q3, line_width=0, fillcolor="rgba(56,189,248,0.12)", layer="below")
    fig.add_vline(x=median_val, line_color="#FF6B6B", line_width=2)
    fig.add_vline(x=mean_val,   line_color="#22C55E", line_width=1, line_dash="dash")
    fig.add_vline(x=low_fence,  line_color="#64748B", line_width=1, line_dash="dot")
    fig.add_vline(x=high_fence, line_color="#64748B", line_width=1, line_dash="dot")

    def _stat(x, label):
        return go.Scatter(x=[x], y=[name], mode="markers", marker=dict(size=12, color="rgba(0,0,0,0)"),
                          showlegend=False, hovertemplate=f"{label}: %{{x:.4g}}{unit_label}<extra></extra>")
    for x, lab in [(q1,"Q1"), (median_val,"Median"), (q3,"Q3"), (low_fence,"1.5×IQR fence (low)"), (high_fence,"1.5×IQR fence (high)"), (mean_val,"Mean")]:
        fig.add_trace(_stat(x, lab))

    fig.update_layout(template=("plotly_dark" if ui=="dark" else "plotly_white"),
                      height=320, margin=dict(l=10, r=160, t=36, b=16),
                      showlegend=True, legend=dict(x=1.02, xanchor="left", y=0.5, orientation="v", title="Legend"),
                      title=dict(text=f"Distribution: {title}", x=0.01, xanchor="left"),
                      xaxis=dict(title=f"Value{unit_label}", zeroline=False),
                      yaxis=dict(showticklabels=False), hovermode="closest")
    return plotly_plot(fig, include_plotlyjs="cdn", output_type="div", auto_open=False)

def _render_boolean_static(series_bool: pd.Series, title: str, ui: str = "light") -> str:
    s = series_bool; total = int(len(s))
    true_c = int((s == True).sum()); false_c = int((s == False).sum()); miss_c = total - (true_c + false_c)
    def pct(x): return (x / total * 100.0) if total > 0 else 0.0
    true_p, false_p, miss_p = pct(true_c), pct(false_c), pct(miss_c)

    try:
        if ui == "dark": plt.style.use("dark_background"); grid_alpha = 0.35
        else: plt.style.use("seaborn-v0_8-whitegrid"); grid_alpha = 0.5
    except Exception:
        plt.style.use("default"); grid_alpha = 0.45

    fig, ax = plt.subplots(figsize=(8.5, 1.8))
    left = 0.0
    for w, c in zip([true_p,false_p,miss_p], ["#22C55E", "#EF4444", "#94A3B8"]):
        ax.barh([0], [w], left=left, color=c, height=0.5); left += w
    ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel("Percent of rows")
    ax.set_title(f"Boolean Distribution: {title}", fontsize=12, fontweight="bold", pad=8)
    ax.grid(axis="x", linestyle="--", alpha=grid_alpha)
    for s in ["top","right","left"]: ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", transparent=True, dpi=192, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig); plt.style.use("default")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

def _render_boolean_interactive(series_bool: pd.Series, title: str, ui: str = "light") -> str:
    if not PLOTLY_AVAILABLE: return ""
    s = series_bool; total = int(len(s))
    true_c = int((s == True).sum()); false_c = int((s == False).sum()); miss_c = total - (true_c + false_c)
    def pct(x): return (x / total * 100.0) if total > 0 else 0.0
    true_p, false_p, miss_p = pct(true_c), pct(false_c), pct(miss_c)

    fig = go.Figure(); ylab = [str(title or "values")]
    fig.add_trace(go.Bar(y=ylab, x=[false_p], orientation="h", name="False",  marker_color="#EF4444",
                         hovertemplate=f"False: {false_c} ({false_p:.1f}%)<extra></extra>", showlegend=False))
    fig.add_trace(go.Bar(y=ylab, x=[miss_p],  orientation="h", name="Missing", marker_color="#94A3B8",
                         hovertemplate=f"Missing: {miss_c} ({miss_p:.1f}%)<extra></extra>", showlegend=False))
    fig.add_trace(go.Bar(y=ylab, x=[true_p],  orientation="h", name="True",    marker_color="#22C55E",
                         hovertemplate=f"True: {true_c} ({true_p:.1f}%)<extra></extra>", showlegend=False))
    fig.update_layout(barmode="stack", template=("plotly_dark" if ui=="dark" else "plotly_white"),
                      height=180, margin=dict(l=10, r=10, t=36, b=16),
                      title=dict(text=f"Boolean Distribution: {title}", x=0.01, xanchor="left"),
                      xaxis=dict(range=[0, 100], title="Percent of rows"),
                      yaxis=dict(showticklabels=False), hovermode="closest")
    return plotly_plot(fig, include_plotlyjs="cdn", output_type="div", auto_open=False)

# --- Column analysis --------------------------------------------------------
def analyze_column(series: pd.Series, col_type: str) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {"type": col_type, "missing_count": int(series.isnull().sum())}
    if col_type == "Numerical":
        series_numeric = _to_floatable(series).dropna()
        if not series_numeric.empty:
            q1, q3 = series_numeric.quantile([0.25, 0.75]); iqr = q3 - q1
            outliers = ((series_numeric < (q1 - 1.5 * iqr)) | (series_numeric > (q3 + 1.5 * iqr))).sum()
            analysis["stats"] = {
                "mean": float(series_numeric.mean()),
                "std_dev": float(series_numeric.std()),
                "min": float(series_numeric.min()),
                "q1": float(q1),
                "median": float(series_numeric.median()),
                "q3": float(q3),
                "max": float(series_numeric.max()),
                "iqr": float(iqr),
                "skewness": float(series_numeric.skew()),
                "kurtosis": float(series_numeric.kurtosis()),
                "outlier_count": int(outliers),
            }
            html_div = _render_interactive_boxplot(series_numeric, series, str(series.name or "values"), ui="light")
            png_uri  = _render_static_boxplot_clean(series_numeric, str(series.name or "values"), ui="light")
            if html_div:
                analysis["boxplot_html"] = html_div; analysis["viz_html"] = html_div
            analysis["boxplot"] = png_uri; analysis["viz_png"] = png_uri

    elif col_type == "Categorical":
        series_str = series.dropna().astype(str)
        if not series_str.empty:
            total_count = len(series)
            value_counts = series_str.value_counts()
            top_n_counts = value_counts.nlargest(CATEGORICAL_TOP_N)
            probs = value_counts / len(series_str)
            entropy = -np.sum(probs * np.log2(probs))
            analysis.update({
                "unique_count": int(series_str.nunique()),
                "distinct_percent": float(series_str.nunique() / total_count * 100) if total_count > 0 else 0.0,
                "mode": series_str.mode().iloc[0] if not series_str.mode().empty else None,
                "is_high_cardinality": bool((series_str.nunique() / total_count) > HIGH_CARDINALITY_THRESHOLD) if total_count > 0 else False,
                "value_counts": top_n_counts.to_dict(),
                "percentages": (top_n_counts / len(series_str) * 100).to_dict(),
                "entropy": float(entropy),
            })

    elif col_type in ["DATE", "TIMESTAMP"]:
        analysis["type"] = "Temporal"
        try:
            dt = pd.to_datetime(series, errors="coerce", utc=True).dropna()
        except Exception:
            dt = pd.to_datetime(series, errors="coerce", utc=True).dropna()
        if dt.empty: return analysis
        span_days = (dt.max() - dt.min()).total_seconds() / (3600 * 24)
        if span_days < 2:       freq, agg_level, fmt = "H", "Hourly", "%Y-%m-%d %H:00"
        elif span_days <= 90:   freq, agg_level, fmt = "D", "Daily", "%Y-%m-%d"
        elif span_days <= 365*3:freq, agg_level, fmt = "W", "Weekly", "%Y-%m-%d (Week)"
        else:                   freq, agg_level, fmt = "M", "Monthly", "%Y-%m"
        grouped = dt.dt.to_period(freq).dt.to_timestamp() if freq in ("W","M") else dt.dt.floor(freq)
        time_counts = grouped.value_counts().sort_index()
        analysis.update({
            "min_date": str(dt.min()), "max_date": str(dt.max()),
            "avg_events_per_day": float(len(dt) / (span_days or 1)),
            "aggregation_level": agg_level,
            "time_counts": {k.strftime(fmt): int(v) for k, v in time_counts.items()},
        })

    elif col_type == "BOOLEAN":
        s_bool = _to_boolean(series)
        total = int(len(s_bool))
        true_c = int((s_bool == True).sum()); false_c = int((s_bool == False).sum()); miss_c = total - (true_c + false_c)
        valid = max(1, (true_c + false_c))
        analysis["stats"] = {
            "true_count": true_c, "false_count": false_c, "missing_count": miss_c,
            "true_percent_total": (true_c / total * 100.0) if total > 0 else 0.0,
            "false_percent_total": (false_c / total * 100.0) if total > 0 else 0.0,
            "missing_percent_total": (miss_c / total * 100.0) if total > 0 else 0.0,
            "true_rate_valid": (true_c / valid * 100.0),
            "false_rate_valid": (false_c / valid * 100.0),
        }
        html_div = _render_boolean_interactive(s_bool, str(series.name or "values"), ui="light")
        png_uri  = _render_boolean_static(s_bool, str(series.name or "values"), ui="light")
        if html_div:
            analysis["viz_html"] = html_div; analysis["boxplot_html"] = html_div
        analysis["viz_png"] = png_uri; analysis["boxplot"] = png_uri
    return analysis

# --- File processing --------------------------------------------------------
def analyze_file(path: str) -> Dict[str, Any]:
    ext = Path(path).suffix.lower()
    try:
        df = (pd.read_csv(path, low_memory=False) if ext == ".csv" else pd.read_excel(path))
    except Exception as e:
        raise ValueError(f"Failed to parse file. Error: {e}")

    rows, cols = df.shape
    total_cells = rows * cols
    summary = {
        "row_count": int(rows),
        "column_count": int(cols),
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_percentage": float((df.isnull().sum().sum() / total_cells) * 100) if total_cells > 0 else 0.0,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage": int(df.memory_usage(deep=True).sum()),
    }

    schema_out, analysis_out = [], {}
    for col in df.columns:
        col_name = str(col)
        series = df[col].dropna()
        if not series.empty and series.nunique() == len(series):
            continue
        high_level_type = infer_column_type(df[col])
        pg_type = map_to_postgres(df[col], high_level_type)
        schema_out.append({"column_name": col_name, "inferred_type": pg_type})
        analysis_out[col_name] = analyze_column(df[col], high_level_type)

    return {"summary": summary, "schema": schema_out, "analysis": analysis_out}

def convert_numpy_types(obj):
    if isinstance(obj, dict):  return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [convert_numpy_types(i) for i in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    if obj is None or (isinstance(obj, float) and pd.isna(obj)): return None
    return obj

# --- Endpoints --------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root():
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/uploadfile/", summary="Analyze a structured data file (isolated workspace)")
async def create_upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Please use CSV, XLS, or XLSX.")

    tmp_path = None
    ws_id, ws_path = _create_workspace()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = Path(tmp.name)
            await file.seek(0)
            shutil.copyfileobj(file.file, tmp)

        data_csv = ws_path / "data.csv"
        if ext == ".csv":
            shutil.copyfile(tmp_path, data_csv)
            df_for_schema = pd.read_csv(data_csv, low_memory=False)
        else:
            df_for_schema = pd.read_excel(tmp_path)
            df_for_schema.to_csv(data_csv, index=False)

        result = analyze_file(str(data_csv))
        sanitized_result = convert_numpy_types(result)

        schema_array = []
        for col in df_for_schema.columns:
            series = df_for_schema[col]
            high_type = infer_column_type(series)
            if high_type == "BOOLEAN":
                type_str = "BOOLEAN"
            elif high_type == "DATE":
                type_str = "DATE"
            elif high_type == "TIMESTAMP":
                type_str = "TIMESTAMP"
            elif high_type == "Numerical":
                num_series = _to_floatable(series).dropna()
                if not num_series.empty and (num_series % 1).abs().lt(1e-9).all():
                    vmin, vmax = num_series.min(), num_series.max()
                    type_str = "INTEGER" if (-2147483648 <= vmin <= 2147483647 and -2147483648 <= vmax <= 2147483647) else "BIGINT"
                else:
                    type_str = "DOUBLE PRECISION"
            else:
                type_str = "TEXT"
            schema_array.append({"column-name": str(col), "type": type_str})

        schema_path = ws_path / "data_schema.json"
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_array, f, ensure_ascii=False, indent=2)

        sanitized_result["workspace_id"] = ws_id
        return JSONResponse(content=sanitized_result)

    except ValueError as e:
        _delete_workspace(ws_id)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _delete_workspace(ws_id)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")
    finally:
        if tmp_path and tmp_path.exists():
            try: tmp_path.unlink()
            except Exception: pass

@app.post("/demo/run", summary="Run chatbot.js inside isolated workspace and return outputs")
async def run_chatbot_demo(payload: Dict[str, Any]):
    """
    Body:
    {
      "workspace_id": "<required>",
      "question": "...",
      "output": "table|viz|both",
      "params": {...},
      "model": "gemini-2.5-flash",
      "sandbox": true,
      "timeout_sec": 900
    }
    """
    import os, subprocess, shlex, time

    ws_id = (payload or {}).get("workspace_id")
    if not ws_id or not isinstance(ws_id, str):
        raise HTTPException(status_code=400, detail="Missing required 'workspace_id'.")
    ws_path = _get_workspace(ws_id)

    root_dir = Path(__file__).resolve().parent
    script_path = root_dir / "chatbot.js"
    data_csv = ws_path / "data.csv"
    schema_path = ws_path / "data_schema.json"

    out_file = "data_plan.json"
    result_json = ws_path / "data_result.json"
    result_png  = ws_path / "data_result.png"
    result_html = ws_path / "data_result.html"

    question = (payload or {}).get("question")
    if not question or not isinstance(question, str):
        raise HTTPException(status_code=400, detail="Please provide a 'question' string in the JSON body.")

    output = (payload or {}).get("output")
    if output is not None and output not in {"table", "viz", "both"}:
        raise HTTPException(status_code=400, detail="Invalid 'output'. Use one of: 'table', 'viz', 'both'.")

    params = (payload or {}).get("params") or {}
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="'params' must be a JSON object if provided.")

    model = (payload or {}).get("model", "gemini-2.5-flash")
    sandbox = bool((payload or {}).get("sandbox", True))
    timeout_sec = int((payload or {}).get("timeout_sec", 900))

    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"chatbot.js not found at {script_path}")
    if not data_csv.exists():
        raise HTTPException(status_code=400, detail="CSV not found in workspace. Upload a file first.")
    if not schema_path.exists():
        raise HTTPException(status_code=400, detail="Schema not found in workspace. Upload a file first to create it.")

    # Build command; run with cwd=workspace so outputs stay isolated
    cmd = ["node", str(script_path), str(data_csv), question, "--out", out_file, "--model", model]
    if sandbox: cmd += ["--run-sandbox", str(data_csv)]
    else:       cmd += ["--no-sandbox"]
    if output:  cmd += ["--output", output]
    if params:  cmd += ["--params", json.dumps(params, separators=(",", ":"))]

    # Clean previous outputs in workspace
    for p in (result_json, result_png, result_html, ws_path / out_file):
        if p.exists():
            try: p.unlink()
            except Exception: pass

    env = os.environ.copy()
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ws_path), env=env, capture_output=True, text=True, timeout=timeout_sec, check=False)
    duration = time.time() - t0

    def _read_json_safe(path: Path):
        if not path.exists(): return None
        try: return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"_error": f"Failed to parse {path.name}: {e}", "_raw": path.read_text(encoding="utf-8", errors="ignore")[:2000]}

    plan_obj = _read_json_safe(ws_path / out_file)
    result_obj = _read_json_safe(result_json)

    resp = {
        "ok": proc.returncode == 0 and (result_obj is not None or sandbox is False),
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "returncode": proc.returncode,
        "duration_sec": round(duration, 3),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "plan": plan_obj,
        "result": result_obj,
        "files": {
            "plan_json": str((ws_path / out_file)),
            "result_json": str(result_json),
            "result_png": str(result_png) if result_png.exists() else None,
            "result_html": str(result_html) if result_html.exists() else None,
        },
        "notes": [
            "Ensure GOOGLE_API_KEY is present in environment or .env next to chatbot.js.",
            "Execution ran in an isolated workspace; the workspace persists until /workspace/close is called by the client.",
        ],
    }

    if proc.returncode != 0 and not resp["result"]:
        for blob in (proc.stdout, proc.stderr):
            if "AI EXPLANATION OF FAILURE" in blob:
                resp["ai_explanation"] = blob.split("AI EXPLANATION OF FAILURE", 1)[-1].strip()
                break

    return JSONResponse(content=resp)

@app.post("/workspace/close", summary="Close and delete a workspace")
async def close_workspace(payload: Dict[str, Any]):
    ws_id = (payload or {}).get("workspace_id")
    if not ws_id or not isinstance(ws_id, str):
        raise HTTPException(status_code=400, detail="Missing required 'workspace_id'.")
    _delete_workspace(ws_id)
    return {"ok": True, "workspace_id": ws_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
