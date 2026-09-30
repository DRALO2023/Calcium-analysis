import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks


st.set_page_config(page_title="Calcium Analysis", layout="wide")
st.title("Calcium Imaging Post-Analysis")
st.caption(
    "Multi-file calcium analysis with file-specific ROI detection, "
    "background correction, ΔF/F0 normalization, and click-based manual events."
)

if "uploader_token" not in st.session_state:
    st.session_state.uploader_token = 0
if "manual_events" not in st.session_state:
    st.session_state.manual_events = {}
if "current_manual_points" not in st.session_state:
    st.session_state.current_manual_points = {}


def reset_analysis():
    next_token = st.session_state.get("uploader_token", 0) + 1
    st.session_state.clear()
    st.session_state.uploader_token = next_token
    st.session_state.manual_events = {}
    st.session_state.current_manual_points = {}


def read_table(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def clean_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_columns(df):
    columns = list(df.columns)
    normalized = {c: clean_name(c) for c in columns}

    label_candidates = [
        c for c in columns
        if normalized[c] in {"label", "condition", "sample", "experiment", "group", "treatment"}
    ]
    label_col = label_candidates[0] if label_candidates else None

    bg_candidates = [
        c for c in columns
        if "background" in normalized[c] and "mean" in normalized[c]
    ]
    if not bg_candidates:
        bg_candidates = [c for c in columns if normalized[c] in {"background", "bg", "meanbg"}]
    background_col = bg_candidates[0] if bg_candidates else None

    area_cols = [c for c in columns if normalized[c].startswith("area")]
    signal_cols = [
        c for c in columns
        if c != background_col and re.fullmatch(r"mean\d+", normalized[c])
    ]

    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    excluded = set(signal_cols + area_cols)
    if label_col is not None:
        excluded.add(label_col)
    if background_col is not None:
        excluded.add(background_col)
    time_candidates = [c for c in numeric_cols if c not in excluded]

    return {
        "label_column": label_col,
        "background_column": background_col,
        "signal_columns": signal_cols,
        "area_columns": area_cols,
        "time_column": time_candidates[0] if time_candidates else None,
    }


def first_valid(series):
    values = series.dropna()
    return values.iloc[0] if not values.empty else np.nan


def make_time_vector(df, time_column, frame_interval):
    if time_column is not None and time_column in df.columns:
        frames = pd.to_numeric(df[time_column], errors="coerce")
    else:
        frames = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    if frames.notna().sum() < 2:
        frames = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    else:
        frames = frames.ffill().bfill()
    time = (frames - frames.iloc[0]) * frame_interval
    return frames, time


def calculate_f0(values, time, settings):
    values = pd.to_numeric(values, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return np.nan
    mode = settings["f0_mode"]
    if mode == "First valid value":
        return float(valid.iloc[0])
    if mode == "Mean of first N valid rows":
        return float(valid.iloc[: max(1, settings["f0_n"])].mean())
    if mode == "Minimum value in trace":
        return float(valid.min())
    if mode == "Lower quartile (25th percentile)":
        return float(valid.quantile(0.25))
    if mode == "Mean within baseline-time window":
        mask = (time >= settings["baseline_start"]) & (time <= settings["baseline_end"])
        window = values.loc[mask].dropna()
        return float(window.mean()) if not window.empty else np.nan
    return float(valid.iloc[0])


def corrected_and_dff(raw, background, time, settings):
    raw = pd.to_numeric(raw, errors="coerce")
    if settings["background_mode"] == "Subtract empty-area background before ΔF/F0":
        corrected = raw - pd.to_numeric(background, errors="coerce")
    else:
        corrected = raw.copy()

    policy = settings["negative_policy"]
    if policy == "Clip corrected fluorescence below zero to zero":
        corrected = corrected.clip(lower=0)
    if policy == "Exclude ROI if any corrected value is below zero" and (corrected < 0).any():
        return corrected, pd.Series(np.nan, index=corrected.index), np.nan

    f0 = calculate_f0(corrected, time, settings)
    if not np.isfinite(f0) or f0 <= 0:
        return corrected, pd.Series(np.nan, index=corrected.index), f0
    return corrected, (corrected - f0) / f0, f0


def auc_positive(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan
    return float(np.trapezoid(np.maximum(y, 0), x)) if hasattr(np, "trapezoid") else float(np.trapz(np.maximum(y, 0), x))


def detect_candidates(frames, time, values, settings):
    x = np.asarray(frames, dtype=float)
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(t) & np.isfinite(y)
    x, t, y = x[mask], t[mask], y[mask]
    if len(y) < 3:
        return pd.DataFrame()
    indices, props = find_peaks(
        y,
        height=settings["min_peak_height"],
        prominence=settings["min_peak_prominence"],
        distance=max(1, settings["min_peak_distance"]),
    )
    return pd.DataFrame({
        "Auto event": range(1, len(indices) + 1),
        "Peak frame": x[indices].astype(int),
        "Peak time": t[indices],
        "Peak ΔF/F0": y[indices],
        "Peak prominence": props.get("prominences", np.nan),
    })


def process_file(df, file_name, settings):
    detected = detect_columns(df)
    signals = detected["signal_columns"]
    frames, time = make_time_vector(df, settings["time_column"], settings["frame_interval"])

    label_column = settings["label_column"]
    if label_column is not None and label_column in df.columns:
        label = str(first_valid(df[label_column]))
    else:
        label = re.sub(r"\.[^.]+$", "", file_name)
    if not label or label.lower() == "nan":
        label = re.sub(r"\.[^.]+$", "", file_name)

    output = pd.DataFrame({"Frame": frames, "Time": time})
    bg_col = settings["background_column"]
    if bg_col is not None and bg_col in df.columns:
        background = pd.to_numeric(df[bg_col], errors="coerce")
        output["Background"] = background
    else:
        background = pd.Series(0.0, index=df.index)

    warning_messages = []
    summaries = []
    cell_candidate_tables = []

    for signal in signals:
        raw = pd.to_numeric(df[signal], errors="coerce")
        corrected, dff, f0 = corrected_and_dff(raw, background, time, settings)
        output[f"Raw | {signal}"] = raw
        output[f"Corrected | {signal}"] = corrected
        output[f"dF/F0 | {signal}"] = dff

        if not np.isfinite(f0) or f0 <= 0:
            warning_messages.append(
                f"{file_name} — {signal}: ΔF/F0 not calculated because corrected F0 is missing, zero, or negative."
            )
            summaries.append({
                "File": file_name, "Label": label, "ROI": signal, "F0 corrected": f0,
                "Peak ΔF/F0": np.nan, "Peak time": np.nan, "AUC above baseline": np.nan,
                "Auto candidate count": 0, "Status": "Excluded: invalid F0",
            })
            continue

        candidates = detect_candidates(frames, time, dff, settings)
        if not candidates.empty:
            candidates.insert(0, "ROI", signal)
            candidates.insert(0, "Label", label)
            candidates.insert(0, "File", file_name)
            cell_candidate_tables.append(candidates)

        valid = dff.dropna()
        peak_index = dff.idxmax() if not valid.empty else None
        summaries.append({
            "File": file_name,
            "Label": label,
            "ROI": signal,
            "F0 corrected": f0,
            "Peak ΔF/F0": float(dff.loc[peak_index]) if peak_index is not None else np.nan,
            "Peak time": float(time.loc[peak_index]) if peak_index is not None else np.nan,
            "AUC above baseline": auc_positive(time, dff),
            "Auto candidate count": len(candidates),
            "Status": "Included",
        })

    dff_cols = [c for c in output.columns if c.startswith("dF/F0 | ")]
    output["Average dF/F0"] = output[dff_cols].mean(axis=1, skipna=True) if dff_cols else np.nan
    output["SEM dF/F0"] = output[dff_cols].sem(axis=1, ddof=1) if dff_cols else np.nan

    average_candidates = detect_candidates(frames, time, output["Average dF/F0"], settings)
    if not average_candidates.empty:
        average_candidates.insert(0, "Trace", "File average")
        average_candidates.insert(0, "Label", label)
        average_candidates.insert(0, "File", file_name)

    return {
        "file_name": file_name,
        "label": label,
        "processed_df": output,
        "summary_df": pd.DataFrame(summaries),
        "cell_candidates": pd.concat(cell_candidate_tables, ignore_index=True) if cell_candidate_tables else pd.DataFrame(),
        "average_candidates": average_candidates,
        "warnings": warning_messages,
    }


def trace_key(file_name, trace_column):
    return f"{file_name}::{trace_column}"


def initialize_manual_trace(key):
    if key not in st.session_state.manual_events:
        st.session_state.manual_events[key] = []
    if key not in st.session_state.current_manual_points:
        st.session_state.current_manual_points[key] = {"point1": None, "point2": None, "point3": None}


def trace_point(processed_df, trace_column, frame):
    rows = processed_df[processed_df["Frame"].astype(float) == float(frame)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    value = row[trace_column]
    if not np.isfinite(value):
        return None
    return {"frame": int(row["Frame"]), "time": float(row["Time"]), "value": float(value)}


def selection_figure(processed_df, trace_column, result, current, saved, positive_only):
    y = processed_df[trace_column].to_numpy(dtype=float)
    y_plot = np.maximum(y, 0) if positive_only else y
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=processed_df["Time"], y=y_plot, mode="lines+markers",
        name=f"{result['label']} | {result['file_name']} | {trace_column}",
        line=dict(color="#1f77b4", width=2.5), marker=dict(size=6),
        customdata=processed_df["Frame"].to_numpy(),
        hovertemplate="Frame: %{customdata}<br>Time: %{x:.4f}<br>ΔF/F0: %{y:.4f}<extra></extra>",
    ))

    def add_event(p1, p2, p3, label, opacity=1.0):
        fig.add_trace(go.Scatter(
            x=[p1["time"], p2["time"]], y=[p1["value"], p2["value"]],
            mode="lines+markers+text", line=dict(color=f"rgba(0,128,0,{opacity})", width=3),
            marker=dict(color=["green", "red"], size=[11, 13]),
            text=[f"{label} P1", f"{label} P2"], textposition="top center", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[p2["time"], p3["time"]], y=[p2["value"], p3["value"]],
            mode="lines+markers+text", line=dict(color=f"rgba(128,0,128,{opacity})", width=3, dash="dash"),
            marker=dict(color=["red", "purple"], size=[13, 11]),
            text=[f"{label} P2", f"{label} P3"], textposition="bottom center", showlegend=False,
        ))

    for event in saved:
        add_event(event["point1"], event["point2"], event["point3"], event["event_id"], 0.85)

    p1, p2, p3 = current.get("point1"), current.get("point2"), current.get("point3")
    if p1 is not None:
        fig.add_trace(go.Scatter(x=[p1["time"]], y=[p1["value"]], mode="markers+text", marker=dict(color="green", size=14), text=["Current P1"], textposition="bottom center", showlegend=False))
    if p2 is not None:
        fig.add_trace(go.Scatter(x=[p2["time"]], y=[p2["value"]], mode="markers+text", marker=dict(color="red", size=15), text=["Current P2"], textposition="top center", showlegend=False))
    if p3 is not None:
        fig.add_trace(go.Scatter(x=[p3["time"]], y=[p3["value"]], mode="markers+text", marker=dict(color="purple", size=14), text=["Current P3"], textposition="bottom center", showlegend=False))
    if p1 is not None and p2 is not None:
        fig.add_trace(go.Scatter(x=[p1["time"], p2["time"]], y=[p1["value"], p2["value"]], mode="lines", line=dict(color="green", width=3), showlegend=False))
    if p2 is not None and p3 is not None:
        fig.add_trace(go.Scatter(x=[p2["time"], p3["time"]], y=[p2["value"], p3["value"]], mode="lines", line=dict(color="purple", width=3, dash="dash"), showlegend=False))

    fig.update_layout(
        height=620, template="plotly_white", dragmode="select",
        xaxis_title=f"Time ({settings['time_unit']})", yaxis_title="ΔF/F0",
    )
    return fig


def manual_metrics(events):
    rows = []
    for event in events:
        p1, p2, p3 = event["point1"], event["point2"], event["point3"]
        rise_time = p2["time"] - p1["time"]
        decay_time = p3["time"] - p2["time"]
        rise_amp = p2["value"] - p1["value"]
        decay_amp = p3["value"] - p2["value"]
        uptake = rise_amp / rise_time if rise_time > 0 else np.nan
        decay = decay_amp / decay_time if decay_time > 0 else np.nan
        rows.append({
            "Event ID": event["event_id"],
            "Point 1 frame": p1["frame"], "Point 1 time": p1["time"], "Point 1 ΔF/F0": p1["value"],
            "Point 2 frame": p2["frame"], "Point 2 time": p2["time"], "Point 2 ΔF/F0": p2["value"],
            "Point 3 frame": p3["frame"], "Point 3 time": p3["time"], "Point 3 ΔF/F0": p3["value"],
            "Uptake duration": rise_time, "Uptake amplitude": rise_amp, "Endpoint uptake rate": uptake,
            "Decay duration": decay_time, "Decay amplitude": decay_amp, "Observed decay/release rate": decay,
            "Release magnitude": -decay if np.isfinite(decay) else np.nan,
            "Endpoint type": event["endpoint_type"], "Notes": event["notes"],
        })
    return pd.DataFrame(rows)


def all_files_figure(results, settings):
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
    fig = go.Figure()
    for i, result in enumerate(results):
        p = result["processed_df"]
        y = p["Average dF/F0"].to_numpy()
        if settings["positive_only"]:
            y = np.maximum(y, 0)
        fig.add_trace(go.Scatter(
            x=p["Time"], y=y, mode="lines", line=dict(color=colors[i % len(colors)], width=2.5),
            name=f"{result['label']} | {result['file_name']}",
        ))
    fig.update_layout(height=560, template="plotly_white", xaxis_title=f"Time ({settings['time_unit']})", yaxis_title="Average ΔF/F0", legend_title="Uploaded file")
    return fig


def safe_sheet_name(name, used):
    base = re.sub(r"[\\/*?:\[\]]", "_", str(name))[:31] or "sheet"
    candidate, number = base, 1
    while candidate in used:
        suffix = f"_{number}"
        candidate = base[:31 - len(suffix)] + suffix
        number += 1
    used.add(candidate)
    return candidate


def workbook_bytes(results, settings, summary, auto_candidates, click_metrics):
    output = io.BytesIO()
    used = set()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([settings]).to_excel(writer, index=False, sheet_name="analysis_settings")
        pd.DataFrame([{"File": r["file_name"], "Label": r["label"]} for r in results]).to_excel(writer, index=False, sheet_name="file_metadata")
        if not summary.empty:
            summary.to_excel(writer, index=False, sheet_name="cell_summary")
        if not auto_candidates.empty:
            auto_candidates.to_excel(writer, index=False, sheet_name="auto_candidates")
        if not click_metrics.empty:
            click_metrics.to_excel(writer, index=False, sheet_name="manual_events")
        for result in results:
            result["processed_df"].to_excel(writer, index=False, sheet_name=safe_sheet_name(f"trace_{result['file_name']}", used))
    output.seek(0)
    return output.getvalue()


with st.sidebar:
    st.header("Time")
    frame_interval = st.number_input("Frame interval", min_value=0.000001, value=1.0, step=0.1)
    time_unit = st.selectbox("Time unit", ["seconds", "minutes", "milliseconds"])

    st.header("Background and F0")
    background_mode = st.radio("Background method", ["Subtract empty-area background before ΔF/F0", "No background correction"], index=0)
    negative_policy = st.radio(
        "Negative corrected-signal handling",
        ["Keep negative values (recommended)", "Clip corrected fluorescence below zero to zero", "Exclude ROI if any corrected value is below zero"],
        index=0,
    )
    f0_mode = st.selectbox("F0 definition", ["First valid value", "Mean of first N valid rows", "Minimum value in trace", "Lower quartile (25th percentile)", "Mean within baseline-time window"], index=1)
    f0_n, baseline_start, baseline_end = 5, None, None
    if f0_mode == "Mean of first N valid rows":
        f0_n = int(st.number_input("Number of baseline rows", min_value=1, value=5, step=1))
    if f0_mode == "Mean within baseline-time window":
        baseline_start = st.number_input(f"Baseline start ({time_unit})", value=0.0)
        baseline_end = st.number_input(f"Baseline end ({time_unit})", value=5.0)

    st.header("Automatic candidates")
    min_peak_height = st.number_input("Minimum candidate peak height (ΔF/F0)", min_value=0.0, value=0.05, step=0.01)
    min_peak_prominence = st.number_input("Minimum candidate prominence (ΔF/F0)", min_value=0.0, value=0.05, step=0.01)
    min_peak_distance = int(st.number_input("Minimum frames between candidates", min_value=1, value=3, step=1))

    st.header("Display")
    positive_only = st.checkbox("Plot only positive ΔF/F0", value=False)
    st.button("Reset analysis / upload new files", on_click=reset_analysis, use_container_width=True)

uploaded_files = st.file_uploader(
    "Upload calcium time-series files", type=["csv", "xlsx", "xls"], accept_multiple_files=True,
    key=f"files_{st.session_state.uploader_token}",
    help="Every uploaded file can have its own number of Mean1...MeanN ROI columns.",
)
if not uploaded_files:
    st.info("Upload one or more CSV, XLSX, or XLS calcium files to start.")
    st.stop()

loaded = []
for uploaded in uploaded_files:
    try:
        loaded.append({"file_name": uploaded.name, "df": read_table(uploaded)})
    except Exception as exc:
        st.error(f"Could not read {uploaded.name}: {exc}")
if not loaded:
    st.stop()

st.subheader("Detected input structure")
detection_rows = []
for item in loaded:
    d = detect_columns(item["df"])
    detection_rows.append({
        "File": item["file_name"], "Suggested label": d["label_column"], "Suggested time": d["time_column"],
        "Suggested background": d["background_column"], "ROI Mean columns": ", ".join(d["signal_columns"]), "ROI count": len(d["signal_columns"]),
    })
st.dataframe(pd.DataFrame(detection_rows), use_container_width=True)

first_df = loaded[0]["df"]
first_detected = detect_columns(first_df)
columns = list(first_df.columns)
with st.expander("Shared column mapping", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        label_options = ["<auto/file name>"] + columns
        label_index = columns.index(first_detected["label_column"]) + 1 if first_detected["label_column"] in columns else 0
        selected_label = st.selectbox("Label column", label_options, index=label_index)
    with c2:
        time_options = ["<auto row index>"] + columns
        time_index = columns.index(first_detected["time_column"]) + 1 if first_detected["time_column"] in columns else 0
        selected_time = st.selectbox("Frame/time column", time_options, index=time_index)
    with c3:
        bg_options = ["<none>"] + columns
        bg_index = bg_options.index(first_detected["background_column"]) if first_detected["background_column"] in bg_options else 0
        selected_background = st.selectbox("Empty-area background column", bg_options, index=bg_index)
    with c4:
        st.info("ROI traces are detected separately in each file. All Mean1, Mean2, ..., MeanN columns are analyzed automatically.")

if background_mode == "Subtract empty-area background before ΔF/F0" and selected_background == "<none>":
    st.error("Select Mean(background), or choose 'No background correction'.")
    st.stop()

settings = {
    "label_column": None if selected_label == "<auto/file name>" else selected_label,
    "time_column": None if selected_time == "<auto row index>" else selected_time,
    "background_column": None if selected_background == "<none>" else selected_background,
    "frame_interval": frame_interval, "time_unit": time_unit, "background_mode": background_mode,
    "negative_policy": negative_policy, "f0_mode": f0_mode, "f0_n": f0_n,
    "baseline_start": baseline_start, "baseline_end": baseline_end,
    "min_peak_height": min_peak_height, "min_peak_prominence": min_peak_prominence,
    "min_peak_distance": min_peak_distance, "positive_only": positive_only,
}

errors, warnings = [], []
for item in loaded:
    df, name = item["df"], item["file_name"]
    d = detect_columns(df)
    if not d["signal_columns"]:
        errors.append(f"{name}: no Mean1...MeanN cellular ROI columns were detected.")
    if settings["background_column"] is not None and settings["background_column"] not in df.columns:
        errors.append(f"{name}: missing selected background column '{settings['background_column']}'.")
    if settings["time_column"] is not None and settings["time_column"] not in df.columns:
        errors.append(f"{name}: missing selected time/frame column '{settings['time_column']}'.")
    if settings["label_column"] is not None and settings["label_column"] not in df.columns:
        warnings.append(f"{name}: selected label column absent; filename will be used as label.")
for message in warnings:
    st.warning(message)
if errors:
    st.error("Some files cannot be analyzed:\n\n" + "\n".join(errors))
    st.stop()

results = [process_file(item["df"], item["file_name"], settings) for item in loaded]
for result in results:
    for message in result["warnings"]:
        st.warning(message)

st.subheader("All uploaded file-average traces")
all_plot = all_files_figure(results, settings)
st.plotly_chart(all_plot, use_container_width=True)

summary_tables = [r["summary_df"] for r in results if not r["summary_df"].empty]
combined_summary = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
auto_tables = []
for result in results:
    if not result["cell_candidates"].empty:
        auto_tables.append(result["cell_candidates"])
    if not result["average_candidates"].empty:
        auto_tables.append(result["average_candidates"])
combined_auto = pd.concat(auto_tables, ignore_index=True) if auto_tables else pd.DataFrame()

st.subheader("Cell/ROI summary")
st.dataframe(combined_summary, use_container_width=True)
with st.expander("Automatic candidate peaks (optional guide)"):
    st.caption("Candidates are not used for manual point selection. They are only shown for optional quality control.")
    st.dataframe(combined_auto, use_container_width=True)

st.subheader("Click-based manual event selection")
st.markdown(
    "For each event, manually select **Point 1** (lowest/start point), **Point 2** (chosen peak), and **Point 3** (post-peak endpoint). "
    "Uptake is calculated from Point 1 → Point 2. Observed decay/release is calculated from Point 2 → Point 3. "
    "Point 3 does not have to be zero."
)

selected_file_name = st.selectbox("File for manual selection", [r["file_name"] for r in results])
selected_result = next(r for r in results if r["file_name"] == selected_file_name)
processed = selected_result["processed_df"]
trace_options = ["Average dF/F0"] + [c for c in processed.columns if c.startswith("dF/F0 | ")]
selected_trace = st.selectbox("Individual trace", trace_options)
key = trace_key(selected_file_name, selected_trace)
initialize_manual_trace(key)
current = st.session_state.current_manual_points[key]
saved = st.session_state.manual_events[key]

next_label = st.radio(
    "Select next point",
    ["Point 1: uptake start / lowest point", "Point 2: selected peak", "Point 3: release endpoint / lowest post-peak point"],
    horizontal=True,
)
point_map = {
    "Point 1: uptake start / lowest point": "point1",
    "Point 2: selected peak": "point2",
    "Point 3: release endpoint / lowest post-peak point": "point3",
}

st.info("Use the Plotly graph toolbar **Select Points** tool, then click a marker on the graph. The selected acquired frame is used exactly.")
fig = selection_figure(processed, selected_trace, selected_result, current, saved, settings["positive_only"])
selection = st.plotly_chart(
    fig, use_container_width=True, key=f"selector_{key}", on_select="rerun", selection_mode="points",
    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
)

if selection and selection.selection and selection.selection.get("points"):
    point_data = selection.selection["points"][0]
    frame = point_data.get("customdata")
    chosen = trace_point(processed, selected_trace, frame)
    if chosen is not None:
        st.session_state.current_manual_points[key][point_map[next_label]] = chosen
        st.rerun()

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Undo last point", key=f"undo_{key}", use_container_width=True):
        points = st.session_state.current_manual_points[key]
        if points["point3"] is not None:
            points["point3"] = None
        elif points["point2"] is not None:
            points["point2"] = None
        else:
            points["point1"] = None
        st.rerun()
with c2:
    if st.button("Reset current event", key=f"reset_current_{key}", use_container_width=True):
        st.session_state.current_manual_points[key] = {"point1": None, "point2": None, "point3": None}
        st.rerun()
with c3:
    if st.button("Delete last saved event", key=f"delete_last_{key}", use_container_width=True):
        if st.session_state.manual_events[key]:
            st.session_state.manual_events[key].pop()
        st.rerun()
with c4:
    if st.button("Reset all events for this trace", key=f"reset_all_{key}", use_container_width=True):
        st.session_state.manual_events[key] = []
        st.session_state.current_manual_points[key] = {"point1": None, "point2": None, "point3": None}
        st.rerun()

endpoint_type = st.selectbox(
    "Point 3 endpoint type",
    ["Post-peak trough", "Returned to baseline", "Before next rise", "Recording end", "Sustained plateau", "Manual endpoint"],
    key=f"endpoint_{key}",
)
notes = st.text_input("Event notes", key=f"notes_{key}")
complete = all(current[p] is not None for p in ["point1", "point2", "point3"])
if not complete:
    st.caption("Current event is incomplete: select Point 1, Point 2, and Point 3.")

if st.button("Save current three-point event", disabled=not complete, key=f"save_{key}", use_container_width=True):
    p1, p2, p3 = current["point1"], current["point2"], current["point3"]
    if not (p1["time"] < p2["time"] < p3["time"]):
        st.error("Points must be in chronological order: Point 1 before Point 2 before Point 3.")
    else:
        event_number = len(st.session_state.manual_events[key]) + 1
        st.session_state.manual_events[key].append({
            "event_id": f"E{event_number}", "point1": p1, "point2": p2, "point3": p3,
            "endpoint_type": endpoint_type, "notes": notes,
        })
        st.session_state.current_manual_points[key] = {"point1": None, "point2": None, "point3": None}
        st.rerun()

saved_metrics = manual_metrics(st.session_state.manual_events[key])
st.subheader("Saved manual event metrics for selected trace")
if saved_metrics.empty:
    st.info("No saved manual events for this trace yet.")
else:
    st.dataframe(saved_metrics, use_container_width=True)

all_manual_metric_tables = []
for state_key, events in st.session_state.manual_events.items():
    metrics = manual_metrics(events)
    if not metrics.empty:
        metrics.insert(0, "Trace key", state_key)
        all_manual_metric_tables.append(metrics)
all_manual_metrics = pd.concat(all_manual_metric_tables, ignore_index=True) if all_manual_metric_tables else pd.DataFrame()

st.subheader("Downloads")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
excel = workbook_bytes(results, settings, combined_summary, combined_auto, all_manual_metrics)
d1, d2, d3, d4 = st.columns(4)
with d1:
    st.download_button("Download Excel workbook", excel, f"calcium_analysis_{timestamp}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
with d2:
    st.download_button("Download cell summary CSV", combined_summary.to_csv(index=False).encode("utf-8"), f"calcium_cell_summary_{timestamp}.csv", "text/csv", use_container_width=True)
with d3:
    st.download_button("Download automatic candidates CSV", combined_auto.to_csv(index=False).encode("utf-8"), f"calcium_auto_candidates_{timestamp}.csv", "text/csv", use_container_width=True)
with d4:
    st.download_button("Download manual events CSV", all_manual_metrics.to_csv(index=False).encode("utf-8"), f"calcium_manual_events_{timestamp}.csv", "text/csv", use_container_width=True)
st.download_button("Download all-file graph HTML", all_plot.to_html(include_plotlyjs="cdn"), f"calcium_all_files_{timestamp}.html", "text/html")
