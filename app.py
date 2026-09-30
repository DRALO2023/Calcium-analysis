import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.signal import find_peaks

st.set_page_config(page_title="Calcium Analysis", layout="wide")

st.title("Calcium Imaging Post-Analysis")
st.caption(
    "Multi-file calcium analysis with empty-area background subtraction, ΔF/F0 normalization, "
    "and event-wise uptake/release kinetics."
)

if "uploader_token" not in st.session_state:
    st.session_state.uploader_token = 0


def reset_analysis():
    for key in list(st.session_state.keys()):
        if key != "uploader_token":
            del st.session_state[key]
    st.session_state.uploader_token += 1


def safe_sheet_name(name, used=None):
    name = re.sub(r"[\\/*?:\[\]]", "_", str(name))[:31] or "sheet"
    if used is None:
        return name
    candidate = name
    n = 1
    while candidate in used:
        suffix = f"_{n}"
        candidate = f"{name[:31-len(suffix)]}{suffix}"
        n += 1
    used.add(candidate)
    return candidate


def read_table(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def clean_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def detect_columns(df):
    columns = list(df.columns)
    normalized = {c: clean_name(c) for c in columns}

    label_candidates = [
        c for c in columns
        if normalized[c] in {"label", "condition", "sample", "experiment", "group"}
    ]
    label_col = label_candidates[0] if label_candidates else None

    bg_candidates = [
        c for c in columns
        if "background" in normalized[c] and "mean" in normalized[c]
    ]
    if not bg_candidates:
        bg_candidates = [c for c in columns if normalized[c] in {"background", "bg"}]
    background_col = bg_candidates[0] if bg_candidates else None

    area_cols = [c for c in columns if normalized[c].startswith("area")]
    mean_cols = []
    for c in columns:
        norm = normalized[c]
        if norm == "meanbackground" or "background" in norm:
            continue
        if re.fullmatch(r"mean\d+", norm):
            mean_cols.append(c)

    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    excluded = set(mean_cols + area_cols + ([background_col] if background_col else []) + ([label_col] if label_col else []))
    time_candidates = [c for c in numeric_cols if c not in excluded]
    time_col = time_candidates[0] if time_candidates else None

    return {
        "label_col": label_col,
        "background_col": background_col,
        "mean_cols": mean_cols,
        "area_cols": area_cols,
        "time_col": time_col,
    }


def first_valid(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.iloc[0] if not s.empty else np.nan


def get_f0(series, mode, n_rows, baseline_start=None, baseline_end=None):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return np.nan
    if mode == "First valid value":
        return float(valid.iloc[0])
    if mode == "Mean of first N valid rows":
        return float(valid.iloc[:max(1, n_rows)].mean())
    if mode == "Minimum value in trace":
        return float(valid.min())
    if mode == "Lower quartile (25th percentile)":
        return float(valid.quantile(0.25))
    if mode == "Mean within baseline-time window":
        mask = (s.index >= baseline_start) & (s.index <= baseline_end)
        window = s.loc[mask].dropna()
        return float(window.mean()) if not window.empty else np.nan
    return float(valid.iloc[0])


def interpolate_zero_crossing(x1, y1, x2, y2):
    if not np.isfinite([x1, y1, x2, y2]).all() or y2 == y1:
        return float(x2)
    return float(x1 + (0 - y1) * (x2 - x1) / (y2 - y1))


def slope_from_segment(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.unique(x).size < 2:
        return np.nan
    return float(stats.linregress(x, y).slope)


def event_boundaries(x, y, peak_idx):
    start_idx = 0
    start_time = float(x[0])
    for i in range(peak_idx - 1, -1, -1):
        if y[i] <= 0:
            start_idx = i
            if i < peak_idx and y[i + 1] > 0:
                start_time = interpolate_zero_crossing(x[i], y[i], x[i + 1], y[i + 1])
            else:
                start_time = float(x[i])
            break

    end_idx = None
    end_time = np.nan
    for i in range(peak_idx + 1, len(y)):
        if y[i] <= 0:
            end_idx = i
            if y[i - 1] > 0:
                end_time = interpolate_zero_crossing(x[i - 1], y[i - 1], x[i], y[i])
            else:
                end_time = float(x[i])
            break
    return start_idx, start_time, end_idx, end_time


def detect_events(time_values, trace, min_height, min_prominence, min_distance, min_points):
    x = np.asarray(time_values, dtype=float)
    y = np.asarray(trace, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return pd.DataFrame()

    peaks, properties = find_peaks(
        y,
        height=min_height,
        prominence=min_prominence,
        distance=max(1, int(min_distance))
    )
    rows = []
    for order, peak_idx in enumerate(peaks, start=1):
        start_idx, start_time, end_idx, end_time = event_boundaries(x, y, peak_idx)
        rise_x = x[start_idx:peak_idx + 1]
        rise_y = y[start_idx:peak_idx + 1]
        uptake_rate = slope_from_segment(rise_x, rise_y) if len(rise_x) >= min_points else np.nan
        max_rise = np.nanmax(np.diff(rise_y) / np.diff(rise_x)) if len(rise_x) >= 2 and np.all(np.diff(rise_x) > 0) else np.nan

        if end_idx is not None:
            decay_x = x[peak_idx:end_idx + 1]
            decay_y = y[peak_idx:end_idx + 1]
            release_rate = slope_from_segment(decay_x, decay_y) if len(decay_x) >= min_points else np.nan
            max_release = np.nanmin(np.diff(decay_y) / np.diff(decay_x)) if len(decay_x) >= 2 and np.all(np.diff(decay_x) > 0) else np.nan
            duration = end_time - start_time
            auc = float(np.trapz(np.maximum(y[start_idx:end_idx + 1], 0), x[start_idx:end_idx + 1]))
            release_status = "Returned to baseline"
        else:
            release_rate = np.nan
            max_release = np.nan
            duration = np.nan
            auc = np.nan
            release_status = "No return to baseline"

        if y[peak_idx] <= 0 or (peak_idx - start_idx + 1) < min_points:
            continue

        rows.append({
            "Event": order,
            "Start time": start_time,
            "Peak time": float(x[peak_idx]),
            "End time": end_time,
            "Peak ΔF/F0": float(y[peak_idx]),
            "Prominence": float(properties["prominences"][order - 1]),
            "Duration": duration,
            "AUC above baseline": auc,
            "Uptake slope": uptake_rate,
            "Max rise rate": max_rise,
            "Release slope": release_rate,
            "Release magnitude": -release_rate if np.isfinite(release_rate) else np.nan,
            "Max decay rate": max_release,
            "Release status": release_status,
        })
    return pd.DataFrame(rows)


def process_file(df, file_name, config):
    time_col = config["time_col"]
    label_col = config["label_col"]
    signal_cols = config["signal_cols"]
    background_col = config["background_col"]

    raw_time = pd.to_numeric(df[time_col], errors="coerce") if time_col else pd.Series(np.arange(len(df)), index=df.index)
    if raw_time.notna().sum() >= 2:
        frame = raw_time.fillna(method="ffill").fillna(method="bfill")
        time = (frame - frame.iloc[0]) * config["frame_interval"]
    else:
        frame = pd.Series(np.arange(len(df)), index=df.index)
        time = frame * config["frame_interval"]

    label = str(first_valid(df[label_col])) if label_col else ""
    if not label or label == "nan":
        label = re.sub(r"\.[^.]+$", "", file_name)

    output = pd.DataFrame({"Frame": frame, "Time": time})
    background = pd.to_numeric(df[background_col], errors="coerce") if background_col else None
    if background is not None:
        output["Background"] = background

    trace_rows = []
    event_frames = []
    warnings = []
    for col in signal_cols:
        raw = pd.to_numeric(df[col], errors="coerce")
        if config["background_mode"] == "Subtract empty-area background before ΔF/F0":
            corrected = raw - background
        else:
            corrected = raw.copy()

        f0_series = corrected.copy()
        f0_series.index = time.to_numpy()
        f0 = get_f0(
            f0_series,
            config["f0_mode"],
            config["f0_n"],
            config["baseline_start"],
            config["baseline_end"],
        )
        output[f"Raw | {col}"] = raw
        output[f"Corrected | {col}"] = corrected

        if not np.isfinite(f0) or f0 <= 0:
            output[f"dF/F0 | {col}"] = np.nan
            warnings.append(f"{file_name}: {col} excluded because corrected F0 is non-positive or missing.")
            continue

        dff = (corrected - f0) / f0
        output[f"dF/F0 | {col}"] = dff
        events = detect_events(
            output["Time"].to_numpy(), dff.to_numpy(),
            config["min_height"], config["min_prominence"],
            config["min_distance"], config["min_event_points"]
        )
        if not events.empty:
            events.insert(0, "ROI", col)
            events.insert(0, "Label", label)
            events.insert(0, "File", file_name)
            event_frames.append(events)

        trace_rows.append({
            "File": file_name,
            "Label": label,
            "ROI": col,
            "F0 corrected": f0,
            "Peak ΔF/F0": float(np.nanmax(dff)) if dff.notna().any() else np.nan,
            "Peak time": float(output.loc[dff.idxmax(), "Time"]) if dff.notna().any() else np.nan,
            "AUC above baseline": float(np.trapz(np.maximum(dff.fillna(0), 0), output["Time"])) if dff.notna().any() else np.nan,
            "Event count": len(events),
        })

    dff_cols = [c for c in output.columns if c.startswith("dF/F0 | ")]
    output["Average dF/F0"] = output[dff_cols].mean(axis=1, skipna=True) if dff_cols else np.nan
    output["SEM dF/F0"] = output[dff_cols].sem(axis=1, ddof=1) if dff_cols else np.nan

    average_events = detect_events(
        output["Time"].to_numpy(), output["Average dF/F0"].to_numpy(),
        config["min_height"], config["min_prominence"],
        config["min_distance"], config["min_event_points"]
    )
    if not average_events.empty:
        average_events.insert(0, "Trace", "File average")
        average_events.insert(0, "Label", label)
        average_events.insert(0, "File", file_name)

    return {
        "file": file_name,
        "label": label,
        "processed": output,
        "cell_summary": pd.DataFrame(trace_rows),
        "cell_events": pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(),
        "average_events": average_events,
        "warnings": warnings,
    }


def average_by_label(results):
    rows = []
    for result in results:
        p = result["processed"]
        rows.append(pd.DataFrame({
            "File": result["file"],
            "Label": result["label"],
            "Time": p["Time"],
            "Average dF/F0": p["Average dF/F0"],
        }))
    long_df = pd.concat(rows, ignore_index=True)
    group = long_df.groupby(["Label", "Time"], as_index=False)["Average dF/F0"].agg(["mean", "sem", "count"]).reset_index()
    group.columns = ["Label", "Time", "Mean dF/F0", "SEM dF/F0", "N files"]
    return long_df, group


def make_plot(results, grouped, show_sem, positive_only, time_unit, show_events):
    fig = go.Figure()
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
    labels = list(grouped["Label"].drop_duplicates())
    for i, label in enumerate(labels):
        color = palette[i % len(palette)]
        g = grouped[grouped["Label"] == label].sort_values("Time")
        y = g["Mean dF/F0"].to_numpy()
        if positive_only:
            y = np.maximum(y, 0)
        fig.add_trace(go.Scatter(x=g["Time"], y=y, mode="lines", name=label, line=dict(color=color, width=3)))
        if show_sem and g["N files"].max() > 1:
            sem = g["SEM dF/F0"].fillna(0).to_numpy()
            fig.add_trace(go.Scatter(x=g["Time"], y=y + sem, mode="lines", line=dict(color=color, width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=g["Time"], y=y - sem, mode="lines", line=dict(color=color, width=0), fill="tonexty", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)", showlegend=False))

    if show_events:
        for result in results:
            events = result["average_events"]
            if not events.empty:
                p = result["processed"]
                for _, ev in events.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[ev["Peak time"]], y=[ev["Peak ΔF/F0"]], mode="markers",
                        marker=dict(symbol="x", size=9, color="black"), showlegend=False,
                        hovertemplate=f"{result['label']}<br>Peak: %{{y:.3f}}<br>Time: %{{x:.3f}}<extra></extra>"
                    ))
    fig.update_layout(
        height=520,
        template="plotly_white",
        xaxis_title=f"Time ({time_unit})",
        yaxis_title="ΔF/F0",
        legend_title="Label",
    )
    return fig


def to_excel_bytes(results, grouped, config):
    bio = io.BytesIO()
    used = set()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame([config]).to_excel(writer, index=False, sheet_name="analysis_settings")
        metadata = pd.DataFrame([{"File": r["file"], "Label": r["label"]} for r in results])
        metadata.to_excel(writer, index=False, sheet_name="file_metadata")
        grouped.to_excel(writer, index=False, sheet_name="group_average")
        summaries = [r["cell_summary"] for r in results if not r["cell_summary"].empty]
        if summaries:
            pd.concat(summaries, ignore_index=True).to_excel(writer, index=False, sheet_name="cell_summary")
        events = [r["cell_events"] for r in results if not r["cell_events"].empty]
        if events:
            pd.concat(events, ignore_index=True).to_excel(writer, index=False, sheet_name="cell_events")
        avg_events = [r["average_events"] for r in results if not r["average_events"].empty]
        if avg_events:
            pd.concat(avg_events, ignore_index=True).to_excel(writer, index=False, sheet_name="average_events")
        for r in results:
            sheet = safe_sheet_name(f"trace_{r['file']}", used)
            r["processed"].to_excel(writer, index=False, sheet_name=sheet)
    bio.seek(0)
    return bio.getvalue()


with st.sidebar:
    st.header("Analysis settings")
    frame_interval = st.number_input("Frame interval", min_value=0.000001, value=1.0, step=0.1)
    time_unit = st.selectbox("Time unit", ["seconds", "minutes", "milliseconds"], index=0)
    background_mode = st.radio(
        "Background correction",
        ["Subtract empty-area background before ΔF/F0", "No background correction"],
        index=0,
        help="Recommended: subtract Mean(background) at each frame before determining F0 and calculating ΔF/F0."
    )
    f0_mode = st.selectbox(
        "F0 definition",
        ["First valid value", "Mean of first N valid rows", "Minimum value in trace", "Lower quartile (25th percentile)", "Mean within baseline-time window"],
        index=1,
    )
    f0_n = 5
    baseline_start, baseline_end = None, None
    if f0_mode == "Mean of first N valid rows":
        f0_n = int(st.number_input("Number of baseline rows", min_value=1, value=5, step=1))
    if f0_mode == "Mean within baseline-time window":
        baseline_start = st.number_input("Baseline start (frame index)", value=0.0)
        baseline_end = st.number_input("Baseline end (frame index)", value=5.0)
    st.divider()
    st.header("Significant-event settings")
    min_height = st.number_input("Minimum peak height (ΔF/F0)", min_value=0.0, value=0.05, step=0.01)
    min_prominence = st.number_input("Minimum peak prominence (ΔF/F0)", min_value=0.0, value=0.05, step=0.01)
    min_distance = int(st.number_input("Minimum frames between peaks", min_value=1, value=3, step=1))
    min_event_points = int(st.number_input("Minimum points in rise/decay", min_value=2, value=3, step=1))
    show_sem = st.checkbox("Show SEM across files sharing a label", value=True)
    positive_only = st.checkbox("Plot only positive ΔF/F0", value=False)
    show_events = st.checkbox("Mark accepted peaks", value=True)
    st.button("Reset analysis / upload new files", on_click=reset_analysis, use_container_width=True)

uploaded_files = st.file_uploader(
    "Upload calcium time-series files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    key=f"files_{st.session_state.uploader_token}",
    help="Expected format: Label, frame/time column, Mean1...MeanN, and Mean(background). Area columns are retained as metadata but are not signals."
)

if not uploaded_files:
    st.info("Upload one or more calcium time-series files to begin.")
    st.stop()

loaded = []
for f in uploaded_files:
    try:
        loaded.append((f.name, read_table(f)))
    except Exception as exc:
        st.error(f"Could not read {f.name}: {exc}")

if not loaded:
    st.stop()

st.subheader("Detected columns")
preview_rows = []
for file_name, df in loaded:
    d = detect_columns(df)
    preview_rows.append({
        "File": file_name,
        "Suggested label": d["label_col"],
        "Suggested time": d["time_col"],
        "Suggested signal columns": ", ".join(map(str, d["mean_cols"])),
        "Suggested background": d["background_col"],
        "Area columns retained": ", ".join(map(str, d["area_cols"])),
    })
st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

first_file, first_df = loaded[0]
detected = detect_columns(first_df)
all_cols = first_df.columns.tolist()

with st.expander("Column mapping (applied to all files)", expanded=True):
    map1, map2, map3, map4 = st.columns(4)
    with map1:
        label_col = st.selectbox("Label column", ["<none>"] + all_cols, index=(all_cols.index(detected["label_col"]) + 1 if detected["label_col"] in all_cols else 0))
    with map2:
        time_col = st.selectbox("Frame/time column", ["<auto index>"] + all_cols, index=(all_cols.index(detected["time_col"]) + 1 if detected["time_col"] in all_cols else 0))
    with map3:
        background_options = ["<none>"] + all_cols
        background_col = st.selectbox("Empty-area background column", background_options, index=(background_options.index(detected["background_col"]) if detected["background_col"] in background_options else 0))
    with map4:
        signal_cols = st.multiselect("Cell signal columns", all_cols, default=detected["mean_cols"])

if background_mode.startswith("Subtract") and background_col == "<none>":
    st.error("Select the Mean(background) column or choose 'No background correction'.")
    st.stop()
if not signal_cols:
    st.error("Select at least one cellular Mean column.")
    st.stop()

config = {
    "time_col": None if time_col == "<auto index>" else time_col,
    "label_col": None if label_col == "<none>" else label_col,
    "background_col": None if background_col == "<none>" else background_col,
    "signal_cols": signal_cols,
    "frame_interval": frame_interval,
    "time_unit": time_unit,
    "background_mode": background_mode,
    "f0_mode": f0_mode,
    "f0_n": f0_n,
    "baseline_start": baseline_start,
    "baseline_end": baseline_end,
    "min_height": min_height,
    "min_prominence": min_prominence,
    "min_distance": min_distance,
    "min_event_points": min_event_points,
}

missing = []
for file_name, df in loaded:
    needed = signal_cols + ([config["background_col"]] if config["background_col"] else []) + ([config["time_col"]] if config["time_col"] else [])
    absent = [c for c in needed if c not in df.columns]
    if absent:
        missing.append(f"{file_name}: missing {', '.join(map(str, absent))}")
if missing:
    st.error("The common mapping cannot be applied to every file:

" + "
".join(missing))
    st.stop()

results = [process_file(df, name, config) for name, df in loaded]
all_warnings = [warning for r in results for warning in r["warnings"]]
if all_warnings:
    for warning in all_warnings:
        st.warning(warning)

long_df, grouped = average_by_label(results)
fig = make_plot(results, grouped, show_sem, positive_only, time_unit, show_events)
st.subheader("Average calcium traces")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Processed data preview")
selected_file = st.selectbox("View processed file", [r["file"] for r in results])
selected = next(r for r in results if r["file"] == selected_file)
st.caption(f"Label: {selected['label']}")
st.dataframe(selected["processed"], use_container_width=True, height=340)

st.subheader("Per-cell summary")
cell_summary = pd.concat([r["cell_summary"] for r in results if not r["cell_summary"].empty], ignore_index=True)
st.dataframe(cell_summary, use_container_width=True)

st.subheader("Significant calcium events and rates")
st.markdown(
    "- **Uptake slope** is a linear-regression slope from the last zero/baseline crossing before an accepted positive peak to that peak.
"
    "- **Release slope** is calculated only from the peak to the first return to zero or below; it is negative.
"
    "- Events without a return to baseline are retained but their release metrics are reported as missing.
"
    "- Peak acceptance requires the selected minimum height, prominence, separation, and minimum rise length."
)
cell_events = pd.concat([r["cell_events"] for r in results if not r["cell_events"].empty], ignore_index=True) if any(not r["cell_events"].empty for r in results) else pd.DataFrame()
avg_events = pd.concat([r["average_events"] for r in results if not r["average_events"].empty], ignore_index=True) if any(not r["average_events"].empty for r in results) else pd.DataFrame()
rate_tab1, rate_tab2 = st.tabs(["Cell/ROI events", "File-average events"])
with rate_tab1:
    if cell_events.empty:
        st.info("No individual-ROI events met the current significance criteria.")
    else:
        st.dataframe(cell_events, use_container_width=True)
with rate_tab2:
    if avg_events.empty:
        st.info("No file-average events met the current significance criteria.")
    else:
        st.dataframe(avg_events, use_container_width=True)

st.subheader("Downloads")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_data = to_excel_bytes(results, grouped, config)
d1, d2, d3, d4 = st.columns(4)
with d1:
    st.download_button("Download workbook", excel_data, f"calcium_analysis_{ts}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
with d2:
    st.download_button("Download group averages CSV", grouped.to_csv(index=False).encode("utf-8"), f"calcium_group_averages_{ts}.csv", "text/csv", use_container_width=True)
with d3:
    st.download_button("Download cell summaries CSV", cell_summary.to_csv(index=False).encode("utf-8"), f"calcium_cell_summary_{ts}.csv", "text/csv", use_container_width=True)
with d4:
    st.download_button("Download event metrics CSV", cell_events.to_csv(index=False).encode("utf-8"), f"calcium_events_{ts}.csv", "text/csv", use_container_width=True)

st.download_button(
    "Download interactive graph HTML",
    fig.to_html(include_plotlyjs="cdn"),
    f"calcium_trace_{ts}.html",
    "text/html",
    use_container_width=False,
)
