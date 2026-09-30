import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# App setup and persistent session state
# ============================================================

st.set_page_config(
    page_title="Calcium Analysis",
    layout="wide",
)

st.title("Calcium Imaging Post-Analysis")
st.caption(
    "Manual P1–P4 calcium-event curation with multi-file ROI detection, "
    "background correction, and ΔF/F0 normalization."
)

if "uploader_token" not in st.session_state:
    st.session_state.uploader_token = 0

if "manual_events" not in st.session_state:
    st.session_state.manual_events = {}

if "current_points" not in st.session_state:
    st.session_state.current_points = {}


def reset_analysis():
    """
    Clear uploaded-file widgets, stored manual events, and temporary points.
    """
    token = st.session_state.get("uploader_token", 0) + 1

    st.session_state.clear()

    st.session_state.uploader_token = token
    st.session_state.manual_events = {}
    st.session_state.current_points = {}


# ============================================================
# Input and column detection
# ============================================================

def read_table(uploaded_file):
    """
    Read a CSV or Excel file.
    """
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file)


def clean_name(value):
    """
    Normalize a column name for automatic matching.
    """
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def detect_columns(df):
    """
    Detect likely Label, background, cellular Mean ROI, Area, and time columns.

    Cellular ROI traces:
        Mean1, Mean2, Mean3, ..., MeanN

    Background:
        Mean(background), Mean Background, background mean, etc.

    Area columns are retained as metadata, not used as fluorescence signals.
    """
    columns = list(df.columns)

    normalized = {
        column: clean_name(column)
        for column in columns
    }

    label_candidates = [
        column
        for column in columns
        if normalized[column] in {
            "label",
            "condition",
            "sample",
            "experiment",
            "group",
            "treatment",
        }
    ]

    background_candidates = [
        column
        for column in columns
        if (
            "background" in normalized[column]
            and "mean" in normalized[column]
        )
    ]

    if not background_candidates:
        background_candidates = [
            column
            for column in columns
            if normalized[column] in {
                "background",
                "bg",
                "meanbg",
            }
        ]

    background_column = (
        background_candidates[0]
        if background_candidates
        else None
    )

    signal_columns = [
        column
        for column in columns
        if (
            column != background_column
            and re.fullmatch(
                r"mean\d+",
                normalized[column],
            )
        )
    ]

    area_columns = [
        column
        for column in columns
        if normalized[column].startswith("area")
    ]

    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    excluded_columns = set(
        signal_columns
        + area_columns
        + (
            [background_column]
            if background_column is not None
            else []
        )
        + (
            [label_candidates[0]]
            if label_candidates
            else []
        )
    )

    time_candidates = [
        column
        for column in numeric_columns
        if column not in excluded_columns
    ]

    return {
        "label": (
            label_candidates[0]
            if label_candidates
            else None
        ),
        "background": background_column,
        "signals": signal_columns,
        "areas": area_columns,
        "time": (
            time_candidates[0]
            if time_candidates
            else None
        ),
    }


def first_valid(series):
    """
    Return the first non-missing value.
    """
    values = series.dropna()

    if values.empty:
        return np.nan

    return values.iloc[0]


# ============================================================
# Time, background, and ΔF/F0 calculations
# ============================================================

def make_time(df, time_column, interval):
    """
    Convert image/frame number to elapsed time.

    If a time/frame column is selected:
        elapsed time = (frame - first frame) × interval

    Otherwise:
        row index is used as the frame number.
    """
    if (
        time_column is not None
        and time_column in df.columns
    ):
        frames = pd.to_numeric(
            df[time_column],
            errors="coerce",
        )
    else:
        frames = pd.Series(
            np.arange(len(df)),
            index=df.index,
            dtype=float,
        )

    if frames.notna().sum() >= 2:
        frames = frames.ffill().bfill()
    else:
        frames = pd.Series(
            np.arange(len(df)),
            index=df.index,
            dtype=float,
        )

    elapsed_time = (
        frames - frames.iloc[0]
    ) * interval

    return frames, elapsed_time


def calculate_f0(signal, time, settings):
    """
    Calculate baseline F0 from a corrected fluorescence trace.
    """
    signal = pd.to_numeric(
        signal,
        errors="coerce",
    )

    valid = signal.dropna()

    if valid.empty:
        return np.nan

    mode = settings["f0_mode"]

    if mode == "First valid value":
        return float(valid.iloc[0])

    if mode == "Mean of first N valid rows":
        return float(
            valid.iloc[
                :max(1, settings["f0_n"])
            ].mean()
        )

    if mode == "Minimum value in trace":
        return float(valid.min())

    if mode == "Lower quartile (25th percentile)":
        return float(valid.quantile(0.25))

    if mode == "Mean within baseline-time window":
        selected_values = signal[
            (time >= settings["baseline_start"])
            & (time <= settings["baseline_end"])
        ].dropna()

        if selected_values.empty:
            return np.nan

        return float(selected_values.mean())

    return float(valid.iloc[0])


def make_dff(raw, background, time, settings):
    """
    Background-correct a fluorescence trace and calculate ΔF/F0.

    Default:
        F corrected = F ROI - F background

        ΔF/F0 = (F corrected - F0) / F0

    Negative corrected values are retained by default.
    """
    raw = pd.to_numeric(
        raw,
        errors="coerce",
    )

    if (
        settings["background_mode"]
        == "Subtract empty-area background before ΔF/F0"
    ):
        corrected = (
            raw
            - pd.to_numeric(
                background,
                errors="coerce",
            )
        )
    else:
        corrected = raw.copy()

    if (
        settings["negative_policy"]
        == "Clip corrected fluorescence below zero to zero"
    ):
        corrected = corrected.clip(lower=0)

    if (
        settings["negative_policy"]
        == "Exclude ROI if any corrected value is below zero"
        and (corrected < 0).any()
    ):
        return (
            corrected,
            pd.Series(
                np.nan,
                index=corrected.index,
            ),
            np.nan,
        )

    f0 = calculate_f0(
        corrected,
        time,
        settings,
    )

    if not np.isfinite(f0) or f0 <= 0:
        return (
            corrected,
            pd.Series(
                np.nan,
                index=corrected.index,
            ),
            f0,
        )

    dff = (
        corrected - f0
    ) / f0

    return corrected, dff, f0


def auc_positive(time, values):
    """
    Calculate positive AUC only.
    """
    x = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    y_positive = np.maximum(y, 0)

    if hasattr(np, "trapezoid"):
        return float(
            np.trapezoid(
                y_positive,
                x,
            )
        )

    return float(
        np.trapz(
            y_positive,
            x,
        )
    )


# ============================================================
# File processing
# ============================================================

def process_file(df, filename, settings):
    """
    Process all automatically detected Mean1...MeanN traces in one file.
    """
    detected = detect_columns(df)

    frames, time = make_time(
        df,
        settings["time_column"],
        settings["frame_interval"],
    )

    label_column = settings["label_column"]

    if (
        label_column is not None
        and label_column in df.columns
    ):
        label = str(
            first_valid(
                df[label_column]
            )
        )
    else:
        label = re.sub(
            r"\.[^.]+$",
            "",
            filename,
        )

    if (
        not label
        or label.lower() == "nan"
    ):
        label = re.sub(
            r"\.[^.]+$",
            "",
            filename,
        )

    processed = pd.DataFrame(
        {
            "Frame": frames,
            "Time": time,
        }
    )

    background_column = settings["background_column"]

    if (
        background_column is not None
        and background_column in df.columns
    ):
        background = pd.to_numeric(
            df[background_column],
            errors="coerce",
        )

        processed["Background"] = background
    else:
        background = pd.Series(
            0.0,
            index=df.index,
        )

    warnings = []
    summary_rows = []

    for signal_column in detected["signals"]:
        raw = pd.to_numeric(
            df[signal_column],
            errors="coerce",
        )

        corrected, dff, f0 = make_dff(
            raw,
            background,
            time,
            settings,
        )

        processed[
            f"Raw | {signal_column}"
        ] = raw

        processed[
            f"Corrected | {signal_column}"
        ] = corrected

        processed[
            f"dF/F0 | {signal_column}"
        ] = dff

        if not np.isfinite(f0) or f0 <= 0:
            status = "Excluded: invalid corrected F0"

            warnings.append(
                f"{filename} — {signal_column} excluded from ΔF/F0 "
                "because corrected F0 is missing, zero, or negative."
            )

            peak = np.nan
            peak_time = np.nan
            auc = np.nan

        else:
            status = "Included"

            valid_dff = dff.dropna()

            if valid_dff.empty:
                peak = np.nan
                peak_time = np.nan
                auc = np.nan
            else:
                peak_index = dff.idxmax()

                peak = float(
                    dff.loc[peak_index]
                )

                peak_time = float(
                    time.loc[peak_index]
                )

                auc = auc_positive(
                    time,
                    dff,
                )

        summary_rows.append(
            {
                "File": filename,
                "Label": label,
                "ROI": signal_column,
                "F0 corrected": f0,
                "Peak ΔF/F0": peak,
                "Peak time": peak_time,
                "AUC above baseline": auc,
                "Status": status,
            }
        )

    dff_columns = [
        column
        for column in processed.columns
        if column.startswith("dF/F0 | ")
    ]

    if dff_columns:
        processed["Average dF/F0"] = (
            processed[dff_columns]
            .mean(axis=1, skipna=True)
        )

        processed["SEM dF/F0"] = (
            processed[dff_columns]
            .sem(axis=1, ddof=1)
        )
    else:
        processed["Average dF/F0"] = np.nan
        processed["SEM dF/F0"] = np.nan

    return {
        "file": filename,
        "label": label,
        "processed": processed,
        "summary": pd.DataFrame(summary_rows),
        "warnings": warnings,
    }


# ============================================================
# Manual four-point event functions
# ============================================================

def event_key(filename, trace):
    """
    Unique state key for one file and one selected trace.
    """
    return f"{filename}::{trace}"


def init_event_state(key):
    """
    Create persistent session-state containers for this trace.
    """
    if key not in st.session_state.manual_events:
        st.session_state.manual_events[key] = []

    if key not in st.session_state.current_points:
        st.session_state.current_points[key] = {
            "p1": None,
            "p2": None,
            "p3": None,
            "p4": None,
        }


def point_from_frame(df, trace, frame):
    """
    Return frame/time/value metadata for one manually selected graph point.
    """
    if frame is None:
        return None

    frame_values = pd.to_numeric(
        df["Frame"],
        errors="coerce",
    )

    matched = df[
        np.isclose(
            frame_values,
            float(frame),
            equal_nan=False,
        )
    ]

    if matched.empty:
        return None

    row = matched.iloc[0]

    value = row[trace]

    if not np.isfinite(value):
        return None

    return {
        "frame": int(row["Frame"]),
        "time": float(row["Time"]),
        "value": float(value),
    }


def add_segment(
    fig,
    p_left,
    p_right,
    color,
    dash,
    text_left,
    text_right,
):
    """
    Draw a selected rate segment between two points.
    """
    fig.add_trace(
        go.Scatter(
            x=[
                p_left["time"],
                p_right["time"],
            ],
            y=[
                p_left["value"],
                p_right["value"],
            ],
            mode="lines+markers+text",
            line=dict(
                color=color,
                width=3,
                dash=dash,
            ),
            marker=dict(
                color=color,
                size=11,
            ),
            text=[
                text_left,
                text_right,
            ],
            textposition="top center",
            showlegend=False,
        )
    )


def manual_figure(
    df,
    trace,
    result,
    current,
    saved,
    positive_only,
    unit,
):
    """
    Create an interactive trace graph showing saved and current P1-P4 points.
    """
    y = df[trace].to_numpy(dtype=float)

    if positive_only:
        y_plot = np.maximum(y, 0)
    else:
        y_plot = y

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=y_plot,
            mode="lines+markers",
            marker=dict(size=6),
            line=dict(
                color="#1f77b4",
                width=2.5,
            ),
            name=(
                f"{result['label']} | "
                f"{result['file']} | "
                f"{trace}"
            ),
            customdata=df["Frame"].to_numpy(),
            hovertemplate=(
                "Frame: %{customdata}"
                "<br>Time: %{x:.4f}"
                "<br>ΔF/F0: %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    # Show all saved events for this selected trace.
    for event in saved:
        add_segment(
            fig,
            event["p1"],
            event["p2"],
            "green",
            "solid",
            f"{event['id']} P1",
            f"{event['id']} P2",
        )

        add_segment(
            fig,
            event["p2"],
            event["p3"],
            "orange",
            "dot",
            f"{event['id']} P2",
            f"{event['id']} P3",
        )

        add_segment(
            fig,
            event["p3"],
            event["p4"],
            "purple",
            "dash",
            f"{event['id']} P3",
            f"{event['id']} P4",
        )

    point_labels = {
        "p1": (
            "green",
            "P1: pre-rise low",
        ),
        "p2": (
            "orange",
            "P2: end fast rise",
        ),
        "p3": (
            "red",
            "P3: main peak",
        ),
        "p4": (
            "purple",
            "P4: post-peak end",
        ),
    }

    # Show unfinished/current event points.
    for point_name, (
        color,
        label,
    ) in point_labels.items():
        point = current.get(point_name)

        if point is None:
            continue

        fig.add_trace(
            go.Scatter(
                x=[point["time"]],
                y=[point["value"]],
                mode="markers+text",
                marker=dict(
                    color=color,
                    size=15,
                ),
                text=[label],
                textposition="bottom center",
                showlegend=False,
            )
        )

    p1 = current.get("p1")
    p2 = current.get("p2")
    p3 = current.get("p3")
    p4 = current.get("p4")

    if p1 is not None and p2 is not None:
        add_segment(
            fig,
            p1,
            p2,
            "green",
            "solid",
            "P1",
            "P2",
        )

    if p2 is not None and p3 is not None:
        add_segment(
            fig,
            p2,
            p3,
            "orange",
            "dot",
            "P2",
            "P3",
        )

    if p3 is not None and p4 is not None:
        add_segment(
            fig,
            p3,
            p4,
            "purple",
            "dash",
            "P3",
            "P4",
        )

    fig.update_layout(
        height=620,
        template="plotly_white",
        dragmode="select",
        xaxis_title=f"Time ({unit})",
        yaxis_title="ΔF/F0",
    )

    return fig


def event_metrics(events):
    """
    Calculate P1-P4 metrics for a list of saved manual events.
    """
    rows = []

    for event in events:
        p1 = event["p1"]
        p2 = event["p2"]
        p3 = event["p3"]
        p4 = event["p4"]

        fast_duration = (
            p2["time"] - p1["time"]
        )

        total_duration = (
            p3["time"] - p1["time"]
        )

        decay_duration = (
            p4["time"] - p3["time"]
        )

        fast_amplitude = (
            p2["value"] - p1["value"]
        )

        total_amplitude = (
            p3["value"] - p1["value"]
        )

        late_amplitude = (
            p3["value"] - p2["value"]
        )

        decay_amplitude = (
            p4["value"] - p3["value"]
        )

        fast_rate = (
            fast_amplitude / fast_duration
            if fast_duration > 0
            else np.nan
        )

        total_rate = (
            total_amplitude / total_duration
            if total_duration > 0
            else np.nan
        )

        decay_rate = (
            decay_amplitude / decay_duration
            if decay_duration > 0
            else np.nan
        )

        rows.append(
            {
                "Event ID": event["id"],
                "P1 frame": p1["frame"],
                "P1 time": p1["time"],
                "P1 ΔF/F0": p1["value"],
                "P2 frame": p2["frame"],
                "P2 time": p2["time"],
                "P2 ΔF/F0": p2["value"],
                "P3 frame": p3["frame"],
                "P3 time": p3["time"],
                "P3 ΔF/F0": p3["value"],
                "P4 frame": p4["frame"],
                "P4 time": p4["time"],
                "P4 ΔF/F0": p4["value"],
                "Fast upstroke duration P1→P2": fast_duration,
                "Fast upstroke amplitude P1→P2": fast_amplitude,
                "Fast upstroke rate P1→P2": fast_rate,
                "Time to peak P1→P3": total_duration,
                "Peak amplitude above P1": total_amplitude,
                "Overall rise rate P1→P3": total_rate,
                "Late-rise amplitude P2→P3": late_amplitude,
                "Observed decay duration P3→P4": decay_duration,
                "Observed decay amplitude P3→P4": decay_amplitude,
                "Observed decay/release rate P3→P4": decay_rate,
                "Release magnitude": (
                    -decay_rate
                    if np.isfinite(decay_rate)
                    else np.nan
                ),
                "P4 endpoint type": event["endpoint_type"],
                "Notes": event["notes"],
            }
        )

    return pd.DataFrame(rows)


def build_cumulative_event_table(
    manual_events,
    results,
):
    """
    Build one table of every saved manual event from every file and trace.

    This table remains populated when the user changes the selected graph.
    """
    event_tables = []

    file_to_label = {
        result["file"]: result["label"]
        for result in results
    }

    for stored_key, saved_events in manual_events.items():
        if not saved_events:
            continue

        metrics = event_metrics(saved_events)

        if metrics.empty:
            continue

        if "::" in stored_key:
            stored_file, stored_trace = stored_key.split(
                "::",
                1,
            )
        else:
            stored_file = stored_key
            stored_trace = ""

        stored_label = file_to_label.get(
            stored_file,
            "",
        )

        metrics.insert(
            0,
            "File",
            stored_file,
        )

        metrics.insert(
            1,
            "Label",
            stored_label,
        )

        metrics.insert(
            2,
            "Trace",
            stored_trace,
        )

        event_tables.append(metrics)

    if not event_tables:
        return pd.DataFrame()

    return pd.concat(
        event_tables,
        ignore_index=True,
    )


# ============================================================
# Overview graph and export functions
# ============================================================

def overview_figure(
    results,
    positive_only,
    unit,
):
    """
    Plot every uploaded file as a separate average trace.
    """
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#e377c2",
    ]

    fig = go.Figure()

    for index, result in enumerate(results):
        df = result["processed"]

        y = df[
            "Average dF/F0"
        ].to_numpy()

        if positive_only:
            y = np.maximum(y, 0)

        fig.add_trace(
            go.Scatter(
                x=df["Time"],
                y=y,
                mode="lines",
                line=dict(
                    color=palette[
                        index % len(palette)
                    ],
                    width=2.5,
                ),
                name=(
                    f"{result['label']} | "
                    f"{result['file']}"
                ),
            )
        )

    fig.update_layout(
        height=560,
        template="plotly_white",
        xaxis_title=f"Time ({unit})",
        yaxis_title="Average ΔF/F0",
        legend_title="Uploaded file",
    )

    return fig


def safe_sheet_name(name, used):
    """
    Create a unique valid Excel sheet name.
    """
    base = re.sub(
        r"[\\/*?:\[\]]",
        "_",
        str(name),
    )[:31] or "sheet"

    candidate = base
    counter = 1

    while candidate in used:
        suffix = f"_{counter}"

        candidate = (
            base[:31 - len(suffix)]
            + suffix
        )

        counter += 1

    used.add(candidate)

    return candidate


def make_workbook(
    results,
    settings,
    cell_summary,
    manual_events_df,
):
    """
    Create Excel workbook containing all processed traces,
    per-ROI summary, settings, and all saved manual events.
    """
    bio = io.BytesIO()
    used = set()

    with pd.ExcelWriter(
        bio,
        engine="openpyxl",
    ) as writer:
        pd.DataFrame(
            [settings]
        ).to_excel(
            writer,
            index=False,
            sheet_name="analysis_settings",
        )

        pd.DataFrame(
            [
                {
                    "File": result["file"],
                    "Label": result["label"],
                }
                for result in results
            ]
        ).to_excel(
            writer,
            index=False,
            sheet_name="file_metadata",
        )

        if not cell_summary.empty:
            cell_summary.to_excel(
                writer,
                index=False,
                sheet_name="cell_summary",
            )

        if not manual_events_df.empty:
            manual_events_df.to_excel(
                writer,
                index=False,
                sheet_name="manual_events",
            )

        for result in results:
            sheet_name = safe_sheet_name(
                f"trace_{result['file']}",
                used,
            )

            result["processed"].to_excel(
                writer,
                index=False,
                sheet_name=sheet_name,
            )

    bio.seek(0)

    return bio.getvalue()


# ============================================================
# Sidebar settings
# ============================================================

with st.sidebar:
    st.header("Time")

    frame_interval = st.number_input(
        "Frame interval",
        min_value=0.000001,
        value=1.0,
        step=0.1,
    )

    time_unit = st.selectbox(
        "Time unit",
        [
            "seconds",
            "minutes",
            "milliseconds",
        ],
        index=0,
    )

    st.header("Background correction")

    background_mode = st.radio(
        "Background method",
        [
            "Subtract empty-area background before ΔF/F0",
            "No background correction",
        ],
        index=0,
    )

    negative_policy = st.radio(
        "Negative corrected-signal handling",
        [
            "Keep negative values (recommended)",
            "Clip corrected fluorescence below zero to zero",
            "Exclude ROI if any corrected value is below zero",
        ],
        index=0,
    )

    st.header("F0")

    f0_mode = st.selectbox(
        "F0 definition",
        [
            "First valid value",
            "Mean of first N valid rows",
            "Minimum value in trace",
            "Lower quartile (25th percentile)",
            "Mean within baseline-time window",
        ],
        index=1,
    )

    f0_n = 5
    baseline_start = None
    baseline_end = None

    if f0_mode == "Mean of first N valid rows":
        f0_n = int(
            st.number_input(
                "Number of baseline rows",
                min_value=1,
                value=5,
                step=1,
            )
        )

    if f0_mode == "Mean within baseline-time window":
        baseline_start = st.number_input(
            f"Baseline start ({time_unit})",
            value=0.0,
        )

        baseline_end = st.number_input(
            f"Baseline end ({time_unit})",
            value=5.0,
        )

    st.header("Display")

    positive_only = st.checkbox(
        "Plot only positive ΔF/F0",
        value=False,
    )

    st.button(
        "Reset analysis / upload new files",
        on_click=reset_analysis,
        use_container_width=True,
    )


# ============================================================
# Upload and read input
# ============================================================

uploads = st.file_uploader(
    "Upload calcium time-series files",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
    accept_multiple_files=True,
    key=f"files_{st.session_state.uploader_token}",
    help=(
        "Each file may contain a different number of ROI columns named "
        "Mean1, Mean2, ..., MeanN."
    ),
)

if not uploads:
    st.info(
        "Upload one or more calcium CSV/XLSX files to begin."
    )
    st.stop()

loaded = []

for upload in uploads:
    try:
        loaded.append(
            {
                "file": upload.name,
                "df": read_table(upload),
            }
        )
    except Exception as exc:
        st.error(
            f"Could not read {upload.name}: {exc}"
        )

if not loaded:
    st.stop()


# ============================================================
# Show automatic column detection
# ============================================================

st.subheader("Detected input structure")

detected_rows = []

for item in loaded:
    detected = detect_columns(item["df"])

    detected_rows.append(
        {
            "File": item["file"],
            "Suggested label": detected["label"],
            "Suggested time": detected["time"],
            "Suggested background": detected["background"],
            "Detected ROI columns": ", ".join(
                detected["signals"]
            ),
            "ROI count": len(
                detected["signals"]
            ),
        }
    )

st.dataframe(
    pd.DataFrame(detected_rows),
    use_container_width=True,
)


# ============================================================
# Shared mapping
# ============================================================

first_df = loaded[0]["df"]
first_detected = detect_columns(first_df)
columns = list(first_df.columns)

with st.expander(
    "Shared column mapping",
    expanded=True,
):
    mapping_col1, mapping_col2, mapping_col3, mapping_col4 = st.columns(4)

    with mapping_col1:
        label_options = [
            "<auto/file name>",
        ] + columns

        label_index = (
            columns.index(
                first_detected["label"]
            )
            + 1
            if first_detected["label"] in columns
            else 0
        )

        selected_label = st.selectbox(
            "Label column",
            label_options,
            index=label_index,
        )

    with mapping_col2:
        time_options = [
            "<auto row index>",
        ] + columns

        time_index = (
            columns.index(
                first_detected["time"]
            )
            + 1
            if first_detected["time"] in columns
            else 0
        )

        selected_time = st.selectbox(
            "Frame/time column",
            time_options,
            index=time_index,
        )

    with mapping_col3:
        background_options = [
            "<none>",
        ] + columns

        background_index = (
            background_options.index(
                first_detected["background"]
            )
            if first_detected["background"] in background_options
            else 0
        )

        selected_background = st.selectbox(
            "Empty-area background column",
            background_options,
            index=background_index,
        )

    with mapping_col4:
        st.info(
            "Every file independently detects all Mean1, Mean2, ..., MeanN "
            "ROI columns. No automatic event selection is used."
        )


if (
    background_mode
    == "Subtract empty-area background before ΔF/F0"
    and selected_background == "<none>"
):
    st.error(
        "Select Mean(background) or choose "
        "'No background correction'."
    )
    st.stop()


settings = {
    "label_column": (
        None
        if selected_label == "<auto/file name>"
        else selected_label
    ),
    "time_column": (
        None
        if selected_time == "<auto row index>"
        else selected_time
    ),
    "background_column": (
        None
        if selected_background == "<none>"
        else selected_background
    ),
    "frame_interval": frame_interval,
    "time_unit": time_unit,
    "background_mode": background_mode,
    "negative_policy": negative_policy,
    "f0_mode": f0_mode,
    "f0_n": f0_n,
    "baseline_start": baseline_start,
    "baseline_end": baseline_end,
    "positive_only": positive_only,
}


# ============================================================
# Validate every file
# ============================================================

errors = []
warnings = []

for item in loaded:
    df = item["df"]
    filename = item["file"]

    detected = detect_columns(df)

    if not detected["signals"]:
        errors.append(
            f"{filename}: no ROI columns named "
            "Mean1, Mean2, etc. were found."
        )

    if (
        settings["background_column"] is not None
        and settings["background_column"] not in df.columns
    ):
        errors.append(
            f"{filename}: missing selected background column "
            f"'{settings['background_column']}'."
        )

    if (
        settings["time_column"] is not None
        and settings["time_column"] not in df.columns
    ):
        errors.append(
            f"{filename}: missing selected frame/time column "
            f"'{settings['time_column']}'."
        )

    if (
        settings["label_column"] is not None
        and settings["label_column"] not in df.columns
    ):
        warnings.append(
            f"{filename}: label column is absent; filename will be used."
        )

for warning in warnings:
    st.warning(warning)

if errors:
    st.error(
        "Some uploaded files cannot be analyzed:\n\n"
        + "\n".join(errors)
    )
    st.stop()


# ============================================================
# Process all files
# ============================================================

results = [
    process_file(
        item["df"],
        item["file"],
        settings,
    )
    for item in loaded
]

for result in results:
    for warning in result["warnings"]:
        st.warning(warning)


# ============================================================
# Overview plots and cell summary
# ============================================================

st.subheader("All uploaded file-average traces")

overview = overview_figure(
    results,
    positive_only,
    time_unit,
)

st.plotly_chart(
    overview,
    use_container_width=True,
)

summary_tables = [
    result["summary"]
    for result in results
    if not result["summary"].empty
]

cell_summary = (
    pd.concat(
        summary_tables,
        ignore_index=True,
    )
    if summary_tables
    else pd.DataFrame()
)

st.subheader("Cell/ROI summary")

st.dataframe(
    cell_summary,
    use_container_width=True,
)


# ============================================================
# Manual event curation
# ============================================================

st.subheader("Manual four-point event selection")

st.markdown(
    "**P1:** pre-rise low point. "
    "**P2:** end of the visually straight fast-upstroke. "
    "**P3:** main peak. "
    "**P4:** post-peak endpoint or trough.\n\n"
    "Fast upstroke = P1→P2; overall rise = P1→P3; "
    "observed decay/release = P3→P4. "
    "No point is selected automatically."
)

selected_file = st.selectbox(
    "File for manual curation",
    [
        result["file"]
        for result in results
    ],
)

selected_result = next(
    result
    for result in results
    if result["file"] == selected_file
)

processed = selected_result["processed"]

trace_options = [
    "Average dF/F0",
] + [
    column
    for column in processed.columns
    if column.startswith("dF/F0 | ")
]

selected_trace = st.selectbox(
    "Trace",
    trace_options,
)

key = event_key(
    selected_file,
    selected_trace,
)

init_event_state(key)

current = st.session_state.current_points[key]
saved = st.session_state.manual_events[key]

next_point = st.radio(
    "Select next point",
    [
        "P1: pre-rise low point",
        "P2: end of linear fast upstroke",
        "P3: main peak",
        "P4: post-peak endpoint / low point",
    ],
    horizontal=True,
)

point_map = {
    "P1: pre-rise low point": "p1",
    "P2: end of linear fast upstroke": "p2",
    "P3: main peak": "p3",
    "P4: post-peak endpoint / low point": "p4",
}

st.info(
    "Use Plotly's **Select Points** tool in the graph toolbar, "
    "then click a plotted frame marker. The app records the exact "
    "acquired frame."
)

figure = manual_figure(
    processed,
    selected_trace,
    selected_result,
    current,
    saved,
    positive_only,
    time_unit,
)

selection = st.plotly_chart(
    figure,
    use_container_width=True,
    key=f"manual_{key}",
    on_select="rerun",
    selection_mode="points",
    config={
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "lasso2d",
        ],
    },
)

if (
    selection
    and selection.selection
    and selection.selection.get("points")
):
    selected = selection.selection["points"][0]

    chosen = point_from_frame(
        processed,
        selected_trace,
        selected.get("customdata"),
    )

    if chosen is not None:
        st.session_state.current_points[key][
            point_map[next_point]
        ] = chosen

        st.rerun()


# ------------------------------------------------------------
# Manual selection controls
# ------------------------------------------------------------

control_col1, control_col2, control_col3, control_col4 = st.columns(4)

with control_col1:
    if st.button(
        "Undo last point",
        key=f"undo_{key}",
        use_container_width=True,
    ):
        points = st.session_state.current_points[key]

        for point_name in [
            "p4",
            "p3",
            "p2",
            "p1",
        ]:
            if points[point_name] is not None:
                points[point_name] = None
                break

        st.rerun()

with control_col2:
    if st.button(
        "Reset current event",
        key=f"reset_current_{key}",
        use_container_width=True,
    ):
        st.session_state.current_points[key] = {
            "p1": None,
            "p2": None,
            "p3": None,
            "p4": None,
        }

        st.rerun()

with control_col3:
    if st.button(
        "Delete last saved event",
        key=f"delete_{key}",
        use_container_width=True,
    ):
        if st.session_state.manual_events[key]:
            st.session_state.manual_events[key].pop()

        st.rerun()

with control_col4:
    if st.button(
        "Reset all events for this trace",
        key=f"reset_all_{key}",
        use_container_width=True,
    ):
        st.session_state.manual_events[key] = []

        st.session_state.current_points[key] = {
            "p1": None,
            "p2": None,
            "p3": None,
            "p4": None,
        }

        st.rerun()


endpoint_type = st.selectbox(
    "P4 endpoint type",
    [
        "Post-peak trough",
        "Returned to baseline",
        "Before next rise",
        "Recording end",
        "Sustained plateau",
        "Manual endpoint",
    ],
    key=f"endpoint_{key}",
)

notes = st.text_input(
    "Notes",
    key=f"notes_{key}",
)

complete = all(
    current[point_name] is not None
    for point_name in [
        "p1",
        "p2",
        "p3",
        "p4",
    ]
)

if not complete:
    st.caption(
        "Current event is incomplete. "
        "Select P1, P2, P3, and P4."
    )

if st.button(
    "Save current four-point event",
    disabled=not complete,
    key=f"save_{key}",
    use_container_width=True,
):
    p1 = current["p1"]
    p2 = current["p2"]
    p3 = current["p3"]
    p4 = current["p4"]

    if not (
        p1["time"]
        < p2["time"]
        < p3["time"]
        < p4["time"]
    ):
        st.error(
            "Points must be selected in order: "
            "P1 before P2 before P3 before P4."
        )
    else:
        event_number = (
            len(
                st.session_state.manual_events[key]
            )
            + 1
        )

        st.session_state.manual_events[key].append(
            {
                "id": f"E{event_number}",
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p4": p4,
                "endpoint_type": endpoint_type,
                "notes": notes,
            }
        )

        # Clear only temporary unfinished points. Saved events remain.
        st.session_state.current_points[key] = {
            "p1": None,
            "p2": None,
            "p3": None,
            "p4": None,
        }

        st.rerun()


# ============================================================
# Current trace event table
# ============================================================

selected_metrics = event_metrics(
    st.session_state.manual_events[key]
)

st.subheader("Saved events for the selected trace")

if selected_metrics.empty:
    st.info(
        "No completed four-point events have been saved for the "
        "currently selected trace."
    )
else:
    selected_metrics_display = selected_metrics.copy()

    selected_metrics_display.insert(
        0,
        "File",
        selected_file,
    )

    selected_metrics_display.insert(
        1,
        "Label",
        selected_result["label"],
    )

    selected_metrics_display.insert(
        2,
        "Trace",
        selected_trace,
    )

    st.dataframe(
        selected_metrics_display,
        use_container_width=True,
    )


# ============================================================
# Cumulative event table: all files and all traces
# ============================================================

manual_events_df = build_cumulative_event_table(
    st.session_state.manual_events,
    results,
)

st.subheader(
    "Cumulative saved manual events: all files and traces"
)

if manual_events_df.empty:
    st.info(
        "No completed manual events have been saved yet. "
        "After choosing P1, P2, P3, and P4, click "
        "'Save current four-point event'."
    )
else:
    st.dataframe(
        manual_events_df,
        use_container_width=True,
        height=420,
    )


# ============================================================
# Downloads
# ============================================================

st.subheader("Downloads")

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

workbook = make_workbook(
    results,
    settings,
    cell_summary,
    manual_events_df,
)

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        "Download Excel workbook",
        workbook,
        f"calcium_analysis_{timestamp}.xlsx",
        (
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )

with download_col2:
    st.download_button(
        "Download cell summary CSV",
        cell_summary.to_csv(
            index=False
        ).encode("utf-8"),
        f"calcium_cell_summary_{timestamp}.csv",
        "text/csv",
        use_container_width=True,
    )

with download_col3:
    st.download_button(
        "Download all manual events CSV",
        manual_events_df.to_csv(
            index=False
        ).encode("utf-8"),
        f"calcium_manual_events_{timestamp}.csv",
        "text/csv",
        use_container_width=True,
    )

st.download_button(
    "Download all-file graph HTML",
    overview.to_html(
        include_plotlyjs="cdn"
    ),
    f"calcium_all_files_{timestamp}.html",
    "text/html",
)
