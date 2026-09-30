import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# App setup
# ============================================================

st.set_page_config(
    page_title="Calcium Analysis",
    layout="wide",
)

st.title("Calcium Imaging Post-Analysis")
st.caption(
    "Manual P1–P4 calcium-event curation with multi-file ROI detection, "
    "background correction, ΔF/F0 normalization, and multiple manually "
    "defined stimulation events per trace."
)

if "uploader_token" not in st.session_state:
    st.session_state.uploader_token = 0

if "trace_events" not in st.session_state:
    st.session_state.trace_events = {}

if "current_points" not in st.session_state:
    st.session_state.current_points = {}


def reset_analysis():
    """
    Reset file upload, current selections, and saved manual events.
    """
    next_token = st.session_state.get(
        "uploader_token",
        0,
    ) + 1

    st.session_state.clear()

    st.session_state.uploader_token = next_token
    st.session_state.trace_events = {}
    st.session_state.current_points = {}


# ============================================================
# File reading and column detection
# ============================================================

def read_table(uploaded_file):
    """
    Read CSV, XLSX, or XLS uploaded file.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file)


def clean_name(value):
    """
    Normalize column names for automatic matching.
    """
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def detect_columns(df):
    """
    Identify likely Label, time, background, Mean ROI, and Area columns.

    Cell ROI signals must match:
        Mean1, Mean2, Mean3, ..., MeanN

    Background can match:
        Mean(background), Mean Background, Background Mean, etc.
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
    Return the first nonmissing value in a series.
    """
    values = series.dropna()

    if values.empty:
        return np.nan

    return values.iloc[0]


# ============================================================
# Time and fluorescence calculations
# ============================================================

def make_time(df, time_column, frame_interval):
    """
    Make frame and elapsed-time vectors.
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
    ) * frame_interval

    return frames, elapsed_time


def calculate_f0(signal, time, settings):
    """
    Calculate F0 from the corrected fluorescence trace.
    """
    signal = pd.to_numeric(
        signal,
        errors="coerce",
    )

    valid_values = signal.dropna()

    if valid_values.empty:
        return np.nan

    mode = settings["f0_mode"]

    if mode == "First valid value":
        return float(valid_values.iloc[0])

    if mode == "Mean of first N valid rows":
        return float(
            valid_values.iloc[
                :max(1, settings["f0_n"])
            ].mean()
        )

    if mode == "Minimum value in trace":
        return float(valid_values.min())

    if mode == "Lower quartile (25th percentile)":
        return float(
            valid_values.quantile(0.25)
        )

    if mode == "Mean within baseline-time window":
        values = signal[
            (
                time >= settings["baseline_start"]
            )
            & (
                time <= settings["baseline_end"]
            )
        ].dropna()

        if values.empty:
            return np.nan

        return float(values.mean())

    return float(valid_values.iloc[0])


def make_dff(raw, background, time, settings):
    """
    Calculate corrected fluorescence and ΔF/F0.

    Recommended default:
        F corrected = F ROI - F background
        ΔF/F0 = (F corrected - F0) / F0
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
    Calculate AUC above zero.
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
# Process each uploaded file
# ============================================================

def process_file(df, filename, settings):
    """
    Process every Mean1...MeanN ROI signal in a single file.
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

    if not label or label.lower() == "nan":
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
# Manual event state and point selection
# ============================================================

def trace_key(filename, trace):
    """
    Unique session key for a selected file and trace.
    """
    return f"{filename}::{trace}"


def empty_points():
    """
    Empty P1-P4 selection.
    """
    return {
        "p1": None,
        "p2": None,
        "p3": None,
        "p4": None,
    }


def initialize_trace_events(
    key,
    number_of_events,
):
    """
    Create exactly the requested number of manual event slots.

    Existing event point selections are retained where possible
    when the number of events increases.
    """
    if key not in st.session_state.trace_events:
        st.session_state.trace_events[key] = {}

    stored_events = st.session_state.trace_events[key]

    # Add missing event slots.
    for event_number in range(
        1,
        number_of_events + 1,
    ):
        event_id = f"E{event_number}"

        if event_id not in stored_events:
            stored_events[event_id] = {
                "p1": None,
                "p2": None,
                "p3": None,
                "p4": None,
                "endpoint_type": "Post-peak trough",
                "notes": "",
            }

    # Remove event slots beyond the currently selected count.
    valid_event_ids = {
        f"E{event_number}"
        for event_number in range(
            1,
            number_of_events + 1,
        )
    }

    event_ids_to_remove = [
        event_id
        for event_id in stored_events
        if event_id not in valid_event_ids
    ]

    for event_id in event_ids_to_remove:
        del stored_events[event_id]


def point_from_frame(
    df,
    trace,
    frame,
):
    """
    Convert one graph-selected frame into frame/time/ΔF/F0 metadata.
    """
    if frame is None:
        return None

    numeric_frames = pd.to_numeric(
        df["Frame"],
        errors="coerce",
    )

    matching_rows = df[
        np.isclose(
            numeric_frames,
            float(frame),
            equal_nan=False,
        )
    ]

    if matching_rows.empty:
        return None

    row = matching_rows.iloc[0]

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
    left_point,
    right_point,
    color,
    dash,
    left_text,
    right_text,
):
    """
    Add one P1-P2, P2-P3, or P3-P4 segment to a Plotly graph.
    """
    if (
        left_point is None
        or right_point is None
    ):
        return

    fig.add_trace(
        go.Scatter(
            x=[
                left_point["time"],
                right_point["time"],
            ],
            y=[
                left_point["value"],
                right_point["value"],
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
                left_text,
                right_text,
            ],
            textposition="top center",
            showlegend=False,
        )
    )


def build_manual_figure(
    df,
    trace,
    result,
    all_events,
    active_event_id,
    positive_only,
    time_unit,
):
    """
    Show full trace, all saved/partially selected events,
    and highlight the currently active event.
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

    point_colors = {
        "p1": "green",
        "p2": "orange",
        "p3": "red",
        "p4": "purple",
    }

    point_labels = {
        "p1": "P1",
        "p2": "P2",
        "p3": "P3",
        "p4": "P4",
    }

    for event_id, event in all_events.items():
        p1 = event["p1"]
        p2 = event["p2"]
        p3 = event["p3"]
        p4 = event["p4"]

        opacity = 1.0 if event_id == active_event_id else 0.45

        if p1 is not None and p2 is not None:
            fig.add_trace(
                go.Scatter(
                    x=[
                        p1["time"],
                        p2["time"],
                    ],
                    y=[
                        p1["value"],
                        p2["value"],
                    ],
                    mode="lines+markers+text",
                    line=dict(
                        color=f"rgba(0,128,0,{opacity})",
                        width=3,
                    ),
                    marker=dict(
                        color="green",
                        size=10,
                    ),
                    text=[
                        f"{event_id} P1",
                        f"{event_id} P2",
                    ],
                    textposition="top center",
                    showlegend=False,
                )
            )

        if p2 is not None and p3 is not None:
            fig.add_trace(
                go.Scatter(
                    x=[
                        p2["time"],
                        p3["time"],
                    ],
                    y=[
                        p2["value"],
                        p3["value"],
                    ],
                    mode="lines+markers+text",
                    line=dict(
                        color=f"rgba(255,165,0,{opacity})",
                        width=2,
                        dash="dot",
                    ),
                    marker=dict(
                        color="orange",
                        size=10,
                    ),
                    text=[
                        f"{event_id} P2",
                        f"{event_id} P3",
                    ],
                    textposition="bottom center",
                    showlegend=False,
                )
            )

        if p3 is not None and p4 is not None:
            fig.add_trace(
                go.Scatter(
                    x=[
                        p3["time"],
                        p4["time"],
                    ],
                    y=[
                        p3["value"],
                        p4["value"],
                    ],
                    mode="lines+markers+text",
                    line=dict(
                        color=f"rgba(128,0,128,{opacity})",
                        width=3,
                        dash="dash",
                    ),
                    marker=dict(
                        color="purple",
                        size=10,
                    ),
                    text=[
                        f"{event_id} P3",
                        f"{event_id} P4",
                    ],
                    textposition="top center",
                    showlegend=False,
                )
            )

        # Show isolated points before segment is complete.
        for point_name in [
            "p1",
            "p2",
            "p3",
            "p4",
        ]:
            point = event[point_name]

            if point is None:
                continue

            fig.add_trace(
                go.Scatter(
                    x=[point["time"]],
                    y=[point["value"]],
                    mode="markers+text",
                    marker=dict(
                        color=point_colors[point_name],
                        size=15 if event_id == active_event_id else 10,
                    ),
                    text=[
                        f"{event_id} {point_labels[point_name]}"
                    ],
                    textposition="bottom center",
                    showlegend=False,
                )
            )

    fig.update_layout(
        height=650,
        template="plotly_white",
        dragmode="select",
        xaxis_title=f"Time ({time_unit})",
        yaxis_title="ΔF/F0",
    )

    return fig


# ============================================================
# Manual-event calculations
# ============================================================

def calculate_event_metrics(
    event_id,
    event,
):
    """
    Calculate kinetics only when all P1-P4 points are complete
    and appear in chronological order.
    """
    p1 = event["p1"]
    p2 = event["p2"]
    p3 = event["p3"]
    p4 = event["p4"]

    if any(
        point is None
        for point in [
            p1,
            p2,
            p3,
            p4,
        ]
    ):
        return None

    if not (
        p1["time"]
        < p2["time"]
        < p3["time"]
        < p4["time"]
    ):
        return {
            "Event ID": event_id,
            "Status": (
                "Invalid point order: require "
                "P1 < P2 < P3 < P4"
            ),
        }

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

    overall_rate = (
        total_amplitude / total_duration
        if total_duration > 0
        else np.nan
    )

    decay_rate = (
        decay_amplitude / decay_duration
        if decay_duration > 0
        else np.nan
    )

    return {
        "Event ID": event_id,
        "Status": "Complete",
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
        "Overall rise rate P1→P3": overall_rate,
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


def build_trace_event_table(
    event_dict,
):
    """
    Build metrics table for all manually defined events on one trace.
    """
    rows = []

    for event_id, event in event_dict.items():
        row = calculate_event_metrics(
            event_id,
            event,
        )

        if row is not None:
            rows.append(row)
        else:
            rows.append(
                {
                    "Event ID": event_id,
                    "Status": "Incomplete",
                }
            )

    return pd.DataFrame(rows)


def build_cumulative_event_table(
    trace_events,
    results,
):
    """
    Combine every complete/incomplete event from all files and traces.
    """
    file_to_label = {
        result["file"]: result["label"]
        for result in results
    }

    tables = []

    for stored_key, event_dict in trace_events.items():
        metrics = build_trace_event_table(
            event_dict
        )

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

        metrics.insert(
            0,
            "File",
            stored_file,
        )

        metrics.insert(
            1,
            "Label",
            file_to_label.get(
                stored_file,
                "",
            ),
        )

        metrics.insert(
            2,
            "Trace",
            stored_trace,
        )

        tables.append(metrics)

    if not tables:
        return pd.DataFrame()

    return pd.concat(
        tables,
        ignore_index=True,
    )


# ============================================================
# Plotting and exports
# ============================================================

def overview_figure(
    results,
    positive_only,
    time_unit,
):
    """
    Plot each uploaded file separately.
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
        processed = result["processed"]

        y = processed[
            "Average dF/F0"
        ].to_numpy()

        if positive_only:
            y = np.maximum(y, 0)

        fig.add_trace(
            go.Scatter(
                x=processed["Time"],
                y=y,
                mode="lines",
                name=(
                    f"{result['label']} | "
                    f"{result['file']}"
                ),
                line=dict(
                    color=palette[
                        index % len(palette)
                    ],
                    width=2.5,
                ),
            )
        )

    fig.update_layout(
        height=560,
        template="plotly_white",
        xaxis_title=f"Time ({time_unit})",
        yaxis_title="Average ΔF/F0",
        legend_title="Uploaded file",
    )

    return fig


def safe_sheet_name(
    name,
    used_names,
):
    """
    Create valid unique Excel sheet names.
    """
    base = re.sub(
        r"[\\/*?:\[\]]",
        "_",
        str(name),
    )[:31] or "sheet"

    candidate = base
    counter = 1

    while candidate in used_names:
        suffix = f"_{counter}"

        candidate = (
            base[:31 - len(suffix)]
            + suffix
        )

        counter += 1

    used_names.add(candidate)

    return candidate


def make_workbook(
    results,
    settings,
    cell_summary,
    manual_events_df,
):
    """
    Export all processed traces plus all manually curated events.
    """
    bio = io.BytesIO()
    used_names = set()

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
                used_names,
            )

            result["processed"].to_excel(
                writer,
                index=False,
                sheet_name=sheet_name,
            )

    bio.seek(0)

    return bio.getvalue()


# ============================================================
# Sidebar
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
# Upload files
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
        "Each file may have its own number of cellular ROI columns "
        "named Mean1, Mean2, ..., MeanN."
    ),
)

if not uploads:
    st.info(
        "Upload one or more calcium CSV/XLSX files to begin."
    )
    st.stop()


# ============================================================
# Read files
# ============================================================

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
# Detect and map columns
# ============================================================

st.subheader("Detected input structure")

detection_rows = []

for item in loaded:
    detected = detect_columns(item["df"])

    detection_rows.append(
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
    pd.DataFrame(detection_rows),
    use_container_width=True,
)

first_df = loaded[0]["df"]
first_detected = detect_columns(first_df)
first_columns = list(first_df.columns)

with st.expander(
    "Shared column mapping",
    expanded=True,
):
    column1, column2, column3, column4 = st.columns(4)

    with column1:
        label_options = [
            "<auto/file name>",
        ] + first_columns

        label_index = (
            first_columns.index(
                first_detected["label"]
            )
            + 1
            if first_detected["label"] in first_columns
            else 0
        )

        selected_label = st.selectbox(
            "Label column",
            label_options,
            index=label_index,
        )

    with column2:
        time_options = [
            "<auto row index>",
        ] + first_columns

        time_index = (
            first_columns.index(
                first_detected["time"]
            )
            + 1
            if first_detected["time"] in first_columns
            else 0
        )

        selected_time = st.selectbox(
            "Frame/time column",
            time_options,
            index=time_index,
        )

    with column3:
        background_options = [
            "<none>",
        ] + first_columns

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

    with column4:
        st.info(
            "Cellular signals are detected independently per file. "
            "All Mean1, Mean2, ..., MeanN ROI columns are analyzed. "
            "Events are selected manually."
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
# Validate and process every file
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
            f"{filename}: missing selected time/frame column "
            f"'{settings['time_column']}'."
        )

    if (
        settings["label_column"] is not None
        and settings["label_column"] not in df.columns
    ):
        warnings.append(
            f"{filename}: label column is absent; "
            "filename will be used."
        )

for warning in warnings:
    st.warning(warning)

if errors:
    st.error(
        "Some uploaded files cannot be analyzed:\n\n"
        + "\n".join(errors)
    )
    st.stop()

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
# Overview and summary
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
# Manual event selection
# ============================================================

st.subheader("Manual stimulation-event selection")

st.markdown(
    "For each trace, first choose the number of stimulation-response "
    "events. Then select **P1, P2, P3, and P4** separately for each event.\n\n"
    "- **P1:** pre-rise low point\n"
    "- **P2:** end of visually straight fast-upstroke\n"
    "- **P3:** main peak\n"
    "- **P4:** post-peak endpoint/trough\n\n"
    "Fast upstroke = P1→P2; overall rise = P1→P3; "
    "observed decay/release = P3→P4."
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

key = trace_key(
    selected_file,
    selected_trace,
)

number_of_events = int(
    st.number_input(
        "Number of stimulation-response events in this trace",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        help=(
            "For example, select 3 if this trace has three ATP "
            "stimulations/responses to analyze separately."
        ),
        key=f"event_count_{key}",
    )
)

initialize_trace_events(
    key,
    number_of_events,
)

all_events = st.session_state.trace_events[key]

event_options = [
    f"E{event_number}"
    for event_number in range(
        1,
        number_of_events + 1,
    )
]

active_event_id = st.selectbox(
    "Event/stimulation currently being curated",
    event_options,
    key=f"active_event_{key}",
)

active_event = all_events[active_event_id]

event_label = st.text_input(
    "Stimulation label",
    value=active_event.get(
        "stimulation_label",
        active_event_id,
    ),
    key=f"stimulation_label_{key}_{active_event_id}",
    help=(
        "For example: ATP 1, ATP 2, ATP + CaCl2, "
        "washout, or spontaneous response."
    ),
)

all_events[active_event_id][
    "stimulation_label"
] = event_label

next_point = st.radio(
    f"Select next point for {active_event_id}",
    [
        "P1: pre-rise low point",
        "P2: end of linear fast upstroke",
        "P3: main peak",
        "P4: post-peak endpoint / low point",
    ],
    horizontal=True,
    key=f"next_point_{key}_{active_event_id}",
)

point_map = {
    "P1: pre-rise low point": "p1",
    "P2: end of linear fast upstroke": "p2",
    "P3: main peak": "p3",
    "P4: post-peak endpoint / low point": "p4",
}

st.info(
    "Use Plotly's **Select Points** tool in the graph toolbar, "
    "then click a plotted marker. The app records the exact acquired frame."
)

manual_plot = build_manual_figure(
    processed,
    selected_trace,
    selected_result,
    all_events,
    active_event_id,
    positive_only,
    time_unit,
)

selection = st.plotly_chart(
    manual_plot,
    use_container_width=True,
    key=f"manual_plot_{key}",
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
    selected_point = selection.selection["points"][0]

    selected_frame = selected_point.get(
        "customdata"
    )

    chosen_point = point_from_frame(
        processed,
        selected_trace,
        selected_frame,
    )

    if chosen_point is not None:
        all_events[active_event_id][
            point_map[next_point]
        ] = chosen_point

        st.rerun()


# ------------------------------------------------------------
# Active-event controls
# ------------------------------------------------------------

control1, control2, control3 = st.columns(3)

with control1:
    if st.button(
        f"Undo last point for {active_event_id}",
        key=f"undo_{key}_{active_event_id}",
        use_container_width=True,
    ):
        for point_name in [
            "p4",
            "p3",
            "p2",
            "p1",
        ]:
            if (
                all_events[active_event_id][
                    point_name
                ]
                is not None
            ):
                all_events[active_event_id][
                    point_name
                ] = None
                break

        st.rerun()

with control2:
    if st.button(
        f"Reset points for {active_event_id}",
        key=f"reset_event_{key}_{active_event_id}",
        use_container_width=True,
    ):
        all_events[active_event_id].update(
            empty_points()
        )

        st.rerun()

with control3:
    if st.button(
        "Reset all events for this trace",
        key=f"reset_all_{key}",
        use_container_width=True,
    ):
        st.session_state.trace_events[key] = {}

        initialize_trace_events(
            key,
            number_of_events,
        )

        st.rerun()


# ------------------------------------------------------------
# Event annotations and current status
# ------------------------------------------------------------

endpoint_type = st.selectbox(
    f"P4 endpoint type for {active_event_id}",
    [
        "Post-peak trough",
        "Returned to baseline",
        "Before next rise",
        "Recording end",
        "Sustained plateau",
        "Manual endpoint",
    ],
    index=[
        "Post-peak trough",
        "Returned to baseline",
        "Before next rise",
        "Recording end",
        "Sustained plateau",
        "Manual endpoint",
    ].index(
        all_events[active_event_id].get(
            "endpoint_type",
            "Post-peak trough",
        )
    ),
    key=f"endpoint_{key}_{active_event_id}",
)

all_events[active_event_id][
    "endpoint_type"
] = endpoint_type

notes = st.text_input(
    f"Notes for {active_event_id}",
    value=all_events[active_event_id].get(
        "notes",
        "",
    ),
    key=f"notes_{key}_{active_event_id}",
)

all_events[active_event_id][
    "notes"
] = notes

current_metric = calculate_event_metrics(
    active_event_id,
    all_events[active_event_id],
)

if current_metric is None:
    st.caption(
        f"{active_event_id} is incomplete. "
        "Select P1, P2, P3, and P4."
    )
elif current_metric.get("Status") == "Complete":
    st.success(
        f"{active_event_id} is complete and ready for export."
    )
else:
    st.error(
        current_metric.get(
            "Status",
            "Invalid event.",
        )
    )


# ============================================================
# Current trace and cumulative event tables
# ============================================================

current_trace_table = build_trace_event_table(
    all_events
)

st.subheader(
    "Events for the selected trace"
)

st.dataframe(
    current_trace_table,
    use_container_width=True,
)

manual_events_df = build_cumulative_event_table(
    st.session_state.trace_events,
    results,
)

st.subheader(
    "Cumulative manual events: all files and traces"
)

if manual_events_df.empty:
    st.info(
        "No manual event slots have been created yet."
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

download1, download2, download3 = st.columns(3)

with download1:
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

with download2:
    st.download_button(
        "Download cell summary CSV",
        cell_summary.to_csv(
            index=False
        ).encode("utf-8"),
        f"calcium_cell_summary_{timestamp}.csv",
        "text/csv",
        use_container_width=True,
    )

with download3:
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
