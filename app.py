import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.signal import find_peaks


# ============================================================
# App setup
# ============================================================

st.set_page_config(
    page_title="Calcium Analysis",
    layout="wide",
)

st.title("Calcium Imaging Post-Analysis")
st.caption(
    "Multi-file calcium analysis with per-file ROI detection, "
    "empty-area background correction, ΔF/F0 normalization, "
    "automatic candidate events, and manual uptake/decay curation."
)

if "uploader_token" not in st.session_state:
    st.session_state.uploader_token = 0


def reset_analysis():
    """
    Reset all user selections and clear uploaded files.
    """
    new_token = st.session_state.uploader_token + 1
    st.session_state.clear()
    st.session_state.uploader_token = new_token


# ============================================================
# Input functions
# ============================================================

def read_table(uploaded_file):
    """
    Read a CSV, XLSX, or XLS file.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file)


def normalize_column_name(column_name):
    """
    Normalize a column name for detection.
    """
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(column_name).lower(),
    )


def get_first_valid_value(series):
    """
    Return the first non-empty value from a series.
    """
    valid = series.dropna()

    if valid.empty:
        return np.nan

    return valid.iloc[0]


def detect_columns(df):
    """
    Detect likely Label, Time, Mean ROI, Area, and background columns.

    ROI fluorescence columns must match:
        Mean1, Mean2, Mean3, ..., MeanN

    Mean(background) is excluded from ROI signals.
    """
    columns = list(df.columns)
    normalized = {
        column: normalize_column_name(column)
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

    label_column = (
        label_candidates[0]
        if label_candidates
        else None
    )

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

    area_columns = [
        column
        for column in columns
        if normalized[column].startswith("area")
    ]

    signal_columns = []

    for column in columns:
        name = normalized[column]

        if column == background_column:
            continue

        if re.fullmatch(r"mean\d+", name):
            signal_columns.append(column)

    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    excluded_columns = set(
        signal_columns
        + area_columns
    )

    if label_column is not None:
        excluded_columns.add(label_column)

    if background_column is not None:
        excluded_columns.add(background_column)

    time_candidates = [
        column
        for column in numeric_columns
        if column not in excluded_columns
    ]

    time_column = (
        time_candidates[0]
        if time_candidates
        else None
    )

    return {
        "label_column": label_column,
        "time_column": time_column,
        "background_column": background_column,
        "signal_columns": signal_columns,
        "area_columns": area_columns,
    }


# ============================================================
# Time and fluorescence calculations
# ============================================================

def make_time_vector(df, selected_time_column, frame_interval):
    """
    Convert a frame/index column into elapsed time.

    If no time column is selected, row order becomes the frame index.
    """
    if selected_time_column is not None:
        raw_frame = pd.to_numeric(
            df[selected_time_column],
            errors="coerce",
        )
    else:
        raw_frame = pd.Series(
            np.arange(len(df)),
            index=df.index,
            dtype=float,
        )

    if raw_frame.notna().sum() >= 2:
        raw_frame = raw_frame.ffill().bfill()
    else:
        raw_frame = pd.Series(
            np.arange(len(df)),
            index=df.index,
            dtype=float,
        )

    elapsed_time = (
        raw_frame - raw_frame.iloc[0]
    ) * frame_interval

    return raw_frame, elapsed_time


def calculate_f0(
    corrected_signal,
    time_values,
    f0_mode,
    f0_n,
    baseline_start,
    baseline_end,
):
    """
    Calculate F0 from a corrected fluorescence trace.
    """
    values = pd.to_numeric(
        corrected_signal,
        errors="coerce",
    )

    times = pd.to_numeric(
        time_values,
        errors="coerce",
    )

    valid_values = values.dropna()

    if valid_values.empty:
        return np.nan

    if f0_mode == "First valid value":
        return float(valid_values.iloc[0])

    if f0_mode == "Mean of first N valid rows":
        return float(
            valid_values.iloc[:max(1, int(f0_n))].mean()
        )

    if f0_mode == "Minimum value in trace":
        return float(valid_values.min())

    if f0_mode == "Lower quartile (25th percentile)":
        return float(valid_values.quantile(0.25))

    if f0_mode == "Mean within baseline-time window":
        if baseline_start is None or baseline_end is None:
            return np.nan

        baseline_mask = (
            (times >= baseline_start)
            & (times <= baseline_end)
            & values.notna()
        )

        baseline_values = values.loc[baseline_mask]

        if baseline_values.empty:
            return np.nan

        return float(baseline_values.mean())

    return float(valid_values.iloc[0])


def calculate_corrected_trace_and_dff(
    raw_signal,
    background_signal,
    settings,
    time_values,
):
    """
    Calculate corrected fluorescence and ΔF/F0.

    Default:
        F corrected = F ROI - F background
        ΔF/F0 = (F corrected - F0) / F0
    """
    raw_signal = pd.to_numeric(
        raw_signal,
        errors="coerce",
    )

    if (
        settings["background_mode"]
        == "Subtract empty-area background before ΔF/F0"
    ):
        background_signal = pd.to_numeric(
            background_signal,
            errors="coerce",
        )

        corrected_signal = raw_signal - background_signal

    else:
        corrected_signal = raw_signal.copy()

    policy = settings["negative_signal_policy"]

    if policy == "Clip corrected fluorescence below zero to zero":
        corrected_signal = corrected_signal.clip(lower=0)

    if policy == "Exclude ROI if any corrected value is below zero":
        if (corrected_signal < 0).any():
            dff = pd.Series(
                np.nan,
                index=corrected_signal.index,
            )

            return corrected_signal, dff, np.nan

    f0 = calculate_f0(
        corrected_signal=corrected_signal,
        time_values=time_values,
        f0_mode=settings["f0_mode"],
        f0_n=settings["f0_n"],
        baseline_start=settings["baseline_start"],
        baseline_end=settings["baseline_end"],
    )

    # Preserve all corrected values, but do not calculate a normalized
    # trace when the selected baseline denominator is invalid.
    if not np.isfinite(f0) or f0 <= 0:
        dff = pd.Series(
            np.nan,
            index=corrected_signal.index,
        )

        return corrected_signal, dff, f0

    dff = (
        corrected_signal - f0
    ) / f0

    return corrected_signal, dff, f0


# ============================================================
# Automatic candidate event detection
# ============================================================

def calculate_linear_slope(x_values, y_values):
    """
    Return linear-regression slope.
    """
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    if len(np.unique(x)) < 2:
        return np.nan

    slope, _, _, _, _ = stats.linregress(x, y)

    return float(slope)


def calculate_auc_above_zero(x_values, y_values):
    """
    Positive area under a ΔF/F0 curve.
    """
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    y_positive = np.maximum(y, 0)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_positive, x))

    return float(np.trapz(y_positive, x))


def detect_auto_event_candidates(
    frame_values,
    time_values,
    dff_values,
    settings,
):
    """
    Detect candidate local maxima.

    These are preliminary candidates for manual curation,
    not necessarily final biological calcium events.
    """
    frames = np.asarray(frame_values, dtype=float)
    times = np.asarray(time_values, dtype=float)
    values = np.asarray(dff_values, dtype=float)

    valid = (
        np.isfinite(frames)
        & np.isfinite(times)
        & np.isfinite(values)
    )

    frames = frames[valid]
    times = times[valid]
    values = values[valid]

    if len(values) < 3:
        return pd.DataFrame()

    peak_indices, properties = find_peaks(
        values,
        height=settings["min_peak_height"],
        prominence=settings["min_peak_prominence"],
        distance=max(
            1,
            int(settings["min_peak_distance"]),
        ),
    )

    candidate_rows = []

    for candidate_number, peak_index in enumerate(
        peak_indices,
        start=1,
    ):
        candidate_rows.append(
            {
                "Auto event": candidate_number,
                "Peak frame": int(frames[peak_index]),
                "Peak time": float(times[peak_index]),
                "Peak ΔF/F0": float(values[peak_index]),
                "Peak prominence": float(
                    properties["prominences"][
                        candidate_number - 1
                    ]
                ),
            }
        )

    return pd.DataFrame(candidate_rows)


# ============================================================
# Manual event metrics
# ============================================================

def get_trace_value_by_frame(processed_df, frame_value, trace_column):
    """
    Obtain one trace value from an exact selected frame.
    """
    matching_rows = processed_df[
        processed_df["Frame"] == frame_value
    ]

    if matching_rows.empty:
        return np.nan, np.nan

    row = matching_rows.iloc[0]

    return (
        float(row["Time"]),
        float(row[trace_column]),
    )


def calculate_manual_event_metrics(
    processed_df,
    trace_column,
    curated_event_df,
):
    """
    Calculate uptake and release/decay metrics for manually chosen points.

    Selected points:
        Start frame -> Peak frame -> End frame

    The endpoint does not need to reach zero. It can be a local trough,
    next response boundary, recording end, or a manually selected point.
    """
    if curated_event_df.empty:
        return pd.DataFrame()

    metric_rows = []

    for _, row in curated_event_df.iterrows():
        include_event = bool(row.get("Include", True))

        if not include_event:
            continue

        start_frame = row.get("Start frame", np.nan)
        peak_frame = row.get("Peak frame", np.nan)
        end_frame = row.get("End frame", np.nan)

        if (
            pd.isna(start_frame)
            or pd.isna(peak_frame)
            or pd.isna(end_frame)
        ):
            continue

        start_time, start_value = get_trace_value_by_frame(
            processed_df,
            start_frame,
            trace_column,
        )

        peak_time, peak_value = get_trace_value_by_frame(
            processed_df,
            peak_frame,
            trace_column,
        )

        end_time, end_value = get_trace_value_by_frame(
            processed_df,
            end_frame,
            trace_column,
        )

        if not np.isfinite(
            [
                start_time,
                start_value,
                peak_time,
                peak_value,
                end_time,
                end_value,
            ]
        ).all():
            continue

        if not (
            start_time < peak_time < end_time
        ):
            status = (
                "Invalid: require Start time < Peak time < End time"
            )

            metric_rows.append(
                {
                    "Event ID": row.get("Event ID", ""),
                    "Status": status,
                }
            )

            continue

        uptake_mask = (
            (processed_df["Time"] >= start_time)
            & (processed_df["Time"] <= peak_time)
        )

        decay_mask = (
            (processed_df["Time"] >= peak_time)
            & (processed_df["Time"] <= end_time)
        )

        uptake_x = processed_df.loc[
            uptake_mask,
            "Time",
        ].to_numpy()

        uptake_y = processed_df.loc[
            uptake_mask,
            trace_column,
        ].to_numpy()

        decay_x = processed_df.loc[
            decay_mask,
            "Time",
        ].to_numpy()

        decay_y = processed_df.loc[
            decay_mask,
            trace_column,
        ].to_numpy()

        uptake_time = peak_time - start_time
        uptake_amplitude = peak_value - start_value

        endpoint_uptake_rate = (
            uptake_amplitude / uptake_time
            if uptake_time > 0
            else np.nan
        )

        regression_uptake_slope = calculate_linear_slope(
            uptake_x,
            uptake_y,
        )

        release_time = end_time - peak_time
        release_amplitude = end_value - peak_value

        observed_decay_rate = (
            release_amplitude / release_time
            if release_time > 0
            else np.nan
        )

        regression_decay_slope = calculate_linear_slope(
            decay_x,
            decay_y,
        )

        if len(uptake_x) >= 2:
            uptake_diffs = (
                np.diff(uptake_y)
                / np.diff(uptake_x)
            )

            max_rise_rate = float(
                np.nanmax(uptake_diffs)
            )
        else:
            max_rise_rate = np.nan

        if len(decay_x) >= 2:
            decay_diffs = (
                np.diff(decay_y)
                / np.diff(decay_x)
            )

            max_decay_rate = float(
                np.nanmin(decay_diffs)
            )
        else:
            max_decay_rate = np.nan

        event_auc = calculate_auc_above_zero(
            processed_df.loc[
                (
                    (processed_df["Time"] >= start_time)
                    & (processed_df["Time"] <= end_time)
                ),
                "Time",
            ].to_numpy(),
            processed_df.loc[
                (
                    (processed_df["Time"] >= start_time)
                    & (processed_df["Time"] <= end_time)
                ),
                trace_column,
            ].to_numpy(),
        )

        metric_rows.append(
            {
                "Event ID": row.get("Event ID", ""),
                "Include": include_event,
                "Event category": row.get(
                    "Event category",
                    "",
                ),
                "Endpoint type": row.get(
                    "Endpoint type",
                    "",
                ),
                "Notes": row.get("Notes", ""),
                "Start frame": start_frame,
                "Start time": start_time,
                "Start ΔF/F0": start_value,
                "Peak frame": peak_frame,
                "Peak time": peak_time,
                "Peak ΔF/F0": peak_value,
                "End frame": end_frame,
                "End time": end_time,
                "End ΔF/F0": end_value,
                "Uptake duration": uptake_time,
                "Uptake amplitude": uptake_amplitude,
                "Endpoint uptake rate": endpoint_uptake_rate,
                "Regression uptake slope": regression_uptake_slope,
                "Maximum rise rate": max_rise_rate,
                "Decay duration": release_time,
                "Decay amplitude": release_amplitude,
                "Observed decay rate": observed_decay_rate,
                "Regression decay slope": regression_decay_slope,
                "Release magnitude": (
                    -observed_decay_rate
                    if np.isfinite(observed_decay_rate)
                    else np.nan
                ),
                "Maximum decay rate": max_decay_rate,
                "Event AUC above zero": event_auc,
                "Status": "Included",
            }
        )

    return pd.DataFrame(metric_rows)


# ============================================================
# Per-file processing
# ============================================================

def process_calcium_file(
    df,
    file_name,
    settings,
):
    """
    Process one uploaded file.

    Every detected Mean1...MeanN column in this file is analyzed.
    """
    file_detection = detect_columns(df)

    file_signal_columns = (
        file_detection["signal_columns"]
    )

    label_column = settings["label_column"]
    time_column = settings["time_column"]
    background_column = settings["background_column"]

    raw_frames, elapsed_time = make_time_vector(
        df=df,
        selected_time_column=time_column,
        frame_interval=settings["frame_interval"],
    )

    if (
        label_column is not None
        and label_column in df.columns
    ):
        label_value = get_first_valid_value(
            df[label_column]
        )

        label = str(label_value)

    else:
        label = re.sub(
            r"\.[^.]+$",
            "",
            file_name,
        )

    if label.strip() == "" or label.lower() == "nan":
        label = re.sub(
            r"\.[^.]+$",
            "",
            file_name,
        )

    processed_df = pd.DataFrame(
        {
            "Frame": raw_frames,
            "Time": elapsed_time,
        }
    )

    if (
        background_column is not None
        and background_column in df.columns
    ):
        background_signal = pd.to_numeric(
            df[background_column],
            errors="coerce",
        )

        processed_df["Background"] = background_signal

    else:
        background_signal = pd.Series(
            0.0,
            index=df.index,
        )

    warnings = []
    cell_summary_rows = []
    auto_event_tables = []

    for signal_column in file_signal_columns:
        raw_signal = pd.to_numeric(
            df[signal_column],
            errors="coerce",
        )

        corrected_signal, dff_signal, f0 = (
            calculate_corrected_trace_and_dff(
                raw_signal=raw_signal,
                background_signal=background_signal,
                settings=settings,
                time_values=elapsed_time,
            )
        )

        processed_df[
            f"Raw | {signal_column}"
        ] = raw_signal

        processed_df[
            f"Corrected | {signal_column}"
        ] = corrected_signal

        processed_df[
            f"dF/F0 | {signal_column}"
        ] = dff_signal

        if not np.isfinite(f0) or f0 <= 0:
            warnings.append(
                f"{file_name} — {signal_column} excluded from "
                "ΔF/F0 calculations because corrected F0 was "
                "missing, zero, or negative."
            )

            cell_summary_rows.append(
                {
                    "File": file_name,
                    "Label": label,
                    "ROI": signal_column,
                    "F0 corrected": f0,
                    "Peak ΔF/F0": np.nan,
                    "Peak time": np.nan,
                    "AUC above baseline": np.nan,
                    "Auto candidate count": 0,
                    "Status": "Excluded: invalid F0",
                }
            )

            continue

        auto_events = detect_auto_event_candidates(
            frame_values=raw_frames.to_numpy(),
            time_values=elapsed_time.to_numpy(),
            dff_values=dff_signal.to_numpy(),
            settings=settings,
        )

        if not auto_events.empty:
            auto_events.insert(0, "ROI", signal_column)
            auto_events.insert(0, "Label", label)
            auto_events.insert(0, "File", file_name)

            auto_event_tables.append(auto_events)

        valid_dff = dff_signal.dropna()

        if valid_dff.empty:
            peak_value = np.nan
            peak_time = np.nan
            auc_value = np.nan

        else:
            peak_index = dff_signal.idxmax()

            peak_value = float(
                dff_signal.loc[peak_index]
            )

            peak_time = float(
                elapsed_time.loc[peak_index]
            )

            auc_value = calculate_auc_above_zero(
                elapsed_time.to_numpy(),
                dff_signal.to_numpy(),
            )

        cell_summary_rows.append(
            {
                "File": file_name,
                "Label": label,
                "ROI": signal_column,
                "F0 corrected": f0,
                "Peak ΔF/F0": peak_value,
                "Peak time": peak_time,
                "AUC above baseline": auc_value,
                "Auto candidate count": len(auto_events),
                "Status": "Included",
            }
        )

    dff_columns = [
        column
        for column in processed_df.columns
        if column.startswith("dF/F0 | ")
    ]

    if dff_columns:
        processed_df["Average dF/F0"] = (
            processed_df[dff_columns]
            .mean(axis=1, skipna=True)
        )

        processed_df["SEM dF/F0"] = (
            processed_df[dff_columns]
            .sem(axis=1, ddof=1)
        )

    else:
        processed_df["Average dF/F0"] = np.nan
        processed_df["SEM dF/F0"] = np.nan

    average_auto_events = detect_auto_event_candidates(
        frame_values=processed_df["Frame"].to_numpy(),
        time_values=processed_df["Time"].to_numpy(),
        dff_values=processed_df[
            "Average dF/F0"
        ].to_numpy(),
        settings=settings,
    )

    if not average_auto_events.empty:
        average_auto_events.insert(
            0,
            "Trace",
            "File average",
        )

        average_auto_events.insert(
            0,
            "Label",
            label,
        )

        average_auto_events.insert(
            0,
            "File",
            file_name,
        )

    if auto_event_tables:
        cell_auto_events_df = pd.concat(
            auto_event_tables,
            ignore_index=True,
        )

    else:
        cell_auto_events_df = pd.DataFrame()

    return {
        "file_name": file_name,
        "label": label,
        "processed_df": processed_df,
        "cell_summary_df": pd.DataFrame(
            cell_summary_rows
        ),
        "cell_auto_events_df": cell_auto_events_df,
        "average_auto_events_df": average_auto_events,
        "warnings": warnings,
    }


# ============================================================
# Plotting
# ============================================================

def build_file_trace_plot(
    results,
    settings,
):
    """
    Plot each uploaded file separately.
    """
    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#bcbd22",
        "#7f7f7f",
    ]

    fig = go.Figure()

    for index, result in enumerate(results):
        color = colors[index % len(colors)]

        processed_df = result["processed_df"]

        x_values = processed_df["Time"].to_numpy()
        y_values = processed_df[
            "Average dF/F0"
        ].to_numpy()

        if settings["positive_only"]:
            y_values = np.maximum(y_values, 0)

        trace_name = (
            f"{result['label']} | {result['file_name']}"
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=trace_name,
                line=dict(
                    color=color,
                    width=2.5,
                ),
            )
        )

        if settings["show_auto_peaks"]:
            auto_events = result[
                "average_auto_events_df"
            ]

            if not auto_events.empty:
                fig.add_trace(
                    go.Scatter(
                        x=auto_events["Peak time"],
                        y=auto_events["Peak ΔF/F0"],
                        mode="markers",
                        marker=dict(
                            symbol="x",
                            size=10,
                            color=color,
                        ),
                        name=(
                            f"Auto candidates | "
                            f"{result['file_name']}"
                        ),
                        showlegend=False,
                        hovertemplate=(
                            "File: " + result["file_name"]
                            + "<br>Auto event: %{text}"
                            + "<br>Peak time: %{x:.3f}"
                            + "<br>Peak ΔF/F0: %{y:.3f}"
                            + "<extra></extra>"
                        ),
                        text=auto_events[
                            "Auto event"
                        ].astype(str),
                    )
                )

    fig.update_layout(
        height=600,
        template="plotly_white",
        xaxis_title=f"Time ({settings['time_unit']})",
        yaxis_title="Average ΔF/F0",
        legend_title="Uploaded file",
    )

    return fig


def add_manual_markers_to_plot(
    fig,
    processed_df,
    trace_column,
    curated_events,
):
    """
    Add manual start, peak, and endpoint markers to a Plotly figure.
    """
    if curated_events.empty:
        return fig

    for _, row in curated_events.iterrows():
        if not bool(row.get("Include", True)):
            continue

        start_frame = row.get("Start frame", np.nan)
        peak_frame = row.get("Peak frame", np.nan)
        end_frame = row.get("End frame", np.nan)

        start_time, start_value = get_trace_value_by_frame(
            processed_df,
            start_frame,
            trace_column,
        )

        peak_time, peak_value = get_trace_value_by_frame(
            processed_df,
            peak_frame,
            trace_column,
        )

        end_time, end_value = get_trace_value_by_frame(
            processed_df,
            end_frame,
            trace_column,
        )

        event_id = row.get("Event ID", "Manual event")

        if np.isfinite([start_time, start_value]).all():
            fig.add_trace(
                go.Scatter(
                    x=[start_time],
                    y=[start_value],
                    mode="markers+text",
                    marker=dict(
                        color="green",
                        size=11,
                    ),
                    text=[f"{event_id} start"],
                    textposition="bottom center",
                    showlegend=False,
                )
            )

        if np.isfinite([peak_time, peak_value]).all():
            fig.add_trace(
                go.Scatter(
                    x=[peak_time],
                    y=[peak_value],
                    mode="markers+text",
                    marker=dict(
                        color="red",
                        size=12,
                    ),
                    text=[f"{event_id} peak"],
                    textposition="top center",
                    showlegend=False,
                )
            )

        if np.isfinite([end_time, end_value]).all():
            fig.add_trace(
                go.Scatter(
                    x=[end_time],
                    y=[end_value],
                    mode="markers+text",
                    marker=dict(
                        color="purple",
                        size=11,
                    ),
                    text=[f"{event_id} end"],
                    textposition="bottom center",
                    showlegend=False,
                )
            )

        if np.isfinite(
            [
                start_time,
                start_value,
                peak_time,
                peak_value,
            ]
        ).all():
            fig.add_trace(
                go.Scatter(
                    x=[start_time, peak_time],
                    y=[start_value, peak_value],
                    mode="lines",
                    line=dict(
                        color="green",
                        width=2,
                    ),
                    showlegend=False,
                )
            )

        if np.isfinite(
            [
                peak_time,
                peak_value,
                end_time,
                end_value,
            ]
        ).all():
            fig.add_trace(
                go.Scatter(
                    x=[peak_time, end_time],
                    y=[peak_value, end_value],
                    mode="lines",
                    line=dict(
                        color="purple",
                        width=2,
                        dash="dash",
                    ),
                    showlegend=False,
                )
            )

    return fig


def build_selected_trace_plot(
    processed_df,
    trace_column,
    file_name,
    label,
    settings,
    curated_events,
):
    """
    Plot a selected file-average or selected ROI trace.
    """
    fig = go.Figure()

    x_values = processed_df["Time"].to_numpy()
    y_values = processed_df[
        trace_column
    ].to_numpy()

    if settings["positive_only"]:
        y_values = np.maximum(y_values, 0)

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name=f"{label} | {file_name} | {trace_column}",
            line=dict(
                color="#1f77b4",
                width=3,
            ),
        )
    )

    fig = add_manual_markers_to_plot(
        fig=fig,
        processed_df=processed_df,
        trace_column=trace_column,
        curated_events=curated_events,
    )

    fig.update_layout(
        height=600,
        template="plotly_white",
        xaxis_title=f"Time ({settings['time_unit']})",
        yaxis_title="ΔF/F0",
        legend_title="Trace",
    )

    return fig


# ============================================================
# Export
# ============================================================

def safe_sheet_name(name, used_names):
    """
    Make a valid unique Excel worksheet name.
    """
    cleaned = re.sub(
        r"[\\/*?:\[\]]",
        "_",
        str(name),
    )

    cleaned = cleaned[:31] or "sheet"

    candidate = cleaned
    counter = 1

    while candidate in used_names:
        suffix = f"_{counter}"

        candidate = (
            cleaned[:31 - len(suffix)]
            + suffix
        )

        counter += 1

    used_names.add(candidate)

    return candidate


def make_excel_bytes(
    results,
    settings,
    combined_summary,
    combined_auto_events,
    curated_metrics,
):
    """
    Build a full downloadable Excel workbook.
    """
    output = io.BytesIO()
    used_sheet_names = set()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        pd.DataFrame(
            [settings]
        ).to_excel(
            writer,
            index=False,
            sheet_name="analysis_settings",
        )

        metadata = pd.DataFrame(
            [
                {
                    "File": result["file_name"],
                    "Label": result["label"],
                }
                for result in results
            ]
        )

        metadata.to_excel(
            writer,
            index=False,
            sheet_name="file_metadata",
        )

        if not combined_summary.empty:
            combined_summary.to_excel(
                writer,
                index=False,
                sheet_name="cell_summary",
            )

        if not combined_auto_events.empty:
            combined_auto_events.to_excel(
                writer,
                index=False,
                sheet_name="auto_candidates",
            )

        if not curated_metrics.empty:
            curated_metrics.to_excel(
                writer,
                index=False,
                sheet_name="manual_curated_events",
            )

        for result in results:
            sheet_name = safe_sheet_name(
                f"trace_{result['file_name']}",
                used_sheet_names,
            )

            result["processed_df"].to_excel(
                writer,
                index=False,
                sheet_name=sheet_name,
            )

    output.seek(0)

    return output.getvalue()


# ============================================================
# Sidebar settings
# ============================================================

with st.sidebar:
    st.header("Time settings")

    frame_interval = st.number_input(
        "Frame interval",
        min_value=0.000001,
        value=1.0,
        step=0.1,
        help=(
            "Time between frames. For example, enter 2 "
            "if images were acquired every 2 seconds."
        ),
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

    st.divider()

    st.header("Background correction")

    background_mode = st.radio(
        "Background method",
        [
            "Subtract empty-area background before ΔF/F0",
            "No background correction",
        ],
        index=0,
    )

    negative_signal_policy = st.radio(
        "Negative corrected-signal handling",
        [
            "Keep negative values (recommended)",
            "Clip corrected fluorescence below zero to zero",
            "Exclude ROI if any corrected value is below zero",
        ],
        index=0,
        help=(
            "Applied after background subtraction. "
            "The default preserves values below zero."
        ),
    )

    st.divider()

    st.header("F0 settings")

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

    st.divider()

    st.header("Automatic candidate settings")

    min_peak_height = st.number_input(
        "Minimum candidate peak height (ΔF/F0)",
        min_value=0.0,
        value=0.05,
        step=0.01,
    )

    min_peak_prominence = st.number_input(
        "Minimum candidate prominence (ΔF/F0)",
        min_value=0.0,
        value=0.05,
        step=0.01,
    )

    min_peak_distance = int(
        st.number_input(
            "Minimum frames between candidates",
            min_value=1,
            value=3,
            step=1,
        )
    )

    st.divider()

    st.header("Display settings")

    positive_only = st.checkbox(
        "Plot only positive ΔF/F0",
        value=False,
    )

    show_auto_peaks = st.checkbox(
        "Show automatic candidate peaks",
        value=True,
    )

    st.divider()

    st.button(
        "Reset analysis / upload new files",
        on_click=reset_analysis,
        use_container_width=True,
    )


# ============================================================
# Upload
# ============================================================

uploaded_files = st.file_uploader(
    "Upload calcium time-series files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    key=f"calcium_files_{st.session_state.uploader_token}",
    help=(
        "Each file may contain any number of cellular columns named "
        "Mean1, Mean2, Mean3, etc. Mean(background) is used as "
        "the empty-area background trace."
    ),
)

if not uploaded_files:
    st.info(
        "Upload one or more calcium imaging CSV/XLSX files to begin."
    )
    st.stop()


# ============================================================
# Read uploaded files
# ============================================================

loaded_files = []

for uploaded_file in uploaded_files:
    try:
        file_df = read_table(uploaded_file)

        loaded_files.append(
            {
                "file_name": uploaded_file.name,
                "df": file_df,
            }
        )

    except Exception as error:
        st.error(
            f"Could not read {uploaded_file.name}: {error}"
        )

if not loaded_files:
    st.stop()


# ============================================================
# Auto-detected structure
# ============================================================

st.subheader("Detected column structure")

detection_rows = []

for item in loaded_files:
    detection = detect_columns(item["df"])

    detection_rows.append(
        {
            "File": item["file_name"],
            "Detected label": detection["label_column"],
            "Detected frame/time": detection["time_column"],
            "Detected background": detection["background_column"],
            "Detected ROI Mean columns": ", ".join(
                detection["signal_columns"]
            ),
            "Number of ROI traces": len(
                detection["signal_columns"]
            ),
        }
    )

st.dataframe(
    pd.DataFrame(detection_rows),
    use_container_width=True,
)


# ============================================================
# Shared mappings
# ============================================================

first_df = loaded_files[0]["df"]
first_detection = detect_columns(first_df)
first_columns = first_df.columns.tolist()

with st.expander(
    "Shared column mapping",
    expanded=True,
):
    map_col1, map_col2, map_col3, map_col4 = st.columns(4)

    with map_col1:
        label_options = ["<auto/file name>"] + first_columns

        default_label_index = 0

        if first_detection["label_column"] in first_columns:
            default_label_index = (
                first_columns.index(
                    first_detection["label_column"]
                )
                + 1
            )

        selected_label_column = st.selectbox(
            "Label column",
            label_options,
            index=default_label_index,
        )

    with map_col2:
        time_options = ["<auto row index>"] + first_columns

        default_time_index = 0

        if first_detection["time_column"] in first_columns:
            default_time_index = (
                first_columns.index(
                    first_detection["time_column"]
                )
                + 1
            )

        selected_time_column = st.selectbox(
            "Frame/time column",
            time_options,
            index=default_time_index,
        )

    with map_col3:
        background_options = ["<none>"] + first_columns

        default_background_index = 0

        if first_detection["background_column"] in first_columns:
            default_background_index = (
                background_options.index(
                    first_detection["background_column"]
                )
            )

        selected_background_column = st.selectbox(
            "Empty-area background column",
            background_options,
            index=default_background_index,
        )

    with map_col4:
        st.info(
            "All ROI traces are detected independently in every uploaded "
            "file. The app analyzes every column matching Mean1, Mean2, "
            "Mean3, etc. A file may contain 5, 10, 20, or more ROIs."
        )


if (
    background_mode
    == "Subtract empty-area background before ΔF/F0"
    and selected_background_column == "<none>"
):
    st.error(
        "Select Mean(background) as the background column or choose "
        "'No background correction'."
    )
    st.stop()


settings = {
    "label_column": (
        None
        if selected_label_column == "<auto/file name>"
        else selected_label_column
    ),
    "time_column": (
        None
        if selected_time_column == "<auto row index>"
        else selected_time_column
    ),
    "background_column": (
        None
        if selected_background_column == "<none>"
        else selected_background_column
    ),
    "frame_interval": frame_interval,
    "time_unit": time_unit,
    "background_mode": background_mode,
    "negative_signal_policy": negative_signal_policy,
    "f0_mode": f0_mode,
    "f0_n": f0_n,
    "baseline_start": baseline_start,
    "baseline_end": baseline_end,
    "min_peak_height": min_peak_height,
    "min_peak_prominence": min_peak_prominence,
    "min_peak_distance": min_peak_distance,
    "positive_only": positive_only,
    "show_auto_peaks": show_auto_peaks,
}


# ============================================================
# Validate mappings for every file
# ============================================================

mapping_errors = []
mapping_warnings = []

for item in loaded_files:
    file_name = item["file_name"]
    df = item["df"]

    detection = detect_columns(df)

    if not detection["signal_columns"]:
        mapping_errors.append(
            f"{file_name}: no Mean1...MeanN ROI columns detected."
        )

    if (
        settings["background_column"] is not None
        and settings["background_column"] not in df.columns
    ):
        mapping_errors.append(
            f"{file_name}: missing selected background column "
            f"'{settings['background_column']}'."
        )

    if (
        settings["time_column"] is not None
        and settings["time_column"] not in df.columns
    ):
        mapping_errors.append(
            f"{file_name}: missing selected time/frame column "
            f"'{settings['time_column']}'."
        )

    if (
        settings["label_column"] is not None
        and settings["label_column"] not in df.columns
    ):
        mapping_warnings.append(
            f"{file_name}: label column "
            f"'{settings['label_column']}' not found; filename will "
            "be used as the label."
        )

for warning in mapping_warnings:
    st.warning(warning)

if mapping_errors:
    st.error(
        "Some uploaded files cannot be analyzed:\n\n"
        + "\n".join(mapping_errors)
    )
    st.stop()


# ============================================================
# Process every uploaded file
# ============================================================

results = []

for item in loaded_files:
    file_settings = settings.copy()

    result = process_calcium_file(
        df=item["df"],
        file_name=item["file_name"],
        settings=file_settings,
    )

    results.append(result)

all_warnings = []

for result in results:
    all_warnings.extend(result["warnings"])

for warning in all_warnings:
    st.warning(warning)


# ============================================================
# Main graph: all uploaded files
# ============================================================

st.subheader("All uploaded file-average traces")

main_plot = build_file_trace_plot(
    results=results,
    settings=settings,
)

st.plotly_chart(
    main_plot,
    use_container_width=True,
)


# ============================================================
# Summaries and automatic candidates
# ============================================================

summary_tables = [
    result["cell_summary_df"]
    for result in results
    if not result["cell_summary_df"].empty
]

if summary_tables:
    combined_summary = pd.concat(
        summary_tables,
        ignore_index=True,
    )
else:
    combined_summary = pd.DataFrame()

auto_event_tables = []

for result in results:
    if not result["cell_auto_events_df"].empty:
        auto_event_tables.append(
            result["cell_auto_events_df"]
        )

    if not result["average_auto_events_df"].empty:
        auto_event_tables.append(
            result["average_auto_events_df"]
        )

if auto_event_tables:
    combined_auto_events = pd.concat(
        auto_event_tables,
        ignore_index=True,
    )
else:
    combined_auto_events = pd.DataFrame()

st.subheader("Cell/ROI summary")

st.dataframe(
    combined_summary,
    use_container_width=True,
)

st.subheader("Automatic candidate peaks")

st.caption(
    "These are candidate local maxima for review. They are not automatically "
    "treated as final uptake/release events."
)

if combined_auto_events.empty:
    st.info(
        "No automatic candidate peaks met the current height, prominence, "
        "and distance criteria."
    )
else:
    st.dataframe(
        combined_auto_events,
        use_container_width=True,
    )


# ============================================================
# Manual event curation
# ============================================================

st.subheader("Manual event curation")

st.write(
    "Select biologically meaningful start, peak, and endpoint frames. "
    "The endpoint may be zero, a post-peak trough, the point before a "
    "new rise, or the recording end. Manual selections determine the "
    "final uptake and observed-decay metrics."
)

curation_file_name = st.selectbox(
    "File for manual curation",
    [result["file_name"] for result in results],
)

curation_result = next(
    result
    for result in results
    if result["file_name"] == curation_file_name
)

curation_processed_df = curation_result["processed_df"]

trace_options = ["Average dF/F0"] + [
    column
    for column in curation_processed_df.columns
    if column.startswith("dF/F0 | ")
]

selected_trace_column = st.selectbox(
    "Trace for manual curation",
    trace_options,
)

frame_options = (
    curation_processed_df["Frame"]
    .dropna()
    .astype(int)
    .tolist()
)

default_peak_frames = []

if selected_trace_column == "Average dF/F0":
    default_auto_events = curation_result[
        "average_auto_events_df"
    ]

    if not default_auto_events.empty:
        default_peak_frames = (
            default_auto_events["Peak frame"]
            .astype(int)
            .tolist()
        )

else:
    selected_roi_name = selected_trace_column.replace(
        "dF/F0 | ",
        "",
    )

    default_auto_events = curation_result[
        "cell_auto_events_df"
    ]

    if not default_auto_events.empty:
        default_auto_events = default_auto_events[
            default_auto_events["ROI"] == selected_roi_name
        ]

        default_peak_frames = (
            default_auto_events["Peak frame"]
            .astype(int)
            .tolist()
        )

if not default_peak_frames:
    default_peak_frames = [
        frame_options[
            int(len(frame_options) / 2)
        ]
    ]

default_manual_events = pd.DataFrame(
    {
        "Event ID": [
            f"E{number}"
            for number in range(
                1,
                len(default_peak_frames) + 1,
            )
        ],
        "Start frame": [
            max(
                frame_options[0],
                peak_frame - 1,
            )
            for peak_frame in default_peak_frames
        ],
        "Peak frame": default_peak_frames,
        "End frame": [
            min(
                frame_options[-1],
                peak_frame + 1,
            )
            for peak_frame in default_peak_frames
        ],
        "Event category": [
            "Manual event"
            for _ in default_peak_frames
        ],
        "Endpoint type": [
            "Manual endpoint"
            for _ in default_peak_frames
        ],
        "Include": [
            True
            for _ in default_peak_frames
        ],
        "Notes": [
            ""
            for _ in default_peak_frames
        ],
    }
)

st.caption(
    "Edit frames directly. Use the plus button in the table to add events. "
    "For every included event, Start frame must be before Peak frame, and "
    "Peak frame must be before End frame."
)

edited_manual_events = st.data_editor(
    default_manual_events,
    num_rows="dynamic",
    use_container_width=True,
    key=(
        f"manual_events_"
        f"{curation_file_name}_"
        f"{selected_trace_column}"
    ),
    column_config={
        "Event ID": st.column_config.TextColumn(
            "Event ID"
        ),
        "Start frame": st.column_config.SelectboxColumn(
            "Start frame",
            options=frame_options,
        ),
        "Peak frame": st.column_config.SelectboxColumn(
            "Peak frame",
            options=frame_options,
        ),
        "End frame": st.column_config.SelectboxColumn(
            "End frame",
            options=frame_options,
        ),
        "Event category": st.column_config.SelectboxColumn(
            "Event category",
            options=[
                "Manual event",
                "Complete response",
                "Partial decay",
                "Overlapping response",
                "Sustained plateau",
                "Rejected / noise",
            ],
        ),
        "Endpoint type": st.column_config.SelectboxColumn(
            "Endpoint type",
            options=[
                "Returned to baseline",
                "Post-peak trough",
                "Before next rise",
                "Recording end",
                "Sustained plateau",
                "Manual endpoint",
            ],
        ),
        "Include": st.column_config.CheckboxColumn(
            "Include"
        ),
        "Notes": st.column_config.TextColumn(
            "Notes"
        ),
    },
)

manual_metrics = calculate_manual_event_metrics(
    processed_df=curation_processed_df,
    trace_column=selected_trace_column,
    curated_event_df=edited_manual_events,
)

manual_plot = build_selected_trace_plot(
    processed_df=curation_processed_df,
    trace_column=selected_trace_column,
    file_name=curation_result["file_name"],
    label=curation_result["label"],
    settings=settings,
    curated_events=edited_manual_events,
)

st.subheader("Manual-event trace and selected points")

st.plotly_chart(
    manual_plot,
    use_container_width=True,
)

st.subheader("Manual uptake and observed-decay metrics")

if manual_metrics.empty:
    st.info(
        "Add valid manual events with Start frame < Peak frame < End frame."
    )
else:
    st.dataframe(
        manual_metrics,
        use_container_width=True,
    )


# ============================================================
# Downloads
# ============================================================

st.subheader("Downloads")

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

excel_bytes = make_excel_bytes(
    results=results,
    settings=settings,
    combined_summary=combined_summary,
    combined_auto_events=combined_auto_events,
    curated_metrics=manual_metrics,
)

download_col1, download_col2, download_col3, download_col4 = st.columns(4)

with download_col1:
    st.download_button(
        "Download Excel workbook",
        data=excel_bytes,
        file_name=f"calcium_analysis_{timestamp}.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )

with download_col2:
    st.download_button(
        "Download cell summary CSV",
        data=combined_summary.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=f"calcium_cell_summary_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_col3:
    st.download_button(
        "Download auto candidates CSV",
        data=combined_auto_events.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=f"calcium_auto_candidates_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_col4:
    st.download_button(
        "Download manual metrics CSV",
        data=manual_metrics.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=f"calcium_manual_events_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.download_button(
    "Download all-file graph HTML",
    data=main_plot.to_html(
        include_plotlyjs="cdn"
    ),
    file_name=f"calcium_all_files_{timestamp}.html",
    mime="text/html",
)
