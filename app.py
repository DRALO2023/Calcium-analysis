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
    "Multi-file calcium analysis with empty-area background subtraction, "
    "ΔF/F0 normalization, peak detection, and event-wise uptake/release rates."
)


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

if "uploader_token" not in st.session_state:
    st.session_state.uploader_token = 0


def reset_analysis():
    """Clear selections and reset file upload widgets."""
    token = st.session_state.uploader_token + 1
    st.session_state.clear()
    st.session_state.uploader_token = token


# -------------------------------------------------------------------
# Input and column detection
# -------------------------------------------------------------------

def read_table(uploaded_file):
    """Read CSV or Excel upload into a DataFrame."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file)


def normalize_column_name(column_name):
    """Create a simple normalized column name for matching."""
    return re.sub(r"[^a-z0-9]", "", str(column_name).lower())


def get_first_valid_value(series):
    """Return the first non-missing numeric or text value."""
    valid = series.dropna()

    if valid.empty:
        return np.nan

    return valid.iloc[0]


def detect_columns(df):
    """
    Suggest likely Label, time, background, area, and signal columns.

    Expected calcium format:
    Label
    Frame / Time
    Area1, Mean1, Area2, Mean2, ...
    Area(background), Mean(background)
    """
    columns = list(df.columns)
    normalized = {col: normalize_column_name(col) for col in columns}

    label_candidates = [
        col for col in columns
        if normalized[col] in {
            "label",
            "condition",
            "sample",
            "experiment",
            "group",
            "treatment",
        }
    ]
    label_col = label_candidates[0] if label_candidates else None

    background_candidates = [
        col for col in columns
        if "background" in normalized[col] and "mean" in normalized[col]
    ]

    if not background_candidates:
        background_candidates = [
            col for col in columns
            if normalized[col] in {"background", "bg", "meanbg"}
        ]

    background_col = (
        background_candidates[0]
        if background_candidates
        else None
    )

    area_cols = [
        col for col in columns
        if normalized[col].startswith("area")
    ]

    signal_cols = []

    for col in columns:
        normalized_name = normalized[col]

        if col == background_col:
            continue

        if re.fullmatch(r"mean\d+", normalized_name):
            signal_cols.append(col)

    numeric_cols = [
        col for col in columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    excluded_columns = set(signal_cols + area_cols)

    if label_col is not None:
        excluded_columns.add(label_col)

    if background_col is not None:
        excluded_columns.add(background_col)

    time_candidates = [
        col for col in numeric_cols
        if col not in excluded_columns
    ]

    time_col = time_candidates[0] if time_candidates else None

    return {
        "label_col": label_col,
        "time_col": time_col,
        "background_col": background_col,
        "signal_cols": signal_cols,
        "area_cols": area_cols,
    }


# -------------------------------------------------------------------
# Fluorescence and ΔF/F0 calculations
# -------------------------------------------------------------------

def get_f0(
    corrected_signal,
    time_values,
    mode,
    n_rows=5,
    baseline_start=None,
    baseline_end=None,
):
    """
    Calculate F0 from a background-corrected signal.

    Parameters
    ----------
    corrected_signal : pandas Series
        Background-corrected fluorescence trace.
    time_values : pandas Series
        Time vector in user-selected units.
    mode : str
        Selected F0 method.
    n_rows : int
        Number of valid baseline points for early-frame averaging.
    baseline_start : float
        Start of baseline time window.
    baseline_end : float
        End of baseline time window.
    """
    values = pd.to_numeric(corrected_signal, errors="coerce")
    times = pd.to_numeric(time_values, errors="coerce")

    valid_values = values.dropna()

    if valid_values.empty:
        return np.nan

    if mode == "First valid value":
        return float(valid_values.iloc[0])

    if mode == "Mean of first N valid rows":
        return float(valid_values.iloc[:max(1, int(n_rows))].mean())

    if mode == "Minimum value in trace":
        return float(valid_values.min())

    if mode == "Lower quartile (25th percentile)":
        return float(valid_values.quantile(0.25))

    if mode == "Mean within baseline-time window":
        if baseline_start is None or baseline_end is None:
            return np.nan

        mask = (
            (times >= baseline_start)
            & (times <= baseline_end)
            & values.notna()
        )

        window_values = values.loc[mask]

        if window_values.empty:
            return np.nan

        return float(window_values.mean())

    return float(valid_values.iloc[0])


def calculate_dff(raw_signal, background_signal, settings, time_values):
    """
    Calculate background-corrected fluorescence and ΔF/F0.

    Recommended calculation:
        F_corrected = F_ROI - F_background
        ΔF/F0 = (F_corrected - F0_corrected) / F0_corrected
    """
    raw_signal = pd.to_numeric(raw_signal, errors="coerce")

    if settings["background_mode"] == "Subtract empty-area background before ΔF/F0":
        background_signal = pd.to_numeric(
            background_signal,
            errors="coerce",
        )
        corrected_signal = raw_signal - background_signal
    else:
        corrected_signal = raw_signal.copy()

    f0 = get_f0(
        corrected_signal=corrected_signal,
        time_values=time_values,
        mode=settings["f0_mode"],
        n_rows=settings["f0_n"],
        baseline_start=settings["baseline_start"],
        baseline_end=settings["baseline_end"],
    )

    if not np.isfinite(f0) or f0 <= 0:
        dff = pd.Series(np.nan, index=corrected_signal.index)
        return corrected_signal, dff, f0

    dff = (corrected_signal - f0) / f0

    return corrected_signal, dff, f0


# -------------------------------------------------------------------
# Event detection and rate calculations
# -------------------------------------------------------------------

def interpolate_zero_crossing(x1, y1, x2, y2):
    """
    Estimate the time at which the trace crosses zero between two points.
    """
    if not np.isfinite([x1, y1, x2, y2]).all():
        return float(x2)

    if y2 == y1:
        return float(x2)

    return float(
        x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
    )


def calculate_slope(x, y):
    """
    Calculate linear-regression slope for a trace segment.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    if len(np.unique(x)) < 2:
        return np.nan

    slope, _, _, _, _ = stats.linregress(x, y)

    return float(slope)


def find_event_start(x, y, peak_index):
    """
    Find the previous point at or below zero before a positive peak.

    Returns:
        start_index, interpolated_start_time
    """
    for index in range(peak_index - 1, -1, -1):
        if y[index] <= 0:
            if index < peak_index and y[index + 1] > 0:
                zero_time = interpolate_zero_crossing(
                    x[index],
                    y[index],
                    x[index + 1],
                    y[index + 1],
                )
                return index, zero_time

            return index, float(x[index])

    return 0, float(x[0])


def find_event_end(x, y, peak_index):
    """
    Find the first return to zero or below after a positive peak.

    Returns:
        end_index, interpolated_end_time

    If no return to baseline occurs, returns:
        None, NaN
    """
    for index in range(peak_index + 1, len(y)):
        if y[index] <= 0:
            if y[index - 1] > 0:
                zero_time = interpolate_zero_crossing(
                    x[index - 1],
                    y[index - 1],
                    x[index],
                    y[index],
                )
                return index, zero_time

            return index, float(x[index])

    return None, np.nan


def calculate_auc_above_zero(x, y):
    """
    Calculate positive area under the curve only.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    positive_y = np.maximum(y, 0)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(positive_y, x))

    return float(np.trapz(positive_y, x))


def detect_calcium_events(time_values, dff_values, settings):
    """
    Detect significant positive calcium-response events.

    An accepted event:
    - Must exceed zero and minimum peak height.
    - Must meet minimum prominence.
    - Must be separated by configured distance.
    - Gets uptake rate from baseline/zero to peak.
    - Gets release rate only if it returns to zero after the peak.
    """
    x = np.asarray(time_values, dtype=float)
    y = np.asarray(dff_values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return pd.DataFrame()

    peak_indices, peak_properties = find_peaks(
        y,
        height=settings["min_peak_height"],
        prominence=settings["min_peak_prominence"],
        distance=max(1, int(settings["min_peak_distance"])),
    )

    event_rows = []

    for event_number, peak_index in enumerate(peak_indices, start=1):
        peak_value = float(y[peak_index])

        if peak_value <= 0:
            continue

        start_index, start_time = find_event_start(
            x,
            y,
            peak_index,
        )

        end_index, end_time = find_event_end(
            x,
            y,
            peak_index,
        )

        rise_x = x[start_index:peak_index + 1]
        rise_y = y[start_index:peak_index + 1]

        if len(rise_x) >= settings["min_event_points"]:
            uptake_slope = calculate_slope(rise_x, rise_y)
        else:
            uptake_slope = np.nan

        if (
            len(rise_x) >= 2
            and np.all(np.diff(rise_x) > 0)
        ):
            instantaneous_rise = np.diff(rise_y) / np.diff(rise_x)
            max_rise_rate = float(np.nanmax(instantaneous_rise))
        else:
            max_rise_rate = np.nan

        if end_index is not None:
            release_x = x[peak_index:end_index + 1]
            release_y = y[peak_index:end_index + 1]

            if len(release_x) >= settings["min_event_points"]:
                release_slope = calculate_slope(
                    release_x,
                    release_y,
                )
            else:
                release_slope = np.nan

            if (
                len(release_x) >= 2
                and np.all(np.diff(release_x) > 0)
            ):
                instantaneous_release = (
                    np.diff(release_y)
                    / np.diff(release_x)
                )
                max_decay_rate = float(
                    np.nanmin(instantaneous_release)
                )
            else:
                max_decay_rate = np.nan

            event_duration = float(end_time - start_time)

            event_auc = calculate_auc_above_zero(
                x[start_index:end_index + 1],
                y[start_index:end_index + 1],
            )

            release_status = "Returned to baseline"

        else:
            release_slope = np.nan
            max_decay_rate = np.nan
            event_duration = np.nan
            event_auc = np.nan
            release_status = "No return to baseline"

        prominence = float(
            peak_properties["prominences"][event_number - 1]
        )

        event_rows.append(
            {
                "Event": event_number,
                "Start time": start_time,
                "Peak time": float(x[peak_index]),
                "End time": end_time,
                "Peak ΔF/F0": peak_value,
                "Prominence": prominence,
                "Duration": event_duration,
                "AUC above baseline": event_auc,
                "Uptake slope": uptake_slope,
                "Maximum rise rate": max_rise_rate,
                "Release slope": release_slope,
                "Release magnitude": (
                    -release_slope
                    if np.isfinite(release_slope)
                    else np.nan
                ),
                "Maximum decay rate": max_decay_rate,
                "Release status": release_status,
            }
        )

    return pd.DataFrame(event_rows)


# -------------------------------------------------------------------
# Per-file processing
# -------------------------------------------------------------------

def make_time_vector(df, time_column, frame_interval):
    """
    Create elapsed-time vector.

    If a numeric time/frame column exists:
        elapsed_time = (value - first_value) * frame_interval

    Otherwise:
        elapsed_time = row_index * frame_interval
    """
    if time_column is not None:
        raw_time = pd.to_numeric(
            df[time_column],
            errors="coerce",
        )
    else:
        raw_time = pd.Series(
            np.arange(len(df)),
            index=df.index,
            dtype=float,
        )

    if raw_time.notna().sum() >= 2:
        raw_time = raw_time.ffill().bfill()
    else:
        raw_time = pd.Series(
            np.arange(len(df)),
            index=df.index,
            dtype=float,
        )

    elapsed_time = (
        raw_time - raw_time.iloc[0]
    ) * frame_interval

    return raw_time, elapsed_time


def process_calcium_file(df, file_name, settings):
    """
    Process one calcium imaging file and return all trace/event outputs.
    """
    label_column = settings["label_column"]
    time_column = settings["time_column"]
    background_column = settings["background_column"]
    signal_columns = settings["signal_columns"]

    frame_values, time_values = make_time_vector(
        df=df,
        time_column=time_column,
        frame_interval=settings["frame_interval"],
    )

    if label_column is not None:
        label_value = get_first_valid_value(df[label_column])
        label = str(label_value)
    else:
        label = ""

    if label.strip() == "" or label.lower() == "nan":
        label = re.sub(r"\.[^.]+$", "", file_name)

    processed_df = pd.DataFrame(
        {
            "Frame": frame_values,
            "Time": time_values,
        }
    )

    if background_column is not None:
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
    cell_event_tables = []

    for signal_column in signal_columns:
        raw_signal = pd.to_numeric(
            df[signal_column],
            errors="coerce",
        )

        corrected_signal, dff_signal, f0 = calculate_dff(
            raw_signal=raw_signal,
            background_signal=background_signal,
            settings=settings,
            time_values=time_values,
        )

        processed_df[f"Raw | {signal_column}"] = raw_signal
        processed_df[f"Corrected | {signal_column}"] = corrected_signal
        processed_df[f"dF/F0 | {signal_column}"] = dff_signal

        if not np.isfinite(f0) or f0 <= 0:
            warnings.append(
                f"{file_name} — {signal_column} was excluded because "
                "its corrected F0 was missing, zero, or negative."
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
                    "Event count": 0,
                    "Status": "Excluded: invalid corrected F0",
                }
            )

            continue

        events = detect_calcium_events(
            time_values=time_values.to_numpy(),
            dff_values=dff_signal.to_numpy(),
            settings=settings,
        )

        if not events.empty:
            events.insert(0, "ROI", signal_column)
            events.insert(0, "Label", label)
            events.insert(0, "File", file_name)

            cell_event_tables.append(events)

        valid_dff = dff_signal.dropna()

        if valid_dff.empty:
            peak_value = np.nan
            peak_time = np.nan
            auc_value = np.nan
        else:
            peak_index = dff_signal.idxmax()
            peak_value = float(dff_signal.loc[peak_index])
            peak_time = float(time_values.loc[peak_index])

            auc_value = calculate_auc_above_zero(
                time_values.to_numpy(),
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
                "Event count": len(events),
                "Status": "Included",
            }
        )

    dff_columns = [
        column for column in processed_df.columns
        if column.startswith("dF/F0 | ")
    ]

    if dff_columns:
        processed_df["Average dF/F0"] = processed_df[
            dff_columns
        ].mean(axis=1, skipna=True)

        processed_df["SEM dF/F0"] = processed_df[
            dff_columns
        ].sem(axis=1, ddof=1)

    else:
        processed_df["Average dF/F0"] = np.nan
        processed_df["SEM dF/F0"] = np.nan

    average_events = detect_calcium_events(
        time_values=processed_df["Time"].to_numpy(),
        dff_values=processed_df["Average dF/F0"].to_numpy(),
        settings=settings,
    )

    if not average_events.empty:
        average_events.insert(0, "Trace", "File average")
        average_events.insert(0, "Label", label)
        average_events.insert(0, "File", file_name)

    cell_summary_df = pd.DataFrame(cell_summary_rows)

    if cell_event_tables:
        cell_events_df = pd.concat(
            cell_event_tables,
            ignore_index=True,
        )
    else:
        cell_events_df = pd.DataFrame()

    return {
        "file_name": file_name,
        "label": label,
        "processed_df": processed_df,
        "cell_summary_df": cell_summary_df,
        "cell_events_df": cell_events_df,
        "average_events_df": average_events,
        "warnings": warnings,
    }


# -------------------------------------------------------------------
# Grouping and plotting
# -------------------------------------------------------------------

def build_file_average_long_table(results):
    """
    Make one long table containing the average trace from each file.
    """
    rows = []

    for result in results:
        processed = result["processed_df"]

        file_trace = pd.DataFrame(
            {
                "File": result["file_name"],
                "Label": result["label"],
                "Time": processed["Time"],
                "Average dF/F0": processed["Average dF/F0"],
            }
        )

        rows.append(file_trace)

    return pd.concat(rows, ignore_index=True)


def build_group_average_table(file_average_long):
    """
    Average file-level traces within each Label.

    This avoids treating every cell as an independent biological replicate.
    """
    grouped = (
        file_average_long
        .groupby(["Label", "Time"], as_index=False)
        .agg(
            Mean_dF_F0=("Average dF/F0", "mean"),
            SEM_dF_F0=("Average dF/F0", "sem"),
            N_files=("File", "nunique"),
        )
    )

    return grouped


def build_trace_plot(
    grouped_df,
    results,
    settings,
):
    """
    Plot label-level mean traces, optional SEM, and optional file-average peaks.
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

    labels = grouped_df["Label"].drop_duplicates().tolist()

    for index, label in enumerate(labels):
        color = palette[index % len(palette)]

        label_data = grouped_df[
            grouped_df["Label"] == label
        ].sort_values("Time")

        y_values = label_data["Mean_dF_F0"].to_numpy()

        if settings["positive_only"]:
            y_values = np.maximum(y_values, 0)

        fig.add_trace(
            go.Scatter(
                x=label_data["Time"],
                y=y_values,
                mode="lines",
                name=label,
                line=dict(
                    color=color,
                    width=3,
                ),
            )
        )

        if (
            settings["show_sem"]
            and label_data["N_files"].max() > 1
        ):
            sem_values = (
                label_data["SEM_dF_F0"]
                .fillna(0)
                .to_numpy()
            )

            upper = y_values + sem_values
            lower = y_values - sem_values

            if settings["positive_only"]:
                upper = np.maximum(upper, 0)
                lower = np.maximum(lower, 0)

            fig.add_trace(
                go.Scatter(
                    x=label_data["Time"],
                    y=upper,
                    mode="lines",
                    line=dict(
                        color=color,
                        width=0,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=label_data["Time"],
                    y=lower,
                    mode="lines",
                    line=dict(
                        color=color,
                        width=0,
                    ),
                    fill="tonexty",
                    fillcolor=(
                        f"rgba({int(color[1:3], 16)},"
                        f"{int(color[3:5], 16)},"
                        f"{int(color[5:7], 16)},0.18)"
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    if settings["show_events"]:
        for result in results:
            event_df = result["average_events_df"]

            if event_df.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=event_df["Peak time"],
                    y=event_df["Peak ΔF/F0"],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=10,
                        color="black",
                    ),
                    name=f"{result['label']} accepted peaks",
                    showlegend=False,
                    hovertemplate=(
                        "File: " + result["file_name"]
                        + "<br>Label: " + result["label"]
                        + "<br>Peak time: %{x:.3f}"
                        + "<br>Peak ΔF/F0: %{y:.3f}"
                        + "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        height=540,
        template="plotly_white",
        xaxis_title=f"Time ({settings['time_unit']})",
        yaxis_title="ΔF/F0",
        legend_title="Label",
    )

    return fig


# -------------------------------------------------------------------
# Export functions
# -------------------------------------------------------------------

def safe_excel_sheet_name(name, used_names):
    """Create unique Excel-compatible sheet names."""
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", str(name))
    cleaned = cleaned[:31] or "sheet"

    candidate = cleaned
    counter = 1

    while candidate in used_names:
        suffix = f"_{counter}"
        candidate = (
            cleaned[:31 - len(suffix)] + suffix
        )
        counter += 1

    used_names.add(candidate)

    return candidate


def make_excel_bytes(results, group_average_df, settings):
    """
    Create an Excel workbook with raw processed traces,
    cell summaries, and event metrics.
    """
    output = io.BytesIO()
    used_sheet_names = set()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        settings_df = pd.DataFrame([settings])
        settings_df.to_excel(
            writer,
            index=False,
            sheet_name="analysis_settings",
        )

        metadata_df = pd.DataFrame(
            [
                {
                    "File": result["file_name"],
                    "Label": result["label"],
                }
                for result in results
            ]
        )

        metadata_df.to_excel(
            writer,
            index=False,
            sheet_name="file_metadata",
        )

        group_average_df.to_excel(
            writer,
            index=False,
            sheet_name="group_average",
        )

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

            combined_summary.to_excel(
                writer,
                index=False,
                sheet_name="cell_summary",
            )

        event_tables = [
            result["cell_events_df"]
            for result in results
            if not result["cell_events_df"].empty
        ]

        if event_tables:
            combined_events = pd.concat(
                event_tables,
                ignore_index=True,
            )

            combined_events.to_excel(
                writer,
                index=False,
                sheet_name="cell_events",
            )

        average_event_tables = [
            result["average_events_df"]
            for result in results
            if not result["average_events_df"].empty
        ]

        if average_event_tables:
            combined_average_events = pd.concat(
                average_event_tables,
                ignore_index=True,
            )

            combined_average_events.to_excel(
                writer,
                index=False,
                sheet_name="average_events",
            )

        for result in results:
            sheet_name = safe_excel_sheet_name(
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


# -------------------------------------------------------------------
# Sidebar settings
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Time settings")

    frame_interval = st.number_input(
        "Frame interval",
        min_value=0.000001,
        value=1.0,
        step=0.1,
        help=(
            "Time between consecutive image frames. "
            "For example, use 2 if each frame was acquired every 2 seconds."
        ),
    )

    time_unit = st.selectbox(
        "Time unit",
        ["seconds", "minutes", "milliseconds"],
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
        help=(
            "Recommended: subtract Mean(background) from each cellular "
            "Mean trace at every frame before calculating F0 and ΔF/F0."
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

    st.header("Peak and rate settings")

    min_peak_height = st.number_input(
        "Minimum peak height (ΔF/F0)",
        min_value=0.0,
        value=0.05,
        step=0.01,
    )

    min_peak_prominence = st.number_input(
        "Minimum peak prominence (ΔF/F0)",
        min_value=0.0,
        value=0.05,
        step=0.01,
    )

    min_peak_distance = int(
        st.number_input(
            "Minimum frames between peaks",
            min_value=1,
            value=3,
            step=1,
        )
    )

    min_event_points = int(
        st.number_input(
            "Minimum points in rise/decay",
            min_value=2,
            value=3,
            step=1,
        )
    )

    st.divider()

    st.header("Display settings")

    show_sem = st.checkbox(
        "Show SEM across files sharing a label",
        value=True,
    )

    positive_only = st.checkbox(
        "Plot only positive ΔF/F0",
        value=False,
    )

    show_events = st.checkbox(
        "Mark accepted peaks",
        value=True,
    )

    st.divider()

    st.button(
        "Reset analysis / upload new files",
        on_click=reset_analysis,
        use_container_width=True,
    )


# -------------------------------------------------------------------
# File upload
# -------------------------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload calcium time-series files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    key=f"calcium_files_{st.session_state.uploader_token}",
    help=(
        "Expected format: Label, frame/time column, Mean1...MeanN, "
        "and Mean(background). Area columns are retained as metadata "
        "but are not treated as fluorescence signals."
    ),
)

if not uploaded_files:
    st.info(
        "Upload one or more calcium time-series files to begin analysis."
    )
    st.stop()


# -------------------------------------------------------------------
# Read uploads
# -------------------------------------------------------------------

loaded_files = []

for uploaded_file in uploaded_files:
    try:
        df = read_table(uploaded_file)

        loaded_files.append(
            {
                "file_name": uploaded_file.name,
                "df": df,
            }
        )

    except Exception as error:
        st.error(
            f"Could not read {uploaded_file.name}: {error}"
        )

if not loaded_files:
    st.stop()


# -------------------------------------------------------------------
# Show automatic detection
# -------------------------------------------------------------------

st.subheader("Detected column structure")

detection_rows = []

for item in loaded_files:
    detected = detect_columns(item["df"])

    detection_rows.append(
        {
            "File": item["file_name"],
            "Suggested label column": detected["label_col"],
            "Suggested frame/time column": detected["time_col"],
            "Suggested cell Mean columns": ", ".join(
                map(str, detected["signal_cols"])
            ),
            "Suggested background column": detected["background_col"],
            "Detected Area columns": ", ".join(
                map(str, detected["area_cols"])
            ),
        }
    )

st.dataframe(
    pd.DataFrame(detection_rows),
    use_container_width=True,
)


# -------------------------------------------------------------------
# Shared column mapping
# -------------------------------------------------------------------

first_df = loaded_files[0]["df"]
first_detected = detect_columns(first_df)
all_columns = first_df.columns.tolist()

with st.expander(
    "Column mapping (applied to all uploaded files)",
    expanded=True,
):
    mapping_col1, mapping_col2, mapping_col3, mapping_col4 = st.columns(4)

    with mapping_col1:
        label_options = ["<none>"] + all_columns

        default_label_index = 0

        if first_detected["label_col"] in all_columns:
            default_label_index = (
                all_columns.index(first_detected["label_col"]) + 1
            )

        selected_label_column = st.selectbox(
            "Label column",
            label_options,
            index=default_label_index,
        )

    with mapping_col2:
        time_options = ["<auto index>"] + all_columns

        default_time_index = 0

        if first_detected["time_col"] in all_columns:
            default_time_index = (
                all_columns.index(first_detected["time_col"]) + 1
            )

        selected_time_column = st.selectbox(
            "Frame/time column",
            time_options,
            index=default_time_index,
        )

    with mapping_col3:
        background_options = ["<none>"] + all_columns

        default_background_index = 0

        if first_detected["background_col"] in all_columns:
            default_background_index = background_options.index(
                first_detected["background_col"]
            )

        selected_background_column = st.selectbox(
            "Empty-area background column",
            background_options,
            index=default_background_index,
        )

    with mapping_col4:
        selected_signal_columns = st.multiselect(
            "Cell signal columns",
            all_columns,
            default=first_detected["signal_cols"],
        )


if (
    background_mode == "Subtract empty-area background before ΔF/F0"
    and selected_background_column == "<none>"
):
    st.error(
        "Select Mean(background) as the empty-area background column, "
        "or select 'No background correction'."
    )
    st.stop()

if not selected_signal_columns:
    st.error("Select at least one cellular fluorescence signal column.")
    st.stop()


# -------------------------------------------------------------------
# Build settings and validate mapping
# -------------------------------------------------------------------

settings = {
    "label_column": (
        None
        if selected_label_column == "<none>"
        else selected_label_column
    ),
    "time_column": (
        None
        if selected_time_column == "<auto index>"
        else selected_time_column
    ),
    "background_column": (
        None
        if selected_background_column == "<none>"
        else selected_background_column
    ),
    "signal_columns": selected_signal_columns,
    "frame_interval": frame_interval,
    "time_unit": time_unit,
    "background_mode": background_mode,
    "f0_mode": f0_mode,
    "f0_n": f0_n,
    "baseline_start": baseline_start,
    "baseline_end": baseline_end,
    "min_peak_height": min_peak_height,
    "min_peak_prominence": min_peak_prominence,
    "min_peak_distance": min_peak_distance,
    "min_event_points": min_event_points,
    "show_sem": show_sem,
    "positive_only": positive_only,
    "show_events": show_events,
}

mapping_errors = []

for item in loaded_files:
    file_name = item["file_name"]
    df = item["df"]

    needed_columns = list(selected_signal_columns)

    if settings["background_column"] is not None:
        needed_columns.append(settings["background_column"])

    if settings["time_column"] is not None:
        needed_columns.append(settings["time_column"])

    if settings["label_column"] is not None:
        needed_columns.append(settings["label_column"])

    missing_columns = [
        column for column in needed_columns
        if column not in df.columns
    ]

    if missing_columns:
        mapping_errors.append(
            f"{file_name}: missing columns: "
            f"{', '.join(map(str, missing_columns))}"
        )

if mapping_errors:
    st.error(
        "The selected column mapping cannot be applied to every file:\n\n"
        + "\n".join(mapping_errors)
    )
    st.stop()


# -------------------------------------------------------------------
# Process all files
# -------------------------------------------------------------------

results = []

for item in loaded_files:
    result = process_calcium_file(
        df=item["df"],
        file_name=item["file_name"],
        settings=settings,
    )

    results.append(result)

all_warnings = []

for result in results:
    all_warnings.extend(result["warnings"])

for warning in all_warnings:
    st.warning(warning)


# -------------------------------------------------------------------
# Build averages and plot
# -------------------------------------------------------------------

file_average_long = build_file_average_long_table(results)

group_average_df = build_group_average_table(
    file_average_long
)

trace_plot = build_trace_plot(
    grouped_df=group_average_df,
    results=results,
    settings=settings,
)

st.subheader("Average calcium traces")

st.plotly_chart(
    trace_plot,
    use_container_width=True,
)


# -------------------------------------------------------------------
# Processed-file preview
# -------------------------------------------------------------------

st.subheader("Processed data preview")

selected_file_name = st.selectbox(
    "Select uploaded file",
    [result["file_name"] for result in results],
)

selected_result = next(
    result
    for result in results
    if result["file_name"] == selected_file_name
)

st.caption(
    f"Extracted label: {selected_result['label']}"
)

st.dataframe(
    selected_result["processed_df"],
    use_container_width=True,
    height=350,
)


# -------------------------------------------------------------------
# Cell/ROI summaries
# -------------------------------------------------------------------

st.subheader("Cell/ROI summary")

cell_summary_tables = [
    result["cell_summary_df"]
    for result in results
    if not result["cell_summary_df"].empty
]

if cell_summary_tables:
    combined_cell_summary = pd.concat(
        cell_summary_tables,
        ignore_index=True,
    )

    st.dataframe(
        combined_cell_summary,
        use_container_width=True,
    )

else:
    combined_cell_summary = pd.DataFrame()

    st.info("No valid cell/ROI summaries were generated.")


# -------------------------------------------------------------------
# Events and uptake/release rates
# -------------------------------------------------------------------

st.subheader("Significant calcium events and kinetic rates")

st.markdown(
    "- **Uptake slope:** linear-regression slope from the last zero/baseline "
    "crossing before the accepted peak to the peak.\n"
    "- **Release slope:** linear-regression slope from the peak to the first "
    "return to zero or below. Release slopes are expected to be negative.\n"
    "- **Release magnitude:** the positive value of the negative release slope.\n"
    "- **No return to baseline:** the event remains in the output, but its "
    "release metrics are not calculated.\n"
    "- **Accepted peaks:** must meet your selected height, prominence, "
    "minimum separation, and minimum segment-length criteria."
)

cell_event_tables = [
    result["cell_events_df"]
    for result in results
    if not result["cell_events_df"].empty
]

average_event_tables = [
    result["average_events_df"]
    for result in results
    if not result["average_events_df"].empty
]

if cell_event_tables:
    combined_cell_events = pd.concat(
        cell_event_tables,
        ignore_index=True,
    )
else:
    combined_cell_events = pd.DataFrame()

if average_event_tables:
    combined_average_events = pd.concat(
        average_event_tables,
        ignore_index=True,
    )
else:
    combined_average_events = pd.DataFrame()

events_tab1, events_tab2 = st.tabs(
    [
        "Individual cell/ROI events",
        "File-average events",
    ]
)

with events_tab1:
    if combined_cell_events.empty:
        st.info(
            "No individual cell/ROI events met the current significance criteria."
        )
    else:
        st.dataframe(
            combined_cell_events,
            use_container_width=True,
        )

with events_tab2:
    if combined_average_events.empty:
        st.info(
            "No file-average events met the current significance criteria."
        )
    else:
        st.dataframe(
            combined_average_events,
            use_container_width=True,
        )


# -------------------------------------------------------------------
# Downloads
# -------------------------------------------------------------------

st.subheader("Downloads")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

excel_bytes = make_excel_bytes(
    results=results,
    group_average_df=group_average_df,
    settings=settings,
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
        "Download group averages CSV",
        data=group_average_df.to_csv(index=False).encode("utf-8"),
        file_name=f"calcium_group_average_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_col3:
    st.download_button(
        "Download cell summary CSV",
        data=combined_cell_summary.to_csv(index=False).encode("utf-8"),
        file_name=f"calcium_cell_summary_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_col4:
    st.download_button(
        "Download event metrics CSV",
        data=combined_cell_events.to_csv(index=False).encode("utf-8"),
        file_name=f"calcium_events_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.download_button(
    "Download interactive graph HTML",
    data=trace_plot.to_html(include_plotlyjs="cdn"),
    file_name=f"calcium_trace_{timestamp}.html",
    mime="text/html",
)
