import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Calcium Post Analysis", layout="wide")
st.title("Calcium Imaging Post-Analysis")
st.caption("Upload an Excel or CSV file with time and mean intensity values for each cell. F0 is fixed as the first image for each cell.")

with st.expander("Expected input format", expanded=True):
    st.markdown("""
    Your file should be in **wide format** with:
    - one time column, for example: `time`, `Time`, `frame`, or `Frame`
    - one column per cell, for example: `cell_1`, `cell_2`, `cell_3`

    Example:

    | time | cell_1 | cell_2 | cell_3 |
    |---|---:|---:|---:|
    | 0 | 120 | 98 | 115 |
    | 1 | 130 | 105 | 121 |
    | 2 | 155 | 120 | 140 |
    """)

uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])


def load_table(file):
    name = file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(file)
    return pd.read_excel(file)


def auc_trapz(y, x):
    return float(np.trapz(y, x))


def first_fade_time(y, x, peak_idx, threshold_fraction):
    peak = y[peak_idx]
    threshold = peak * threshold_fraction
    for i in range(peak_idx + 1, len(y)):
        if y[i] <= threshold:
            return x[i]
    return np.nan


if uploaded is not None:
    df = load_table(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head())

    cols = list(df.columns)
    default_time = None
    for cand in ["time", "Time", "frame", "Frame", "t"]:
        if cand in cols:
            default_time = cand
            break
    time_col = st.selectbox("Time column", cols, index=cols.index(default_time) if default_time else 0)
    cell_cols = st.multiselect("Cell intensity columns", [c for c in cols if c != time_col], default=[c for c in cols if c != time_col])

    if cell_cols:
        subtract_bg = st.checkbox("Subtract background column", value=False)
        bg_col = None
        if subtract_bg:
            bg_options = [c for c in cols if c != time_col and c not in cell_cols] + cell_cols
            bg_col = st.selectbox("Background column", bg_options)

        st.subheader("Metric settings")
        peak_window = st.slider("Peak search start row", min_value=0, max_value=max(len(df)-1, 0), value=0)
        fade_fraction = st.slider("Fade threshold as fraction of peak", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

        time_vals = pd.to_numeric(df[time_col], errors='coerce').to_numpy()
        metrics = []
        traces = []

        for cell in cell_cols:
            y_raw = pd.to_numeric(df[cell], errors='coerce').to_numpy(dtype=float)
            if subtract_bg and bg_col:
                bg = pd.to_numeric(df[bg_col], errors='coerce').to_numpy(dtype=float)
                y = y_raw - bg
            else:
                y = y_raw.copy()

            f0 = y[0]
            if pd.isna(f0) or f0 == 0:
                dff = np.full_like(y, np.nan, dtype=float)
            else:
                dff = (y - f0) / f0

            peak_search = dff[int(peak_window):]
            if len(peak_search) == 0 or np.all(np.isnan(peak_search)):
                peak_time = np.nan
                peak_val = np.nan
                fade_time = np.nan
                auc_val = np.nan
            else:
                local_idx = int(np.nanargmax(peak_search))
                peak_idx = local_idx + int(peak_window)
                peak_time = time_vals[peak_idx]
                peak_val = dff[peak_idx]
                fade_time = first_fade_time(dff, time_vals, peak_idx, fade_fraction)
                auc_val = auc_trapz(np.nan_to_num(dff, nan=0.0), time_vals)

            latency = peak_time - time_vals[0] if np.isfinite(peak_time) else np.nan

            metrics.append({
                'cell': cell,
                'F0_first_image': f0,
                'peak_dF_F0': peak_val,
                'time_to_peak': peak_time,
                'latency_from_start': latency,
                'fade_time': fade_time,
                'AUC': auc_val,
            })

            traces.append(pd.DataFrame({'time': time_vals, 'cell': cell, 'dF_F0': dff}))

        metrics_df = pd.DataFrame(metrics)
        traces_df = pd.concat(traces, ignore_index=True)
        avg_df = traces_df.groupby('time', as_index=False)['dF_F0'].mean()

        st.subheader("Average trace")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=avg_df['time'], y=avg_df['dF_F0'], mode='lines', name='Average dF/F0'))
        fig.update_layout(xaxis_title='Time', yaxis_title='dF/F0', height=450)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show individual cell traces"):
            fig2 = px.line(traces_df, x='time', y='dF_F0', color='cell')
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Per-cell metrics")
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("Average metrics across selected cells")
        summary = metrics_df[["peak_dF_F0", "time_to_peak", "latency_from_start", "fade_time", "AUC"]].mean(numeric_only=True).to_frame("mean_value")
        st.dataframe(summary)

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            traces_df.to_excel(writer, sheet_name='traces', index=False)
            metrics_df.to_excel(writer, sheet_name='cell_metrics', index=False)
            avg_df.to_excel(writer, sheet_name='average_trace', index=False)
            summary.to_excel(writer, sheet_name='summary')
        st.download_button('Download analysis workbook', data=out.getvalue(), file_name='calcium_post_analysis.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
else:
    st.info('Upload a file to begin.')
