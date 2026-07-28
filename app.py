
import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Calcium Analysis", layout="wide")

st.title("Calcium Imaging Post-Analysis")
st.write("Upload one control file and one treatment file. The app computes average ΔF/F0 traces, peak time, fade times, and AUC.")

if "uploader_token" not in st.session_state:
    st.session_state.uploader_token = 0


def reset_analysis():
    keys_to_clear = [
        "control_time_col", "control_cell_cols", "treatment_time_col", "treatment_cell_cols",
        "f0_mode", "f0_n", "add_stats"
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.uploader_token += 1


with st.sidebar:
    st.header("Analysis")
    f0_mode = st.selectbox(
        "F0 definition",
        ["First row", "Mean of first N rows"],
        key="f0_mode"
    )
    f0_n = 5
    if f0_mode == "Mean of first N rows":
        f0_n = st.number_input("Number of rows for F0", min_value=1, value=5, step=1, key="f0_n")
    add_stats = st.checkbox("Add statistics (SEM on graph, p-values)", value=False, key="add_stats")
    st.button("Reset analysis / Upload new files", on_click=reset_analysis, use_container_width=True)


def read_table(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def get_f0(series, mode, n_rows):
    s = pd.to_numeric(series, errors='coerce')
    if mode == "First row":
        return float(s.iloc[0])
    return float(s.iloc[:n_rows].mean())


def compute_dff(df, time_col, cell_cols, mode, n_rows):
    time_vals = pd.to_numeric(df[time_col], errors='coerce')
    out = pd.DataFrame({"Time": time_vals})
    for col in cell_cols:
        vals = pd.to_numeric(df[col], errors='coerce')
        f0 = get_f0(vals, mode, n_rows)
        if pd.isna(f0) or f0 == 0:
            out[col] = np.nan
        else:
            out[col] = (vals - f0) / f0
    out = out.dropna(subset=["Time"]).reset_index(drop=True)
    return out


def summarize_average_trace(avg_df):
    x = pd.to_numeric(avg_df["Time"], errors='coerce').to_numpy()
    y = pd.to_numeric(avg_df["Average_dF_F0"], errors='coerce').to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return {}
    peak_idx = int(np.nanargmax(y))
    peak_x = float(x[peak_idx])
    peak_y = float(y[peak_idx])

    fade_a_x = np.nan
    fade_a_y = np.nan
    for i in range(peak_idx + 1, len(y)):
        if y[i] < y[i - 1]:
            fade_a_x = float(x[i])
            fade_a_y = float(y[i])
            break

    target_b = peak_y * 0.5
    fade_b_x = np.nan
    fade_b_y = np.nan
    for i in range(peak_idx + 1, len(y)):
        if y[i] <= target_b:
            fade_b_x = float(x[i])
            fade_b_y = float(y[i])
            break

    fade_c_x = np.nan
    fade_c_y = np.nan
    for i in range(peak_idx + 1, len(y)):
        if y[i] <= 0:
            fade_c_x = float(x[i])
            fade_c_y = float(y[i])
            break

    # AUC: trapezoidal integration over the entire trace
    auc_val = float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))

    return {
        "Peak time": peak_x,
        "Peak value": peak_y,
        "Fade A time": fade_a_x,
        "Fade A value": fade_a_y,
        "Fade B time": fade_b_x,
        "Fade B value": fade_b_y,
        "Fade C time": fade_c_x,
        "Fade C value": fade_c_y,
        "AUC": auc_val,
    }


def make_avg_trace(dff_df):
    cell_cols = [c for c in dff_df.columns if c != "Time"]
    avg = dff_df[cell_cols].mean(axis=1, skipna=True)
    return pd.DataFrame({"Time": dff_df["Time"], "Average_dF_F0": avg})


def build_single_plot_with_sem(avg_df, sem_df, label, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=avg_df["Time"], y=avg_df["Average_dF_F0"],
        mode="lines", name=label,
        line=dict(color=color, width=3)
    ))
    upper = avg_df["Average_dF_F0"] + sem_df["SEM"]
    lower = avg_df["Average_dF_F0"] - sem_df["SEM"]
    fig.add_trace(go.Scatter(
        x=avg_df["Time"], y=upper,
        mode="lines", name=f"{label} + SEM",
        line=dict(color=color, width=0, dash="dot"),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=avg_df["Time"], y=lower,
        mode="lines", name=f"{label} - SEM",
        line=dict(color=color, width=0, dash="dot"),
        fill="tonexty",
        fillcolor=f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.2)",
        showlegend=False
    ))
    fig.update_layout(height=420, xaxis_title="Time", yaxis_title="Average ΔF/F0", template="plotly_white")
    return fig


def build_overlay_plot_with_sem(control_avg, treatment_avg, control_sem, treatment_sem):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=control_avg["Time"], y=control_avg["Average_dF_F0"],
        mode="lines", name="Control",
        line=dict(color="#1f77b4", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=treatment_avg["Time"], y=treatment_avg["Average_dF_F0"],
        mode="lines", name="Treatment",
        line=dict(color="#d62728", width=3)
    ))
    ctrl_upper = control_avg["Average_dF_F0"] + control_sem["SEM"]
    ctrl_lower = control_avg["Average_dF_F0"] - control_sem["SEM"]
    fig.add_trace(go.Scatter(
        x=control_avg["Time"], y=ctrl_upper,
        mode="lines", name="Control + SEM",
        line=dict(color="#1f77b4", width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=control_avg["Time"], y=ctrl_lower,
        mode="lines", name="Control - SEM",
        line=dict(color="#1f77b4", width=0),
        fill="tonexty",
        fillcolor="rgba(31,119,180,0.2)",
        showlegend=False
    ))
    trt_upper = treatment_avg["Average_dF_F0"] + treatment_sem["SEM"]
    trt_lower = treatment_avg["Average_dF_F0"] - treatment_sem["SEM"]
    fig.add_trace(go.Scatter(
        x=treatment_avg["Time"], y=trt_upper,
        mode="lines", name="Treatment + SEM",
        line=dict(color="#d62728", width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=treatment_avg["Time"], y=trt_lower,
        mode="lines", name="Treatment - SEM",
        line=dict(color="#d62728", width=0),
        fill="tonexty",
        fillcolor="rgba(214,39,40,0.2)",
        showlegend=False
    ))
    fig.update_layout(height=480, xaxis_title="Time", yaxis_title="Average ΔF/F0", template="plotly_white")
    return fig


def metrics_table(control_summary, treatment_summary):
    rows = []
    for cond, summary in [("Control", control_summary), ("Treatment", treatment_summary)]:
        for metric_name in ["Peak", "Fade A", "Fade B", "Fade C", "AUC"]:
            if metric_name in ["Peak", "AUC"]:
                rows.append({
                    "Condition": cond,
                    "Metric": metric_name,
                    "X_time": np.nan if metric_name == "AUC" else summary.get("Peak time", np.nan),
                    "Y_value": summary.get("Peak value" if metric_name == "Peak" else "AUC", np.nan),
                })
            else:
                rows.append({
                    "Condition": cond,
                    "Metric": metric_name,
                    "X_time": summary.get(f"{metric_name} time", np.nan),
                    "Y_value": summary.get(f"{metric_name} value", np.nan),
                })
    return pd.DataFrame(rows)


def to_excel_bytes(control_dff, treatment_dff, control_avg, treatment_avg, metric_df):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        control_dff.to_excel(writer, index=False, sheet_name="control_dff")
        treatment_dff.to_excel(writer, index=False, sheet_name="treatment_dff")
        control_avg.to_excel(writer, index=False, sheet_name="control_average")
        treatment_avg.to_excel(writer, index=False, sheet_name="treatment_average")
        metric_df.to_excel(writer, index=False, sheet_name="metrics")
    bio.seek(0)
    return bio.getvalue()


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


col1, col2 = st.columns(2)
with col1:
    control_file = st.file_uploader("Upload control file", type=["csv", "xlsx", "xls"], key=f"control_{st.session_state.uploader_token}")
with col2:
    treatment_file = st.file_uploader("Upload treatment file", type=["csv", "xlsx", "xls"], key=f"treatment_{st.session_state.uploader_token}")

if control_file and treatment_file:
    control_df = read_table(control_file)
    treatment_df = read_table(treatment_file)

    st.subheader("Column selection")
    c1, c2 = st.columns(2)
    with c1:
        control_time_col = st.selectbox("Control time column", control_df.columns.tolist(), key="control_time_col")
        control_cell_cols = st.multiselect(
            "Control cell columns",
            [c for c in control_df.columns if c != control_time_col],
            default=[c for c in control_df.columns if c != control_time_col],
            key="control_cell_cols"
        )
    with c2:
        treatment_time_col = st.selectbox("Treatment time column", treatment_df.columns.tolist(), key="treatment_time_col")
        treatment_cell_cols = st.multiselect(
            "Treatment cell columns",
            [c for c in treatment_df.columns if c != treatment_time_col],
            default=[c for c in treatment_df.columns if c != treatment_time_col],
            key="treatment_cell_cols"
        )

    if control_cell_cols and treatment_cell_cols:
        control_dff = compute_dff(control_df, control_time_col, control_cell_cols, f0_mode, int(f0_n))
        treatment_dff = compute_dff(treatment_df, treatment_time_col, treatment_cell_cols, f0_mode, int(f0_n))

        control_avg = make_avg_trace(control_dff)
        treatment_avg = make_avg_trace(treatment_dff)

        control_summary = summarize_average_trace(control_avg)
        treatment_summary = summarize_average_trace(treatment_avg)
        metric_df = metrics_table(control_summary, treatment_summary)

        # Compute SEM
        sem_control = control_dff[control_cell_cols].apply(lambda col: col.std(ddof=1) / np.sqrt(len(col)), axis=1)
        sem_control = pd.DataFrame({"Time": control_dff["Time"], "SEM": sem_control})
        sem_treatment = treatment_dff[treatment_cell_cols].apply(lambda col: col.std(ddof=1) / np.sqrt(len(col)), axis=1)
        sem_treatment = pd.DataFrame({"Time": treatment_dff["Time"], "SEM": sem_treatment})

        if add_stats:
            st.subheader("Average trace: Control (with SEM)")
            fig_control = build_single_plot_with_sem(control_avg, sem_control, "Control", "#1f77b4")
            st.plotly_chart(fig_control, use_container_width=True)

            st.subheader("Average trace: Treatment (with SEM)")
            fig_treatment = build_single_plot_with_sem(treatment_avg, sem_treatment, "Treatment", "#d62728")
            st.plotly_chart(fig_treatment, use_container_width=True)

            st.subheader("Overlay: Control vs Treatment (with SEM)")
            fig_overlay = build_overlay_plot_with_sem(control_avg, treatment_avg, sem_control, sem_treatment)
            st.plotly_chart(fig_overlay, use_container_width=True)
        else:
            st.subheader("Average trace: Control")
            fig_control = go.Figure()
            fig_control.add_trace(go.Scatter(x=control_avg["Time"], y=control_avg["Average_dF_F0"], mode="lines", name="Control", line=dict(color="#1f77b4", width=3)))
            fig_control.update_layout(height=420, xaxis_title="Time", yaxis_title="Average ΔF/F0", template="plotly_white")
            st.plotly_chart(fig_control, use_container_width=True)

            st.subheader("Average trace: Treatment")
            fig_treatment = go.Figure()
            fig_treatment.add_trace(go.Scatter(x=treatment_avg["Time"], y=treatment_avg["Average_dF_F0"], mode="lines", name="Treatment", line=dict(color="#d62728", width=3)))
            fig_treatment.update_layout(height=420, xaxis_title="Time", yaxis_title="Average ΔF/F0", template="plotly_white")
            st.plotly_chart(fig_treatment, use_container_width=True)

            st.subheader("Overlay: Control vs Treatment")
            fig_overlay = go.Figure()
            fig_overlay.add_trace(go.Scatter(x=control_avg["Time"], y=control_avg["Average_dF_F0"], mode="lines", name="Control", line=dict(color="#1f77b4", width=3)))
            fig_overlay.add_trace(go.Scatter(x=treatment_avg["Time"], y=treatment_avg["Average_dF_F0"], mode="lines", name="Treatment", line=dict(color="#d62728", width=3)))
            fig_overlay.update_layout(height=480, xaxis_title="Time", yaxis_title="Average ΔF/F0", template="plotly_white")
            st.plotly_chart(fig_overlay, use_container_width=True)

        st.subheader("Peak, fade, and AUC table")
        st.write("**Fade definitions:**")
        st.markdown(
            "- **Fade A**: first time point after the peak where the average trace decreases.\\n"
            "- **Fade B**: first time point after the peak where the average trace reaches 50% of peak.\\n"
            "- **Fade C**: first time point after the peak where the average trace reaches 0 or below.\\n"
            "- **AUC**: area under the ΔF/F0 vs time curve, calculated by trapezoidal integration over the entire trace."
        )
        st.dataframe(metric_df, use_container_width=True)

        if add_stats:
            st.subheader("Statistical comparison (peak values)")
            ctrl_peak_vals = control_dff[control_cell_cols].max().dropna()
            trt_peak_vals = treatment_dff[treatment_cell_cols].max().dropna()
            if len(ctrl_peak_vals) > 1 and len(trt_peak_vals) > 1:
                try:
                    t_stat, p_val = stats.ttest_ind(ctrl_peak_vals, trt_peak_vals, equal_var=False)
                except Exception:
                    t_stat, p_val = np.nan, np.nan
                st.write(f"Peak value comparison (Welch's t-test): t = {t_stat:.3f}, p = {p_val:.4f}")
            else:
                st.write("Not enough cells to compute a meaningful peak-value comparison.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_data = to_excel_bytes(control_dff, treatment_dff, control_avg, treatment_avg, metric_df)

        cdl1, cdl2, cdl3, cdl4 = st.columns(4)
        with cdl1:
            st.download_button("Download metrics CSV", data=csv_bytes(metric_df), file_name=f"metrics_{ts}.csv", mime="text/csv", use_container_width=True)
        with cdl2:
            st.download_button("Download control avg CSV", data=csv_bytes(control_avg), file_name=f"control_average_{ts}.csv", mime="text/csv", use_container_width=True)
        with cdl3:
            st.download_button("Download treatment avg CSV", data=csv_bytes(treatment_avg), file_name=f"treatment_average_{ts}.csv", mime="text/csv", use_container_width=True)
        with cdl4:
            st.download_button("Download Excel workbook", data=excel_data, file_name=f"calcium_analysis_{ts}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        html_control = fig_control.to_html(include_plotlyjs='cdn')
        html_treatment = fig_treatment.to_html(include_plotlyjs='cdn')
        html_overlay = fig_overlay.to_html(include_plotlyjs='cdn')

        g1, g2, g3 = st.columns(3)
        with g1:
            st.download_button("Download control graph HTML", data=html_control, file_name=f"control_graph_{ts}.html", mime="text/html", use_container_width=True)
        with g2:
            st.download_button("Download treatment graph HTML", data=html_treatment, file_name=f"treatment_graph_{ts}.html", mime="text/html", use_container_width=True)
        with g3:
            st.download_button("Download overlay graph HTML", data=html_overlay, file_name=f"overlay_graph_{ts}.html", mime="text/html", use_container_width=True)
    else:
        st.warning("Please select at least one cell column for each condition.")
else:
    st.info("Upload both control and treatment files to start the analysis.")
