import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Calcium Analysis", layout="wide")
st.title("Calcium Imaging Post-Analysis")
st.caption("Manual P1–P4 event curation; event AUC is defined on the file-average trace and applied to every ROI.")

for k, v in {"uploader_token": 0, "trace_events": {}}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_analysis():
    token = st.session_state.get("uploader_token", 0) + 1
    st.session_state.clear()
    st.session_state["uploader_token"] = token
    st.session_state["trace_events"] = {}


def read_table(f):
    return pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)


def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def detect_columns(df):
    cols = list(df.columns)
    n = {c: norm(c) for c in cols}
    labels = [c for c in cols if n[c] in {"label", "condition", "sample", "experiment", "group", "treatment"}]
    bgs = [c for c in cols if "background" in n[c] and "mean" in n[c]]
    if not bgs:
        bgs = [c for c in cols if n[c] in {"background", "bg", "meanbg"}]
    bg = bgs[0] if bgs else None
    signals = [c for c in cols if c != bg and re.fullmatch(r"mean\d+", n[c])]
    areas = [c for c in cols if n[c].startswith("area")]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    excluded = set(signals + areas + ([bg] if bg else []) + ([labels[0]] if labels else []))
    candidates = [c for c in numeric if c not in excluded]
    return {"label": labels[0] if labels else None, "background": bg, "signals": signals, "time": candidates[0] if candidates else None}


def first_valid(s):
    v = s.dropna()
    return v.iloc[0] if not v.empty else np.nan


def make_time(df, time_col, interval):
    frames = pd.to_numeric(df[time_col], errors="coerce") if time_col in df.columns else pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    frames = frames.ffill().bfill() if frames.notna().sum() >= 2 else pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    return frames, (frames - frames.iloc[0]) * interval


def get_f0(x, t, cfg):
    x = pd.to_numeric(x, errors="coerce")
    valid = x.dropna()
    if valid.empty:
        return np.nan
    mode = cfg["f0_mode"]
    if mode == "First valid value": return float(valid.iloc[0])
    if mode == "Mean of first N valid rows": return float(valid.iloc[:max(1, cfg["f0_n"])].mean())
    if mode == "Minimum value in trace": return float(valid.min())
    if mode == "Lower quartile (25th percentile)": return float(valid.quantile(.25))
    if mode == "Mean within baseline-time window":
        q = x[(t >= cfg["baseline_start"]) & (t <= cfg["baseline_end"])].dropna()
        return float(q.mean()) if not q.empty else np.nan
    return float(valid.iloc[0])


def dff(raw, bg, t, cfg):
    raw = pd.to_numeric(raw, errors="coerce")
    corrected = raw - pd.to_numeric(bg, errors="coerce") if cfg["background_mode"] == "Subtract empty-area background before ΔF/F0" else raw.copy()
    if cfg["negative_policy"] == "Clip corrected fluorescence below zero to zero": corrected = corrected.clip(lower=0)
    if cfg["negative_policy"] == "Exclude ROI if any corrected value is below zero" and (corrected < 0).any():
        return corrected, pd.Series(np.nan, index=corrected.index), np.nan
    f0 = get_f0(corrected, t, cfg)
    if not np.isfinite(f0) or f0 <= 0:
        return corrected, pd.Series(np.nan, index=corrected.index), f0
    return corrected, (corrected - f0) / f0, f0


def positive_auc(t, y):
    x, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2: return np.nan
    return float(np.trapezoid(np.maximum(y, 0), x)) if hasattr(np, "trapezoid") else float(np.trapz(np.maximum(y, 0), x))


def process_file(df, filename, cfg):
    d = detect_columns(df)
    frames, time = make_time(df, cfg["time_column"], cfg["frame_interval"])
    label = str(first_valid(df[cfg["label_column"]])) if cfg["label_column"] in df.columns else re.sub(r"\.[^.]+$", "", filename)
    if not label or label.lower() == "nan": label = re.sub(r"\.[^.]+$", "", filename)
    out = pd.DataFrame({"Frame": frames, "Time": time})
    bg = pd.to_numeric(df[cfg["background_column"]], errors="coerce") if cfg["background_column"] in df.columns else pd.Series(0., index=df.index)
    if cfg["background_column"] in df.columns: out["Background"] = bg
    summary, warnings = [], []
    for signal in d["signals"]:
        raw = pd.to_numeric(df[signal], errors="coerce")
        corrected, trace, f0 = dff(raw, bg, time, cfg)
        out[f"Raw | {signal}"] = raw
        out[f"Corrected | {signal}"] = corrected
        out[f"dF/F0 | {signal}"] = trace
        if not np.isfinite(f0) or f0 <= 0:
            peak, peak_time, auc, status = np.nan, np.nan, np.nan, "Excluded: invalid corrected F0"
            warnings.append(f"{filename} — {signal}: invalid corrected F0; ΔF/F0 was not calculated.")
        else:
            status = "Included"; valid = trace.dropna()
            if valid.empty: peak, peak_time, auc = np.nan, np.nan, np.nan
            else:
                i = trace.idxmax(); peak, peak_time, auc = float(trace.loc[i]), float(time.loc[i]), positive_auc(time, trace)
        summary.append({"File": filename, "Label": label, "ROI": signal, "F0 corrected": f0, "Peak ΔF/F0": peak, "Peak time": peak_time, "Whole-trace positive AUC": auc, "Status": status})
    dcols = [c for c in out if c.startswith("dF/F0 | ")]
    out["Average dF/F0"] = out[dcols].mean(axis=1, skipna=True) if dcols else np.nan
    out["SEM dF/F0"] = out[dcols].sem(axis=1, ddof=1) if dcols else np.nan
    return {"file": filename, "label": label, "processed": out, "summary": pd.DataFrame(summary), "warnings": warnings}


def tkey(file, trace): return f"{file}::{trace}"

def empty_event(): return {"p1": None, "p2": None, "p3": None, "p4": None, "stimulation_label": "", "endpoint_type": "Post-peak trough", "notes": ""}

def init_events(key, n):
    store = st.session_state.trace_events.setdefault(key, {})
    for i in range(1, n + 1): store.setdefault(f"E{i}", empty_event())
    for event_id in list(store):
        if event_id not in {f"E{i}" for i in range(1, n + 1)}: del store[event_id]

def picked_point(df, trace, frame):
    if frame is None: return None
    rows = df[np.isclose(pd.to_numeric(df["Frame"], errors="coerce"), float(frame), equal_nan=False)]
    if rows.empty: return None
    r = rows.iloc[0]
    if not np.isfinite(r[trace]): return None
    return {"frame": int(r["Frame"]), "time": float(r["Time"]), "value": float(r[trace])}

def add_line(fig, a, b, color, dash, name):
    if a is None or b is None: return
    fig.add_trace(go.Scatter(x=[a["time"], b["time"]], y=[a["value"], b["value"]], mode="lines+markers+text", line=dict(color=color, width=3, dash=dash), marker=dict(color=color, size=11), text=[name.split("→")[0], name.split("→")[1]], textposition="top center", showlegend=False))

def manual_plot(df, trace, result, events, active, positive, unit):
    y = df[trace].to_numpy(float); y = np.maximum(y, 0) if positive else y
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Time"], y=y, mode="lines+markers", line=dict(color="#1f77b4", width=2.5), marker=dict(size=6), name=f"{result['label']} | {result['file']} | {trace}", customdata=df["Frame"], hovertemplate="Frame: %{customdata}<br>Time: %{x:.4f}<br>ΔF/F0: %{y:.4f}<extra></extra>"))
    colors = {"p1": "green", "p2": "orange", "p3": "red", "p4": "purple"}
    for eid, ev in events.items():
        opacity = 1 if eid == active else .40
        for left, right, col, dash in [("p1", "p2", "green", "solid"), ("p2", "p3", "orange", "dot"), ("p3", "p4", "purple", "dash")]:
            a, b = ev[left], ev[right]
            if a is not None and b is not None:
                fig.add_trace(go.Scatter(x=[a["time"], b["time"]], y=[a["value"], b["value"]], mode="lines+markers+text", line=dict(color=col, width=3, dash=dash), opacity=opacity, marker=dict(color=col, size=10), text=[f"{eid} {left.upper()}", f"{eid} {right.upper()}"], textposition="top center", showlegend=False))
        for p, point in [(p, ev[p]) for p in ["p1", "p2", "p3", "p4"]]:
            if point is not None:
                fig.add_trace(go.Scatter(x=[point["time"]], y=[point["value"]], mode="markers+text", marker=dict(color=colors[p], size=15 if eid == active else 10), text=[f"{eid} {p.upper()}"], textposition="bottom center", showlegend=False, opacity=opacity))
    fig.update_layout(height=650, template="plotly_white", dragmode="select", xaxis_title=f"Time ({unit})", yaxis_title="ΔF/F0")
    return fig

def one_event_metrics(eid, ev):
    p1,p2,p3,p4 = ev["p1"],ev["p2"],ev["p3"],ev["p4"]
    if any(x is None for x in [p1,p2,p3,p4]): return {"Event ID": eid, "Status": "Incomplete"}
    if not (p1["time"] < p2["time"] < p3["time"] < p4["time"]): return {"Event ID": eid, "Status": "Invalid point order"}
    fd,td,dd = p2["time"]-p1["time"],p3["time"]-p1["time"],p4["time"]-p3["time"]
    fa,ta,da = p2["value"]-p1["value"],p3["value"]-p1["value"],p4["value"]-p3["value"]
    dr = da/dd if dd>0 else np.nan
    return {"Event ID":eid,"Stimulation label":ev.get("stimulation_label",eid),"Status":"Complete","P1 frame":p1["frame"],"P1 time":p1["time"],"P1 ΔF/F0":p1["value"],"P2 frame":p2["frame"],"P2 time":p2["time"],"P2 ΔF/F0":p2["value"],"P3 frame":p3["frame"],"P3 time":p3["time"],"P3 ΔF/F0":p3["value"],"P4 frame":p4["frame"],"P4 time":p4["time"],"P4 ΔF/F0":p4["value"],"Fast upstroke duration P1→P2":fd,"Fast upstroke amplitude P1→P2":fa,"Fast upstroke rate P1→P2":fa/fd if fd>0 else np.nan,"Time to peak P1→P3":td,"Peak amplitude above P1":ta,"Overall rise rate P1→P3":ta/td if td>0 else np.nan,"Late-rise amplitude P2→P3":p3["value"]-p2["value"],"Observed decay duration P3→P4":dd,"Observed decay amplitude P3→P4":da,"Observed decay/release rate P3→P4":dr,"Release magnitude":-dr if np.isfinite(dr) else np.nan,"P4 endpoint type":ev.get("endpoint_type",""),"Notes":ev.get("notes","")}

def event_table(events): return pd.DataFrame([one_event_metrics(eid,ev) for eid,ev in events.items()])

def cumulative_events(trace_events, results):
    labels = {r["file"]:r["label"] for r in results}; tabs=[]
    for key, events in trace_events.items():
        file, trace = key.split("::",1)
        tab = event_table(events); tab.insert(0,"File",file); tab.insert(1,"Label",labels.get(file,"")); tab.insert(2,"Trace",trace); tabs.append(tab)
    return pd.concat(tabs,ignore_index=True) if tabs else pd.DataFrame()

def window_auc(df, column, start_frame, end_frame):
    part = df[(df["Frame"] >= start_frame) & (df["Frame"] <= end_frame)]
    return positive_auc(part["Time"], part[column]) if len(part) >= 2 else np.nan

def event_auc_for_file(result, average_events):
    df=result["processed"]; roi_cols=[c for c in df if c.startswith("dF/F0 | ")]
    detail=[]; summary=[]
    for eid, ev in average_events.items():
        p1,p4=ev.get("p1"),ev.get("p4")
        if p1 is None or p4 is None or p1["frame"] >= p4["frame"]: continue
        values=[]
        for col in roi_cols:
            value=window_auc(df,col,p1["frame"],p4["frame"])
            detail.append({"File":result["file"],"Label":result["label"],"Trace used to define window":"Average dF/F0","Event ID":eid,"Stimulation label":ev.get("stimulation_label",eid),"ROI":col.replace("dF/F0 | ",""),"P1 frame":p1["frame"],"P4 frame":p4["frame"],"Event AUC":value})
            if np.isfinite(value): values.append(value)
        n=len(values); mean=float(np.mean(values)) if n else np.nan; sem=float(np.std(values,ddof=1)/np.sqrt(n)) if n>1 else np.nan
        summary.append({"File":result["file"],"Label":result["label"],"Trace used to define window":"Average dF/F0","Event ID":eid,"Stimulation label":ev.get("stimulation_label",eid),"P1 frame":p1["frame"],"P4 frame":p4["frame"],"Valid ROI count":n,"Mean ROI event AUC":mean,"SEM ROI event AUC":sem,"AUC of average trace":window_auc(df,"Average dF/F0",p1["frame"],p4["frame"])})
    return pd.DataFrame(detail),pd.DataFrame(summary)

def overview(results, positive, unit):
    colors=["#1f77b4","#d62728","#2ca02c","#9467bd","#ff7f0e","#17becf"]
    fig=go.Figure()
    for i,r in enumerate(results):
        y=r["processed"]["Average dF/F0"].to_numpy(); y=np.maximum(y,0) if positive else y
        fig.add_trace(go.Scatter(x=r["processed"]["Time"],y=y,mode="lines",line=dict(color=colors[i%len(colors)],width=2.5),name=f"{r['label']} | {r['file']}"))
    fig.update_layout(height=550,template="plotly_white",xaxis_title=f"Time ({unit})",yaxis_title="Average ΔF/F0",legend_title="Uploaded file")
    return fig

def safe_sheet(name, used):
    b=re.sub(r"[\\/*?:\[\]]","_",str(name))[:31] or "sheet"; c=b; i=1
    while c in used: s=f"_{i}"; c=b[:31-len(s)]+s; i+=1
    used.add(c); return c

def excel_bytes(results,cfg,cell_summary,manual_df,auc_detail,auc_summary):
    bio=io.BytesIO(); used=set()
    with pd.ExcelWriter(bio,engine="openpyxl") as w:
        pd.DataFrame([cfg]).to_excel(w,index=False,sheet_name="analysis_settings")
        pd.DataFrame([{"File":r["file"],"Label":r["label"]} for r in results]).to_excel(w,index=False,sheet_name="file_metadata")
        if not cell_summary.empty: cell_summary.to_excel(w,index=False,sheet_name="cell_summary")
        if not manual_df.empty: manual_df.to_excel(w,index=False,sheet_name="manual_events")
        if not auc_detail.empty: auc_detail.to_excel(w,index=False,sheet_name="per_roi_event_auc")
        if not auc_summary.empty: auc_summary.to_excel(w,index=False,sheet_name="event_auc_summary")
        for r in results: r["processed"].to_excel(w,index=False,sheet_name=safe_sheet(f"trace_{r['file']}",used))
    bio.seek(0); return bio.getvalue()

with st.sidebar:
    st.header("Time")
    interval=st.number_input("Frame interval",min_value=.000001,value=1.0,step=.1)
    unit=st.selectbox("Time unit",["seconds","minutes","milliseconds"])
    st.header("Background correction")
    bg_mode=st.radio("Background method",["Subtract empty-area background before ΔF/F0","No background correction"])
    negative=st.radio("Negative corrected-signal handling",["Keep negative values (recommended)","Clip corrected fluorescence below zero to zero","Exclude ROI if any corrected value is below zero"])
    st.header("F0")
    f0_mode=st.selectbox("F0 definition",["First valid value","Mean of first N valid rows","Minimum value in trace","Lower quartile (25th percentile)","Mean within baseline-time window"],index=1)
    n=5; bs=be=None
    if f0_mode=="Mean of first N valid rows": n=int(st.number_input("Number of baseline rows",min_value=1,value=5,step=1))
    if f0_mode=="Mean within baseline-time window": bs=st.number_input(f"Baseline start ({unit})",value=0.); be=st.number_input(f"Baseline end ({unit})",value=5.)
    st.header("Display"); positive=st.checkbox("Plot only positive ΔF/F0",value=False)
    st.button("Reset analysis / upload new files",on_click=reset_analysis,use_container_width=True)

uploads=st.file_uploader("Upload calcium time-series files",type=["csv","xlsx","xls"],accept_multiple_files=True,key=f"up_{st.session_state.uploader_token}")
if not uploads: st.info("Upload one or more calcium files to begin."); st.stop()
loaded=[]
for f in uploads:
    try: loaded.append({"file":f.name,"df":read_table(f)})
    except Exception as e: st.error(f"Could not read {f.name}: {e}")
if not loaded: st.stop()

st.subheader("Detected input structure")
rows=[]
for item in loaded:
    d=detect_columns(item["df"]); rows.append({"File":item["file"],"Suggested label":d["label"],"Suggested time":d["time"],"Suggested background":d["background"],"Detected ROI columns":", ".join(d["signals"]),"ROI count":len(d["signals"])})
st.dataframe(pd.DataFrame(rows),use_container_width=True)
first=loaded[0]["df"]; detected=detect_columns(first); cols=list(first.columns)
with st.expander("Shared column mapping",expanded=True):
    c1,c2,c3,c4=st.columns(4)
    with c1:
        op=["<auto/file name>"]+cols; ix=cols.index(detected["label"])+1 if detected["label"] in cols else 0; label_col=st.selectbox("Label column",op,index=ix)
    with c2:
        op=["<auto row index>"]+cols; ix=cols.index(detected["time"])+1 if detected["time"] in cols else 0; time_col=st.selectbox("Frame/time column",op,index=ix)
    with c3:
        op=["<none>"]+cols; ix=op.index(detected["background"]) if detected["background"] in op else 0; bg_col=st.selectbox("Empty-area background column",op,index=ix)
    with c4: st.info("All Mean1...MeanN ROI columns are analyzed independently in each file. Event windows are entirely manual.")
if bg_mode=="Subtract empty-area background before ΔF/F0" and bg_col=="<none>": st.error("Select Mean(background) or choose no background correction."); st.stop()
cfg={"label_column":None if label_col=="<auto/file name>" else label_col,"time_column":None if time_col=="<auto row index>" else time_col,"background_column":None if bg_col=="<none>" else bg_col,"frame_interval":interval,"time_unit":unit,"background_mode":bg_mode,"negative_policy":negative,"f0_mode":f0_mode,"f0_n":n,"baseline_start":bs,"baseline_end":be,"positive_only":positive}
errors=[]
for item in loaded:
    d=detect_columns(item["df"]); name=item["file"]
    if not d["signals"]: errors.append(f"{name}: no Mean1...MeanN ROI columns found.")
    if cfg["background_column"] is not None and cfg["background_column"] not in item["df"].columns: errors.append(f"{name}: missing selected background column '{cfg['background_column']}'.")
    if cfg["time_column"] is not None and cfg["time_column"] not in item["df"].columns: errors.append(f"{name}: missing selected time column '{cfg['time_column']}'.")
if errors: st.error("Some files cannot be analyzed:\n\n"+"\n".join(errors)); st.stop()
results=[process_file(item["df"],item["file"],cfg) for item in loaded]
for r in results:
    for msg in r["warnings"]: st.warning(msg)

st.subheader("All uploaded file-average traces")
fig=overview(results,positive,unit); st.plotly_chart(fig,use_container_width=True)
sums=[r["summary"] for r in results if not r["summary"].empty]; cell_summary=pd.concat(sums,ignore_index=True) if sums else pd.DataFrame()
st.subheader("Cell/ROI summary"); st.dataframe(cell_summary,use_container_width=True)

st.subheader("Manual stimulation-event definition")
st.markdown("Define event windows on **Average dF/F0** to calculate per-ROI event AUC mean ± SEM. Manual P1–P4 kinetics can still be recorded on any selected trace. **P1:** pre-rise low; **P2:** end of fast straight upstroke; **P3:** peak; **P4:** post-peak endpoint.")
selected_file=st.selectbox("File for manual curation",[r["file"] for r in results]); result=next(r for r in results if r["file"]==selected_file); processed=result["processed"]
trace_options=["Average dF/F0"]+[c for c in processed if c.startswith("dF/F0 | ")]; trace=st.selectbox("Trace",trace_options); key=tkey(selected_file,trace)
count=int(st.number_input("Number of stimulation-response events in this trace",1,20,1,1,key=f"count_{key}")); init_events(key,count); events=st.session_state.trace_events[key]
active=st.selectbox("Event currently being curated",[f"E{i}" for i in range(1,count+1)],key=f"active_{key}"); ev=events[active]
ev["stimulation_label"]=st.text_input("Stimulation label",value=ev.get("stimulation_label") or active,key=f"label_{key}_{active}")
point_name=st.radio("Select next point",["P1: pre-rise low point","P2: end of linear fast upstroke","P3: main peak","P4: post-peak endpoint / low point"],horizontal=True,key=f"point_{key}_{active}")
point_map={"P1: pre-rise low point":"p1","P2: end of linear fast upstroke":"p2","P3: main peak":"p3","P4: post-peak endpoint / low point":"p4"}
st.info("Use Plotly's **Select Points** tool and click an acquired-frame marker. No event point is proposed automatically.")
sel=st.plotly_chart(manual_plot(processed,trace,result,events,active,positive,unit),use_container_width=True,key=f"plot_{key}",on_select="rerun",selection_mode="points",config={"displaylogo":False,"modeBarButtonsToRemove":["lasso2d"]})
if sel and sel.selection and sel.selection.get("points"):
    point=picked_point(processed,trace,sel.selection["points"][0].get("customdata"))
    if point is not None: events[active][point_map[point_name]]=point; st.rerun()
c1,c2,c3=st.columns(3)
with c1:
    if st.button(f"Undo last point for {active}",key=f"undo_{key}_{active}",use_container_width=True):
        for p in ["p4","p3","p2","p1"]:
            if events[active][p] is not None: events[active][p]=None; break
        st.rerun()
with c2:
    if st.button(f"Reset points for {active}",key=f"reset_{key}_{active}",use_container_width=True):
        old={k:events[active][k] for k in ["stimulation_label","endpoint_type","notes"]}; events[active]=empty_event(); events[active].update(old); st.rerun()
with c3:
    if st.button("Reset all events for this trace",key=f"allreset_{key}",use_container_width=True):
        st.session_state.trace_events[key]={}; init_events(key,count); st.rerun()
options=["Post-peak trough","Returned to baseline","Before next rise","Recording end","Sustained plateau","Manual endpoint"]
ev["endpoint_type"]=st.selectbox("P4 endpoint type",options,index=options.index(ev.get("endpoint_type","Post-peak trough")),key=f"end_{key}_{active}")
ev["notes"]=st.text_input("Notes",value=ev.get("notes",""),key=f"notes_{key}_{active}")
current=one_event_metrics(active,ev)
if current["Status"]=="Complete": st.success(f"{active} is complete and is included in calculation/export.")
elif current["Status"]=="Incomplete": st.caption(f"{active} is incomplete. Select P1-P4.")
else: st.error(current["Status"])

current_table=event_table(events); st.subheader("Events for selected trace"); st.dataframe(current_table,use_container_width=True)
manual_df=cumulative_events(st.session_state.trace_events,results)

# Option B: only Average dF/F0 event windows define cross-ROI event AUC.
auc_detail_tabs=[]; auc_summary_tabs=[]
for r in results:
    avg_key=tkey(r["file"],"Average dF/F0")
    detail, summary=event_auc_for_file(r,st.session_state.trace_events.get(avg_key,{}))
    if not detail.empty: auc_detail_tabs.append(detail)
    if not summary.empty: auc_summary_tabs.append(summary)
auc_detail=pd.concat(auc_detail_tabs,ignore_index=True) if auc_detail_tabs else pd.DataFrame()
auc_summary=pd.concat(auc_summary_tabs,ignore_index=True) if auc_summary_tabs else pd.DataFrame()
st.subheader("Event AUC summary: mean ± SEM across ROI traces")
st.caption("Only P1→P4 windows manually defined on each file's Average dF/F0 trace are applied to all ROIs. Mean and SEM therefore describe ROI-level event AUC values within that file.")
if auc_summary.empty: st.info("Select P1 and P4 (and complete P2/P3) for an event on the Average dF/F0 trace to calculate ROI event AUC mean ± SEM.")
else: st.dataframe(auc_summary,use_container_width=True)
with st.expander("Per-ROI event AUC values"):
    st.dataframe(auc_detail,use_container_width=True)
st.subheader("Cumulative manual events: all files and traces")
st.dataframe(manual_df,use_container_width=True,height=360)

st.subheader("Downloads")
ts=datetime.now().strftime("%Y%m%d_%H%M%S"); xlsx=excel_bytes(results,cfg,cell_summary,manual_df,auc_detail,auc_summary)
d1,d2,d3,d4=st.columns(4)
with d1: st.download_button("Download Excel workbook",xlsx,f"calcium_analysis_{ts}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
with d2: st.download_button("Download cell summary CSV",cell_summary.to_csv(index=False).encode(),f"cell_summary_{ts}.csv","text/csv",use_container_width=True)
with d3: st.download_button("Download event AUC summary CSV",auc_summary.to_csv(index=False).encode(),f"event_auc_summary_{ts}.csv","text/csv",use_container_width=True)
with d4: st.download_button("Download per-ROI event AUC CSV",auc_detail.to_csv(index=False).encode(),f"per_roi_event_auc_{ts}.csv","text/csv",use_container_width=True)
st.download_button("Download manual events CSV",manual_df.to_csv(index=False).encode(),f"manual_events_{ts}.csv","text/csv")
st.download_button("Download all-file graph HTML",fig.to_html(include_plotlyjs="cdn"),f"calcium_all_files_{ts}.html","text/html")
