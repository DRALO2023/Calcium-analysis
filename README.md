# Calcium Analysis Streamlit App (v4)

This version includes:
- Fade A/B/C definitions on the page
- AUC (area under the curve) calculation for each condition
- Optional SEM bands on graphs
- Optional Welch's t-test for peak values
- Reset button to upload new files

## AUC definition
AUC is computed as the trapezoidal integral of the average ΔF/F0 trace over the entire time range.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
