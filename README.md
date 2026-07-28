
# Calcium Analysis Streamlit App

This app lets you:
- upload one control dataset and one treatment dataset
- compute ΔF/F0 for all selected cell columns
- average ΔF/F0 across cells for each condition
- plot control alone, treatment alone, and both together
- calculate peak time/value and fade A, B, C from the average trace
- download metrics, average traces, and an Excel workbook
- reset the analysis and upload new files

## Expected input format
Each file should contain:
- one time column
- one or more cell intensity columns

Supported formats:
- CSV
- XLSX
- XLS

## Fade definitions
- Fade A: first point after the peak where the average trace decreases
- Fade B: first point after the peak where the average trace reaches 50% of peak
- Fade C: first point after the peak where the average trace reaches 0 or below

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
