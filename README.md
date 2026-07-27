# Calcium Post Analysis Streamlit App

This starter app:
- uploads Excel/CSV tables of time and cell mean intensity values
- calculates dF/F0 using the first image as F0
- optionally subtracts a background column
- plots average and individual cell traces
- calculates peak intensity, time to peak, fade time, and AUC
- exports results to Excel

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```
