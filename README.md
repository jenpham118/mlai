# AMD Next-Close Predictor

A stock price prediction project for AMD (NASDAQ). A Jupyter notebook trains and compares five regression models to predict AMD's next-day closing price, and a small standalone website displays the result.

## What's in this folder

| File | Purpose |
|---|---|
| `final_notebook.ipynb` | The notebook — data loading, feature engineering, model training/comparison, and final prediction |
| `amd_predictor.html` | The demo website — displays the model's prediction |
| `prediction_output.json` | Generated automatically by the notebook — the website reads from this |

## Requirements

- Python 3.9–3.12
- Jupyter Notebook or JupyterLab
- Internet connection (to download AMD price data from Yahoo Finance)

Install the required Python packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn yfinance tensorflow
```

## Step 1 — Run the notebook

1. Open `final_notebook.ipynb` in Jupyter.
2. Run every cell from the top, in order. The easiest way:
   - **Jupyter Notebook/Lab:** `Kernel` → `Restart Kernel and Run All Cells`
   - **VS Code:** click the "Run All" button at the top of the notebook
3. Wait for it to finish (the RNN/LSTM training cells take the longest — a few minutes).
4. Confirm a new file called `prediction_output.json` has appeared in the same folder as the notebook. This is created by the very last cell, and it's what the website reads from.

> **Important:** the notebook must be run all the way through, top to bottom, in one go. Running cells out of order (or skipping ahead) will cause errors, since later sections depend on variables and files created by earlier ones.

## Step 2 — Run the website

The website needs to be served by a local web server rather than opened directly by double-clicking, because browsers block a webpage from reading local files (like `prediction_output.json`) any other way.

1. Make sure `amd_predictor.html` is in the **same folder** as `prediction_output.json`.
2. Open a terminal (or Anaconda Prompt) in that folder:
   ```bash
   cd path\to\this\folder
   ```
3. Start a local server:
   ```bash
   python -m http.server 8000
   ```
   You should see a message like `Serving HTTP on :: port 8000...` — leave this window open.
4. Open a browser and go to:
   ```
   http://localhost:8000/amd_predictor.html
   ```

## Step 3 — Use it

1. Enter today's date (pre-filled automatically).
2. Click **"Predict Next Close"**.
3. The site shows:
   - The predicted closing price for the next actual trading day (weekends are automatically skipped)
   - The predicted $ and % change versus the last known close
   - Which model produced the prediction
   - A chart of the real recent price history with the prediction highlighted

## Notes

- The prediction only updates when the notebook is re-run and `prediction_output.json` is regenerated — the website itself doesn't recompute anything live.
- To stop the local server, go back to the terminal and press `Ctrl+C`.
- This is a coursework demo, not financial advice — predictions are statistical estimates and can be wrong.

<img src="hey_girl.jpg">