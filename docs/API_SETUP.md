# Run the single-customer prediction locally

The dashboard on GitHub Pages is a static website. Python model predictions run in a small API service, so run both the dashboard and the API for a local demo.

## 1. Open a terminal at the project root

This is the folder containing `src`, `docs`, `data`, and `requirements.txt`. Activate the project's virtual environment if it is not active yet.

## 2. Train and save the model

Run:

```bash
python src/train_churn_model.py
```

This compares Logistic Regression, Random Forest, and Gradient Boosting, writes the evaluation data to `docs/data/ml_predictions.json`, and saves the selected fitted model to `models/churn_live_model.joblib`. The model file stays on your computer and is excluded from Git.

## 3. Start the prediction API in a second terminal

Keep the first terminal open. Open another terminal at the project root, activate the same virtual environment, and run:

```bash
python -m uvicorn src.api:app --reload --port 8000
```

Leave this terminal running. The API address is `http://127.0.0.1:8000`.

## 4. Open the dashboard and make a separate prediction

Serve the project files over HTTP as you already do, then open:

`http://127.0.0.1:8765/docs/index.html#predict-customer`

The dashboard fills in the local API address when opened on `localhost` or `127.0.0.1). Click **Check connection**, enter the customer's service details, then click **Estimate churn risk**. The answer appears in the single-customer card and does not change or get added to the historical charts.

If your dashboard server uses another local port, the API allows localhost addresses on any port. Keep the page and API terminals running while using the form.

## Using the public GitHub Pages dashboard

The public page cannot run Python by itself. To make the form work for everyone on GitHub Pages, the API must also be deployed to a host that provides an HTTPS address. Enter that address in **Prediction API address** on the dashboard. The API host must allow the website origin `https://ashishraj108-new.github.io`; the service supports the `CHURN_ALLOWED_ORIGINS` environment variable for extra allowed origins.

No public API address is configured yet, so the live dashboard will show the form and explain how to connect it, while historical charts continue to work normally.
