# Single-customer prediction setup

The dashboard is hosted on GitHub Pages. A Python API handles the model prediction. For local use, run the dashboard server and API on your computer. For the public dashboard, deploy the API to Render so VS Code does not need to stay open.

## Use it locally

1. Open a terminal in the project root, the folder containing `src`, `docs`, `data`, and `requirements.txt`. Activate your virtual environment.
2. Pull the latest project files, then train the model:

   ```bash
   git pull origin main
   python src/train_churn_model.py
   ```

   The trainer writes the evaluation data to `docs/data/ml_predictions.json` and saves the selected model to `models/churn_live_model.joblib`.

3. In a second terminal at the same project root, start the API:

   ```bash
   python -m uvicorn src.api:app --reload --port 8000
   ```

4. Keep the API and the local dashboard server running. Open `http://127.0.0.1:8765/docs/index.html#predict-customer`, click **Check connection**, fill the customer details, and click **Estimate churn risk**.

The prediction appears separately from the historical charts and the entered customer details are not saved by this project.

## Keep the public dashboard working without VS Code

A GitHub Pages site cannot run Python. This repository includes a Render Blueprint (`render.yaml`) that describes a free Python API service. Render builds the model from the CSV when deploying, then serves the API over HTTPS.

1. Sign in to [Render](https://dashboard.render.com/) using your GitHub account.
2. In Render, choose **New → Blueprint** and connect `AshishRaj108-new/customer-churn-analysis` on the `main` branch.
3. Review the service named `ashish-customer-churn-api`; make sure its plan is **Free**, then choose **Deploy Blueprint**.
4. Wait for the build to finish. It installs the small API dependency list and runs the three-model trainer to create the selected model for the API.
5. Open the deployed service and copy its HTTPS address, for example `https://ashish-customer-churn-api.onrender.com`. Use the exact address shown in your Render dashboard.
6. Open the GitHub Pages dashboard, go to the one-customer form, paste that base address into **Prediction API address** (do not add `/health` or `/predict`), then click **Check connection**.

The API already allows requests from `https://ashishraj108-new.github.io`. Render's free service sleeps after 15 minutes with no incoming traffic; the first request after it sleeps can take about one minute to wake it. This means VS Code can be closed after deployment, but the first prediction after a quiet period may be slow. Free instances are intended for demos and hobby projects, not production.

For the exact service setup behavior and current free-tier limits, see [Render's FastAPI deployment guide](https://render.com/docs/deploy-fastapi), [Blueprint setup guide](https://render.com/docs/infrastructure-as-code), and [free service limits](https://render.com/docs/free).
