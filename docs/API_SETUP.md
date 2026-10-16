# Dashboard API and AI chatbot setup

The public dashboard is hosted on GitHub Pages. Its prediction and chatbot requests go to the same Render API, so visitors do not need to open VS Code or enter a local address. The API is public and shared by everyone who opens the dashboard.

## Use the dashboard

Open the [public dashboard](https://ashishraj108-new.github.io/customer-churn-analysis/docs/index.html). The Render API address is filled in automatically. The page checks the API connection when it opens; after the free service wakes, it should say **Connected** and **AI chat is ready**.

The API provides:

- `/predict` for an individual churn estimate and SHAP factors.
- `/chat` for the Gemini-powered dashboard assistant, grounded with aggregate statistics from the Telco CSV and the model comparison.

The chatbot can explain a prediction made in the same browser tab. It does not change the score or the historical dashboard charts.

## One-time Gemini key setup for the public chatbot

The Render API is already deployed, but the chatbot needs a Google Gemini API key. Create the key yourself and keep it private:

1. Open [Google AI Studio API keys](https://aistudio.google.com/app/apikey) and create an API key.
2. Open the Render service named `ashish-customer-churn-api`.
3. Open **Environment** and add a variable:
   - **Key:** `GEMINI_API_KEY`
   - **Value:** paste your Google AI Studio key.
4. Save the change and wait for the service to show **Live**. If Render does not start a new deploy automatically, use **Manual Deploy → Deploy latest commit**.
5. Refresh the public dashboard. The connection check should say **AI chat is ready**. Then ask a sample question, such as “Which contract type has the highest churn rate?”

The key stays on Render's server. Do not put it in `docs/index.html`, GitHub, or a chat message. Every visitor shares the same API and Gemini usage quota. The demo API limits chat to 12 requests per minute across the running service.

### Privacy note

Chat questions and recent chat turns, aggregate dataset/model summaries, and—when present—the latest prediction score and its top SHAP factors are sent to Gemini to create an answer. Do not enter customer names, phone numbers, email addresses, account numbers, or other private details. This dashboard does not save chat history. Google's free Gemini tier says submitted content may be used to improve Google products; check the [current Gemini pricing and data-use details](https://ai.google.dev/gemini-api/docs/pricing) before enabling the key for public use.

## Run the prediction API locally

1. Open a terminal in the project root, the folder containing `src`, `docs`, `data`, and `requirements.txt`. Activate your virtual environment.
2. Pull the latest project files and train the model:

   ```bash
   git pull origin main
   python src/train_churn_model.py
   ```

3. In a second terminal at the same project root, start the API:

   ```bash
   python -m uvicorn src.api:app --reload --port 8000
   ```

4. Start the local dashboard server if it is not already running, then open `http://127.0.0.1:8765/docs/index.html#predict-customer`. The local page uses the local API address automatically.

Local prediction does not require Gemini. The chatbot requires `GEMINI_API_KEY` to be set for whichever API server is handling the request.

## Render free-service behavior

The Render free service can sleep after inactivity. The first request after it sleeps may take about a minute to wake it. This can delay the automatic connection check and the first chat question. Free-tier model usage also has provider-set limits.

For hosting details, see [Render's FastAPI deployment guide](https://render.com/docs/deploy-fastapi), [Blueprint guide](https://render.com/docs/infrastructure-as-code), and [free service limits](https://render.com/docs/free).
