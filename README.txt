Customer Churn dashboard update — Ashish Raj

The ZIP contains TWO files named index.html, but they go in different folders:
- ZIP root\index.html -> C:\Users\HP\churn-prediction\index.html (new project home page; add it)
- ZIP docs\index.html -> C:\Users\HP\churn-prediction\docs\index.html (new dashboard; replace the existing dashboard)

Your previous dashboard is saved separately here:
- ZIP docs\index-before-update.html -> C:\Users\HP\churn-prediction\docs\index-before-update.html

Easy install:
1. Right-click the ZIP and choose Extract All.
2. Choose C:\Users\HP\churn-prediction as the destination, keeping folders as shown.
3. If Windows asks about docs\index.html, choose Replace. The old dashboard is already included as index-before-update.html.
4. Keep your existing data and notebooks folders; this update does not include or replace them.

Open the project page at http://127.0.0.1:8765/ and the dashboard at http://127.0.0.1:8765/docs/ (if your local server is already running).

To start a local preview in Command Prompt:
cd /d C:\Users\HP\churn-prediction
venv\Scripts\python.exe -m http.server 8000
Then open http://localhost:8000/ . Keep internet access available for the dashboard chart library.
