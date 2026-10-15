# Customer Churn Analysis

An exploratory analysis and interactive dashboard for the Telco Customer Churn dataset.

**Author:** Ashish Raj  
**Repository:** [AshishRaj108-new/customer-churn-analysis](https://github.com/AshishRaj108-new/customer-churn-analysis)  
**Project page:** [Open the project overview](index.html) · [Open the dashboard](docs/index.html)

## Project overview

This project explores customer churn patterns in a telecommunications dataset. The dashboard lets you filter customers and compare historical churn across contract types, tenure, payment methods, internet services, technical support, and subscribed services.

The analysis is descriptive: the retention priority table ranks customer segments by their observed churn rate. It does not predict whether an individual customer will churn.

## Dataset snapshot

These figures are calculated from the CSV included in this repository:

| Measure | Value |
| --- | ---: |
| Customer records | 7,043 |
| Churned customers | 1,869 |
| Retained customers | 5,174 |
| Observed churn rate | 26.54% |

The snapshot describes this dataset only. Dashboard figures update when filters are changed.

## Interactive dashboard

Open [`docs/index.html`](docs/index.html) to explore:

- Filters for contract, internet service, payment method, churn status, and senior citizen status
- Summary cards for churn rate, monthly charges associated with churned customers, average monthly charge for churned customers, and retained customers
- Charts for contract, tenure, payment method, internet service and tech support, revenue split, monthly charges, and subscribed services
- A segment-level retention priority list

**Currency note:** The source CSV has numeric `MonthlyCharges` and `TotalCharges` fields but no currency column. The dashboard displays INR estimates using its stated USD assumption and conversion rate. Treat these financial figures as estimates, not verified rupee-denominated revenue.

## Repository contents

| Path | Description |
| --- | --- |
| [`index.html`](index.html) | Project overview page |
| [`docs/index.html`](docs/index.html) | Interactive churn dashboard |
| [`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`](data/WA_Fn-UseC_-Telco-Customer-Churn.csv) | Customer dataset used by the dashboard and analysis |
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Exploratory data analysis notebook |
| [`requirements.txt`](requirements.txt) | Python packages for the analysis environment |

## Run the dashboard locally

From the project folder, start a small local web server:

```bash
python -m http.server 8000
```

Then open:

- Project overview: <http://localhost:8000/>
- Dashboard: <http://localhost:8000/docs/>

Opening the HTML files directly with `file://` may prevent the dashboard from loading the CSV. The charts use Chart.js from a CDN, so an internet connection is needed for the chart library.

## Run the analysis notebook

From the project folder, install the listed packages and open the notebook:

```bash
python -m pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

Run the notebook cells in order.

## Author

- [GitHub](https://github.com/AshishRaj108-new)
- [LinkedIn](https://www.linkedin.com/in/ashish-raj-38b81b253/)