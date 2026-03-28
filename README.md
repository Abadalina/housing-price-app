# Madrid Airbnb Price Predictor — Streamlit App

Interactive web app that predicts the nightly price of a Madrid Airbnb listing using an XGBoost model trained on real Inside Airbnb data.

---

## What it does

- **Instant price estimate** with a ±15 % confidence range
- **Price gauge** comparing your listing against the Madrid median
- **Neighbourhood context** — bar chart of top 20 neighbourhoods by median price
- **Price simulator** — how predicted price changes with guest capacity
- **Key driver metrics** — capacity, bedrooms, location, review score at a glance

---

## Project structure

```
housing-price-app/
├── app.py                      ← Streamlit application
├── src/
│   └── predict.py              ← load_model, build_input, predict_price
├── models/
│   └── best_price_model.joblib ← XGBoost pipeline (trained in Project 2)
├── data/
│   └── reference.json          ← neighbourhood stats + global median
├── requirements.txt
└── README.md
```

---

## Model

The XGBoost model was trained in [housing-price-ml](https://github.com/alejandroabadal/housing-price-ml) (Project 2 of this portfolio):

| Metric | Value |
|--------|-------|
| R²     | 0.633 |
| MAE    | ~32 € |
| CV R²  | 0.709 ± 0.041 |

Target variable: `log1p(price)` → expm1 to recover euros.

---

## Installation & usage

```bash
git clone https://github.com/alejandroabadal/housing-price-app.git
cd housing-price-app
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![Plotly](https://img.shields.io/badge/Plotly-5.22+-purple)

---

## Part of the Madrid Housing Portfolio

| # | Project | Description |
|---|---------|-------------|
| 1 | [spain-rental-eda](https://github.com/alejandroabadal/spain-rental-eda) | Exploratory data analysis |
| 2 | [housing-price-ml](https://github.com/alejandroabadal/housing-price-ml) | ML price prediction |
| 3 | [rental-price-forecast](https://github.com/alejandroabadal/rental-price-forecast) | Time series forecasting |
| 4 | [airbnb-reviews-nlp](https://github.com/alejandroabadal/airbnb-reviews-nlp) | NLP sentiment & topic analysis |
| 5 | **housing-price-app** | Streamlit deployment ← you are here |

---

## Author

**Alejandro Abadal** — Data Science Student, UOC
[LinkedIn](#) · [GitHub](https://github.com/alejandroabadal)

---

*Data: Inside Airbnb Madrid. For educational purposes.*
