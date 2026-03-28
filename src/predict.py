"""
Prediction helpers for the Streamlit app.
Author: Alejandro Abadal
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

BASE = Path(__file__).parent.parent

def load_model():
    return joblib.load(BASE / "models" / "best_price_model.joblib")

def load_reference():
    with open(BASE / "data" / "reference.json") as f:
        return json.load(f)

def build_input(
    neighbourhood: str,
    room_type: str,
    accommodates: int,
    bedrooms: float,
    beds: float,
    bathrooms: float,
    minimum_nights: int,
    availability_365: int,
    host_is_superhost: bool,
    instant_bookable: bool,
    number_of_reviews: int,
    review_scores_rating: float,
    reviews_per_month: float,
    host_listings_count: int,
    latitude: float,
    longitude: float,
    nb_median_price: float,
) -> pd.DataFrame:
    """Build a single-row dataframe matching the model's feature schema."""
    data = {
        "accommodates":            accommodates,
        "bedrooms":                bedrooms,
        "beds":                    beds,
        "bathrooms":               bathrooms,
        "minimum_nights":          minimum_nights,
        "number_of_reviews":       number_of_reviews,
        "review_scores_rating":    review_scores_rating,
        "reviews_per_month":       reviews_per_month,
        "host_listings_count":     host_listings_count,
        "availability_365":        availability_365,
        "latitude":                latitude,
        "longitude":               longitude,
        "host_is_superhost":       int(host_is_superhost),
        "instant_bookable":        int(instant_bookable),
        "neighbourhood_cleansed":  neighbourhood,
        "room_type":               room_type,
        "beds_per_person":         beds / max(accommodates, 1),
        "has_reviews":             int(number_of_reviews > 0),
        "high_availability":       int(availability_365 > 270),
        "neighbourhood_median_price": nb_median_price,
    }
    return pd.DataFrame([data])

def predict_price(model, input_df: pd.DataFrame) -> dict:
    """Return predicted price and a ±15% confidence range."""
    log_pred = model.predict(input_df)[0]
    price    = float(np.expm1(log_pred))
    return {
        "price":  round(price, 1),
        "low":    round(price * 0.85, 1),
        "high":   round(price * 1.15, 1),
    }
