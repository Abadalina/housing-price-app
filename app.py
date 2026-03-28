"""
Madrid Airbnb Price Predictor — Streamlit App
Author: Alejandro Abadal
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from predict import load_model, load_reference, build_input, predict_price

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Madrid Airbnb Price Predictor",
    layout="wide",
)

# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_reference():
    return load_reference()

model = get_model()
ref   = get_reference()

NEIGHBOURHOODS = ref["neighbourhoods"]
NB_STATS       = ref["nb_stats"]
GLOBAL_MEDIAN  = ref["global_median"]

ROOM_TYPES = [
    "Entire home/apt",
    "Private room",
    "Shared room",
    "Hotel room",
]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size: 2.2rem; font-weight: 800; color: #264653; }
    .subtitle    { font-size: 1rem;   color: #64748b; margin-bottom: 2rem; }
    .price-box   { background: linear-gradient(135deg, #2a9d8f, #264653);
                   border-radius: 16px; padding: 2rem; text-align: center; color: white; }
    .price-value { font-size: 3.5rem; font-weight: 800; line-height: 1; }
    .price-label { font-size: 0.9rem; opacity: 0.8; margin-top: 0.3rem; }
    .price-range { font-size: 1rem; opacity: 0.75; margin-top: 0.5rem; }
    .metric-card { background: #f8fafc; border-radius: 12px; padding: 1rem;
                   border: 1px solid #e2e8f0; text-align: center; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Madrid Airbnb Price Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Enter your listing details to get an estimated nightly price '
    'powered by an XGBoost model trained on real Inside Airbnb data.</p>',
    unsafe_allow_html=True
)

st.divider()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Listing details")

    neighbourhood = st.selectbox("Neighbourhood", NEIGHBOURHOODS, index=NEIGHBOURHOODS.index("Sol") if "Sol" in NEIGHBOURHOODS else 0)
    room_type     = st.selectbox("Room type", ROOM_TYPES)

    st.subheader("Capacity")
    accommodates = st.slider("Guests",   1, 16, 2)
    bedrooms     = st.slider("Bedrooms", 0, 10, 1)
    beds         = st.slider("Beds",     1, 16, 1)
    bathrooms    = st.slider("Bathrooms", 0.5, 6.0, 1.0, step=0.5)

    st.subheader("Availability & rules")
    minimum_nights    = st.slider("Minimum nights",    1, 30, 2)
    availability_365  = st.slider("Availability (days/year)", 0, 365, 180)

    st.subheader("Host profile")
    host_is_superhost  = st.checkbox("Superhost", value=False)
    instant_bookable   = st.checkbox("Instant bookable", value=True)
    host_listings_count = st.number_input("Host total listings", 1, 200, 1)

    st.subheader("Reviews")
    number_of_reviews    = st.number_input("Number of reviews", 0, 2000, 10)
    review_scores_rating = st.slider("Review score (0-5)", 0.0, 5.0, 4.7, step=0.1)
    reviews_per_month    = st.number_input("Reviews per month", 0.0, 30.0, 1.5, step=0.1)

# ── Neighbourhood lookup ──────────────────────────────────────────────────────
nb_info      = NB_STATS.get(neighbourhood, {})
nb_median    = nb_info.get("median", GLOBAL_MEDIAN)
nb_mean      = nb_info.get("mean",   GLOBAL_MEDIAN)
nb_count     = int(nb_info.get("count", 0))

# Madrid center coords as defaults, nudged slightly per neighbourhood index
lat_default = 40.4168
lon_default = -3.7038

# ── Prediction ────────────────────────────────────────────────────────────────
input_df = build_input(
    neighbourhood        = neighbourhood,
    room_type            = room_type,
    accommodates         = accommodates,
    bedrooms             = bedrooms,
    beds                 = beds,
    bathrooms            = bathrooms,
    minimum_nights       = minimum_nights,
    availability_365     = availability_365,
    host_is_superhost    = host_is_superhost,
    instant_bookable     = instant_bookable,
    number_of_reviews    = number_of_reviews,
    review_scores_rating = review_scores_rating,
    reviews_per_month    = reviews_per_month,
    host_listings_count  = host_listings_count,
    latitude             = lat_default,
    longitude            = lon_default,
    nb_median_price      = nb_median,
)

result = predict_price(model, input_df)
price  = result["price"]
low    = result["low"]
high   = result["high"]

vs_global = ((price / GLOBAL_MEDIAN) - 1) * 100
vs_nb     = ((price / nb_median) - 1) * 100 if nb_median else 0

# ── Layout: 3 columns ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    st.markdown(f"""
    <div class="price-box">
        <div class="price-label">Estimated nightly price</div>
        <div class="price-value">{price:.0f} €</div>
        <div class="price-range">Range: {low:.0f} € — {high:.0f} €</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    delta_global = f"{vs_global:+.1f}% vs Madrid median"
    st.metric("Your price", f"{price:.0f} €", delta=delta_global)
    st.metric("Madrid median", f"{GLOBAL_MEDIAN:.0f} €")

with col3:
    st.metric(f"{neighbourhood} median", f"{nb_median:.0f} €")
    st.metric("Listings in neighbourhood", f"{nb_count:,}")

st.divider()

# ── Price gauge ───────────────────────────────────────────────────────────────
col_gauge, col_bars = st.columns(2)

with col_gauge:
    st.subheader("Price gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=price,
        delta={"reference": GLOBAL_MEDIAN, "valueformat": ".0f", "suffix": "€"},
        title={"text": "Predicted price vs Madrid median"},
        gauge={
            "axis": {"range": [0, max(price * 2, GLOBAL_MEDIAN * 2)]},
            "bar":  {"color": "#2a9d8f"},
            "steps": [
                {"range": [0, GLOBAL_MEDIAN * 0.7],           "color": "#e0f2f1"},
                {"range": [GLOBAL_MEDIAN * 0.7, GLOBAL_MEDIAN * 1.3], "color": "#b2dfdb"},
                {"range": [GLOBAL_MEDIAN * 1.3, max(price * 2, GLOBAL_MEDIAN * 2)], "color": "#f8f9fa"},
            ],
            "threshold": {
                "line": {"color": "#e76f51", "width": 3},
                "thickness": 0.75,
                "value": GLOBAL_MEDIAN,
            },
        },
        number={"suffix": " €", "valueformat": ".0f"},
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_bars:
    st.subheader("Neighbourhood context")
    # Top/bottom neighbourhoods for context
    nb_df = pd.DataFrame([
        {"neighbourhood": k, "median": v["median"]}
        for k, v in NB_STATS.items()
    ]).sort_values("median", ascending=False)

    top10    = nb_df.head(10)
    bottom10 = nb_df.tail(10)
    selected_row = nb_df[nb_df["neighbourhood"] == neighbourhood]

    fig_bar = px.bar(
        nb_df.head(20),
        x="median", y="neighbourhood",
        orientation="h",
        color="median",
        color_continuous_scale="Teal",
        labels={"median": "Median price (€)", "neighbourhood": ""},
        title="Top 20 neighbourhoods by median price",
    )
    fig_bar.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10),
                          coloraxis_showscale=False, yaxis=dict(autorange="reversed"))

    # Highlight selected neighbourhood
    if not selected_row.empty and neighbourhood in nb_df.head(20)["neighbourhood"].values:
        idx = nb_df.head(20)[nb_df.head(20)["neighbourhood"] == neighbourhood].index[0]
        fig_bar.add_vline(x=price, line_dash="dash", line_color="#e76f51",
                          annotation_text=f"Your price: {price:.0f}€",
                          annotation_position="top right")
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Price breakdown ───────────────────────────────────────────────────────────
st.subheader("What drives your price?")

col_f1, col_f2, col_f3, col_f4 = st.columns(4)
factors = [
    ("Guests",        accommodates,         "people",       16,    "Higher capacity → higher price"),
    ("Bedrooms",      bedrooms,              "rooms",        10,    "More bedrooms → higher price"),
    ("Location",      nb_median,             "€ nb median",  None,  f"{neighbourhood} median price"),
    ("Review score",  review_scores_rating, "/ 5",          5,     "Higher score → higher price"),
]
for col, (label, value, unit, max_val, tip) in zip([col_f1, col_f2, col_f3, col_f4], factors):
    with col:
        st.metric(label, f"{value} {unit}")
        st.caption(tip)

st.divider()

# ── Price simulator ───────────────────────────────────────────────────────────
st.subheader("Price simulator — how does capacity affect price?")

capacities = list(range(1, 13))
sim_prices = []
for cap in capacities:
    sim_df = build_input(
        neighbourhood=neighbourhood, room_type=room_type,
        accommodates=cap, bedrooms=max(1, cap // 2),
        beds=cap, bathrooms=max(1.0, cap / 3),
        minimum_nights=minimum_nights, availability_365=availability_365,
        host_is_superhost=host_is_superhost, instant_bookable=instant_bookable,
        number_of_reviews=number_of_reviews, review_scores_rating=review_scores_rating,
        reviews_per_month=reviews_per_month, host_listings_count=host_listings_count,
        latitude=lat_default, longitude=lon_default, nb_median_price=nb_median,
    )
    sim_prices.append(predict_price(model, sim_df)["price"])

fig_sim = px.line(
    x=capacities, y=sim_prices,
    markers=True,
    labels={"x": "Number of guests", "y": "Predicted price (€)"},
    title=f"Estimated price by capacity — {neighbourhood}, {room_type}",
    color_discrete_sequence=["#2a9d8f"],
)
fig_sim.add_scatter(x=[accommodates], y=[price], mode="markers",
                    marker=dict(size=14, color="#e76f51", symbol="star"),
                    name="Current config")
fig_sim.update_layout(height=320, showlegend=True)
st.plotly_chart(fig_sim, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: XGBoost trained on Inside Airbnb Madrid data · "
    "Built by **Alejandro Abadal** · "
    "Part of the [Madrid Housing Portfolio](https://github.com/alejandroabadal)"
)
