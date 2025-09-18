# 1. Import libraries
import streamlit as st
import pandas as pd
import joblib

# 2. Load model + feature list
model, feature_names = joblib.load("estate_model.pkl")

# 3. App title
st.title("🏡 EstateGauge: Real Estate Risk & Yield Evaluator")

# 4. User inputs
crime_density = st.number_input("Crime Density (per km²)", 0.0, 1000.0, 50.0)
dist_to_school = st.number_input("Distance to Nearest School (m)", 0.0, 10000.0, 500.0)
zhvi_cagr = st.number_input("5-Year Price CAGR (%)", -10.0, 20.0, 5.0)

# 5. Prepare input dict
user_input = {
    'crime_density': crime_density,
    'dist_to_school': dist_to_school,
    'zhvi_cagr': zhvi_cagr
}

# 6. Build DataFrame aligned with training features
X_new = pd.DataFrame([user_input])
X_new = X_new.reindex(columns=feature_names, fill_value=0)

# 7. Predict
if st.button("Evaluate Property"):
    pred = model.predict(X_new)[0]
    st.success(f"Predicted Score: {pred:.2f}")
