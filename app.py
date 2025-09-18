import streamlit as st
import pandas as pd
import joblib

model = joblib.load("estate_model.pkl")

st.title("EstateGauge: Risk & Yield Evaluator")

year = st.number_input("Year", 2025, 2035, 2025)
month = st.slider("Month", 1, 12, 1)
income = st.number_input("Median Income", 20000, 200000, 60000)
pop = st.number_input("Population", 1000, 1000000, 500000)

X_new = pd.DataFrame([[year,month,pop,income]],
                     columns=['year','month','Population','MedianIncome'])
pred = model.predict(X_new)[0]


st.metric("Predicted Home Value", f"${pred:,.0f}")
# expected order of features
expected_features = ['crime_density', 'dist_to_school', 'zhvi_cagr']

# user input
X_new = pd.DataFrame([user_input_dict])

# reindex so order & missing cols handled
X_new = X_new.reindex(columns=expected_features, fill_value=0)
