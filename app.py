import streamlit as st
import pandas as pd
from clv_calculation import CLVCalculation
from churn_prediction import ChurnPrediction
from sales_forecasting import Forecast
from io import BytesIO
import clv_dashboard
import churn_dashboard
import sales_dashboard
import combined_dashboard

# Ensure session state for dataset persistence
if "data" not in st.session_state:
    st.session_state["data"] = None

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Dashboard", [
    "Main App", "CLV Dashboard", "Churn Dashboard", "Sales Dashboard", "Customer Dashboard"
])

if app_mode == "Main App":
    st.title("AI-Driven Customer Lifetime Value, Churn Prediction, and Sales Prediction")
    st.write("""
    This application processes your uploaded dataset to calculate Customer Lifetime Value (CLV), 
    predict customer churn, and forecast sales.
    """)

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file, engine="openpyxl")
            
            st.session_state["data"] = data  # Store dataset in session state
            st.success("File uploaded successfully!")
            st.write("### Uploaded Data")
            st.dataframe(data)

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    if st.session_state["data"] is not None:
        data = st.session_state["data"]  # Retrieve dataset

        # Initialize model objects
        clv_calc = CLVCalculation()
        churn_pred = ChurnPrediction()
        forecast = Forecast()

        st.header("Step 1: CLV Calculation")
        clv_calc.upload_data(data)
        clv_calc.preprocess_data()
        clv_calc.fit_gamma_gamma_model()
        clv_calc.calculate_clv()
        #clv_calc.plot_clv_distribution()
        
        if "predicted_clv" in clv_calc.customer_data.columns:
            st.session_state["data"] = st.session_state["data"].merge(
                clv_calc.customer_data[["Customer ID", "predicted_clv"]],
                on="Customer ID",
                how="left"
            )
            st.session_state["clv_calculated"] = True  # Flag to indicate CLV is available
            st.success("CLV values successfully stored in session state!")
        else:
            st.error("Error: CLV values could not be stored. Please check CLV calculation.")
        
        st.write("### CLV Results")
        st.dataframe(clv_calc.customer_data.head())
        st.header("Step 2: Churn Prediction")
        churn_pred.data = data.copy()
        churn_pred.handle_infinite_values()
        churn_pred.select_columns()
        churn_pred.feature_scaling()
        churn_pred.preprocess_data()
        churn_pred.train_model()
        churn_pred.predict_churn()
        st.write("### Churn Prediction Results")

        st.header("Step 3: Sales Prediction")
        forecast.data = data.copy()
        forecast.preview_data()
        forecast.data_info()
        forecast.data_statistics()
        forecast.select_columns()
        forecast.time_series_analysis()
        forecast.category_analysis()
        sales_forecast = forecast.forecasting2()
        st.write("### Sales Prediction Results")

elif app_mode == "CLV Dashboard":
    if st.session_state["data"] is None:
        st.warning("⚠️ Please upload a dataset in the Main App first!")
    else:
        clv_dashboard.run(st.session_state["data"])

elif app_mode == "Churn Dashboard":
    if st.session_state["data"] is None:
        st.warning("⚠️ Please upload a dataset in the Main App first!")
    else:
        churn_dashboard.run(st.session_state["data"])

elif app_mode == "Sales Dashboard":
    if st.session_state["data"] is None:
        st.warning("⚠️ Please upload a dataset in the Main App first!")
    else:
        sales_dashboard.run(st.session_state["data"])

elif app_mode == "Customer Dashboard":
    if st.session_state["data"] is None:
        st.warning("⚠️ Please upload a dataset in the Main App first!")
    else:
        combined_dashboard.run(st.session_state["data"])
