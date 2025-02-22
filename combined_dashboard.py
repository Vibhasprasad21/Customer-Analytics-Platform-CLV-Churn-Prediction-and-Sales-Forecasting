import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
from sales_forecasting import Forecast  # Import Forecast class
from marketing_strategies import allocate_sales_teams, optimize_marketing, apply_discounts  # Import strategies

st.set_page_config(layout="wide")

def optimize_marketing(data):
    """
    Allocates marketing budget based on Customer Lifetime Value (CLV).
    """
    return np.where(data["CLV"] > 400, 5000, np.where(data["CLV"] > 200, 3000, 1000))

def run(global_store):
    st.title("Advanced Business Analytics Dashboard")

    # Ensure data exists in session state
    if "data" not in st.session_state or st.session_state["data"] is None:
        st.error("No dataset found! Please process CLV in the Main App first.")
        return
    
    data = st.session_state["data"]
    
    # Ensure CLV is calculated before running marketing strategies
    if "predicted_clv" not in data.columns:
        st.warning("⚠️ Please ensure CLV is calculated in the Main App!")
        return

    try:
        if st.button("Perform Analysis"):
            data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
            data['Signup_Date'] = pd.to_datetime(data['Signup_Date'], errors='coerce')

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Key Performance Indicators")
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)

                with kpi1:
                    st.metric("Total Sales", f"${data['Sales'].sum():,.2f}")
                with kpi2:
                    st.metric("Total Customers", f"{data['Customer ID'].nunique():,}")
                with kpi3:
                    avg_transaction = data['Average_Transaction_Amount'].mean()
                    st.metric("Avg Transaction", f"${avg_transaction:.2f}")
                with kpi4:
                    churn_rate = (data['Churn_Label'] == '1').mean() * 100
                    st.metric("Churn Rate", f"{churn_rate:.1f}%")

            with col2:
                st.subheader("Customer Lifetime Value")
                clv_data = data.groupby('Customer_Segment_Performance').agg({
                    'Monetary_Value': 'mean', 'Customer ID': 'count'
                }).reset_index()

                fig = px.bar(clv_data, x='Customer_Segment_Performance', y='Monetary_Value', 
                             color='Customer ID', title='CLV by Customer Segment')
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Product Performance & Customer Behavior")
            col3, col4 = st.columns(2)

            with col3:
                category_perf = data.groupby('Category').agg({
                    'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum'
                }).reset_index()

                fig = px.treemap(category_perf, path=['Category'],
                                 values='Sales', color='Profit',
                                 title='Category Performance')
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                fig = px.scatter(data, x='Frequency', y='Monetary_Value',
                                 color='Segment', size='Recency',
                                 title='Customer Purchase Patterns',
                                 hover_data=['Customer ID'])
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Advanced Analytics")
            col5, col6 = st.columns(2)

            with col5:
                churn_factors = data.groupby('Churn_Label').agg({
                    'Satisfaction_Score': 'mean', 'Return_Rate': 'mean',
                    'Support_Contact_Rate': 'mean'
                }).reset_index()

                fig = px.parallel_coordinates(churn_factors,
                                             dimensions=['Churn_Label', 'Satisfaction_Score',
                                                         'Return_Rate', 'Support_Contact_Rate'],
                                             title='Churn Risk Factors')
                st.plotly_chart(fig, use_container_width=True)

            with col6:
                channel_perf = data.groupby('Marketing_Channel').agg({
                    'Sales': 'sum', 'Promotion_Response': 'mean', 'Customer ID': 'count'
                }).reset_index()

                fig = px.sunburst(channel_perf, path=['Marketing_Channel'],
                                  values='Sales', color='Promotion_Response',
                                  title='Marketing Channel Performance')
                st.plotly_chart(fig, use_container_width=True)

            # MARKETING STRATEGIES INTEGRATION
            st.subheader("Marketing Strategy Recommendations")

            # Ensure 'predicted_clv' is correctly named as 'CLV' if needed
            if "predicted_clv" in data.columns and "CLV" not in data.columns:
                data.rename(columns={"predicted_clv": "CLV"}, inplace=True)

            sales_team_allocation = allocate_sales_teams(data)
            marketing_optimization = optimize_marketing(data)
            discount_suggestions = apply_discounts(data)

            st.write("### Sales Team Allocation Strategy")
            st.dataframe(sales_team_allocation)

            st.write("### Optimized Marketing Strategy")
            st.dataframe(marketing_optimization)

            st.write("### Discount Strategy Suggestions")
            st.dataframe(discount_suggestions)

            # DOWNLOAD REPORT
            st.sidebar.subheader("Download Marketing Report")

            def generate_report():
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    data.to_excel(writer, sheet_name="Marketing Data", index=False)
                    sales_team_allocation.to_excel(writer, sheet_name="Sales_Team", index=False)
                    marketing_optimization.to_excel(writer, sheet_name="Marketing_Strategy", index=False)
                    discount_suggestions.to_excel(writer, sheet_name="Discount_Strategy", index=False)
                output.seek(0)
                return output

            if st.sidebar.button("Generate Report"):
                report_file = generate_report()
                st.sidebar.download_button(
                    label="Download XLSX",
                    data=report_file,
                    file_name="Marketing_Strategy_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
