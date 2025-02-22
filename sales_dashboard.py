import streamlit as st
import pandas as pd
import plotly.express as px
import io  # For downloading reports

def run(global_store):
        st.title("Sales Performance Dashboard")
        st.sidebar.title("Sales Insights")
    
        if "data" not in st.session_state or st.session_state["data"] is None:
            st.error("No dataset found! Please upload a dataset in the Main App.")
            st.stop()  # Stop execution if no data is available
        data = st.session_state["data"]
        try:
        # Perform Analysis Button
            if st.button("Perform Analysis"):
                data = st.session_state["data"]  # Retrieve dataset from session state
                
                st.success("File uploaded successfully!")
            
            # Sales Over Time
            st.subheader("Sales Trend Over Time")
            fig = px.line(data, x='Order Date', y='Sales', title="Sales Over Time", labels={"Sales": "Total Sales"})
            st.plotly_chart(fig)
            
            # Sales by Region
            st.subheader("Sales by Region")
            region_sales = data.groupby("Region")["Sales"].sum().reset_index()
            fig = px.bar(region_sales, x="Region", y="Sales", title="Sales by Region", color="Sales")
            st.plotly_chart(fig)
            
            # Customer Segmentation
            st.subheader("Customer Segmentation")
            fig = px.pie(data, names="Segment", values="Sales", title="Sales Distribution by Segment")
            st.plotly_chart(fig)
            
            # Profitability Analysis
            st.subheader("Profit vs Discount")
            fig = px.scatter(data, x='Discount', y='Profit', color='Category', title="Impact of Discounts on Profit")
            st.plotly_chart(fig)
            
            # Top Products by Sales
            st.subheader("Top 10 Best-Selling Products")
            top_products = data.groupby("Product Name")["Sales"].sum().nlargest(10).reset_index()
            fig = px.bar(top_products, x="Product Name", y="Sales", title="Top 10 Products", color="Sales")
            st.plotly_chart(fig)
            
            # Customer Behavior Insights
            st.subheader("Customer Purchase Frequency")
            fig = px.histogram(data, x='Purchase_Frequency', title="Distribution of Customer Purchase Frequency")
            st.plotly_chart(fig)
            
            # High Spender Analysis
            st.subheader("High Spender Distribution")
            high_spenders = data[data['High_Spender_Flag'] == 1]
            fig = px.histogram(high_spenders, x='Annualized_Spend', title="Annualized Spend of High Spenders")
            st.plotly_chart(fig)
            
            # Churn Analysis
            #st.subheader("Customer Churn Distribution")
            #churn_counts = data['Churn_Label'].value_counts().reset_index()
            #churn_counts.columns = ['Churn_Label', 'count']  # Rename columns
            #fig = px.pie(churn_counts, names='Churn_Label', values='count', title="Churn Distribution")

            
            # Download Report
            st.sidebar.subheader("Download Sales Report")
            
            def generate_report():
                report_data = data[['Order ID', 'Order Date', 'Product Name', 'Sales', 'Region', 'Profit', 'Customer Name', 'Annualized_Spend']]
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    report_data.to_excel(writer, index=False, sheet_name="Sales Report")
                output.seek(0)
                return output
            
            if st.sidebar.button("Generate Report"):
                report_file = generate_report()
                st.sidebar.download_button(
                    label="Download XLSX",
                    data=report_file,
                    file_name="Sales_Performance_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"Error loading file: {e}")

