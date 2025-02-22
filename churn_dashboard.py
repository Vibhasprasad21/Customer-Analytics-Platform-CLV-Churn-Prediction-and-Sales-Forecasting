import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import io  # For downloading reports
from wordcloud import WordCloud, STOPWORDS

def run(global_store):
    # Check if dataset exists in session state
   

    # Assign data properly
    

        st.title("Customer Churn Analysis")
        st.sidebar.title("Churn Insights")
        if "data" not in st.session_state or st.session_state["data"] is None:
            st.error("No dataset found! Please upload a dataset in the Main App.")
            st.stop()  # Stop execution if no data is available
        data = st.session_state["data"]
    
        try:
            # Perform Analysis Button
            if st.button("Perform Analysis"):
                data = st.session_state["data"]  # Retrieve dataset from session state
                
                st.success("File uploaded successfully!")

            # Ensure necessary columns exist
            required_columns = ['Churn_Label', 'Customer_Tenure', 'Region']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                st.error(f"Missing columns: {', '.join(missing_columns)}. Please upload a valid dataset.")
                st.stop()
            
            # Convert Churn_Label to numeric
            data["Churn_Label"] = pd.to_numeric(data["Churn_Label"], errors="coerce")
            data = data.dropna(subset=["Churn_Label"])  # Remove NaNs if any exist
            
            # Churn Risk Analysis
            if "Churn_Label" in data.columns:
                st.header("Churn Risk Analysis")
                
                # Churn distribution
                churn_counts = data["Churn_Label"].value_counts().reset_index()
                churn_counts.columns = ['Churn Status', 'Count']
                
                fig = px.pie(
                    churn_counts,
                    values="Count",
                    names="Churn Status",
                    title="Customer Churn Distribution",
                    color_discrete_sequence=px.colors.sequential.Reds_r
                )
                st.plotly_chart(fig)
                
                # Churn by segment
                if "Segment" in data.columns:
                    st.subheader("Churn Rate by Segment")
                    
                    # Calculate churn rate by segment
                    churn_by_segment = data.groupby("Segment")["Churn_Label"].mean().reset_index()
                    churn_by_segment.columns = ['Segment', 'Churn Rate']
                    churn_by_segment['Churn Rate'] = churn_by_segment['Churn Rate'] * 100
                    churn_by_segment = churn_by_segment.sort_values('Churn Rate', ascending=False)
                    
                    fig = px.bar(
                        churn_by_segment,
                        x="Segment",
                        y="Churn Rate",
                        title="Churn Rate by Customer Segment",
                        color="Churn Rate",
                        color_continuous_scale=px.colors.sequential.Reds,
                        labels={"Churn Rate": "Churn Rate (%)"}
                    )
                    st.plotly_chart(fig)
                
                # CLV comparison of churned vs retained
                if "Monetary_Value" in data.columns:
                    st.subheader("CLV Comparison: Churned vs Retained Customers")
                    
                    # Calculate average CLV by churn status
                    clv_by_churn = data.groupby("Churn_Label")["Monetary_Value"].mean().reset_index()
                    clv_by_churn.columns = ['Churn Status', 'Average CLV']
                    
                    fig = px.bar(
                        clv_by_churn,
                        x="Churn Status",
                        y="Average CLV",
                        title="Average CLV by Churn Status",
                        color="Churn Status",
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        labels={"Average CLV": "Average CLV ($)"}
                    )
                    st.plotly_chart(fig)
            
            # Churn Rate Pie Chart
            st.sidebar.subheader("Churn Rate Distribution")
            fig = px.pie(data, names='Churn_Label', title="Churned vs Retained Customers")
            st.plotly_chart(fig)
            
            # Churn by Tenure
            st.sidebar.subheader("Churn by Customer Tenure")
            fig = px.histogram(data, x='Customer_Tenure', color='Churn_Label', nbins=20, title="Churn Rate by Tenure")
            st.plotly_chart(fig)
            
            # Churn by Region
            st.sidebar.subheader("Churn by Region")
            churn_by_region = data.groupby(["Region", "Churn_Label"]).size().reset_index(name='Count')
            fig = px.bar(churn_by_region, x="Region", y="Count", color="Churn_Label", barmode="stack", title="Churn by Region")
            st.plotly_chart(fig)
            
            # Word Cloud for Churned Customers' Feedback
            st.sidebar.subheader("Churned Customers' Feedback")
            if not st.sidebar.checkbox("Hide Word Cloud", False):
                if "Marketing_Channel" in data.columns:
                    words = ' '.join(data[data['Churn_Label'] == 1]['Marketing_Channel'].dropna())
                    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(words)
                    plt.figure(figsize=(8, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
                else:
                    st.warning("No 'Marketing_Channel' column found in dataset. Cannot generate Word Cloud.")
            
            # Download Report
            st.sidebar.subheader("Download Churn Report")
            
            def generate_report():
                report_data = data[['Customer Name', 'Churn_Label', 'Customer_Tenure', 'Region']]
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    report_data.to_excel(writer, index=False, sheet_name="Churn Report")
                output.seek(0)
                return output
            
            if st.sidebar.button("Generate Report"):
                report_file = generate_report()
                st.sidebar.download_button(
                    label="Download XLSX",
                    data=report_file,
                    file_name="Churn_Analysis_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"Error loading file: {e}")
