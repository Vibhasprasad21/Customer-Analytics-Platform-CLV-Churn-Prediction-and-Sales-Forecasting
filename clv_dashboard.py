import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io  # For downloading reports
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np

def run(global_store):
        st.title("Customer Lifetime Value (CLV) Dashboard")
        st.sidebar.title("Customer Insights")
    
         # Initialize session state for file storage
        if "data" not in st.session_state or st.session_state["data"] is None:
            st.error("No dataset found! Please upload a dataset in the Main App.")
            st.stop()  # Stop execution if no data is available
        data = st.session_state["data"]
        try:
        # Perform Analysis Button
            if st.button("Perform Analysis"):
                data = st.session_state["data"]  # Retrieve dataset from session state
                
                st.success("File uploaded successfully!")
            
            # CLV Overview Section
            st.header("CLV Overview")
            
            
            
            # Display key CLV metrics
            col1, col2, col3 = st.columns(3)
            
            if "Monetary_Value" in data.columns:
                with col1:
                    avg_clv = data["Monetary_Value"].mean()
                    st.metric("Average CLV", f"${avg_clv:.2f}")
                
                with col2:
                    median_clv = data["Monetary_Value"].median()
                    st.metric("Median CLV", f"${median_clv:.2f}")
                
                with col3:
                    max_clv = data["Monetary_Value"].max()
                    st.metric("Highest CLV", f"${max_clv:.2f}")
                
                # CLV Distribution
                st.subheader("CLV Distribution")
                fig = px.histogram(
                    data, 
                    x="Monetary_Value",
                    title="Customer Lifetime Value Distribution",
                    nbins=50,
                    color_discrete_sequence=['#636EFA']
                )
                fig.update_layout(xaxis_title="Customer Lifetime Value ($)", yaxis_title="Count")
                st.plotly_chart(fig)
                
                # CLV by Segment
                if "Segment" in data.columns:
                    st.subheader("CLV by Customer Segment")
                    segment_clv = data.groupby("Segment")["Monetary_Value"].agg(['mean', 'median', 'count']).reset_index()
                    segment_clv.columns = ['Segment', 'Average CLV', 'Median CLV', 'Customer Count']
                    
                    # Horizontal bar chart of avg CLV by segment
                    fig = px.bar(
                        segment_clv.sort_values('Average CLV'), 
                        y="Segment", 
                        x="Average CLV",
                        title="Average CLV by Customer Segment",
                        text_auto='.2f',
                        color="Average CLV",
                        color_continuous_scale=px.colors.sequential.Blues,
                        height=400
                    )
                    fig.update_layout(xaxis_title="Average CLV ($)", yaxis_title="")
                    st.plotly_chart(fig)
                    
                    # Display segment stats table
                    st.subheader("Segment Performance")
                    segment_clv['Average CLV'] = segment_clv['Average CLV'].map('${:.2f}'.format)
                    segment_clv['Median CLV'] = segment_clv['Median CLV'].map('${:.2f}'.format)
                    st.dataframe(segment_clv, use_container_width=True)
                
                # RFM Analysis Section
                st.header("RFM Analysis Insights")
                
                # RFM Overview
                if all(col in data.columns for col in ["Recency", "Frequency", "Monetary_Value"]):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_recency = data["Recency"].mean()
                        st.metric("Avg Days Since Last Purchase", f"{avg_recency:.1f}")
                        
                    with col2:
                        avg_frequency = data["Frequency"].mean()
                        st.metric("Avg Purchase Frequency", f"{avg_frequency:.2f}")
                        
                    with col3:
                        avg_monetary = data["Monetary_Value"].mean()
                        st.metric("Avg Monetary Value", f"${avg_monetary:.2f}")
                    
                    # RFM Correlations
                    st.subheader("RFM Correlations with CLV")
                    rfm_corrs = [
                        {"Factor": "Recency (R)", "Correlation": data[["Recency", "Monetary_Value"]].corr().iloc[0, 1]},
                        {"Factor": "Frequency (F)", "Correlation": data[["Frequency", "Monetary_Value"]].corr().iloc[0, 1]},
                    ]
                    
                    corr_df = pd.DataFrame(rfm_corrs)
                    fig = px.bar(
                        corr_df,
                        x="Factor",
                        y="Correlation",
                        title="RFM Factors Correlation with CLV",
                        color="Correlation",
                        color_continuous_scale=px.colors.diverging.RdBu,
                        range_color=[-1, 1]
                    )
                    st.plotly_chart(fig)
                    
                    # RFM Scatter Plot
                    st.subheader("Frequency vs Monetary Value")
                    fig = px.scatter(
                        data,
                        x="Frequency",
                        y="Monetary_Value",
                        color="Recency",
                        size="Total_Spend",
                        hover_name="Customer_Name" if "Customer_Name" in data.columns else None,
                        title="RFM Relationship Analysis",
                        color_continuous_scale="Viridis_r",  # Reversed so lower recency (more recent) is better
                        labels={
                            "Frequency": "Purchase Frequency",
                            "Monetary_Value": "Customer Lifetime Value ($)",
                            "Recency": "Days Since Last Purchase"
                        }
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig)
            
                # CLV Percentile Analysis
                st.subheader("CLV Percentile Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Calculate percentiles
                    percentiles = [0.5, 0.8, 0.9, 0.95, 0.99]
                    perc_values = np.percentile(data["Monetary_Value"].dropna(), [p*100 for p in percentiles])
                    
                    perc_df = pd.DataFrame({
                        'Percentile': [f'Top {int((1-p)*100)}%' for p in percentiles],
                        'CLV Threshold': [f'${v:.2f}' for v in perc_values]
                    })
                    st.table(perc_df)
                
                with col2:
                    # Pareto analysis (80/20 rule visualization)
                    data_sorted = data.sort_values('Monetary_Value', ascending=False).reset_index(drop=True)
                    data_sorted['cumulative_clv'] = data_sorted['Monetary_Value'].cumsum()
                    data_sorted['clv_percentage'] = data_sorted['cumulative_clv'] / data_sorted['Monetary_Value'].sum() * 100
                    data_sorted['customer_percentage'] = (data_sorted.index + 1) / len(data_sorted) * 100
                    
                    # Find point closest to 80% of total CLV
                    target_80pct = data_sorted[data_sorted['clv_percentage'] >= 80].iloc[0]
                    customers_for_80pct = target_80pct['customer_percentage']
                    
                    fig = px.line(
                        data_sorted, 
                        x='customer_percentage', 
                        y='clv_percentage',
                        title="CLV Concentration (Pareto Analysis)",
                        labels={'clv_percentage': 'Cumulative % of Total CLV', 'customer_percentage': '% of Customers'}
                    )
                    
                    # Add reference point for 80% CLV
                    fig.add_trace(
                        go.Scatter(
                            x=[customers_for_80pct],
                            y=[80],
                            mode='markers',
                            marker=dict(size=10, color='red'),
                            name=f'Top {customers_for_80pct:.1f}% of customers generate 80% of CLV'
                        )
                    )
                    
                    # Add reference lines
                    fig.add_shape(type="line", x0=0, y0=80, x1=customers_for_80pct, y1=80, line=dict(dash="dash", color="red"))
                    fig.add_shape(type="line", x0=customers_for_80pct, y0=0, x1=customers_for_80pct, y1=80, line=dict(dash="dash", color="red"))
                    
                    st.plotly_chart(fig)
                    
                # Segmentation and Customer Performance
                if "Customer_Segment_Performance" in data.columns:
                    st.subheader("CLV by Customer Segment Performance")
                    segment_perf = data.groupby("Customer_Segment_Performance")["Monetary_Value"].mean().reset_index()
                    segment_perf = segment_perf.sort_values("Monetary_Value", ascending=False)
                    
                    fig = px.bar(
                        segment_perf,
                        x="Customer_Segment_Performance",
                        y="Monetary_Value",
                        title="Average CLV by Customer Segment Performance",
                        color="Monetary_Value",
                        color_continuous_scale=px.colors.sequential.Viridis,
                        labels={"Monetary_Value": "Average CLV ($)", "Customer_Segment_Performance": "Performance Segment"}
                    )
                    st.plotly_chart(fig)
                
                # CLV by Top Product Category
                if "Top_Product_Category" in data.columns:
                    st.subheader("CLV by Top Product Category")
                    
                    # Get top 10 product categories by customer count
                    top_categories = data["Top_Product_Category"].value_counts().head(10).index.tolist()
                    filtered_data = data[data["Top_Product_Category"].isin(top_categories)]
                    
                    # Calculate average CLV by category
                    category_clv = filtered_data.groupby("Top_Product_Category")["Monetary_Value"].mean().reset_index()
                    category_clv = category_clv.sort_values("Monetary_Value", ascending=False)
                    
                    fig = px.bar(
                        category_clv, 
                        x="Top_Product_Category", 
                        y="Monetary_Value",
                        title="Average CLV by Top Product Categories",
                        color="Monetary_Value",
                        color_continuous_scale=px.colors.sequential.Viridis,
                        labels={"Monetary_Value": "Average CLV ($)", "Top_Product_Category": "Product Category"}
                    )
                    st.plotly_chart(fig)
            else:
                st.warning("Column 'Monetary_Value' not found in dataset. Please ensure your dataset contains CLV predictions.")
            
            # Customer Segmentation
            st.header("Customer Segmentation")
            if "Segment" in data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        data, 
                        names="Segment", 
                        title="Customer Segmentation",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig)
                
                with col2:
                    # Count by segment
                    segment_counts = data["Segment"].value_counts().reset_index()
                    segment_counts.columns = ['Segment', 'Count']
                    
                    fig = px.bar(
                        segment_counts, 
                        x="Segment", 
                        y="Count",
                        title="Customer Count by Segment",
                        color="Segment",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig)
            else:
                st.warning("Column 'Segment' not found in dataset.")
            
            # Customer Tenure and Loyalty Analysis
            if "Customer_Tenure" in data.columns and "Segment_Loyalty" in data.columns:
                st.header("Customer Tenure and Loyalty Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Tenure distribution
                    fig = px.histogram(
                        data,
                        x="Customer_Tenure",
                        title="Customer Tenure Distribution",
                        color="Segment" if "Segment" in data.columns else None,
                        nbins=20
                    )
                    st.plotly_chart(fig)
                
                with col2:
                    # Segment loyalty distribution
                    loyalty_counts = data["Segment_Loyalty"].value_counts().reset_index()
                    loyalty_counts.columns = ['Loyalty Level', 'Count']
                    
                    fig = px.pie(
                        loyalty_counts,
                        values="Count",
                        names="Loyalty Level",
                        title="Customer Loyalty Distribution",
                        color_discrete_sequence=px.colors.sequential.Plasma_r
                    )
                    st.plotly_chart(fig)
                
                # Tenure vs CLV
                if "Monetary_Value" in data.columns:
                    st.subheader("CLV by Customer Tenure")
                    
                    # Create tenure bins
                    data['Tenure_Group'] = pd.cut(
                        data['Customer_Tenure'], 
                        bins=[0, 1, 2, 3, 5, 10, float('inf')],
                        labels=['<1 Year', '1-2 Years', '2-3 Years', '3-5 Years', '5-10 Years', '10+ Years']
                    )
                    
                    tenure_clv = data.groupby('Tenure_Group')['Monetary_Value'].mean().reset_index()
                    
                    fig = px.bar(
                        tenure_clv,
                        x="Tenure_Group",
                        y="Monetary_Value",
                        title="Average CLV by Customer Tenure",
                        color="Monetary_Value",
                        color_continuous_scale=px.colors.sequential.Viridis,
                        labels={"Monetary_Value": "Average CLV ($)", "Tenure_Group": "Customer Tenure"}
                    )
                    st.plotly_chart(fig)
                    
                    # Loyalty vs CLV
                    if "Segment_Loyalty" in data.columns and "Monetary_Value" in data.columns:
                        st.subheader("CLV by Customer Loyalty")
                        loyalty_clv = data.groupby("Segment_Loyalty")["Monetary_Value"].mean().reset_index()
                        loyalty_clv = loyalty_clv.sort_values("Monetary_Value")
                        
                        fig = px.bar(
                            loyalty_clv,
                            x="Segment_Loyalty",
                            y="Monetary_Value",
                            title="Average CLV by Loyalty Segment",
                            color="Monetary_Value",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            labels={"Monetary_Value": "Average CLV ($)", "Segment_Loyalty": "Loyalty Level"}
                        )
                        st.plotly_chart(fig)
            
            # Purchase Behavior Analysis
            st.header("Purchase Behavior Analysis")
            
            # Purchase Frequency Analysis
            if "Purchase_Frequency" in data.columns:
                st.subheader("Purchase Frequency Distribution")
                fig = px.histogram(
                    data,
                    x="Purchase_Frequency",
                    title="Customer Purchase Frequency Distribution",
                    nbins=20,
                    color_discrete_sequence=['#636EFA']
                )
                st.plotly_chart(fig)
                
                # Purchase Frequency vs CLV
                if "Monetary_Value" in data.columns:
                    st.subheader("Purchase Frequency vs CLV")
                    
                    fig = px.scatter(
                        data,
                        x="Purchase_Frequency",
                        y="Monetary_Value",
                        color="Segment" if "Segment" in data.columns else None,
                        size="Annualized_Spend" if "Annualized_Spend" in data.columns else None,
                        hover_name="Customer_Name" if "Customer_Name" in data.columns else None,
                        title="Relationship Between Purchase Frequency and CLV",
                        labels={"Purchase_Frequency": "Purchase Frequency", "Monetary_Value": "CLV ($)"}
                    )
                    
                    # Add trendline
                    fig.update_layout(height=600)
                    fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')), selector=dict(mode='markers'))
                    st.plotly_chart(fig)
                    
                    # Calculate correlation
                    corr = data[["Purchase_Frequency", "Monetary_Value"]].corr().iloc[0, 1]
                    st.info(f"Correlation between Purchase Frequency and CLV: {corr:.3f}")
            
            # Top Product Categories
            st.subheader("Top 10 Product Categories")
            if "Top_Product_Category" in data.columns:
                top_products = data["Top_Product_Category"].value_counts().head(10)
                fig = px.bar(
                    top_products, 
                    x=top_products.index, 
                    y=top_products.values, 
                    title="Top 10 Product Categories",
                    color=top_products.values,
                    color_continuous_scale=px.colors.sequential.Plasma,
                    labels={"x": "Product Category", "y": "Number of Customers"}
                )
                st.plotly_chart(fig)
            else:
                st.warning("Column 'Top_Product_Category' not found in dataset.")
            
            # Word Cloud for Frequent Purchases
            st.subheader("Frequent Purchase Word Cloud")
            if "Top_Product_Category" in data.columns:
                words = " ".join(data["Top_Product_Category"].dropna())
                wordcloud = WordCloud(
                    stopwords=STOPWORDS, 
                    background_color="white",
                    width=800,
                    height=400,
                    max_words=100,
                    colormap='viridis'
                ).generate(words)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            
            # High Spender Analysis
            st.header("High Value Customer Analysis")
            if "High_Spender_Flag" in data.columns and "Annualized_Spend" in data.columns:
                # Count of high spenders
                high_spender_count = data["High_Spender_Flag"].sum()
                high_spender_pct = high_spender_count / len(data) * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("High Value Customers", f"{high_spender_count} ({high_spender_pct:.1f}%)")
                
                with col2:
                    if "Monetary_Value" in data.columns:
                        high_clv = data[data["High_Spender_Flag"] == 1]["Monetary_Value"].mean()
                        normal_clv = data[data["High_Spender_Flag"] == 0]["Monetary_Value"].mean()
                        clv_diff_pct = (high_clv - normal_clv) / normal_clv * 100
                        
                        st.metric("Avg CLV Difference", f"+{clv_diff_pct:.1f}%", 
                                delta=f"${high_clv - normal_clv:.2f}")
                
                # High spender spend distribution
                high_spenders = data[data["High_Spender_Flag"] == 1]
                fig = px.histogram(
                    high_spenders, 
                    x="Annualized_Spend", 
                    title="Annualized Spend of High Value Customers",
                    nbins=30,
                    color_discrete_sequence=['#636EFA']
                )
                fig.update_layout(xaxis_title="Annualized Spend ($)", yaxis_title="Count")
                st.plotly_chart(fig)
                
                # High spender attributes
                if "Segment" in data.columns:
                    st.subheader("High Value Customer Segments")
                    
                    high_spender_segments = high_spenders["Segment"].value_counts().reset_index()
                    high_spender_segments.columns = ['Segment', 'Count']
                    high_spender_segments['Percentage'] = high_spender_segments['Count'] / high_spender_count * 100
                    
                    fig = px.pie(
                        high_spender_segments, 
                        values='Count', 
                        names='Segment',
                        title="High Value Customers by Segment",
                        hover_data=['Percentage'],
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig)
                
                # Top categories for high spenders
                if "Top_Product_Category" in high_spenders.columns:
                    st.subheader("Preferred Categories of High Value Customers")
                    high_spender_categories = high_spenders["Top_Product_Category"].value_counts().head(5)
                    
                    fig = px.pie(
                        names=high_spender_categories.index,
                        values=high_spender_categories.values,
                        title="Top Categories for High Value Customers",
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    st.plotly_chart(fig)
            else:
                st.warning("Columns 'High_Spender_Flag' or 'Annualized_Spend' not found in dataset.")
            
            
                
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
            
            # CLV Prediction Factor Analysis
            if "Monetary_Value" in data.columns:
                st.header("CLV Prediction Factor Analysis")
                
                # Select numerical columns for correlation analysis
                potential_factors = [
                    'Recency', 'Frequency', 'Customer_Tenure', 'Annualized_Spend', 
                    'Age', 'Annual_Income', 'Purchase_Frequency', 'Avg_Time_Between_Purchases',
                    'Return_Rate', 'Support_Contact_Rate', 'Email_Engagement_Rate', 
                    'Satisfaction_Score', 'Promotion_Responsiveness', 'Product_Return_Rate'
                ]
                
                available_factors = [col for col in potential_factors if col in data.columns]
                
                if available_factors:
                    # Calculate correlations with CLV
                    correlations = []
                    for factor in available_factors:
                        corr = data[[factor, "Monetary_Value"]].corr().iloc[0, 1]
                        correlations.append({"Factor": factor, "Correlation": corr})
                    
                    corr_df = pd.DataFrame(correlations)
                    corr_df = corr_df.sort_values("Correlation", ascending=False)
                    
                    # Plot correlation factors
                    fig = px.bar(
                        corr_df,
                        x="Factor",
                        y="Correlation",
                        title="Factors Correlated with CLV",
                        color="Correlation",
                        color_continuous_scale=px.colors.diverging.RdBu,
                        range_color=[-1, 1]
                    )
                    fig.update_layout(xaxis_title="Customer Attribute", yaxis_title="Correlation with CLV")
                    st.plotly_chart(fig)
            
            # Demographic Analysis
            st.header("Demographic Impact on CLV")
            
            # Age Group Analysis
            if "Age_Group" in data.columns and "Monetary_Value" in data.columns:
                st.subheader("CLV by Age Group")
                
                age_clv = data.groupby("Age_Group")["Monetary_Value"].mean().reset_index()
                
                fig = px.bar(
                    age_clv,
                    x="Age_Group",
                    y="Monetary_Value",
                    title="Average CLV by Age Group",
                    color="Monetary_Value",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    labels={"Monetary_Value": "Average CLV ($)", "Age_Group": "Age Group"}
                )
                st.plotly_chart(fig)
            
            # Income Segment Analysis
            if "Income_Segment" in data.columns and "Monetary_Value" in data.columns:
                st.subheader("CLV by Income Segment")
                
                income_clv = data.groupby("Income_Segment")["Monetary_Value"].mean().reset_index()
                
                fig = px.bar(
                    income_clv,
                    x="Income_Segment",
                    y="Monetary_Value",
                    title="Average CLV by Income Segment",
                    color="Monetary_Value",
                    color_continuous_scale=px.colors.sequential.Plasma,
                    labels={"Monetary_Value": "Average CLV ($)", "Income_Segment": "Income Segment"}
                )
                st.plotly_chart(fig)
            
            # Gender Analysis if available
            if "Gender" in data.columns and "Monetary_Value" in data.columns:
                st.subheader("CLV by Gender")
                
                gender_clv = data.groupby("Gender")["Monetary_Value"].mean().reset_index()
                
                fig = px.bar(
                    gender_clv,
                    x="Gender",
                    y="Monetary_Value",
                    title="Average CLV by Gender",
                    color="Gender",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    labels={"Monetary_Value": "Average CLV ($)"}
                )
                st.plotly_chart(fig)
            
            # Regional Analysis
            if "Region" in data.columns and "Monetary_Value" in data.columns:
                st.header("Regional CLV Analysis")
                
                # CLV by Region
                region_clv = data.groupby("Region")["Monetary_Value"].mean().reset_index()
                region_clv = region_clv.sort_values("Monetary_Value", ascending=False)
                
                fig = px.bar(
                    region_clv,
                    x="Region",
                    y="Monetary_Value",
                    title="Average CLV by Region",
                    color="Monetary_Value",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    labels={"Monetary_Value": "Average CLV ($)"}
                )
                st.plotly_chart(fig)
                
            def generate_full_report():
                report_data = data.copy()
                buffer = io.BytesIO()
                
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    report_data.to_excel(writer, sheet_name='Analytics', index=False)
                    sales_team_allocation.to_excel(writer, sheet_name='Sales_Team', index=False)
                    marketing_optimization.to_excel(writer, sheet_name='Marketing_Strategy', index=False)
                    discount_suggestions.to_excel(writer, sheet_name='Discount_Strategy', index=False)
                
                buffer.seek(0)
                return buffer
            
            if st.sidebar.button("Generate Report"):
                report = generate_full_report()
                st.sidebar.download_button(
                    label="Download Report (XLSX)",
                    data=report,
                    file_name="Business_Analytics_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")
  