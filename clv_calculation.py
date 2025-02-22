import streamlit as st
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
import plotly.express as px

class CLVCalculation:
    def __init__(self):
        self.data = None
        self.customer_data = None
        self.bg_model = None
        self.gg_model = None

    def upload_data(self, data):
        """Uploads the data for CLV calculation"""
        if data is not None:
            self.data = data
            self.customer_data = self.data.copy()
            st.success("Data uploaded successfully!")
        else:
            st.error("No data uploaded.")
    


    
    '''
    def preprocess_data(self):
        """Prepares the dataset for CLV calculation"""
        if self.data is not None:
            st.write("### Preprocessing Data")

            # Drop rows with missing values
            self.customer_data = self.data.dropna()

            # Rename columns for model compatibility
            self.customer_data.rename(columns={
                "Customer ID": "customer_id",
                "Total_Spend": "monetary_value",
                "Num_of_Purchases": "frequency",
                "Last_Purchase_Days_Ago": "recency",  # Ensure 'recency' column is correctly named
                "Years_as_Customer": "T",
                "Average_Transaction_Amount": "avg_transaction_amount",
                "Time_Between_Purchases": "time_between_purchases"
            }, inplace=True)

            # Additional feature engineering (ensure columns exist in the data)
            if "Customer_Tenure" in self.customer_data.columns:
                self.customer_data['tenure'] = self.customer_data['Customer_Tenure']
            if "Purchase_Frequency" in self.customer_data.columns:
                self.customer_data['purchase_frequency'] = self.customer_data['Purchase_Frequency']
            if "Avg_Time_Between_Purchases" in self.customer_data.columns:
                self.customer_data['avg_time_between_purchases'] = self.customer_data['Avg_Time_Between_Purchases']
            if "Annualized_Spend" in self.customer_data.columns:
                self.customer_data['annualized_spend'] = self.customer_data['Annualized_Spend']
            if "Return_Rate" in self.customer_data.columns:
                self.customer_data['return_rate'] = self.customer_data['Return_Rate']
            if "Support_Contact_Rate" in self.customer_data.columns:
                self.customer_data['support_contact_rate'] = self.customer_data['Support_Contact_Rate']
            if "Satisfaction_Per_Purchase" in self.customer_data.columns:
                self.customer_data['satisfaction_per_purchase'] = self.customer_data['Satisfaction_Per_Purchase']

            st.success("Data preprocessed successfully!")
        else:
            st.error("No data available to preprocess.")

    def fit_models(self):
        """Fits the BG/NBD and Gamma-Gamma models for CLV calculation."""
        if self.customer_data is not None:
            st.write("### Fitting Models")

            # Ensure relevant columns are numeric
            numeric_columns = [
                "Recency", "Customer_Tenure", "Frequency", "Monetary_Value",
                "Average_Transaction_Amount", "Annualized_Spend", "Return_Rate",
                "Support_Contact_Rate", "Satisfaction_Per_Purchase"
            ]

            for col in numeric_columns:
                if col in self.customer_data.columns:
                    self.customer_data[col] = pd.to_numeric(self.customer_data[col], errors='coerce')
                else:
                    st.warning(f"Column '{col}' is missing from the dataset. Defaulting to 0.")
                    self.customer_data[col] = 0

            # Prepare the CLV data by aggregating customer data
            clv_data = self.customer_data.groupby("Customer ID").agg({
                "Recency": "max",
                "Customer_Tenure": "max",
                "Frequency": "sum",
                "Monetary_Value": "mean",
                "Average_Transaction_Amount": "mean",
                "Annualized_Spend": "mean",
                "Return_Rate": "mean",
                "Support_Contact_Rate": "mean",
                "Satisfaction_Per_Purchase": "mean"
            }).reset_index()

            # Remove rows with invalid data
            clv_data = clv_data[
                ~((clv_data["Frequency"] == 0) & (clv_data["Recency"] > 0))  # Invalid frequency/recency combo
            ]
            clv_data = clv_data[clv_data["Recency"] <= clv_data["Customer_Tenure"]]  # Recency cannot exceed tenure
            clv_data = clv_data[clv_data["Frequency"] > 0]  # At least one purchase

            if clv_data.empty:
                st.error("No valid data available after filtering. Please check your dataset.")
                return

            # Fit the BetaGeoFitter model
            self.bg_model = BetaGeoFitter(penalizer_coef=10.0)
            try:
                self.bg_model.fit(clv_data["Frequency"], clv_data["Recency"], clv_data["Customer_Tenure"])
                st.success("BetaGeoFitter model fitted successfully!")
            except Exception as e:
                st.error(f"Error fitting BetaGeoFitter model: {e}")
                return

            # Fit the GammaGammaFitter model
            self.gg_model = GammaGammaFitter()
            try:
                self.gg_model.fit(clv_data["Frequency"], clv_data["Monetary_Value"])
                st.success("GammaGammaFitter model fitted successfully!")
            except Exception as e:
                st.error(f"Error fitting GammaGammaFitter model: {e}")
        else:
            st.error("No data available for modeling.")

    def calculate_clv(self):
        """Calculates Customer Lifetime Value for each customer."""
        if self.bg_model is not None and self.gg_model is not None and self.customer_data is not None:
            st.write("### Calculating CLV")

            try:
                # Ensure predicted_clv column is created
                self.customer_data["predicted_clv"] = self.gg_model.customer_lifetime_value(
                    self.bg_model,
                    frequency=self.customer_data["Frequency"],
                    recency=self.customer_data["Recency"],
                    T=self.customer_data["Customer_Tenure"],
                    monetary_value=self.customer_data["Monetary_Value"],
                    time=12,  # Predict CLV for 12 months
                    freq="D"
                )
                st.dataframe(self.customer_data[["Customer ID", "predicted_clv"]].head())
                self.customer_data.to_excel("clv_results.xlsx", index=False)
                st.success("CLV results saved to clv_results.xlsx")
            except Exception as e:
                st.error(f"Error calculating CLV: {e}")
        else:
            st.error("Models are not fitted or customer data is missing. Please fit the models first.")

    def plot_clv_distribution(self):
        """Plots the distribution of predicted CLV."""
        if self.customer_data is not None and "predicted_clv" in self.customer_data.columns:
            st.write("### CLV Distribution")
            fig = px.histogram(self.customer_data, x="predicted_clv", nbins=50, title="CLV Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No CLV data available to plot.")


# Example usage
# customer_data = pd.read_excel('input_file.xlsx')
# clv_calculator = CLVCalculator()
# clv_calculator.upload_data(customer_data)
# clv_calculator.fit_models()
# clv_calculator.calculate_clv()
# clv_calculator.plot_clv_distribution()

'''
    def preprocess_data(self):
        """Preprocesses the customer data."""
        if self.customer_data is not None:
            st.write("### Preprocessing Data")

            # Ensure required columns exist
            required_columns = [
                "Customer ID", "Frequency", "Monetary_Value", 
                "Recency", "Customer_Tenure"
            ]
            for col in required_columns:
                if col not in self.customer_data.columns:
                    st.error(f"Column '{col}' is missing from the dataset.")
                    return

            # Convert necessary columns to numeric and handle invalid values
            self.customer_data['Frequency'] = pd.to_numeric(
                self.customer_data['Frequency'], errors='coerce'
            ).fillna(0).astype(int)

            self.customer_data['Monetary_Value'] = pd.to_numeric(
                self.customer_data['Monetary_Value'], errors='coerce'
            ).fillna(0)

            self.customer_data['Recency'] = pd.to_numeric(
                self.customer_data['Recency'], errors='coerce'
            ).fillna(0)

            self.customer_data['Customer_Tenure'] = pd.to_numeric(
                self.customer_data['Customer_Tenure'], errors='coerce'
            ).fillna(0)

            # Filter out rows where Frequency is zero or invalid
            initial_row_count = len(self.customer_data)
            self.customer_data = self.customer_data[self.customer_data['Frequency'] > 0]
            filtered_row_count = len(self.customer_data)

            st.write(f"Filtered out {initial_row_count - filtered_row_count} rows with invalid or zero Frequency.")
            st.success("Data preprocessed successfully!")
        else:
            st.error("No data available for preprocessing.")

    def fit_gamma_gamma_model(self):
        """Fits the Gamma-Gamma model for CLV calculation."""
        if self.customer_data is not None:
            st.write("### Fitting Gamma-Gamma Model")

            try:
                # Fit the Gamma-Gamma model
                self.gg_model = GammaGammaFitter()
                self.gg_model.fit(
                    self.customer_data['Frequency'],
                    self.customer_data['Monetary_Value']
                )
                st.success("Gamma-Gamma model fitted successfully!")
            except Exception as e:
                st.error(f"Error fitting Gamma-Gamma model: {str(e)}")
        else:
            st.error("No data available for modeling.")

    def calculate_clv(self):
        """Calculates Customer Lifetime Value (CLV)."""
        if self.gg_model is not None and self.customer_data is not None:
            st.write("### Calculating CLV")

            try:
                # Calculate CLV for each customer
                self.customer_data['predicted_clv'] = self.gg_model.conditional_expected_average_profit(
                    self.customer_data['Frequency'],
                    self.customer_data['Monetary_Value']
                )

                st.dataframe(self.customer_data[['Customer ID', 'predicted_clv']].head())
                self.customer_data.to_excel("clv_results.xlsx", index=False)
                st.success("CLV results saved to clv_results.xlsx")
            except Exception as e:
                st.error(f"Error calculating CLV: {str(e)}")
        else:
            st.error("Gamma-Gamma model is not fitted or customer data is missing. Please fit the model first.")

    def plot_clv_distribution(self):
        """Plots the distribution of predicted CLV."""
        if self.customer_data is not None and 'predicted_clv' in self.customer_data.columns:
            st.write("### CLV Distribution")
            fig = px.histogram(self.customer_data, x="predicted_clv", nbins=50, title="CLV Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No CLV data available to plot.")

# Example usage
# customer_data = pd.read_excel('input_file.xlsx')
# clv_calculator = CLVCalculator()
# clv_calculator.upload_data(customer_data)
# clv_calculator.preprocess_data()
# clv_calculator.fit_gamma_gamma_model()
# clv_calculator.calculate_clv()
# clv_calculator.plot_clv_distribution()
