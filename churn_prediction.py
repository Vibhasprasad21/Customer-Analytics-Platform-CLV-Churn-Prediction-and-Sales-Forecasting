import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import plotly.express as px

class ChurnPrediction:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.features = [
        'Age', 'Gender', 'Annual_Income', 'Total_Spend', 'Years_as_Customer',
        'Num_of_Purchases', 'Average_Transaction_Amount', 'Num_of_Returns',
        'Num_of_Support_Contacts', 'Satisfaction_Score',
        'Email_Opt_In', 'Segment', 'Country', 'City', 'State', 'Region',
        'Product ID', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Profit',
        'Top_Product_Category', 'Promotion_Response', 'Marketing_Channel',
        'Time_Between_Purchases', 'Customer_Tenure', 'Purchase_Frequency',
        'Avg_Time_Between_Purchases', 'Annualized_Spend', 'Return_Rate',
        'Support_Contact_Rate', 'Email_Engagement_Rate', 'Promotion_Responsiveness',
        'Satisfaction_Per_Purchase', 'Recency', 'Frequency', 'Monetary_Value',
        'Income_Segment', 'Age_Group', 'Diversity_of_Purchases', 'High_Spender_Flag',
        'Discount_Dependency', 'Preferred_Marketing_Channel', 'Reduced_Purchase_Activity',
        'Region_Based_Spend', 'Region_Profitability', 'Customer_Density',
        'Signup_Year', 'Signup_Month', 'Purchase_Seasonality', 'Customer_Segment_Performance',
        'Segment_Loyalty'
         ]

        self.target = 'Churn_Label'

    def upload_data(self):
        """Uploads the churn prediction dataset."""
        # Pass a unique key for the file uploader
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"], key="churn_file_uploader")

        if uploaded_file is not None:
            # Process the uploaded file
            if uploaded_file.name.endswith("csv"):
                self.customer_data = pd.read_csv(uploaded_file)
            else:
                self.customer_data = pd.read_excel(uploaded_file)
            
            st.write("### Data Uploaded Successfully")
            st.write(self.customer_data.head())
        else:
            st.warning("Please upload a file to continue.")

    def preview_data(self):
        """Displays the first few rows of the uploaded dataset"""
        if self.data is not None:
            st.write("### Dataset Preview")
            st.dataframe(self.data.head())
        else:
            st.error("No data available to preview.")

    def data_info(self):
        """Displays the dataset info and null value distribution"""
        if self.data is not None:
            st.write("### Dataset Info")
            st.write(f"**Number of rows:** {self.data.shape[0]}")
            st.write(f"**Number of columns:** {self.data.shape[1]}")
            st.write("### Null Value Distribution")
            null_values = self.data.isnull().sum()
            fig = px.bar(null_values, title="Null Value Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No data available for info.")
    def handle_infinite_values(self):
        """Handle infinite or NaN values in the dataset"""
        if self.data is not None:
            # Handle datetime columns separately
            datetime_columns = self.data.select_dtypes(include=['datetime64[ns]']).columns
            for col in datetime_columns:
                # Fill missing datetime values with a placeholder or the earliest date
                self.data[col].fillna(pd.to_datetime('1970-01-01'), inplace=True)

            # Now handle the rest of the columns (numerical ones)
            numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
            self.data[numeric_columns] = self.data[numeric_columns].replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
            self.data.fillna(self.data[numeric_columns].mean(), inplace=True)  # Replace NaN with column mean
            
            st.success("Infinite values and NaN values have been handled.")
        else:
            st.error("No data available to handle infinite or NaN values.")




    def select_columns(self):
        """Selects necessary features and target column"""
        if self.data is not None:
            if self.target not in self.data.columns:
                st.error(f"Target column '{self.target}' is missing in the dataset.")
                return
            self.data = self.data[self.features + [self.target]]
            st.success("Relevant features and target column have been selected.")
        else:
            st.error("No data available for feature selection.")

    def feature_scaling(self):
        """Scales numerical features using Min-Max Scaling"""
        if self.data is not None:
            # Handle infinite values before scaling
            self.handle_infinite_values()

            numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns
            self.data[numeric_features] = self.scaler.fit_transform(self.data[numeric_features])
            st.success("Features have been scaled.")
        else:
            st.error("No data available for feature scaling.")
    def preprocess_data(self):
        """Preprocesses the dataset to ensure compatibility with XGBoost."""
        if self.data is not None:
            # Handle datetime columns
            datetime_cols = self.data.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                self.data[f"{col}_Year"] = self.data[col].dt.year
                self.data[f"{col}_Month"] = self.data[col].dt.month
                self.data[f"{col}_Day"] = self.data[col].dt.day
            self.data.drop(datetime_cols, axis=1, inplace=True)

            # Handle object columns
            object_cols = self.data.select_dtypes(include=['object']).columns
            for col in object_cols:
                if self.data[col].nunique() < 20:  # Use label encoding for low-cardinality categorical columns
                    self.data[col] = self.data[col].astype('category').cat.codes
                else:  # Drop high-cardinality columns or use alternative encoding methods
                    self.data.drop(col, axis=1, inplace=True)

            # Ensure all columns are numeric
            self.data = self.data.apply(pd.to_numeric, errors='coerce')

            # Handle NaN or infinite values after conversion
            self.data.fillna(0, inplace=True)
            self.data.replace([np.inf, -np.inf], 0, inplace=True)

            st.success("Data preprocessing complete.")
        else:
            st.error("No data available for preprocessing.")


    
    def train_model(self):
        """Trains an XGBoost model for churn prediction."""
        if self.data is not None:
            # Preprocess the data
            self.preprocess_data()

            # Check if the target column exists
            if self.target not in self.data.columns:
                st.warning(f"Target column '{self.target}' is missing. Creating and predicting '{self.target}'.")
                self.data[self.target] = np.random.choice([0, 1], size=len(self.data))  # Randomly create target for demo
                st.info("Target column created randomly for demonstration purposes.")

            # Split data into features and target
            X = self.data.drop(self.target, axis=1)
            y = self.data[self.target]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and train the XGBoost model
            self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            self.model.fit(X_train, y_train)

            # Make predictions and evaluate the model
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Model trained successfully with accuracy: {acc:.2f}")
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))
        else:
            st.error("No data available for training.")

    def predict_churn(self):
        """Predicts churn labels for the dataset and saves results to an Excel file."""
        if self.data is not None and self.model is not None:
            try:
                X = self.data.drop(self.target, axis=1)  # Drop target column to get features

                # Ensure the model is ready to make predictions
                if hasattr(self.model, 'predict'):
                    # Predict the churn labels (0 or 1)
                    self.data['Churn_Label'] = self.model.predict(X)

                    # Save results to an Excel file
                    output_file = "churn_predictions.xlsx"
                    self.data.to_excel(output_file, index=False)

                    st.success(f"Churn predictions saved to '{output_file}'.")
                    st.write("### Churn Predictions")

                    # Display the churn predictions (only Churn_Label column)
                    if 'Churn_Label' in self.data.columns:
                        st.dataframe(self.data[['Churn_Label']])
                    else:
                        st.error("Churn predictions not found in the data.")
                else:
                    st.error("Model does not have the 'predict' method.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.error("Model is not trained or data is unavailable.")




    '''
    def visualize_churn(self):
        """Visualizes churn probabilities"""
        if self.data is not None and 'Churn_Probability' in self.data.columns:
            fig = px.histogram(
                self.data, x='Churn_Probability', nbins=20, title="Churn Probability Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No churn predictions available for visualization.")
    '''