import numpy as np
import pandas as pd

def allocate_sales_teams(data):
    """
    Assigns sales team to top CLV customers.
    """
    high_value_customers = data.sort_values("CLV", ascending=False).head(10).copy()
    high_value_customers["Assigned_Sales_Rep"] = [f"Rep {i}" for i in range(1, 11)]
    return high_value_customers

def optimize_marketing(data):
    """
    Allocates marketing budget based on Customer Lifetime Value (CLV).
    """
    return np.where(data["CLV"] > 400, 5000, np.where(data["CLV"] > 200, 3000, 1000))
def apply_discounts(data):
    """
    Identifies high-risk churn customers and offers discounts.
    Also rewards top revenue-generating customers with special offers.
    """
    data["Retention_Offer"] = "None"

    # Identify high-risk churn customers (Churn Prediction > 0.8)
    data.loc[data["Churn_Label"] > 0.8, "Retention_Offer"] = "20% Discount"

    # Reward customers who generate maximum sales (Top 10% spenders)
    high_spenders = data["Annualized_Spend"].quantile(0.90)
    data.loc[data["Annualized_Spend"] >= high_spenders, "Retention_Offer"] = "Exclusive VIP Membership"

    return data
