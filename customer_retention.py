def apply_discounts(data):
    """
    Identifies high-risk churn customers and offers discounts.
    Also rewards top revenue-generating customers with special offers.
    """
    # Identify high-risk churn customers (Churn Prediction > 0.8)
    data["Retention_Offer"] = "None"
    data.loc[data["Churn_Prediction"] > 0.8, "Retention_Offer"] = "20% Discount"

    # Reward customers who generate maximum sales (Top 10% spenders)
    high_spenders = data["Annualized_Spend"].quantile(0.90)
    data.loc[data["Annualized_Spend"] >= high_spenders, "Retention_Offer"] = "Exclusive VIP Membership"

    return data
