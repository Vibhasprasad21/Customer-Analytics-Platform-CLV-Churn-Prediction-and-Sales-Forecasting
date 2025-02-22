def allocate_sales_teams(data):
    """
    Assigns sales team to top CLV customers.
    """
    high_value_customers = data.sort_values("CLV", ascending=False).head(10)
    high_value_customers["Assigned_Sales_Rep"] = [f"Rep {i}" for i in range(1, 11)]
    return high_value_customers
