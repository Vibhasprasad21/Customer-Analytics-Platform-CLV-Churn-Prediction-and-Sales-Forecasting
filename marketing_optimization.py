import numpy as np

def optimize_marketing(data):
    """
    Allocates marketing budget based on CLV and Churn Risk.
    High CLV → Higher Marketing Spend.
    High Churn Risk → Targeted Retention Efforts.
    """
    budget = np.where(data["CLV"] > 20000, 15000, 
                      np.where(data["CLV"] > 10000, 10000, 5000))

    # Increase marketing spend for high-risk churn customers
    budget = np.where(data["Churn_Prediction"] > 0.8, budget + 5000, budget)
    
    return budget
