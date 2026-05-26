import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('xgb_model.pkl')
encoder = joblib.load('encoder.pkl')
feature_columns = joblib.load('feature_columns.pkl')
scaler = joblib.load('scaler.pkl')

st.title("FMCG Purchase Propensity Dashboard")
st.write(
    "Simulate FMCG marketing campaigns and predict "
    "customer purchase likelihood using Machine Learning."
)

select_page = st.sidebar.selectbox(
    "Navigation",
    [
        'Dashboard',
        'Campaign Simulator',
        'Model Insights'
    ]
)

if select_page == 'Dashboard':
    st.subheader("Project Overview")
    st.write("""
    This dashboard predicts whether customers are likely
    to purchase FMCG products based on:
    - Product pricing
    - Discounts
    - Campaign strategies
    - Customer segments
    - Product categories
    """)

elif select_page == 'Campaign Simulator':
    st.subheader("Campaign Simulator")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Product Configuration")
        price = st.number_input(
            "Product Price",
            min_value=10,
            max_value=10000,
            value=500
        )
        discount = st.slider(
            "Discount Percentage",
            0,
            80,
            20
        )
        product_category = st.selectbox(
            "Product Category",
            [
                "Beverages",
                "Dairy",
                "Frozen Food",
                "Household",
                "Personal Care",
                "Snacks"
            ]
        )
    with col2:
            st.subheader("Campaign Context")
            festival_season = st.selectbox(
                "Festival Season",
                [
                    "No",
                    "Yes"
                ]
            )
            ad_campaign = st.selectbox(
                "Ad Campaign Running",
                [
                    "No",
                    "Yes"
                ]
            )
            city = st.selectbox(
                "Target City",
                [
                    "Bangalore",
                    "Chennai",
                    "Delhi",
                    "Hyderabad",
                    "Kolkata",
                    "Mumbai"
                ]
            )
    st.subheader("Customer Segment")
    customer_segment = st.selectbox(
        "Target Customer Segment",
        [
            'Budget Customer',
            'Frequent Buyer',
            'Premium Customer',
            'Impulse Buyer',
            'New Customer'
        ]
    )

    predict_button = st.button(
        "Predict Purchase Probability"
    )

    if predict_button:
        if customer_segment == "Premium Customer":
            gender = np.random.choice(
    ["Male", "Female"]
            )
            income_level = np.random.randint(120000, 250000)
            brand_loyalty_score = np.random.uniform(8, 10)
            previous_purchases = np.random.randint(15, 40)
            avg_monthly_spend = np.random.randint(20000, 60000)
            membership_level = "Platinum"
            payment_mode = "Credit Card"

        elif customer_segment == "Budget Customer":
            gender = np.random.choice(
    ["Male", "Female"]
)
            income_level = np.random.randint(20000, 60000)
            brand_loyalty_score = np.random.uniform(3, 6)
            previous_purchases = np.random.randint(1, 10)
            avg_monthly_spend = np.random.randint(2000, 10000)
            membership_level = "Bronze"
            payment_mode = "Cash"

        elif customer_segment == "Frequent Buyer":
            gender = np.random.choice(
    ["Male", "Female"]
)
            income_level = np.random.randint(50000, 120000)

            brand_loyalty_score = np.random.uniform(7, 10)

            previous_purchases = np.random.randint(20, 50)

            avg_monthly_spend = np.random.randint(10000, 40000)

            membership_level = "Gold"

            payment_mode = "UPI"

        elif customer_segment == "Impulse Buyer":
            gender = np.random.choice(
    ["Male", "Female"]
            )
            income_level = np.random.randint(40000, 100000)

            brand_loyalty_score = np.random.uniform(4, 7)

            previous_purchases = np.random.randint(5, 20)

            avg_monthly_spend = np.random.randint(5000, 20000)

            membership_level = "Silver"

            payment_mode = "Wallet"

        elif customer_segment == "New Customer":
            gender = np.random.choice(
    ["Male", "Female"]
)
            income_level = np.random.randint(30000, 80000)

            brand_loyalty_score = np.random.uniform(1, 4)

            previous_purchases = np.random.randint(0, 3)

            avg_monthly_spend = np.random.randint(1000, 8000)

            membership_level = "Bronze"

            payment_mode = "Debit Card"

        input_data = {

            'Income_Level': income_level,

            'Product_Price': price,

            'Discount_Percentage': discount,

            'Brand_Loyalty_Score': brand_loyalty_score,

            'Previous_Purchases': previous_purchases,

            'Avg_Monthly_Spend': avg_monthly_spend,

            'Stock_Availability': 1,

            'Ad_Clicked': 1 if ad_campaign == "Yes" else 0,

            'Festival_Season': 1 if festival_season == "Yes" else 0,

            'City': city,

            'Gender': gender,

            'Product_Category': product_category,

            'Membership_Level': membership_level,

            'Payment_Mode': payment_mode

        }

        input_df = pd.DataFrame([input_data])

        categorical_cols = [
            'Gender',
            'City',
            'Product_Category',
            'Payment_Mode',
            'Membership_Level'
        ]

        encoded = encoder.transform(
            input_df[categorical_cols]
        ).toarray()

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(
                categorical_cols
            ),
            index=input_df.index
        )

        # Remove original categorical columns
        input_df = input_df.drop(
            columns=categorical_cols
        )

        # Add encoded columns
        input_df = pd.concat(
            [input_df, encoded_df],
            axis=1
        )

        # Match training feature order
        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Scale input
        input_scaled = scaler.transform(
            input_df
        )
        # Prediction
        prediction = model.predict(
            input_scaled
        )[0]
        # Prediction probability
        probability = model.predict_proba(
            input_scaled
        )[0][1]
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(

                f"Customer is likely to BUY the product."
            )
        else:
            st.error(
                f"Customer is NOT likely to buy the product."
            )
        st.metric(
            "Purchase Probability",
            f"{probability * 100:.2f}%"
        )
elif select_page == 'Model Insights':

    st.subheader("Model Insights")

    st.write("""

    Model Used:

    - XGBoost Classifier

    Evaluation Metrics:

    - Accuracy

    - Precision

    - Recall

    - F1 Score

    The model was trained on synthetic FMCG

    customer behavior data to simulate

    purchase propensity prediction.

    """)