import streamlit as st
import requests


BACKEND_URL = "http://127.0.0.1:8000/predict"  

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn")

# Input fields (matching columns exactly)
Tenure = st.number_input("Tenure", min_value=0, step=1)
City_Tier = st.number_input("City Tier", min_value=0, step=1)
Service_Score = st.number_input("Service Score", min_value=0, step=1)
Account_user_count = st.number_input("Account User Count", min_value=0, step=1)
account_segment = st.number_input("Account Segment", min_value=0, step=1)
Complain_ly = st.number_input("Complain Last Year (0/1)", min_value=0, max_value=1)
Day_Since_CC_connect = st.number_input("Days Since CC Connect", min_value=0, step=1)
cashback = st.number_input("Cashback", min_value=0.0, step=0.01)

if st.button("Predict"):
    payload = {
        "Tenure": Tenure,
        "City_Tier": City_Tier,
        "Service_Score": Service_Score,
        "Account_user_count": Account_user_count,
        "account_segment": account_segment,
        "Complain_ly": Complain_ly,
        "Day_Since_CC_connect": Day_Since_CC_connect,
        "cashback": cashback
    }

    st.json(payload)  

    try:
        response = requests.post(BACKEND_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            if "prediction" in result:
                st.success("Prediction Successful!")
                st.write("Predicted churn probability / class:")
                st.json(result['prediction'])
            else:
                st.error(f"Error from backend: {result.get('error')}")
        else:
            st.error(f"Prediction failed with status code {response.status_code}")
    except Exception as e:
        st.error(f"Error: {e}")
