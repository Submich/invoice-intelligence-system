import streamlit as st
import pandas as pd
import numpy as np
import joblib

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

st.set_page_config(
    page_title="Vendor Invoice Intelligence System",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- UI TITLE -------------------

st.title("📊 Vendor Invoice Intelligence System Portal")

st.write(
    "This system leverages machine learning to analyze vendor invoices by predicting freight costs "
    "and detecting abnormal invoice patterns. It helps businesses reduce financial risks, improve "
    "cost forecasting, and automate approval workflows efficiently."
)

# ------------------- BUSINESS IMPACT -------------------

st.subheader("💼 Business Impact")
st.markdown("""
- Improves cost visibility and freight forecasting accuracy  
- Reduces financial leakage caused by abnormal invoices  
- Enables faster and automated invoice approval processes  
""")

# ------------------- SIDEBAR -------------------

st.sidebar.title("Model Selection")

model_choice = st.sidebar.radio(
    "Choose a Model:",
    ("Freight Cost Prediction", "Invoice Manual Approval Flag")
)

# ------------------- FREIGHT COST PREDICTION -------------------

if model_choice == "Freight Cost Prediction":

    st.header("🚚 Freight Cost Prediction")

    st.write("""
    **Objective:** Predict freight cost for vendor invoices to improve planning and negotiation.
    
    - Freight is the cost associated with logistics and transportation  
    - Poor freight estimation impacts margins and supply chain efficiency  
    - Helps vendors and businesses forecast logistics expenses better  
    """)
    # Input
    dollars = st.number_input(
        "Enter Dollars",
        min_value=0.0,
        step=1.0,
        format="%.0f",
        help="Enter the invoice dollar amount to estimate freight cost"
        )
    st.caption("💡 Enter Dollars amount (e.g., 5000)")

    if st.button("Predict Freight Cost"):

        input_data = {
            "Dollars": [dollars]
        }

        prediction = predict_freight_cost(input_data)['Predicted_Freight'].iloc[0]

        st.markdown(
            f"""
            <div style="
                background-color:#e6f4ea;
                padding:20px;
                border-radius:10px;
                text-align:center;
                border:1px solid #c3e6cb;
            ">
                <h3 style="color:#155724;">🚚 Predicted Freight Cost</h3>
                <h1 style="color:#155724;">$ {prediction:.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )


# ------------------- INVOICE FLAGGING -------------------

elif model_choice == "Invoice Manual Approval Flag":

    st.header("🧾 Invoice Manual Approval Flag")

    st.write("""
    **Objective:** Detect abnormal invoices that require manual approval.
    
    - Identifies unusual invoice patterns  
    - Helps prevent fraud and financial discrepancies  
    - Ensures only safe invoices are auto-approved  
    """)

    st.markdown("### 📥 Enter Invoice Details")

    # --- Input Layout (2 columns for better UI) ---
    col1, col2 = st.columns(2)

    with col1:
        invoice_quantity = st.number_input("Invoice Quantity", min_value=0.0,format="%.0f", help="Total quantity in invoice")
        invoice_dollars = st.number_input("Invoice Dollars", min_value=0.0,format="%.0f", help="Total invoice amount")
        freight = st.number_input("Freight", min_value=0.0,format="%.0f", help="Logistics cost")

    with col2:
        total_item_quantity = st.number_input("Total Item Quantity",format="%.0f", min_value=0.0)
        total_item_dollars = st.number_input("Total Item Dollars",format="%.0f", min_value=0.0)

    st.markdown("---")

    if st.button("🔍 Analyze Invoice"):
        input_data = {
        "invoice_quantity": [invoice_quantity],
        "invoice_dollars": [invoice_dollars],
        "Freight": [freight],
        "total_item_quantity": [total_item_quantity],
        "total_item_dollars": [total_item_dollars]
        }

        flag_prediction = predict_invoice_flag(input_data)['Predicted_Flag'].iloc[0]

        st.markdown("### 📊 Analysis Result")

        if flag_prediction == 1:
            st.markdown(
                """
                <div style="
                    background-color:#fdecea;
                    padding:20px;
                    border-radius:10px;
                    border:1px solid #f5c6cb;
                    text-align:center;
                ">
                    <h2 style="color:#721c24;">🚨 Invoice Requires Manual Approval</h2>
                    <p style="color:#721c24;">
                        This invoice shows unusual patterns and should be reviewed manually.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="
                    background-color:#e6f4ea;
                    padding:20px;
                    border-radius:10px;
                    border:1px solid #c3e6cb;
                    text-align:center;
                ">
                    <h2 style="color:#155724;">✅ Invoice Safe for Auto-Approval</h2>
                    <p style="color:#155724;">
                        No anomalies detected. This invoice can be processed automatically.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
