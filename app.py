
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit config
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🔍 Real-Time Financial Fraud Detection System")

# Load model and features
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_features.pkl", "rb") as f:
    model_features = pickle.load(f)

# Sidebar user input
st.sidebar.header("Enter Transaction Details")

def user_input_features():
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
    payment_method = st.sidebar.selectbox("Payment Method", ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"])
    product_category = st.sidebar.selectbox("Product Category", ["Electronics", "Clothing", "Groceries", "Jewelry", "Others"])
    customer_age = st.sidebar.slider("Customer Age", 18, 90, 30)
    device_used = st.sidebar.selectbox("Device Used", ["Mobile", "Desktop", "Tablet"])
    account_age_days = st.sidebar.slider("Account Age (Days)", 0, 3650, 365)
    transaction_hour = st.sidebar.slider("Transaction Hour", 0, 23, 12)

    data = {
        'Transaction Amount': amount,
        'Payment Method': payment_method,
        'Product Category': product_category,
        'Customer Age': customer_age,
        'Device Used': device_used,
        'Account Age Days': account_age_days,
        'Transaction Hour': transaction_hour
    }
    return pd.DataFrame([data])

input_df = user_input_features()
st.subheader("🧾 Input Transaction Data")
st.write(input_df)

# Preprocess input
input_processed = pd.get_dummies(input_df)
input_processed = input_processed.reindex(columns=model_features, fill_value=0)

# Prediction
prediction = model.predict(input_processed)
prediction_proba = model.predict_proba(input_processed)

# Display result
st.subheader("📊 Prediction Result")
if prediction[0] == 1:
    st.error("⚠️ This transaction is predicted to be **Fraudulent**.")
else:
    st.success("✅ This transaction is predicted to be **Genuine**.")

st.write("Prediction Probabilities:")
st.info(f"🟢 Genuine: {prediction_proba[0][0]:.2f} | 🔴 Fraudulent: {prediction_proba[0][1]:.2f}")

# Visualization Section
st.markdown("---")
st.subheader("📈 Visual Analytics")

try:
    df = pd.read_csv("data.csv")
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    df['Transaction Hour'] = df['Transaction Date'].dt.hour

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Fraud vs Genuine Count")
        sns.countplot(data=df, x='Is Fraudulent', palette='Set2')
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.markdown("### Avg. Transaction Amount by Category")
        sns.barplot(data=df, x='Product Category', y='Transaction Amount', estimator='mean', ci=None, palette='muted')
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
        plt.clf()

    st.markdown("### Transaction Hour Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='Transaction Hour', hue='Is Fraudulent', multiple='stack', palette='Set1', bins=24)
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Unable to load data.csv for visualization: {e}")
