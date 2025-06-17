import streamlit as st
import requests

# ---- Config ----
st.set_page_config(page_title="HealthCare Assistant", page_icon="🩺", layout="wide")

# ---- Branding ----
st.markdown("""
    <h1 style='font-size: 2.5rem;'>🩺 <strong>HealthCare Assistant</strong></h1>
    <p style='font-size: 1.1rem;'>A smart medical chatbot for <strong>symptom checking</strong>, <strong>prescription explanation</strong>, and <strong>health literacy</strong>.</p>
""", unsafe_allow_html=True)

# ---- Sidebar Navigation with Icons ----
menu = {
    "🗨️ Symptom Checker": "chat",
    "💊 Prescription Explainer": "prescription",
    "📘 Health Literacy Tutor": "literacy"
}
choice = st.sidebar.radio("**Choose a mode:**", list(menu.keys()))

# ---- Helper ----
def call_backend(endpoint, query):
    try:
        res = requests.post(f"http://127.0.0.1:8000/{endpoint}", json={"query": query}, timeout=30)
        return res.json().get("response", "⚠️ No response from backend.")
    except Exception as e:
        return f"❌ Error: {e}"

# ---- SYMPTOM CHAT ----
if menu[choice] == "chat":
    st.subheader("🗨️ Symptom Checker")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi there! I'm your HealthCare Assistant. How can I help you today?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type your symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 HealthCare Assistant is thinking..."):
                response = call_backend("symptom-check", prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ---- PRESCRIPTION ----
elif menu[choice] == "prescription":
    st.subheader("💊 Prescription Explainer")
    drug = st.text_input("Enter prescription (e.g., Ibuprofen 200mg twice daily)")
    if st.button("Explain Prescription"):
        if drug.strip():
            with st.spinner("Explaining..."):
                response = call_backend("explain-prescription", drug)
                st.success(response)
        else:
            st.warning("Please enter a valid prescription.")

# ---- LITERACY ----
elif menu[choice] == "literacy":
    st.subheader("📘 Health Literacy Tutor")
    question = st.text_area("Ask a health-related question:")
    if st.button("Explain Concept"):
        if question.strip():
            with st.spinner("Explaining..."):
                response = call_backend("health-literacy", question)
                st.success(response)
        else:
            st.warning("Please enter a valid question.")
