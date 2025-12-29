
import streamlit as st
import pandas as pd
import requests
import io
import time

st.set_page_config(page_title="Student Risk Predictor", layout="wide", page_icon="🎓")

# Custom CSS for nicer look
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Performance Risk Predictor")
st.markdown("Early Warning System to identify students at academic risk.")

API_URL = "http://127.0.0.1:8000"

# Check API Status
try:
    health = requests.get(f"{API_URL}/health", timeout=2)
    api_status = health.json().get("status") == "ok"
    model_loaded = health.json().get("model_loaded")
except:
    api_status = False
    model_loaded = False

if not api_status:
    st.error("❌ Backend API is not reachable. Please start the FastAPI server.")
elif not model_loaded:
    st.warning("⚠️ Backend API is running but Model is not loaded. Please run 'main.py' to train and save the model.")
else:
    st.success("✅ System Online")

st.divider()

# Sidebar
st.sidebar.header("Navigation")
mode = st.sidebar.radio("Choose Mode", ["Single Student Prediction", "Batch Analysis (CSV)"])

if mode == "Single Student Prediction":
    st.subheader("Enter Student Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        attendance = st.slider("Attendance Percentage (%)", 0, 100, 85)
        internal = st.number_input("Internal Marks (0-100)", 0, 100, 65)
        
    with col2:
        assignment = st.number_input("Assignment Average (0-100)", 0, 100, 70)
        gpa = st.number_input("Previous Semester GPA (0.00 - 10.00)", 0.0, 10.0, 7.5, step=0.01)
    
    if st.button("Predict Risk Status", type="primary"):
        payload = {
            "attendance_percentage": attendance,
            "assignment_average": assignment,
            "internal_marks": internal,
            "previous_sem_gpa": gpa
        }
        
        with st.spinner("Analyzing..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    prob = result['risk_probability']
                    
                    st.divider()
                    if result['at_risk'] == 1:
                        st.error(f"### ⚠️ Result: AT RISK")
                        st.write(f"This student has a **{prob:.1%}** probability of being at risk.")
                        st.progress(prob)
                    else:
                        st.success(f"### ✅ Result: NOT AT RISK")
                        st.write(f"This student is safe (Risk Probability: {prob:.1%}).")
                        st.progress(prob)
                else:
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

elif mode == "Batch Analysis (CSV)":
    st.subheader("Upload Student Data CSV")
    st.info("Upload a CSV file containing columns: 'attendance_percentage', 'assignment_average', 'internal_marks', 'previous_sem_gpa'")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if st.button("Analyze Batch"):
            files = {"file": uploaded_file.getvalue()}
            with st.spinner("Processing batch..."):
                try:
                    resp = requests.post(f"{API_URL}/predict_batch", files={"file": uploaded_file.getvalue()})
                    
                    if resp.status_code == 200:
                        results_data = resp.json()
                        results_df = pd.DataFrame(results_data)
                        
                        st.divider()
                        st.markdown("### Analysis Results")
                        
                        # Summary Metrics
                        c1, c2, c3 = st.columns(3)
                        total = len(results_df)
                        risk = results_df['Predicted_Risk'].sum()
                        safe = total - risk
                        
                        c1.metric("Total Students", total)
                        c2.metric("At Risk", int(risk), delta_color="inverse")
                        c3.metric("Safe", int(safe), delta_color="normal")
                        
                        # Data Table
                        st.dataframe(results_df.style.apply(lambda x: ['background-color: #ffcccb' if v == 1 else '' for v in x], subset=['Predicted_Risk']))
                        
                        # Download
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Full Report",
                            csv,
                            "risk_analysis_report.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
                    else:
                        st.error(f"Analysis Failed: {resp.text}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
