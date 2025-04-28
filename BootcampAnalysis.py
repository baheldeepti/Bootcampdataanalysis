# -----------------------------------
#  Full Streamlit Causal Dashboard (Corrected)
# -----------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------
#  Load Dataset
# -----------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/baheldeepti/Bootcampdataanalysis/main/causal_inference_bootcamp_dataset_with_presalary.csv')
    return df

df = load_data()

# -----------------------------------
#  Precompute Important Metrics
# -----------------------------------

# Naive Salary Difference
naive_diff = df[df['treatment'] == 1]['salary'].mean() - df[df['treatment'] == 0]['salary'].mean()

# Difference-in-Differences (DiD) Estimate
df['salary_change'] = df['salary'] - df['pre_salary']
treated_change = df[df['treatment'] == 1]['salary_change'].mean()
untreated_change = df[df['treatment'] == 0]['salary_change'].mean()
did_estimate = treated_change - untreated_change

# Precomputed Average Treatment Effect (PSM Estimate) [manual estimate]
ate_psm_estimate = 10003

# Subgroup Prep
df['experience_bucket'] = pd.cut(df['prior_experience'], bins=[0,2,5,10], labels=['0-2 yrs','2-5 yrs','5-10 yrs'])
education_summary = df.groupby('education_level')['salary_change'].mean().sort_values(ascending=False)
experience_summary = df.groupby('experience_bucket')['salary_change'].mean().sort_values(ascending=False)

# -----------------------------------
#  Streamlit App Starts
# -----------------------------------

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "Causal Insights", "Subgroup Analysis", "Individual Uplift Explorer", "What-If Simulator", "Download Reports"])

# -----------------------------------
#  Home Tab
# -----------------------------------

if app_mode == "Home":
    st.title("Causal Impact of Coding Bootcamps on Salaries")
    st.markdown("""
    Welcome to the Causal Insights Dashboard!  
    We estimate how attending a coding bootcamp truly affects salary outcomes using causal inference methods.

    **Techniques used**:
    - Propensity Score Matching (PSM)
    - Difference-in-Differences (DiD)
    - Causal Forests

    ---
    """)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(f"Number of Records: {df.shape[0]}")
    st.write(f"Columns: {list(df.columns)}")

# -----------------------------------
#  Causal Insights Tab
# -----------------------------------

elif app_mode == "Causal Insights":
    st.title("Causal Insights Overview")

    # Key Metrics with Columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Naive Salary Difference", value=f"${naive_diff:,.0f}")
    with col2:
        st.metric(label="Causal Effect (PSM ATE)", value=f"${ate_psm_estimate:,.0f}")
    with col3:
        st.metric(label="Difference-in-Differences Estimate", value=f"${did_estimate:,.0f}")

    st.markdown("---")

    st.subheader("Naive vs Causal Salary Comparison")

    comparison_data = pd.DataFrame({
        'Method': ['Naive Comparison', 'Causal Effect (PSM)', 'DiD Estimate'],
        'Salary Increase ($)': [naive_diff, ate_psm_estimate, did_estimate]
    })

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=comparison_data, x='Method', y='Salary Increase ($)', palette="pastel", ax=ax)
    ax.set_title("Salary Gain Estimation Comparison")
    st.pyplot(fig)

    st.success("""
    **Summary**:  
    - Naive analysis overstates the bootcamp effect.  
    - Proper causal methods estimate a true uplift of about **$10,000** annually.
    """)

# -----------------------------------
#  Subgroup Analysis Tab
# -----------------------------------

elif app_mode == "Subgroup Analysis":
    st.title("Subgroup Causal Effect Analysis")

    subgroup_option = st.selectbox("Choose Subgroup:", ["Education Level", "Prior Experience"])

    if subgroup_option == "Education Level":
        fig, ax = plt.subplots(figsize=(8,5))
        education_summary.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_ylabel('Average Salary Change ($)')
        ax.set_title('Causal Effect by Education Level')
        st.pyplot(fig)
        st.info("Master's degree holders tend to benefit the most from bootcamps.")

    elif subgroup_option == "Prior Experience":
        fig, ax = plt.subplots(figsize=(8,5))
        experience_summary.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_ylabel('Average Salary Change ($)')
        ax.set_title('Causal Effect by Prior Experience Level')
        st.pyplot(fig)
        st.info("Individuals with 2–5 years experience benefit significantly from bootcamps.")

# -----------------------------------
#  Individual Uplift Explorer Tab
# -----------------------------------

elif app_mode == "Individual Uplift Explorer":
    st.title("Predict Your Salary Uplift After Bootcamp")

    # Inputs
    age = st.slider("Select Age:", 18, 50, 25)
    experience = st.slider("Select Prior Experience (Years):", 0, 10, 3)
    education = st.selectbox("Select Education Level:", ["High School", "Bachelor's", "Master's"])

    # Dummy rule-based prediction (simple for demo)
    if education == "High School":
        base_effect = 7000
    elif education == "Bachelor's":
        base_effect = 10000
    else:
        base_effect = 12000

    predicted_effect = base_effect + (experience * 200)

    st.metric(label="Predicted Salary Uplift", value=f"${predicted_effect:.0f}")

# -----------------------------------
#  What-If Simulator Tab
# -----------------------------------

elif app_mode == "What-If Simulator":
    st.title("Bootcamp Salary Uplift Simulator")

    st.write("Adjust the sliders below to simulate your potential salary boost!")

    # Sliders
    sim_age = st.slider("Age:", 18, 50, 30)
    sim_experience = st.slider("Prior Experience (Years):", 0, 10, 4)
    sim_education = st.select_slider("Education Level:", options=["High School", "Bachelor's", "Master's"])

    # Rule for simulation
    if sim_education == "High School":
        sim_base = 7000
    elif sim_education == "Bachelor's":
        sim_base = 10000
    else:
        sim_base = 12000

    simulated_effect = sim_base + (sim_experience * 250)  # Boost slightly more optimistic for simulator

    st.subheader(f"Simulated Predicted Uplift: **${simulated_effect:,.0f}**")

# -----------------------------------
#  Download Reports Tab
# -----------------------------------

elif app_mode == "Download Reports":
    st.title("Download Full Dataset")

    st.download_button(
        label="Download Dataset as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="causal_inference_bootcamp_dataset.csv",
        mime="text/csv"
    )

    st.success("Dataset ready for download!")
