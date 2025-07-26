import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway

# Title
st.title("Post-COVID Cardiovascular Health KAP Survey Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Identify columns
    knowledge_cols = [col for col in df.columns if col.startswith("Answer Yes or No")]
    attitude_cols = [col for col in df.columns if "5-point Likert scale" in col]
    practice_cols = [col for col in df.columns if col.strip().startswith("[")]

    # Rename for consistency
    df.rename(columns={
        "Gender": "Gender",
        "Educational Level": "Education",
        "Age(years)": "Age"
    }, inplace=True)

    # Scoring maps
    knowledge_map = {"TRUE": 1, "FALSE": 0, "DON'T KNOW": 0.5}
    attitude_map = {"Strongly Disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly Agree": 5}
    practice_map = {"Always": 4, "Sometimes": 3, "Rarely": 2, "Never": 1}

    # Scoring
    df[knowledge_cols] = df[knowledge_cols].applymap(lambda x: knowledge_map.get(str(x).strip().upper(), np.nan))
    df[attitude_cols] = df[attitude_cols].applymap(lambda x: attitude_map.get(str(x).strip(), np.nan))
    df[practice_cols] = df[practice_cols].applymap(lambda x: practice_map.get(str(x).strip(), np.nan))

    # Total scores
    df["Knowledge_Total"] = df[knowledge_cols].sum(axis=1)
    df["Attitude_Total"] = df[attitude_cols].sum(axis=1)
    df["Practice_Total"] = df[practice_cols].sum(axis=1)

    st.subheader("Score Distributions")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df["Knowledge_Total"].dropna(), kde=True, bins=10, ax=ax[0])
    ax[0].set_title("Knowledge Score")
    sns.histplot(df["Attitude_Total"].dropna(), kde=True, bins=10, ax=ax[1])
    ax[1].set_title("Attitude Score")
    sns.histplot(df["Practice_Total"].dropna(), kde=True, bins=10, ax=ax[2])
    ax[2].set_title("Practice Score")
    st.pyplot(fig)

    # Cronbach's Alpha
    def cronbach_alpha(df_subset):
        df_subset = df_subset.dropna()
        k = df_subset.shape[1]
        variances = df_subset.var(axis=0, ddof=1)
        total_variance = df_subset.sum(axis=1).var(ddof=1)
        return (k / (k - 1)) * (1 - (variances.sum() / total_variance)) if k > 1 else np.nan

    st.subheader("Reliability Analysis (Cronbach's Alpha)")
    alpha_attitude = cronbach_alpha(df[attitude_cols])
    alpha_practice = cronbach_alpha(df[practice_cols])
    st.write(f"Attitude: {alpha_attitude:.2f}")
    st.write(f"Practice: {alpha_practice:.2f}")

    # Group comparisons
    st.subheader("Group Comparisons")
    df_clean = df.dropna(subset=["Gender", "Education", "Knowledge_Total", "Attitude_Total", "Practice_Total"])

    def gender_ttest(score):
        group1 = df_clean[df_clean["Gender"].str.lower() == "male"][score]
        group2 = df_clean[df_clean["Gender"].str.lower() == "female"][score]
        return ttest_ind(group1, group2, equal_var=False, nan_policy="omit")

    def education_anova(score):
        groups = [g[score].dropna() for _, g in df_clean.groupby("Education")]
        return f_oneway(*groups)

    for score in ["Knowledge_Total", "Attitude_Total", "Practice_Total"]:
        t_stat, p_val = gender_ttest(score)
        st.write(f"**Gender t-test for {score}**: p = {p_val:.4f}")
        f_stat, p_val = education_anova(score)
        st.write(f"**Education ANOVA for {score}**: p = {p_val:.4f}")

    # Correlation heatmap
    st.subheader("Correlation Between KAP Scores")
    corr = df[["Knowledge_Total", "Attitude_Total", "Practice_Total"]].corr().round(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
