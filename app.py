import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📄 AI Resume Screening System")

st.write(
    "Machine Learning based Resume Screening and Candidate "
    "Shortlisting System"
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("ai_resume_screening.csv")

    df.columns = df.columns.str.strip()

    return df


df = load_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Candidate Screening",
        "Model Performance",
        "Feature Importance"
    ]
)


# =========================================================
# PREPARE DATA
# =========================================================

X = df.drop("shortlisted", axis=1)

y = df["shortlisted"].map({
    "Yes": 1,
    "No": 0
})


numeric_features = [
    "years_experience",
    "skills_match_score",
    "project_count",
    "resume_length",
    "github_activity"
]

categorical_features = [
    "education_level"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header("📊 Dashboard")

    total_candidates = len(df)

    shortlisted = (df["shortlisted"] == "Yes").sum()

    not_shortlisted = (df["shortlisted"] == "No").sum()

    shortlist_rate = shortlisted / total_candidates * 100


    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Candidates",
        total_candidates
    )

    col2.metric(
        "Shortlisted",
        shortlisted
    )

    col3.metric(
        "Not Shortlisted",
        not_shortlisted
    )

    col4.metric(
        "Shortlisting Rate",
        f"{shortlist_rate:.2f}%"
    )


    st.divider()


    # Dataset preview

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # Shortlisting distribution

    st.subheader("Candidate Shortlisting Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="shortlisted",
        ax=ax
    )

    ax.set_xlabel("Shortlisted")
    ax.set_ylabel("Number of Candidates")
    ax.set_title("Shortlisting Distribution")

    st.pyplot(fig)


# =========================================================
# CANDIDATE SCREENING
# =========================================================

elif page == "Candidate Screening":

    st.header("🤖 Candidate Screening")

    st.write(
        "Enter candidate information to predict the "
        "shortlisting result."
    )


    col1, col2 = st.columns(2)


    with col1:

        experience = st.number_input(
            "Years of Experience",
            min_value=0,
            max_value=50,
            value=2
        )


        skills = st.slider(
            "Skills Match Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )


        education = st.selectbox(
            "Education Level",
            sorted(df["education_level"].unique())
        )


    with col2:

        projects = st.number_input(
            "Project Count",
            min_value=0,
            max_value=100,
            value=3
        )


        resume_length = st.number_input(
            "Resume Length",
            min_value=1,
            max_value=5000,
            value=500
        )


        github = st.number_input(
            "GitHub Activity",
            min_value=0,
            max_value=5000,
            value=100
        )


    st.divider()


    if st.button(
        "🔍 Screen Candidate",
        type="primary"
    ):

        candidate = pd.DataFrame({

            "years_experience": [experience],

            "skills_match_score": [skills],

            "education_level": [education],

            "project_count": [projects],

            "resume_length": [resume_length],

            "github_activity": [github]
        })


        prediction = pipeline.predict(
            candidate
        )[0]


        probability = pipeline.predict_proba(
            candidate
        )[0][1]


        st.subheader("Screening Result")


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            if prediction == 1:

                st.success(
                    "✅ Candidate Predicted: SHORTLISTED"
                )

            else:

                st.warning(
                    "⚠️ Candidate Predicted: NOT SHORTLISTED"
                )


        with result_col2:

            st.metric(
                "Shortlisting Probability",
                f"{probability * 100:.2f}%"
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.header("📈 Model Performance")


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Model",
        "Random Forest"
    )

    col2.metric(
        "Test Samples",
        len(X_test)
    )

    col3.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


    st.divider()


    # Classification report

    st.subheader("Classification Report")


    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Shortlisted",
            "Shortlisted"
        ],
        output_dict=True
    )


    report_df = pd.DataFrame(report).transpose()


    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )


    # Confusion matrix

    st.subheader("Confusion Matrix")


    cm = confusion_matrix(
        y_test,
        y_pred
    )


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Not Shortlisted",
            "Shortlisted"
        ],
        yticklabels=[
            "Not Shortlisted",
            "Shortlisted"
        ],
        ax=ax
    )


    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


    st.pyplot(fig)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

elif page == "Feature Importance":

    st.header("🔍 Feature Importance")


    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()


    importances = pipeline.named_steps[
        "model"
    ].feature_importances_


    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importances

    })


    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )


    st.dataframe(
        importance_df.round(4),
        use_container_width=True
    )


    # Chart

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        ax=ax
    )


    ax.set_title(
        "Random Forest Feature Importance"
    )


    st.pyplot(fig)
