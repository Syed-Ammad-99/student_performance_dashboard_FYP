import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px

#Title and layout configuration
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Inline CSS to ensure background and container styling match theme
st.markdown(
    """<style>
    .stApp { background-color: #F5F7FB; }
    .css-18e3th9 { background-color: transparent; }
    .stSidebar { background-color: #8698bc; }
    .stMetricValue { text-align: center; }
    </style>""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    "<div style='background-color: #F5F7FB; padding: 15px; border-radius: 8px; text-align: left;'>"
    "<h2 style='color: #000000; margin: 0;'>🎓 Student Performance System</h2>"
    "<p style='color: #000000; margin: 1px 0 0 0; font-size: 14px;'>Data Analytics & ML Dashboard</p>"
    "</div>",
    unsafe_allow_html=True
)

st.sidebar.divider()

#Dataset loading with caching to improve performance
@st.cache_data
def load_data():
    df = pd.read_excel("CSV3 Students Performance Dataset.xlsx")
    return df
df = load_data()


# ML MODEL (Logistic Regression)

def train_model(data):

    ml_df = data.copy()

    # Create target variable (Rule-based for training)
    ml_df["Risk"] = (
        (ml_df["Attendance (%)"] < 70) |
        (ml_df["Total_Score"] < 60)
    ).astype(int)

    features = [
        "Attendance (%)",
        "Midterm_Score",
        "Final_Score",
        "Assignments_Avg",
        "Quizzes_Avg",
        "Projects_Score",
        "Study_Hours_per_Week"
    ]

    X = ml_df[features]
    y = ml_df["Risk"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model, scaler, features

model, scaler, feature_cols = train_model(df)

# Add Predictions

X_all_scaled = scaler.transform(df[feature_cols])
df["Risk_Prediction"] = model.predict(X_all_scaled)
df["Risk_Label"] = df["Risk_Prediction"].map({1: "At Risk", 0: "Safe"})

# Calculate average scores for use in both dashboards
avg_scores = df[
    ["Midterm_Score", "Final_Score", "Assignments_Avg",
     "Quizzes_Avg", "Projects_Score"]].mean().round(2)

# Recommendation Function
def get_recommendations(student):
    """Generates personalized recommendations based on student data."""
    recs = []
    if student["Attendance (%)"] < 70:
        recs.append("Improve attendance to avoid falling behind.")
    if student["Assignments_Avg"] < 60:
        recs.append("Focus on improving assignment scores through more practice.")
    if student["Final_Score"] < 60:
        recs.append("Review final exam topics and consider seeking extra help.")
    if not recs:
        recs.append("Keep up the great work and maintain your current study habits.")
    return recs

#Sidebar filters and login
st.sidebar.title("🔐 Login")
role = st.sidebar.selectbox("Select Role", ["Teacher", "Student"])
student_id = None

if role == "Student":
    student_id = st.sidebar.selectbox(
        "Select Student ID",
        df["Student_ID"].unique()
    )

# login button
login_btn = st.sidebar.button("Login")
if login_btn and role == "Teacher":
    st.title("🧑‍🏫 Teacher Dashboard")

    #Teacher Dashboard Execution
    #KPI Metrics
    with st.container():
        st.markdown("## 📚 Class Overview")
        col1, col2, col3, col4 = st.columns(4)
        # Counts rows → number of students (label + value centered & bold)
        col1.markdown(
            f"<div style='text-align:center; font-weight:bold'>👨‍🎓 Total Students</div>"
            f"<div style='text-align:center; font-size:28px; font-weight:700'>{df.shape[0]}</div>",
            unsafe_allow_html=True
        )
        # Calculates class average attendance
        col2.markdown(
            f"<div style='text-align:center; font-weight:bold'>📅 Avg Attendance</div>"
            f"<div style='text-align:center; font-size:28px; font-weight:700'>{df['Attendance (%)'].mean():.2f}%</div>",
            unsafe_allow_html=True
        )
        # Average performance score
        col3.markdown(
            f"<div style='text-align:center; font-weight:bold'>📈 Avg Score</div>"
            f"<div style='text-align:center; font-size:28px; font-weight:700'>{df['Total_Score'].mean():.2f}</div>",
            unsafe_allow_html=True
        )
        # "At-Risk Students"
        col4.markdown(
            f"<div style='text-align:center; font-weight:bold'>🚨 At Risk</div>"
            f"<div style='text-align:center; font-size:28px; font-weight:700'>{df[df['Risk_Label'] == 'At Risk'].shape[0]}</div>",
            unsafe_allow_html=True
        )

    st.divider()

    #Teacher Performance Chart
    with st.container():
        st.markdown(f"### 📈 Average Academic Performance")
    avg_df = avg_scores.reset_index()
    avg_df.columns = ["Component", "Average Score"]

    fig = px.bar(
        avg_df,
        x="Component",
        y="Average Score",
        text="Average Score",
        color="Average Score",
        color_continuous_scale=[[0, "#8698bc"], [1, "#1f3a6b"]] # Custom blue gradient
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    #Risk Alert Table
    with st.container():
        st.markdown(f"### 🚨 At-Risk Students Only (ML Prediction)")
    risk_table = df[df["Risk_Label"] == "At Risk"][
        ["Student_ID", "First_Name", "Last_Name", "Attendance (%)",
         "Total_Score", "Risk_Label"]
    ]

    st.dataframe(
        risk_table.style.map(
            lambda x: "background-color: #ff8080" if x == "At Risk" else "",
            subset=["Risk_Label"]
        ),
        use_container_width=True
    )

    st.divider()

    #Safe Students Table
    with st.container():
        st.markdown(f"### ✅ Safe Students (ML Prediction)")
    safe_table = df[df["Risk_Label"] == "Safe"][
        ["Student_ID", "First_Name", "Last_Name", "Attendance (%)",
         "Total_Score", "Risk_Label"]
    ]

    st.dataframe(
        safe_table.style.map(
            lambda x: "background-color: #80ff80" if x == "Safe" else "",
            subset=["Risk_Label"]
        ),
        use_container_width=True
    )


#Student Dashboard Execution
elif login_btn and role == "Student":

    student = df[df["Student_ID"] == student_id].iloc[0]
    st.title("🎓 Student Dashboard")
    st.markdown(
    f"""
    ## 👋 Welcome, **{student['First_Name']}**
    📌 Below is a summary of your academic performance and predicted outcome.
    """
    )

    col1, col2, col3 = st.columns(3)
    col1.markdown(
            f"<div style='text-align:center; font-weight:bold'>📅 Attendance (%)</div>"
            f"<div style='text-align:center; font-size:28px; font-weight:700'>{student['Attendance (%)']:.2f}%</div>",
            unsafe_allow_html=True
        )

    col2.markdown(
            f"<div style='text-align:center; font-weight:bold'>📈 Total Score</div>"
            f"<div style='text-align:center; font-size:28px; font-weight:700'>{student['Total_Score']:.2f}</div>",
            unsafe_allow_html=True
        )
    
    col3.markdown(
            f"<div style='text-align:center; font-weight:bold'>📚 Study Hours / Week</div>"
            f"<div style='text-align:center; font-size:28px; font-weight:700'>{student['Study_Hours_per_Week']}</div>",
            unsafe_allow_html=True
        )

    st.divider()

    #Detailed performance breakdown
    with st.container():
        st.markdown(f"### 📘 Performance Breakdown: {student_id}")
    col1, col2, col3, col4 = st.columns(4)
    
    performance_data = {
        "Midterm": student["Midterm_Score"],
        "Final": student["Final_Score"],
        "Assignments": student["Assignments_Avg"],
        "Quizzes": student["Quizzes_Avg"],
        "Projects": student["Projects_Score"]
    }
    
    perf_df = pd.DataFrame({
        "Score Type": list(performance_data.keys()),
        "Score": list(performance_data.values())
    })
    
    fig = px.bar(
        perf_df,
        x="Score Type",
        y="Score",
        text="Score",
        color="Score Type",  # Assign colors based on the score type
        color_discrete_sequence=px.colors.qualitative.Set2  # Use a nice color set
    )
    
    fig.update_traces(textposition="inside", textangle=0)
    fig.update_xaxes(tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    compare_df = pd.DataFrame({
    "Score Type": list(performance_data.keys()),
    "Student": list(performance_data.values()),
    "Class Average": list(avg_scores.values)
    })

    with st.container():
        st.markdown(f"### 📊 Your Performance vs Class Average")
    fig = px.bar(
        compare_df,
        x="Score Type",
        y=["Student", "Class Average"],
        barmode="group",
        color_discrete_map={"Student": "#1f77b4", "Class Average": "#ff7f0e"},
        labels={"value": "Score", "variable": ""},
        text_auto=True
    )
    
    fig.update_layout(
        height=450,
        hovermode="x unified",
        font=dict(size=12),
        xaxis_tickangle=-45
    )
    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    # Academic Risk Status and Recommendations
    with st.container():
        st.markdown("### 🧠 Academic Risk Status & Recommendations")

    if student["Risk_Label"] == "At Risk":
        st.error("⚠ **AT RISK** – Immediate improvement recommended.")
        recommendations = get_recommendations(student)
        for rec in recommendations:
            st.warning(f"👉 {rec}")
    else:
        st.success("✅ **SAFE** – Keep up the good performance!")


#Default Screen
else:
    st.markdown(
        "<div style='background-color: #F5F7FB; padding: 30px; border-radius: 12px; text-align: center;'>"
        "<h1 style='color: #000000; margin: 0;'>🎓 Student Performance Monitoring System</h1>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='background-color: #F5F7FB; padding: 20px; border-radius: 8px; border-left: 5px solid #000000; margin-top: 20px;'>"
        "<p style='color: #0F1724; font-size: 16px; margin: 0;'>"
        "<strong>👉 Please select a role and click <strong style=\"color: #8698bc;\">Login</strong> from the sidebar to proceed.</strong>"
        "</p></div>",
        unsafe_allow_html=True
    )
