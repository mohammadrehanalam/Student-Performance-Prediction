import streamlit as st
import pandas as pd
import pickle
import xgboost

# Load Model
model = pickle.load(open("xgboost_student_model.pkl", "rb"))

st.title("🎓 Student Performance Prediction System")

# Numeric Inputs
hours_studied = st.number_input("Hours Studied", min_value=1, max_value=50, value=20)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=75)
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, value=2)
physical_activity = st.number_input("Physical Activity", min_value=0, max_value=10, value=3)

# Categorical Inputs
access_to_resources = st.selectbox("Access to Resources", ["High", "Low", "Medium"])
extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
motivation = st.selectbox("Motivation Level", ["High", "Low", "Medium"])
internet_access = st.selectbox("Internet Access", ["No", "Yes"])
family_income = st.selectbox("Family Income", ["High", "Low", "Medium"])
teacher_quality = st.selectbox("Teacher Quality", ["High", "Low", "Medium"])
school_type = st.selectbox("School Type", ["Private", "Public"])
peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])
parent_education = st.selectbox(
    "Parental Education Level",
    ["College", "High School", "Postgraduate"]
)
distance = st.selectbox(
    "Distance From Home",
    ["Far", "Moderate", "Near"]
)

# Mappings
access_map = {"High": 0, "Low": 1, "Medium": 2}
extra_map = {"No": 0, "Yes": 1}
motivation_map = {"High": 0, "Low": 1, "Medium": 2}
internet_map = {"No": 0, "Yes": 1}
income_map = {"High": 0, "Low": 1, "Medium": 2}
teacher_map = {"High": 0, "Low": 1, "Medium": 2}
school_map = {"Private": 0, "Public": 1}
peer_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
learning_map = {"No": 0, "Yes": 1}
parent_map = {"College": 0, "High School": 1, "Postgraduate": 2}
distance_map = {"Far": 0, "Moderate": 1, "Near": 2}

if st.button("Predict Exam Score"):

    # Feature Engineering
    study_efficiency = (hours_studied * attendance) / 100
    academic_index = (previous_scores * 0.6) + (attendance * 0.4)
    learning_support = (
        tutoring_sessions
        + access_map[access_to_resources]
        + internet_map[internet_access]
    )

    data = pd.DataFrame([[
        hours_studied,
        attendance,
        access_map[access_to_resources],
        extra_map[extracurricular],
        previous_scores,
        motivation_map[motivation],
        internet_map[internet_access],
        tutoring_sessions,
        income_map[family_income],
        teacher_map[teacher_quality],
        school_map[school_type],
        peer_map[peer_influence],
        physical_activity,
        learning_map[learning_disabilities],
        parent_map[parent_education],
        distance_map[distance],
        study_efficiency,
        academic_index,
        learning_support
    ]], columns=[
        'Hours_Studied',
        'Attendance',
        'Access_to_Resources',
        'Extracurricular_Activities',
        'Previous_Scores',
        'Motivation_Level',
        'Internet_Access',
        'Tutoring_Sessions',
        'Family_Income',
        'Teacher_Quality',
        'School_Type',
        'Peer_Influence',
        'Physical_Activity',
        'Learning_Disabilities',
        'Parental_Education_Level',
        'Distance_from_Home',
        'Study_Efficiency',
        'Academic_Index',
        'Learning_Support'
    ])

    prediction = model.predict(data)[0]

    st.success(f"Predicted Exam Score: {prediction:.2f}")
