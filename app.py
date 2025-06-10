import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", 'rb'))
model = pickle.load(open("model_compressed.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))

# Title
st.title("🎯 Career Aspiration Predictor")
st.write("Enter student details below to predict their best-fit career aspiration.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
part_time_job = st.selectbox("Has Part-Time Job?", ["Yes", "No"])
absence_days = st.number_input("Absence Days", min_value=0, max_value=365)
extracurricular = st.selectbox("In Extracurricular Activities?", ["Yes", "No"])
weekly_self_study_hours = st.number_input("Weekly Self-Study Hours", min_value=0, max_value=168)

scores = {}
for subject in ["Math", "History", "Physics", "Chemistry", "Biology", "English", "Geography"]:
    scores[subject] = st.slider(f"{subject} Score", min_value=0, max_value=100, value=50)

# Process inputs
if st.button("Predict Career Aspiration"):
    gender_map = {"Male": 0, "Female": 1}
    binary_map = {"Yes": 1, "No": 0}

    input_data = [
        gender_map[gender],
        binary_map[part_time_job],
        absence_days,
        binary_map[extracurricular],
        weekly_self_study_hours,
        scores["Math"],
        scores["History"],
        scores["Physics"],
        scores["Chemistry"],
        scores["Biology"],
        scores["English"],
        scores["Geography"]
    ]

    # Total and average scores
    total_score = sum(scores.values())
    average_score = total_score / len(scores)

    input_data.extend([total_score, average_score])

    # Scale and reshape
    scaled_input = scaler.transform([input_data])

    # Predict
    probs = model.predict_proba(scaled_input)[0]
    top_indices = np.argsort(probs)[::-1][:5]

    # Career mapping
    career_labels = [
        "Artist", "Game Developer", "Real Estate Developer", "Business Owner",
        "Designer", "Doctor", "Engineer", "Teacher", "Lawyer", "Psychologist",
        "Scientist", "Chef", "Architect", "Writer", "Athlete", "Musician", "Entrepreneur"
    ]

    st.subheader("Top Recommended Careers")
    st.write("Based on the prediction probabilities:")

    for idx in top_indices:
        st.write(f"**{career_labels[idx]}** with probability **{probs[idx]:.2f}**")
