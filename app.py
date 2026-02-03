import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Student Marks Predictor")
st.write("Enter the number of hours you studied to predict your exam score.")

data = {
    'hours_studied': [1.5, 3.0, 5.0, 8.0, 10.0],
    'exam_score': [45, 60, 75, 85, 95]
}

df = pd.DataFrame(data)

x = df[['hours_studied']]
y = df['exam_score']
model = LinearRegression()
model.fit(x, y)

user_hours = st.slider("Hours Studied", min_value=0.0, max_value=12.0, value=5.0)

if st.button("Predict Score"):
    user_input = [[user_hours]]
    prediction = model.predict(user_input)

    st.success(f"If you study for {user_hours} hours, you will score: {prediction[0]:.2f}")