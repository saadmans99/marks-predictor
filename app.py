import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Student Marks Predictor")
st.write("Enter the number of hours you studied to predict your exam score.")

try: 
    df = pd.read_csv('student_marks.csv')
    df.columns = [c.title() for c in df.columns]
    if st.checkbox('Show raw data'):
        st.write(df)

        st.subheader("Visualizing the data")
        st.scatter_chart(data=df, x='hours_studied', y='exam_score')

        x=df[['hours_studied','previous_scores']]
        y=df['exam_score']
        model = LinearRegression()
        model.fit(x,y)

        col1,col2 = st.columns(2)

        with col1:
            hours = st.number_input("Hours studied", 0.0, 12.0, 5.0)

        with col2:
            previous = st.number_input("Previous scores", 0.0, 100.0, 70.0)    


        st.subheader("Make a Prediction")
        user_hours = st.slider("hours Studied", 0.0, 12.0, 5.0)

        if st.button("Predict Exam Score"):
            user_input = [[user_hours]]
            prediction = model.predict([hours, previous])
            st.success(f"If you study for {user_hours} hours, you will score: {prediction[0]:.2f}")

except FileNotFoundError:
    st.error("The data file 'student_marks.csv' was not found.")
