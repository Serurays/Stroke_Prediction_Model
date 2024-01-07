import streamlit as st
import numpy as np
import pickle


def main():
    st.title("Stroke Prediction")

    # Display a formatted note about the input values
    st.markdown("""
    Note: 
    - Gender: 1 for Male, 0 for Female
    - Work Type: 0 for Govn Job, 1 for Never worked, 2 for Private, 3 for Self-employed, 4 for Children
    - Residence Type: 1 for Urban, 0 for Rural
    - Smoking Status: 0 for Unknown, 1 for Formerly smoked, 2 for Never smoked, 3 for Smokes
    - For the remaining questions, 1 is Yes and 0 is No
    """)

    gender = st.number_input('Gender', min_value=0, max_value=1, step=1)
    age = st.number_input('Age', min_value=0)
    hypertension = st.number_input('Hypertension', min_value=0, max_value=1, step=1)
    heart_disease = st.number_input('Heart Disease', min_value=0, max_value=1, step=1)
    ever_married = st.number_input('Ever Married', min_value=0, max_value=1, step=1)
    work_type = st.number_input('Work Type', min_value=0, max_value=4, step=1)
    Residence_type = st.number_input('Residence Type', min_value=0, max_value=1, step=1)
    avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0)
    bmi = st.number_input('BMI', min_value=0.0)
    smoking_status = st.number_input('Smoking Status', min_value=0, max_value=3, step=1)

    user_input = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                           avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

    pickle_in = open('stroke_prediction_model.pkl', 'rb')
    rf = pickle.load(pickle_in)

    if st.button('Predict'):
        prediction = rf.predict(user_input)

        if prediction[0] == 0:
            st.subheader('No Stroke Risk')
            st.write("Based on the provided information, it seems there is no immediate risk of stroke. However, it's always advisable to consult with a healthcare professional for a more accurate assessment.")
        else:
            st.subheader('Stroke Risk')
            st.write("Based on the provided information, there is a potential risk of stroke. It is strongly recommended to consult with a healthcare professional for a thorough evaluation and guidance on preventive measures.")


if __name__ == '__main__':
    main()