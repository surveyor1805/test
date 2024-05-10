import os

os.chdir('C:/Users/KirO_/Desktop/Python/PyCharmProjects/Homework9')

import pandas as pd
import streamlit as st

from src.utils import prepare_data, train_model, read_model

st.set_page_config(
    page_title="RealEstatePricePredictor"
    )

model_path = 'rf_fitted.pkl'

data_for_input = prepare_data()

square = st.sidebar.number_input("what is total square of your real estate?", 1, 2070, 40)
floor = st.sidebar.number_input("what is your floor?", 1, 66, 20)
rooms = st.sidebar.number_input("how many rooms in your apartment?", 1, 15, 2)

inputDF = pd.DataFrame(
    {
        "total_square": square,
        "rooms": rooms,
        "floor": floor
    },
    index=[0]
)

if not os.path.exists(model_path):
    train_data = prepare_data()
    train_data.to_csv('data.csv')
    train_model(train_data)

model = read_model('lr.pkl')

if st.button("Predict Price"):
    preds = model.predict(inputDF)
    result = int(preds[0])
    st.write(f"Your estate price based on your input is: {result}")
