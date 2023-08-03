import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Revenue Prediction")
st.subheader("Market have been selling different production to its consumer and a collect some revenue from the sale which results in making profit. Through selling the product the management has able to track all the goods sold in the suparmarket. The main purpose for this analysis is to understand the sales and making predictive model so that management can put measures for maximizing profit in future.")


Order_Quantity=st.number_input('Enter order quantity')

Unit_Cost=st.number_input("Enter unit cost")

Unit_Price=st.number_input("Enter unit price")

Profit=st.number_input("Enter profit") 

Cost=st.number_input("Enter cost")

def predict(Sale):
 	data=joblib.load("Revenue_model.sav")
 	return data.predict(Sale)

if st.button("Sales prediction"):
	Sale=predict([[Order_Quantity,Unit_Cost,Unit_Price,Profit,Cost]])
	st.success("Expected Revenue")
	st.text(Sale[0])



