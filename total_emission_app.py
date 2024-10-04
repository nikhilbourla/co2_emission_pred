# Basic libraries
import pandas as pd
import numpy as np
# Loading model files
import joblib
# Ui and logic library
import streamlit as sl

#################################
# lodaing model files
ohe=joblib.load("ohe.pkl")
ss=joblib.load("ss.pkl")
best_model=joblib.load("best_model.pkl")

###################################
# UI code
sl.header("Total Co2 Emission Prediction During Oil Extraction")
sl.write("This app built on the below features..")
df=pd.read_csv("x_input_total_emisssion_Mtco2.csv")
sl.write(df.head(5))
sl.subheader("Enter The Application Details To Estimate Co2")
sl.image("logo.jpg")

# form type input
col1,col2,col3,col4 = sl.columns(4)

with col1:
    region=sl.selectbox("region",df.Region.unique())
with col2:
    depth=sl.number_input("depth")
with col3:
    Oil_Production_Rate=sl.number_input("Oil_Production_Rate")
with col4:
    Extraction_Method=sl.selectbox("Extraction_Method",df["Extraction_Method"].unique())


col5,col6,col7,col8 = sl.columns(4)
with col5:
    Water_Cut=sl.number_input("Water_Cut")
with col6:
    Flaring_Emissions_MtCO2=sl.number_input("Flaring_Emissions_MtCO2")
with col7:
    Venting_Emissions_MtCO2=sl.number_input("Venting_Emissions_MtCO2")
with col8:
    Methane_Emissions_MtCO2e=sl.number_input("Methane_Emissions_MtCO2e")

# Logic Code
if sl.button("Estimate"):
    row=pd.DataFrame([[region,depth,Oil_Production_Rate,Extraction_Method,Water_Cut,Flaring_Emissions_MtCO2,Venting_Emissions_MtCO2,Methane_Emissions_MtCO2e]],columns=df.columns)
    sl.write("given data")
    sl.dataframe(row)
    
    #onehotencoding
    rowohe=ohe.transform(row[["Region","Extraction_Method"]]).toarray()
    rowohe=pd.DataFrame(rowohe, columns=ohe.get_feature_names_out())
    row=row.drop(["Region","Extraction_Method"],axis=1)
    row=pd.concat([row,rowohe],axis=1)
    
    #scaling
    row[["Depth","Oil_Production_Rate","Water_Cut","Flaring_Emissions_MtCO2","Venting_Emissions_MtCO2","Methane_Emissions_MtCO2e"]]=ss.transform(row[["Depth","Oil_Production_Rate","Water_Cut","Flaring_Emissions_MtCO2","Venting_Emissions_MtCO2","Methane_Emissions_MtCO2e"]])
    
    total_emission=round(best_model.predict(row)[0])
    
    sl.write(f"total emission mt:{total_emission}")