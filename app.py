# All imports
import datetime
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

# loading model
model = load_model("FM")


# function to predict results
def predict(model, input_data):
    pred_df = predict_model(estimator=model, data=input_data)
    preds = pred_df['Label'][0]
    return preds


# function to change label from 100000000000000000110011 to Saving Account,Credit Card,Securities,Pensions,Direct Debit
def label_changer(label: str):
    products = ["Saving Account", "Guarantees",
                "Current Accounts", "Derivada Account", "Payroll Account", "Junior Account", "Más particular Account", "Particular Account", "Particular Plus Account", "Short-term deposits", "Medium-term deposits", "Long-term deposits", "E-Account", "Funds", "Mortgage", "Pensions", "Loans", "Taxes", "Credit Card", "Securities", "Home Account", "Payroll", "Pensions", "Direct Debit"]
    prod_list = []
    for i in range(len(label)):
        if(label[i] == "1"):
            prod_list.append(products[i])
    result = ",".join(prod_list)
    return result


def preprocess_data(df):
    # detecting train and test data
    val = 0
    if(df.shape[1] > 24):
        val = 1
    # dropping this two columns as they contain all NaN values
    df = df.drop(columns=['ult_fec_cli_1t', 'conyuemp'])
    # preprocessing for renta
    df['renta'] = df['renta'].replace({"         NA": '0'})
    df['renta'] = df['renta'].astype('float')
    df['renta'] = df['renta'].replace({0: np.NaN})
    # changing proper type of data for each column
    # here fecha_dato and fecha_alta are only having date hence setting them to date category
    df['fecha_dato'] = pd.to_datetime(df['fecha_dato'])
    df['fecha_alta'] = pd.to_datetime(df['fecha_alta'])
    # Below are all numerical values changing them to proper numerical type
    df['age'] = df['age'].astype('int')
    df['antiguedad'] = df['antiguedad'].astype('int')
    # below columns are categorical data but they already having numeric value
    df['indrel'] = df['indrel'].astype('int')
    df['ind_nuevo'] = df['ind_nuevo'].astype('int')
    df['ind_actividad_cliente'] = df['ind_actividad_cliente'].astype(
        'int')
    if(val == 1):
        df['ind_nomina_ult1'] = df['ind_nomina_ult1'].astype('int')
        df['ind_nom_pens_ult1'] = df['ind_nom_pens_ult1'].astype(
            'int')
    df['indrel_1mes'] = df['indrel_1mes'].replace(
        {"1.0": "1", "2.0": "2", "3.0": "3", "4.0": "4"})
    # changing all labels data into string
    if(val == 1):
        for col_name in (df.loc[:, "ind_ahor_fin_ult1":]).columns:
            df[col_name] = df[col_name].astype(str)
        df['labels'] = (df.iloc[:, 22:]).values.sum(axis=1)
    clean_data = df.iloc[:, :22]
    if(val == 1):
        clean_data.insert(len(clean_data.columns),
                          "labels", df["labels"].values)
    return clean_data


def run():
    add_selectbox = st.sidebar.selectbox(
        "How do you want to predict?", ("Single", "Batch"))
    st.sidebar.info(
        "This app is created to recommend banking products to customers")
    st.sidebar.success("Hello There!")

    st.title("SBRecommender")
    if (add_selectbox == "Single"):
        ind_empleado = st.selectbox("Employee Index", options=[
                                    "A", "B", "F", "N", "P"])
        st.write(
            "Active = A, Ex-Employee = B, Filial = F , Not Employee = N , Passive = P")
        pais_residencia = st.text_input(
            "Customer's Country residence", max_chars=2, value="ES")
        sexo = st.selectbox("Customer Gender", options=["V", "H"])
        st.write(
            "Male = V, Female= H")
        age = st.number_input("Customer Age", min_value=10,
                              max_value=120, value=25)
        fecha_alta = st.date_input(
            "Bank account oppening date", datetime.date(2019, 7, 6))
        ind_nuevo = st.selectbox(
            "Is customer registered within 6 months?", options=[0, 1])
        st.write(
            "Yes = 1, No= 0")
        antiguedad = st.number_input(
            "Customer seniority(Months)", min_value=1, value=6)
        indrel = st.selectbox("Primary customer?", options=[1, 99])
        st.write(
            "Primary = 1, Not Primary= 99")
        indrel_1mes = st.selectbox("Type of customer", options=[1, 2, 3, 4])
        st.write(
            "1=(First/Primary customer), 2=(co-owner),P=(Potential),3=(former primary), 4=(former co-owner)")
        tiprel_1mes = st.selectbox(
            "Customer relation type", options=["A", "I", "P", "R"])
        st.write(" A=(active), I=(inactive), P=(former customer),R=(Potential)")
        indresi = st.selectbox(
            "If the residence country is the same than the bank country?", options=["S", "N"])
        st.write("S=Yes,N=No")
        indext = st.selectbox(
            "If the customer's birth country is different than the bank country?", options=["S", "N"])
        st.write("S=Yes,N=No")
        canal_entrada = st.text_input(
            "Channel used by the customer to join", max_chars=3, value="KHE")
        indfall = st.selectbox(
            "User dead or alive?", options=["S", "N"])
        st.write("S=Yes,N=No")
        tipodom = 1
        cod_prov = st.slider(
            "Province code", min_value=1, max_value=99, value=28)
        nomprov = st.text_input("Province name", value="MADRID")
        ind_actividad_cliente = st.selectbox("Activity index", options=[1, 0])
        st.write("1=Active,0=Inactive")
        renta = st.number_input("Gross income of the household",
                                min_value=0.00, max_value=30000000.00, step=0.1)
        segmento = st.selectbox("Customer class", options=[
                                "01 - TOP", "02 - PARTICULARES", "03 - UNIVERSITARIO"])
        if(st.button("Predict")):
            input_dict = {'fecha_dato': fecha_alta, 'ncodpers': age*antiguedad, 'ind_empleado': ind_empleado, 'pais_residencia': pais_residencia, 'sexo': sexo,
                          'age': age, 'fecha_alta': fecha_alta, 'ind_nuevo': ind_nuevo, 'antiguedad': antiguedad, 'indrel': indrel, 'indrel_1mes': indrel_1mes,
                          'tiprel_1mes': tiprel_1mes, 'indresi': indresi, 'indext': indext, 'canal_entrada': canal_entrada, 'indfall': indfall,
                          'tipodom': tipodom, 'cod_prov': cod_prov, 'nomprov': nomprov, 'ind_actividad_cliente': ind_actividad_cliente, 'renta': renta,
                          'segmento': segmento}
            input_df = pd.DataFrame([input_dict])
            output = predict(model=model, input_data=input_df)
            result = label_changer(output)
            st.write(f"Recommended product/products : {result}")

    if(add_selectbox == "Batch"):
        file_upload = st.file_uploader(
            "Upload your csv file for recommendations", type=["csv"])
        if(file_upload is not None):
            data = pd.read_csv(file_upload)
            final_df = preprocess_data(data)
            pred_results = predict_model(estimator=model, data=final_df)
            encode_dict = {}
            for val in pred_results.Label.unique():
                encode_dict[val] = label_changer(val)
            Recommendations = pred_results["Label"].replace(
                encode_dict)
            pred_results["Label"] = Recommendations
            pred_results.rename(
                columns={"Label": "Recommendations"}, inplace=True)
            st.write(pred_results)


run()
