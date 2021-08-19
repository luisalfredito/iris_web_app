#Importamos librerias
import streamlit as st
import pickle
import pandas as pd

#Extraer archivos pickle
with open('lin_reg.pkl','rb') as li:
    lin_reg = pickle.load(li)

with open('log_reg.pkl','rb') as lo:
    log_reg = pickle.load(lo)

with open('svc_m.pkl','rb')as sv:
    svc_m = pickle.load(sv)

#Fucnion para clasificar las Plantas
def classify(num):
    if num==0:
        return 'Setosa'
    elif num==1:
        return 'Versicolor'
    else:
        return 'Virginia'

def main():
    st.title('Modelamiento de Iris')
    st.sidebar.header('Parametros de entrada:')

    #Funcion para poner los parametros en el sidebar
    def user_input_parameters():
        sepal_lenght = st.sidebar.slider('Sepal/Largo', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal/Ancho', 2.0, 4.4, 3.4)
        petal_lenght = st.sidebar.slider('Petal/Largo', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal/Ancho', 0.1, 2.5, 0.2)

        data= { 'sepal_lenght':sepal_lenght,
                'sepal_width':sepal_width,
                'petal_lenght':petal_lenght,
                'petal_width':sepal_width,
                }

        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()

    #Escojer el modelo preferido
    option = ['Regresion lineal', 'Regresion Logistica', 'SVM']
    model = st.sidebar.selectbox('Que modelo desea utilizar?', option)

    st.subheader('Parametros de entrada')
    st.subheader(model)
    st.write(df)

    if st.button('EJECUTAR'):
        if model == 'Regresion Lineal':
            st.success(classify(lin_reg.predict(df)))
        elif model == 'Regresion Logisitica':
            st.success(classify(log_reg.predict(df)))
        else: 
            st.success(classify(svc_m.predict(df)))

if __name__=='__main__':
    main()







