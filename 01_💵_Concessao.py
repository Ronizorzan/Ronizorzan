#Importação das bibliotecas
import streamlit as st
import pandas as pd
from funcoes import *
from tensorflow.keras.models import load_model
from numpy import load


#Configuração da página
st.set_page_config(page_title="Avaliação de Crédito", layout="centered", initial_sidebar_state="expanded")

    


#Variáveis para preencher as caixas de seleção

profissoes = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador', 'Dentista', 'Empresário', 'Engenheiro', 'Médico', 'Programador']
residencia = ['Alugada', 'Outros', 'Própria']
escolaridades = ['Ens.Fundamental', 'Ens.Médio', 'PósouMais', 'Superior']
scores = ['Baixo', 'Bom', 'Justo', 'MuitoBom']
estados_civis = ['Casado', 'Divorciado', 'Solteiro', 'Víuvo']
produtos = ['AgileXplorer', 'DoubleDuty', 'EcoPrestige', 'ElegantCruise', 'SpeedFury', 'TrailConqueror', 'VoyageRoamer', 'WorkMaster']

with st.sidebar:
        acuracia = load("objects/acuracia.npy") #Exibição da acurácia na barra lateral
        st.markdown(f"<span style='font-size: 18px; font-weight: bold'>Acurácia do Modelo: -- </span>\
                    <span style='font-size: 23px; font-weight: bold; color: #008000'>{acuracia*100:.2f}%", unsafe_allow_html=True) 
        st.text("")        
        st.markdown( # Rodapé na barra lateral com as informações do desenvolvedor
        """
        <style>
        .footer {
        background-color: #f8f9fa;
        padding: 15px 20px;
        border-radius: 8px;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-top: 40px;
        color: #343a40;
        }
        .footer a {
        margin: 0 15px;
        display: inline-block;
        }
        .footer img {
        height: 40px;
        width: auto;
        transition: transform 0.3s ease;
        }
        .footer img:hover {
        transform: scale(1.1);
        }
        </style>
        <div class="footer">
        <p><strong>Desenvolvido por: Ronivan</strong></p>
        <a href="https://github.com/Ronizorzan" target="_blank">
            <img src="https://img.icons8.com/ios-filled/50/000000/github.png" alt="GitHub">
        </a>
        <a href="https://www.linkedin.com/in/ronivan-zorzan-barbosa" target="_blank">
            <img src="https://img.icons8.com/color/48/000000/linkedin.png" alt="LinkedIn">
        </a>
        <a href="https://share.streamlit.io/user/ronizorzan" target="_blank">
            <img src="https://images.seeklogo.com/logo-png/44/1/streamlit-logo-png_seeklogo-441815.png" alt="Streamlit Community">
        </a>
        </div>
        """,
        unsafe_allow_html=True)
        
        

#Formação das caixas de seleção e submit_button
with st.form(key = 'prediction_form'):
    with st.expander("Insira as características do novo cliente"):
        profissao = st.selectbox('profissão', profissoes)
        tempoprofissao = st.number_input('Tempo na profissão (em anos)', min_value=0, value=25, step=1, max_value=70)
        renda = st.number_input("Renda", min_value=0.0, value=50000.0, step = 1000.0)
        tiporesidencia = st.selectbox("Tipo de residência", residencia)
        escolaridade = st.selectbox("Escolaridade", escolaridades)
        idade = st.number_input("Idade", min_value=18, value=28, step=1, max_value=110)
        dependentes = st.number_input('Dependentes', min_value = 0, value = 0, step = 1)
        estadocivil = st.selectbox("Estado civil", estados_civis)
        score = st.selectbox("Score", scores)
        produto = st.selectbox("Produto", produtos)
        valorsolicitado = st.number_input("Valor solicitado", min_value=1000.0, value=25000.0, step=1000.0)        
        valortotalbem = st.number_input("Valor total do bem", min_value=1000.0, value=100000.0, step=1000.0)

        submit_button = st.form_submit_button(label= "Consultar")


if submit_button:
    progresso = st.progress(50, "Aguarde um momento.. Processando os dados inseridos")
        
    dados_novos = {
    'profissao': [profissao],
    'tempoprofissao' : [tempoprofissao],
    'renda' : [renda],
    'tiporesidencia': [tiporesidencia],
    'escolaridade': [escolaridade],
    'score': [score],
    'idade': [idade],
    'dependentes': [dependentes],
    'estadocivil': [estadocivil],
    'produto': [produto],
    'valorsolicitado': [valorsolicitado],
    'valortotalbem': [valortotalbem]
    }

    caminho_modelo = 'meu_modelo.h5'
    modelo = load_model(caminho_modelo)
    seletor = joblib.load("objects/seletor.joblib")
    
    df = pd.DataFrame(dados_novos)
    df = load_scalers(df,['tempoprofissao','renda','idade','dependentes','valorsolicitado','valortotalbem'] )
    df = load_encoders(df, ['profissao', 'tiporesidencia',  'escolaridade','score','estadocivil','produto'])
    df = seletor.transform(df)
    
    predictions = modelo.predict(df)
    probabilidade = predictions[0][0] * 100
    classe = "Bom" if probabilidade > 50 else "Ruim"
    if classe=="Bom":
        st.success(f"Resultado: {classe}")
        st.success(f'Probabilidade:{probabilidade:.2f}%')
    else:
        st.error(f"Resultado: {classe}")
        probabilidade = 100 - probabilidade
        st.error(f"Probabilidade: {probabilidade:.2f}%")

    progresso.progress(100, "Dados processados com sucesso...")








