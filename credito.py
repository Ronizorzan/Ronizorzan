#Importação das bibliotecas
import requests
import streamlit as st
import yaml



#Configuração da página
st.set_page_config(page_title="Avaliação de Crédito", layout="centered", initial_sidebar_state="expanded")

#Arquivos de configuração
with open('secrets.toml', 'r') as file:
    config = yaml.safe_load(file)
    url = config['url_api']['url']    


#Variáveis para preencher as caixas de seleção

profissoes = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador', 'Dentista', 'Empresário', 'Engenheiro', 'Médico', 'Programador']
residencia = ['Alugada', 'Outros', 'Própria']
escolaridades = ['Ens.Fundamental', 'Ens.Médio', 'PósouMais', 'Superior']
scores = ['Baixo', 'Bom', 'Justo', 'MuitoBom']
estados_civis = ['Casado', 'Divorciado', 'Solteiro', 'Víuvo']
produtos = ['AgileXplorer', 'DoubleDuty', 'EcoPrestige', 'ElegantCruise', 'SpeedFury', 'TrailConqueror', 'VoyageRoamer', 'WorkMaster']

#Formação das caixas de seleção e submit_button

with st.form(key = 'prediction_form'):
    profissao = st.selectbox('profissão', profissoes)
    tempoprofissao = st.number_input('Tempo na profissão (em anos)', min_value=0, value=0, step=1, max_value=70)
    renda = st.number_input("Renda", min_value=0.0, value=0.0, step = 1000.0)
    tiporesidencia = st.selectbox("Tipo de residência", residencia)
    escolaridade = st.selectbox("Escolaridade", escolaridades)
    idade = st.number_input("Idade", min_value=18, value=25, step=1, max_value=110)
    dependentes = st.number_input('Dependentes', min_value = 0, value = 0, step = 1)
    estadocivil = st.selectbox("Estado civil", estados_civis)
    score = st.selectbox("Score", scores)
    produto = st.selectbox("Produto", produtos)
    valorsolicitado = st.number_input("Valor solicitado", min_value=1000.0, value=10000.0, step=1000.0)        
    valortotalbem = st.number_input("Valor total do bem", min_value=1000.0, value=10000.0, step=1000.0)


    submit_button = st.form_submit_button(label= "Consultar")


if submit_button:
        
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
    'valortotalbem': [valortotalbem],
    'proporcaosolicitadototal': [valorsolicitado / valortotalbem]
}


    response = requests.post(url, json=dados_novos)
    
    if response.status_code == 200:
        predictions = response.json()
        probabilidade = predictions[0][0] * 100
        classe = "Bom" if probabilidade > 50 else "Ruim"
        if classe=="Bom":
            st.success(f"Classe: {classe}")
            st.success(f'Probabilidade:{probabilidade:.2f}%')
        else:
            st.error(f"Classe: {classe}")
            probabilidade = 100 - probabilidade
            st.error(f"Probabilidade: {probabilidade:.2f}%")
                

    else:
        st.error(f"Erro ao obter previsão: {response.status_code}")
        

    




