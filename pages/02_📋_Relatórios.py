import streamlit as st
import numpy as np
from utilidades import *


#Configuração da página
st.set_page_config(page_title="Avaliação de Crédito", layout="wide", initial_sidebar_state="expanded")


matrix_xgb = np.load("objects/confusion_xgb.npy", allow_pickle=False)
matrix_seq = np.load("objects/confusion_seq.npy", allow_pickle=False)
valor_medio_emprest = np.load("objects/valor_medio_emprest.npy")

resultado_xgb = calcular_metricas(matrix_xgb)
resultado_seq = calcular_metricas(matrix_seq)


# Controles da Barra Lateral
with st.sidebar:    

    st.title("Visualização de Resultados")
    with st.expander("Configurações das visualizações", expanded=True):    
        visualizacao = st.radio("Selecione o tipo de visualização", ("Impacto Financeiro",  "Análise de ROI", "Redução da Inadimplência",
                                                                      "Captação de Bons Clientes", "Taxa de Aprovação"))
        #if visualizacao in ["Impacto Financeiro", "Análise de ROI"]:
        taxa_juros = st.slider("Taxa média de juros", min_value=0, max_value=100, value=29,
                                help="Selecione a taxa de juros cobrada por empréstimo\
                                \ne veja como os valores se atualizam no gráfico\
                                \n(Só funciona em Impacto Financeiro e Análise de ROI)") / 100
    visualizar = st.button("Visualizar", type="primary", use_container_width=True)   

    st.markdown("---")
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px;'>
            <span style='font-size: 22px; font-weight: bold; color: #eeeeee;'>🔗Links do desenvolvedor:</span>
        </div>
        """, unsafe_allow_html=True)
    

    # Rodapé na barra lateral com as informações do desenvolvedor    
    st.markdown("""
<style>
.footer {
    background-color: #f8f9fa;
    padding: 15px 20px;
    border-radius: 10px;
    border-left: 9px solid #972328;
    text-align: center;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin-top: 20px;
    color: #343a40;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.footer p {
    font-size: 16px;
    margin-bottom: 15px;
}
.footer a {
    margin: 0 12px;
    display: inline-block;
}
.footer img {
    height: 36px;
    width: 36px;
    border-radius: 6px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.footer img:hover {
    transform: scale(1.1);
    box-shadow: 0 0 8px rgba(151, 35, 40, 0.4);
}
</style>

<div class="footer">
    <p><strong>Desenvolvido por: Ronivan</strong></p>
    <a href="https://github.com/Ronizorzan/Ronizorzan/blob/master/" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="GitHub">
    </a>
    <a href="https://www.linkedin.com/in/ronivan-zorzan-barbosa" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
    </a>
    <a href="mailto:ronizorzan1992@gmail.com" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" alt="Email">
    </a>
    <a href="https://share.streamlit.io/user/ronizorzan" target="_blank">
        <img src="https://images.seeklogo.com/logo-png/44/1/streamlit-logo-png_seeklogo-441815.png" alt="Streamlit">
    </a>
</div>
""", unsafe_allow_html=True)


            

if visualizar:   
    resultados, figura_impacto = calcular_e_plotar_impacto(matrix_xgb, matrix_seq, valor_medio_emprest, taxa_juros)         
    if visualizacao == "Impacto Financeiro": # Gráfico e relatório de impacto financeiro
        st.header("Decisões Inteligentes, Resultados reais")
        st.markdown("<hr style='border: 1px solid #008000'>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.45,0.55], gap="large")                
        with col1:            
            st.pyplot(figura_impacto, use_container_width=True)                           
            st.markdown("<hr style='border: 1px solid #008000'>", unsafe_allow_html=True)                        
            st.markdown("""**O ponto de partida:** A operação atual funciona, mas á um alto custo.
                        O gráfico revela a dura verdade: para cada :green[**5 reais**] ganhos,
                    aproximadamente :red[**3 reais**] são perdidos para a inadimplência.
                    Tecnologias avançadas de Inteligência artificial foram usadas para encontrar a melhor
                    estratégia de aprovação de clientes e para 'estancar' esse vazamento financeiro.""")        
           

        with col2:
            st.markdown("<div style='font-size: 35px; font-weight: bold; color: #FFFFFF'>Relatório de Impacto Detalhado", unsafe_allow_html=True)
            resultados_styled = resultados.style.apply(highlight_models, axis=1).format("{:,.2f}", subset=['Ganhos', 'Perdas', 'Retorno Líquido'])
            st.write(resultados_styled.to_html(escape=False), unsafe_allow_html=True)
            diferenca_baseline = (resultados.iloc[2, 3]) - (resultados.iloc[0, 3])
            diferenca_xgb = (resultados.iloc[2, 3]) - (resultados.iloc[1, 3])
            #st.markdown(f"<div style='font-size: 27px; font-weight: bold; color: #008000'>Retorno\
            #            líquido estimado utilizando o modelo: R$ {resultados.iloc[2,3]:,.2f} ", unsafe_allow_html=True)
            st.metric(f"Lucro adicional gerado pela IA:",
                      value=f"+ R$ {diferenca_baseline:,.2f} ", delta=f"{(resultados.iloc[2,3] / resultados.iloc[0,3] )* 100:,.2f}%", delta_color="normal")
            st.metric(f"Mesmo contra outro modelo avançado, Redes Neurais se mostrou superior, garantindo máxima eficiência.",
                      value=f"+ R$ {diferenca_xgb:,.2f} ", delta=f"{(resultados.iloc[2,3] / resultados.iloc[1,3] )* 100:,.2f}%", delta_color="normal")
            st.markdown("<hr style='border: 1px solid #008000'>", unsafe_allow_html=True)            
            st.markdown("<div style='font-size: 32px; font-weight: bold; color: #FFFFFF'>Da Incerteza ao lucro ", unsafe_allow_html=True)
            st.markdown("""<div style='font-size: 14px; font-weight: sans serif'>O método com redes neurais superou drasticamente o método atual de aprovação, alcançando um novo patamar de excelência.
                        O modelo praticamente elimina os prejuízos com inadimplência,
                        preservando o capital que antes era perdido com inadimplência. Com decisões de aprovação mais precisas,
                        o modelo campeão não só protege, mas impulsiona o retorno líquido.
                        Ele não apenas previu os maus pagadores com precisão cirúrgica, como também otimizou os ganhos com os bons clientes, encontrando o equilíbrio perfeito.""", unsafe_allow_html=True)                                            
    
    if visualizacao== "Análise de ROI":        
        col1, col2 = st.columns([0.45,0.55], gap="medium")        
        with col1:            
            st.header("**Qual abordagem entrega mais valor?**")
            #st.markdown("<hr style='border: 1px solid green'>", unsafe_allow_html=True)
            grafico_roi = plot_roi(resultados)            
            st.pyplot(grafico_roi, use_container_width=True)
            st.markdown("<hr style='border: 1px solid green'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 20px; font-weight: bold; color: #ffffff'>Informações Importantes: ", unsafe_allow_html=True)
            st.markdown("Esse é um projeto de portfólio. Os valores apresentados não consideram eventuais custos operacionais, de infraestrutura ou de pessoal.\
                        Todos os custos e retornos foram calculados com base apenas no desempenho dos modelos e nos valores presentes nos dados utilizados.")

        with col2:
            st.markdown(f"""## 🚀 ROI dos Modelos de Machine Learning                        

Este gráfico compara o **Retorno sobre Investimento (ROI)** de três abordagens de modelagem:

| Modelo           | ROI     | Destaque |
|------------------|---------|----------|
| 🔴 Baseline       | Baixo    | Referência mínima |
| 🟠 XGBoost        | Bom    | Bom equilíbrio entre custo e retorno |
| 🟢 Redes Neurais  | **Excelente** | 🚀 Maior retorno, ideal para escala |

### 📌 Insights-chave:
- **Redes Neurais entregam mais de 10x o ROI do modelo base**, justificando investimentos em infraestrutura e tempo de treinamento.
- **XGBoost é uma opção sólida** para cenários com restrições de tempo ou recursos.
- O **modelo Baseline serve como controle**, mas não é competitivo em retorno.

### 🎯 Recomendação Estratégica:
> Investir em Redes Neurais para projetos com alto potencial de retorno. Para ambientes com limitação de recursos, XGBoost é uma alternativa eficiente.

---

### 💬 Conclusão Final:
Este gráfico não apenas mostra números — ele conta uma história de evolução tecnológica e retorno financeiro. Vamos usar esses dados para embasar decisões com confiança.""")
                        


    if visualizacao == "Redução da Inadimplência": # Gráfico e relatório de Inadimplência
        st.header("Redução da Inadimplência", anchor="red_inadimplencia")
        st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.55, 0.45], gap="medium")
        with col1:            
            figura_inad = plot_inadimplencia(resultado_xgb['inadimplencia_sem_modelos'], 
                                         resultado_xgb['inadimplencia_prevista'], resultado_seq['inadimplencia_prevista'])
            st.pyplot(figura_inad, use_container_width=True)
        
        with col2:
            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Redução drástica na inadimplência", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 20px; font-weight: bold'>Este gráfico compara as três abordagens distintas e revela o poder dos modelos\
                         de inteligência artificial para transformar a gestão de risco e aumentar a rentabilidade nos negócios.", unsafe_allow_html=True)            
            
            st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True) # Linha de separação

            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Comparação de Cenários", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 20px; font-weight: bold'>Taxa de Inadimplência Atual atinge alarmantes: \t </span>\
                      <span style='color: red; font-size: 25px; font-weight: bold'> {resultado_xgb['inadimplencia_sem_modelos']}%</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 20px; font-weight: bold'>Taxa de Inadimplência estimada com o uso do modelo mais\
                        eficaz (Redes Neurais) é de apenas: </span> <span style='color: green; font-size: 25px; font-weight: bold'>\
                        \t {resultado_seq['inadimplencia_prevista']}0%</span>", unsafe_allow_html=True)
            
            st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True) # Linha de separação

            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Impacto final na taxa de Inadimplência", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 25px; font-weight: bold'>Redução total estimada na taxa de inadimplência com o uso de Redes Neurais:</span>\
                     <span style='font-size: 30px; font-weight: bold; color: green'>\
                         -{round(resultado_seq['inadimplencia_sem_modelos'] - resultado_seq['inadimplencia_prevista'], 2)}%</span>", unsafe_allow_html=True)


    elif visualizacao == "Captação de Bons Clientes": # Gráfico e relatório de captação de bons clientes
        st.header("Captação de Bons Clientes")
        st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.6, 0.4], gap="large")
        with col1:
            figura2 = plot_captacao_bons_clientes(resultado_xgb["captacao_bons_clientes"], resultado_seq["captacao_bons_clientes"])        
            st.pyplot(figura2, use_container_width=True)
        with col2:
            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Descrição da visualização", unsafe_allow_html=True)
            st.markdown("<div style=' font-size: 20px; font-weight:bold'>🏆 O gráfico ao lado mostra a capacidade do modelo de identificar\
                        clientes de qualidade. Uma alta captação como a mostrada ao lado é imprescindível \
                        pois pode significar maior qualidade na aprovação e menos risco financeiro. ", unsafe_allow_html=True)            
            
            st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Possível abordagem alternativa", unsafe_allow_html=True)
            st.markdown("<div style=' font-size: 20px; font-weight:bold'>🤖 XGBoost ainda é competitivo! Embora tenha ficado atrás," \
            " a quantidade de bons clientes captados ainda indica um excelente desempenho, mostrando que árvores de decisão otimizadas\
                  continuam sendo uma alternativa sólida em alguns casos.", unsafe_allow_html=True)
            st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)

    
    elif visualizacao == "Taxa de Aprovação": # Gráfico e relatório de taxa de aprovação
        st.header("Taxa de Aprovação")
        st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.6,0.4], gap="large")
        with col1:            
            figura3 = plot_aprovacao_reprovacao(resultado_xgb["taxa_aprovacao"], resultado_xgb['taxa_reprovacao'],
                                                 resultado_seq['taxa_aprovacao'], resultado_seq['taxa_reprovacao'])
            st.pyplot(figura3, use_container_width=True)
        with col2:            
            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Descrição da visualização", unsafe_allow_html=True)                      
            st.markdown("<div style=' font-size: 22px; font-weight:bold'>O gráfico ao lado mostra a taxa de aprovação \
                        de clientes. Note que Redes Neurais aprovou menos clientes que XGBoost, mas \
                        ainda assim conseguiu captar uma maior quantidade de bons clientes e consequentemente reprovou\
                        clientes com alto risco de se tornar inadimplentes, o que traz grandes benefícios, não\
                        só na maior captação de recursos mas também na prevenção de perdas financeiras.</div>", unsafe_allow_html=True )
            st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Taxa de Aprovação dos Modelos", unsafe_allow_html=True)
            st.markdown(f"<div style=' font-size: 22px; font-weight:bold'>Redes Neurais aprovou {resultado_seq['taxa_aprovacao']}%\
                        dos clientes que solicitaram empréstimo enquanto XGB aprovou {resultado_xgb['taxa_aprovacao']}0% ", unsafe_allow_html=True)
                     
                       


