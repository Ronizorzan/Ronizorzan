#Importação das bibliotecas
import streamlit as st
import pandas as pd
from funcoes import *
from utilidades import *

#Configuração da página
st.set_page_config(page_title="Análise de Clientes por Segmento", layout="wide", initial_sidebar_state="expanded")


with st.sidebar:
    data = pd.read_csv("dados_segment.csv")
    segmentacao = st.selectbox("Selecione o Segmento para Análise:", ['Profissão', 'Renda', 'Estado Civil',
                                                'Valor Solicitado', 'Solicitado vs Total Bem', 'Renda vs ValorSolicitado'])
    processar = st.button("Gerar Análise", type="primary", use_container_width=True)
    st.sidebar.markdown("""
### 💡 O Que Esta Análise Revela?

Esta seção funciona como um :orange[**raio-X do nosso risco de crédito**], permitindo-nos entender o porque de alguns clientes se tornarem inadimplentes e outros não.

A análise segmentada revela :orange[**padrões ocultos por trás da inadimplência**] e nos mostra, com clareza, quais perfis e cenários financeiros representam os maiores riscos e, consequentemente, as melhores oportunidades de melhoria.

Com essa inteligência, saímos da reação para a **ação estratégica**: podemos ajustar políticas de crédito, direcionar o marketing e otimizar nosso portfólio de produtos de forma muito mais eficaz.

⚠️ **Ponto de Atenção:**
É importante ressaltar que, embora um fator isolado (como `Profissão`) mostre forte correlação com o risco, 
                        a inadimplência é complexa e pode estar associada a outras variáveis, pois um cliente não é apenas sua profissão, mas a soma de todas as suas características.
                        A causa real do risco geralmente está na **combinação de múltiplos fatores** (ex: uma certa profissão *com* uma determinada faixa de renda e *um certo* valor solicitado).
""")
if processar:
    if segmentacao=='Profissão':
        st.header("Profissão do Cliente é um Indicador Surpreendentemente Forte de Risco de Crédito", divider="orange")
        ax, resumo, grafico = plot_rate_by_group(
        df=data,
        target="classe",
        group=segmentacao,
        bin_strategy="quantile",
        min_count=15,
        sort_by="rate",
        ascending=False,
        title="Inadimplência por Profissão",
        show_ci=False,
        figsize=(9,6)
    )
        col1, col2 = st.columns([0.49,0.51])        
        with col1:
            st.pyplot(grafico, use_container_width=True)
        
        with col2:
            st.markdown("""**Observação-Chave:** Profissões como Arquiteto e Dentista apresentam taxas de inadimplência superiores a :red[**50%**],
                         enquanto Engenheiros se destacam como o grupo com as menores taxas em nossa carteira: :green[**(16,7%)**].

**O Valor da Análise:** Essa visão nos permite ir além dos dados financeiros e entender o contexto do cliente. Ela abre portas para estratégias de marketing e parcerias direcionadas,
                         permitindo-nos "direcionar" ativamente para um perfil de cliente mais atrativo para o negócio.

**Ações Recomendadas:**
Essa análise permite criar uma "Inteligência de Mercado", Mapeando o Risco por Profissão para Atrair os Melhores Clientes."

- **Marketing e Parcerias Estratégicas:** Por que não criar campanhas e parcerias direcionadas a conselhos de engenharia ou empresas de tecnologia, oferecendo condições especiais?
                         Esta é uma forma proativa de aumentar a proporção de bons clientes em nossa base.

- **Motor de Risco:** A "profissão" deve ser uma variável com peso considerável em nosso modelo de "Risco de Crédito". A análise mostra que ela tem um poder preditivo consideravelmente relevante.

- **Análise de Crédito:** Para determinadas profissões, pode-se solicitar documentação adicional que comprove a estabilidade de renda para tomar uma decisão mais bem informada.""")


    if segmentacao=='Renda':
        st.header("Renda Só Faz a Diferença no Topo da Pirâmide", divider="orange")
        ax, resumo, grafico = plot_rate_by_group(
        df=data,
        target="classe",
        group=segmentacao,
        bins=4,
        bin_strategy="quantile",
        min_count=15,
        sort_by="rate",
        ascending=False,
        title="Inadimplência por faixa de Renda",
        show_ci=False,
        figsize=(8,6)
    )
        col1, col2 = st.columns([0.51,0.49])
        with col1:            
            st.pyplot(grafico, use_container_width=True)
        
        with col2:
            st.markdown("""**Observação-Chave:** A inadimplência permanece uniformemente alta :red[**(cerca de 44%)**] para todas as faixas de renda (***abaixo de 46 mil***.)
                         Apenas o grupo com a maior renda (***acima de 46 mil***) mostra uma queda significativa na inadimplência :green[**(28,9%)**].

**O Valor da Análise:** Esta análise quebra o mito de que "renda maior é sempre igual a risco menor". Ela revela a existência de um "ponto de virada".
                         Aumentar a renda de 10 para 30 mil não alterou o comportamento de pagamento em nossa base. O diferencial está em pertencer ao topo da pirâmide.

**Ações Recomendadas:**
- Onde devemos concentrar os esforços? Nosso cliente "ideal" em termos de renda está claramente no topo da pirâmide. Esforços de aquisição e produtos premium 
                        podem ser direcionados para este público, que oferece o melhor equilíbrio entre volume e segurança.

- **Combinação de Variáveis:** Esta análise é mais poderosa quando combinada com a de "Comprometimento Financeiro". Um cliente de alta renda (> R$ 46 mil) que solicita um valor baixo (Alta Renda vs ValorSolicitado baixo) é uma aposta de ouro.
                        Um cliente de renda média ou baixa solicitando um valor que compromete sua renda é uma aposta perigosa.

- **Questionamento do Negócio:** Por que os clientes de renda média e baixa se comportam de maneira tão similar? O valor do nosso produto está mal ajustado para a realidade financeira desses segmentos?
                        Esta é uma pergunta estratégica que pode levar à criação de novos produtos ou a ajustes nos existentes.""")


    if segmentacao=='Estado Civil':
        st.header("Estado Civil Revela Padrões de Risco Surpreendentes", divider="orange")
        ax, resumo, grafico = plot_rate_by_group(
        df=data,
        target="classe",
        group=segmentacao,
        bins=4,
        bin_strategy="quantile",
        min_count=15,
        sort_by="rate",
        ascending=False,
        title="Inadimplência por Score de Crédito",
        show_ci=False
    )
        col1, col2 = st.columns([0.6,0.4])
        with col1:
            st.pyplot(grafico, use_container_width=True)
        
        with col2:
            st.markdown(""" **Observação-Chave:** Clientes casados compõem nosso segmento de maior risco :red[**(58.0%)**], com uma taxa de inadimplência que se aproxima do dobro 
                        da de clientes solteiros :green[**(29.2%)**], nosso grupo mais seguro atualmente.

**O Valor da Análise:** Este insight desafia o senso comum de que "casado" equivale a maior estabilidade financeira. Para nossa carteira de clientes, essa condição sinaliza maior pressão financeira e risco.
                        Essa variável é uma peça-chave para refinar nosso entendimento sobre o perfil do cliente.

**Ações Recomendadas:**
- **Motor de Risco:** O "Estado Civil" deve ser incorporado como uma variável de peso em nossos modelos preditivos de inadimplência.
- **Análise Aprofundada:** Devemos cruzar o estado civil com outras variáveis (como faixa de renda, profissão e valor solicitado) para descobrir a causa raiz. Clientes casados solicitam valores maiores? Possuem mais dependentes?
- **Estratégia de Marketing:** Campanhas com comunicação e ofertas direcionadas ao público solteiro podem ser uma maneira eficaz de atrair uma base de clientes com menor risco inerente.
""")

    if segmentacao=='Valor Solicitado':
        st.header("Emprétimos Menores podem significar Risco Menor.", divider="orange")
        ax, resumo, grafico = plot_rate_by_group(
        df=data,
        target="classe",
        group=segmentacao,
        bins=4,
        bin_strategy="quantile",
        min_count=15,
        sort_by="rate",
        ascending=False,
        title="Inadimplência por Crédito Solicitado",
        show_ci=False
    )
        col1, col2 = st.columns([0.6,0.4])
        with col1:
            st.pyplot(grafico, use_container_width=True)
        
        with col2:
            st.markdown(""" **Observação-Chave:** Pedidos de crédito de alto valor (acima de 123.000) resultam em uma taxa de inadimplência alarmante de :red[**75%**].
                        Em contrapartida, os pedidos de menor valor (até 70.000) são os mais seguros de toda a carteira, com :green[**0% de inadimplência**].

**O Valor da Análise:** Esta análise sugere que ***"emprestar pouco é mais seguro"***, algo bastante intuitivo. Ela indica que o *perfil* do cliente que busca valores menores é menos arriscado,
                         enquanto clientes que solicitam valores maiores tendem a ser mais "arriscados".

**Ações Recomendadas:**
- **Estratégia de Produto:** A rentabilidade do nosso produto de médio valor precisa ser reavaliada. A precificação atual (taxas) é suficiente para cobrir um "risco" de 75%? 
                        Talvez seja necessário um score de aprovação mais rigoroso para esta faixa.
- **Oportunidade de Crescimento:** O segmento de baixo valor é uma ***"mina de ouro"***. Podemos criar produtos premium ou programas de fidelidade para atrair e reter mais clientes que solicitam valores até de R$ 70.000.
- **Análise de Crédito:** A análise de risco para pedidos de **médio à alto valor** deve ser *mais* criteriosa, para proteger a saúde financeira da empresa.
""")
    

    if segmentacao=='Solicitado vs Total Bem':
        st.header("Quanto menor a entrada, maior o risco de Inadimplência Futura.", divider="orange")
        ax, resumo, grafico = plot_rate_by_group(
        df=data,
        target="classe",
        group=segmentacao,
        bin_strategy="quantile",
        bins=4,
        min_count=20,
        sort_by="rate",
        ascending=False,
        title="Inadimplência por proporção financiada",
        show_ci=False
    )
        col1, col2 = st.columns([0.6,0.4])
        with col1:
            st.pyplot(grafico, use_container_width=True)
        
        with col2:
            st.markdown(""" **Observação-Chave:** A proporção do bem que o cliente financia é um indicador fortíssimo de inadimplência.
                        Clientes que deram uma entrada robusta (financiando até 33% do item) apresentaram uma baixa taxa de inadimplência :green[**menos de 15.0%**]. 
                        Em contraste, aqueles que financiaram uma alta proporção (acima de 33%) se tornaram inadimplentes em :red[**83.8%**] dos casos.

**O Valor da Análise:** Esta análise nos entrega uma regra de negócio clara, poderosa e intuitiva. O valor da entrada (ou "sinal") é um reflexo direto do comprometimento e da capacidade do cliente de honrar os compromissos.
                        É uma informação que podemos usar para automatizar e blindar nosso processo de crédito.

**Ações Recomendadas:**
- **Política de Crédito Automatizada:** Implementar regras de corte rígidas na análise de crédito. Financiamentos com proporção acima de 30% devem ser negados ou escalados para análise gerencial.
                        Propostas com financiamento abaixo de 20% podem ser pré-aprovadas.
- **Estratégia de Vendas e Produto:** Podemos incentivar entradas maiores? Oferecer taxas de juros reduzidas para clientes que pagam um sinal maior é uma estratégia "ganha-ganha":
                         melhora a qualidade da nossa carteira e torna a oferta mais atrativa para bons clientes.
""")
        

    if segmentacao=='Renda vs ValorSolicitado':
        st.header("Quanto Menor a Renda em Relação ao Empréstimo, Maior a Certeza do Prejuízo.", divider="orange")
        ax, resumo, grafico = plot_rate_by_group(
        df=data,
        target="classe",
        group=segmentacao,
        bin_strategy="quantile",
        bins=4,
        min_count=20,
        sort_by="rate",
        ascending=False,
        title="Inadimplência por Comprometimento Financeiro",
        show_ci=False,
        figsize=(8,6)
    )
        col1, col2 = st.columns([0.51,0.49], gap="large")
        with col1:
            st.pyplot(grafico, use_container_width=True)
        
        with col2:
            st.markdown("""**Observação-Chave:** O risco de inadimplência está diretamente ligado ao quanto o cliente compromete de sua renda.
                        Clientes cuja renda anual corresponde a até 30% do valor solicitado apresentaram uma taxa de inadimplência altíssima ***(aproximadamente 60%)***, 
                        enquanto aqueles cuja renda anual corresponde à mais da metade do valor solicitado nunca se tornaram inadimplentes.

**O Valor da Análise:** Esta é a métrica mais poderosa para a tomada de decisão. Ela nos dá uma regra clara e quantificável para separar propostas de alto e baixo risco
    antes mesmo de uma análise aprofundada.É o nosso sistema de alerta precoce mais eficaz.

**Ações Recomendadas:**                        
**Política de Crédito:** Devemos usar essa variável para criar faixas de aprovação automáticas?

- **Faixa Segura** :green[**(> 51.54):**] Clientes nesta faixa podem ser pré-aprovados ou passar por um processo "Aprovação-rápida", melhorando a experiência do usuário e nossa agilidade competitiva.

- **Faixa de Alerta** :red[**(< 29.54):**] Propostas nesta faixa devem ser automaticamente recusadas ou enviadas para uma análise manual mais rigorosa. Isso representa a nossa maior oportunidade de redução de perdas.

- **Oportunidade de Produto:** Podemos criar um produto de "crédito expresso" com aprovação instantânea para o perfil de cliente mais seguro, atraindo e fidelizando os melhores pagadores do mercado.
                                        """)



