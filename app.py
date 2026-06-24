import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from analise_arquivo import carregar_e_preparar_dados_unificados, top_especialidades, analisar_glicemia_reinternacao, calcular_dados_conversao_hba1c,calcular_matriz_correlacao_multivariada

st.set_page_config(
    page_title="Análise de Readmissão Hospitalar",
    layout="wide",
)

# Carregamento otimizado usando cache do Streamlit e a função unificada
@st.cache_data
def obter_dados_completos():
    return carregar_e_preparar_dados_unificados()

df, dados = obter_dados_completos()

# --- Funções Genéricas e Dinâmicas de Visualização ---

def criar_tabela_resumo(dados_df, coluna_grupo, coluna_alvo="readmitido"):
    """Gera uma tabela agrupada dinâmica baseada na coluna de grupo e na métrica alvo desejada."""
    tabela = (
        dados_df.groupby(coluna_grupo, observed=False)
        .agg(
            total_pacientes=(coluna_alvo, "count"),
            total_readmitidos=(coluna_alvo, "sum"),
            taxa_readmissao=(coluna_alvo, "mean"),
        )
        .reset_index()
    )
    tabela["taxa_readmissao_pct"] = tabela["taxa_readmissao"] * 100
    return tabela

def criar_grafico(tabela, coluna_grupo, eixo_x, titulo, cores, label_y="Taxa de readmissão (%)"):
    """Cria um gráfico de barras vertical dinâmico usando Plotly Express."""
    fig = px.bar(
        tabela,
        x=coluna_grupo,
        y="taxa_readmissao_pct",
        title=titulo,
        labels={
            coluna_grupo: eixo_x,
            "taxa_readmissao_pct": label_y,
            "total_pacientes": "Total de pacientes",
            "total_readmitidos": "Total readmitidos",
        },
        hover_data={
            coluna_grupo: False,
            "total_pacientes": ":,",
            "total_readmitidos": ":,",
            "taxa_readmissao_pct": ":.2f",
        },
        color=coluna_grupo,
        color_discrete_map=cores,
    )
    fig.update_layout(
        xaxis_title=eixo_x,
        yaxis_title=label_y,
        showlegend=False,
        hoverlabel=dict(bgcolor="black", font_color="white", font_size=13),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig.update_yaxes(range=[0, max(100, tabela["taxa_readmissao_pct"].max() * 1.15)])
    return fig

def mostrar_analise(dados_df, pergunta, hipotese, coluna_grupo, eixo_x, titulo_grafico, cores, coluna_alvo="readmitido", label_y="Taxa de readmissão (%)"):
    """Renderiza a estrutura padrão de perguntas, hipóteses, tabelas e gráficos."""
    st.subheader("Pergunta e hipótese")
    st.markdown(f"**Pergunta:** {pergunta}")
    st.markdown(f"**Hipótese:** {hipotese}")

    tabela = criar_tabela_resumo(dados_df, coluna_grupo, coluna_alvo=coluna_alvo)

    st.subheader("Tabela de resultados")
    st.dataframe(
        tabela.rename(
            columns={
                coluna_grupo: "grupo",
                "total_pacientes": "total de pacientes",
                "total_readmitidos": "total readmitidos",
                "taxa_readmissao_pct": label_y.lower(),
            }
        )[
            [
                "grupo",
                "total de pacientes",
                "total readmitidos",
                label_y.lower(),
            ]
        ].round(2),
        use_container_width=True,
    )
    st.subheader("Gráfico interativo")
    st.caption("Passe o mouse sobre as barras para ver os valores exatos.")
    fig = criar_grafico(tabela, coluna_grupo, eixo_x, titulo_grafico, cores, label_y=label_y)
    st.plotly_chart(fig, use_container_width=True)

# --- Cabeçalho e Métricas Gerais do Dashboard ---

st.title("📊 Análise de Readmissão Hospitalar")
st.markdown(
    """
    Esta análise investiga como o histórico clínico, o volume de medicamentos administrados 
    e as especialidades médicas impactam a taxa de readmissão hospitalar dos pacientes.
    """
)

taxa_geral = dados["readmitido"].mean() * 100
total_pacientes = len(dados)
total_readmitidos = dados["readmitido"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total de pacientes", f"{total_pacientes:,}".replace(",", "."))
col2.metric("Total readmitidos", f"{total_readmitidos:,}".replace(",", "."))
col3.metric("Taxa geral de readmissão", f"{taxa_geral:.2f}%")

with st.expander("📂 Ver primeiras linhas da base bruta"):
    st.dataframe(df.head(), use_container_width=True)

with st.expander("🛠️ Ver dados tratados usados nesta análise"):
    st.dataframe(dados.head(), use_container_width=True)

st.divider()

aba_faixa_etaria, aba_tempo_internacao, aba_medicamentos, aba_especialidades, aba_genero, aba_diagnosticos, aba_internacoes, aba_urgencias, aba_glicemia, aba_correlacao, = st.tabs([
    "1. Faixa Etária",
    "2. Tempo de Internação",
    "3. Medicamentos (H3)",
    "4. Especialidades Médicas (H4)",
    "5. Internação por gênero",
    "6. Diagnósticos anteriores",
    "7. Internações anteriores", 
    "8. Urgências anteriores",
    "9. Controle Glicêmico ",
    "10. Matriz de Correlação "
])



#  CONFIGURAÇÃO DOS FILTROS NA SIDEBAR 

st.sidebar.header("Filtros de Análise")

lista_idades = ["Todos"] + sorted(list(dados['age'].unique()))
idade_sel = st.sidebar.selectbox("Faixa Etária (age):", lista_idades)

lista_generos = ["Todos"] + sorted(list(dados['genero'].unique()))
genero_sel = st.sidebar.selectbox("Gênero (gender):", lista_generos)

lista_racas = ["Todos"] + sorted(list(dados['race'].unique()))
raca_sel = st.sidebar.selectbox("Raça/Etnia (race):", lista_racas)

dados_filtrados = dados.copy()

if idade_sel != "Todos":
    dados_filtrados = dados_filtrados[dados_filtrados['age'] == idade_sel]

if genero_sel != "Todos":
    dados_filtrados = dados_filtrados[dados_filtrados['genero'] == genero_sel]

if raca_sel != "Todos":
    dados_filtrados = dados_filtrados[dados_filtrados['race'] == raca_sel]

dados = dados_filtrados


# 1. ABA FAIXA ETÁRIA (Bianca)

with aba_faixa_etaria:

    mostrar_analise(
        dados_df=dados,
        pergunta="Pacientes mais idosos apresentam maiores taxas de reinternação?",
        hipotese="Pacientes acima de 60 anos possuem maior probabilidade de serem reinternados em até 30 dias.",
        coluna_grupo="faixa_etaria",
        eixo_x="Faixa Etária",
        titulo_grafico="Taxa de readmissão por faixa etária",
        cores={
            "Até 30 anos": "#4C9BE8",
            "31-60 anos": "#F0A500",
            "Acima de 60 anos": "#E05C5C"
        }
    )

    st.divider()

    st.subheader("Distribuição completa das categorias de readmissão")

    dist_idade = (
        dados.groupby(['faixa_etaria', 'readmitted'])
        .size()
        .unstack(fill_value=0)
    )

    dist_idade_pct = (
        dist_idade.div(dist_idade.sum(axis=1), axis=0)
        .mul(100)
        .reset_index()
    )

    dist_long = dist_idade_pct.melt(
        id_vars='faixa_etaria',
        var_name='Readmissao',
        value_name='Porcentagem'
    )

    fig_idade = px.bar(
        dist_long,
        x='Porcentagem',
        y='faixa_etaria',
        color='Readmissao',
        orientation='h',
        title='Distribuição percentual por faixa etária',
        color_discrete_map={
            'NO': '#4C9BE8',
            '>30': '#F0A500',
            '<30': '#E05C5C'
        }
    )

    st.plotly_chart(fig_idade, use_container_width=True)

    with st.expander("📝 Conclusão – Hipótese 1"):
        st.markdown("""
        **Resultado:** O gráfico permite comparar as taxas de readmissão
        entre diferentes grupos etários.

        **Conclusão:** Caso os pacientes acima de 60 anos apresentem
        maiores taxas de readmissão em menos de 30 dias, a hipótese é
        confirmada.
        """)

# 2. ABA TEMPO DE INTERNAÇÃO (Bianca)

with aba_tempo_internacao:

    mostrar_analise(
        dados_df=dados,
        pergunta="O tempo de permanência hospitalar influencia a chance de reinternação?",
        hipotese="Pacientes que permanecem mais tempo internados apresentam maior probabilidade de reinternação.",
        coluna_grupo="faixa_internacao",
        eixo_x="Tempo de Internação",
        titulo_grafico="Tempo de internação x readmissão",
        cores={
            "1-3 dias": "#4C9BE8",
            "4-7 dias": "#F0A500",
            "8-14 dias": "#E05C5C"
        }
    )

    st.divider()

    st.subheader("Distribuição completa das categorias de readmissão")

    dist_tempo = (
        dados.groupby(['faixa_internacao', 'readmitted'])
        .size()
        .unstack(fill_value=0)
    )

    dist_tempo_pct = (
        dist_tempo.div(dist_tempo.sum(axis=1), axis=0)
        .mul(100)
        .reset_index()
    )

    dist_long = dist_tempo_pct.melt(
        id_vars='faixa_internacao',
        var_name='Readmissao',
        value_name='Porcentagem'
    )

    fig_tempo = px.bar(
        dist_long,
        x='Porcentagem',
        y='faixa_internacao',
        color='Readmissao',
        orientation='h',
        title='Distribuição percentual por tempo de internação',
        color_discrete_map={
            'NO': '#4C9BE8',
            '>30': '#F0A500',
            '<30': '#E05C5C'
        }
    )

    st.plotly_chart(fig_tempo, use_container_width=True)

    with st.expander("📝 Conclusão – Hipótese 2"):
        st.markdown("""
        **Resultado:** O gráfico mostra a distribuição das readmissões
        de acordo com o tempo de permanência hospitalar.

        **Conclusão:** Caso os grupos com maior tempo de internação
        apresentem maiores taxas de readmissão, a hipótese é confirmada.
        """)


# 3. ABA MEDICAMENTOS (Lucas)
with aba_medicamentos:
    mostrar_analise(
        dados_df=dados,
        pergunta="Existe relação entre o número de medicamentos administrados e a velocidade de reinternação?",
        hipotese="Pacientes que receberam mais medicamentos apresentam maior taxa de reinternação em menos de 30 dias.",
        coluna_grupo="faixa_medicamentos",
        eixo_x="Faixa de medicamentos administrados",
        titulo_grafico="Readmissão em < 30 dias por faixa de medicamentos",
        cores={label: "#E05C5C" for label in ['1-5', '6-10', '11-15', '16-20', '21-30', '31+']},
        coluna_alvo="readmit_30d",
        label_y="Taxa de readmissão < 30 dias (%)"
    )
    
    st.divider()
    st.subheader("Análise Complementar: Média de medicamentos por desfecho")
    
    med_mean = dados.groupby('readmitted', observed=True)['num_medications'].mean().reindex(['NO', '<30', '>30']).reset_index()
    med_mean['desfecho_pt'] = med_mean['readmitted'].map({'NO': 'Sem readmissão', '<30': 'Readmissão < 30 dias', '>30': 'Readmissão > 30 dias'})
    
    fig3b = px.bar(
        med_mean,
        x='desfecho_pt',
        y='num_medications',
        text='num_medications',
        title="Quantidade média de medicamentos por desfecho clínico",
        labels={'desfecho_pt': 'Desfecho clínico', 'num_medications': 'Média de medicamentos'},
        color='desfecho_pt',
        color_discrete_map={'Sem readmissão': '#4C9BE8', 'Readmissão < 30 dias': '#E05C5C', 'Readmissão > 30 dias': '#F0A500'}
    )
    fig3b.update_traces(texttemplate='%.1f', textposition='outside')
    fig3b.update_layout(
        showlegend=False, 
        margin=dict(l=20, r=20, t=70, b=20),
        hoverlabel=dict(bgcolor="black", font_color="white", font_size=13)
    )
    st.plotly_chart(fig3b, use_container_width=True)

    with st.expander("📝 Conclusão – Hipótese 3"):
        tabela_h3 = criar_tabela_resumo(dados, "faixa_medicamentos", coluna_alvo="readmit_30d")
        taxa_baixa = tabela_h3.iloc[0]['taxa_readmissao_pct']
        taxa_alta  = tabela_h3.iloc[-1]['taxa_readmissao_pct']
        media_30d  = med_mean[med_mean['readmitted'] == '<30']['num_medications'].values[0]
        media_no   = med_mean[med_mean['readmitted'] == 'NO']['num_medications'].values[0]
        tendencia  = "aumenta" if taxa_alta > taxa_baixa else "não varia de forma linear"

        st.markdown(f"""
        **Resultado:** A taxa de readmissão em < 30 dias **{tendencia}** conforme mudamos de faixa de medicamentos. 
        Pacientes na faixa inicial (1–5) possuem uma taxa de **{taxa_baixa:.2f}%**, ao passo que os pacientes na faixa mais alta (31+) registram **{taxa_alta:.2f}%**.
        
        Complementarmente, a média de medicamentos administrados para pacientes reinternados em menos de 30 dias 
        é de **{media_30d:.1f}** medicamentos, comparado a **{media_no:.1f}** daqueles que não foram readmitidos.
        
        **Conclusão:** A hipótese é **parcialmente confirmada**. Embora pacientes com quadros clínicos mais graves tomem mais remédios e apresentem médias ligeiramente superiores, a faixa isolada de medicação não dita de forma absoluta a variação do risco de readmissão rápida.
        """)

# 4. ABA ESPECIALIDADES (Lucas)
with aba_especialidades:        
    # ADICIONADO: Pergunta e Hipótese
    st.subheader("Pergunta e hipótese")
    st.markdown("**Pergunta:** Algumas especialidades médicas apresentam maiores taxas de reinternação?")
    st.markdown("**Hipótese:** Especialidades associadas ao tratamento de doenças crônicas apresentam maiores índices de reinternação.")
    st.divider()

    n_esp = st.slider(
        "Selecione o número de especialidades a exibir (por volume de atendimentos):",
        min_value=5, max_value=20, value=10, step=1
    )
    
    lista_top_esp = top_especialidades(dados, n=n_esp)
    dados_esp = dados[dados['medical_specialty'].isin(lista_top_esp)].copy()
    
    tabela_esp = criar_tabela_resumo(dados_esp, 'medical_specialty', coluna_alvo='readmit_30d').sort_values('taxa_readmissao_pct', ascending=True)
    mediana_taxa = tabela_esp['taxa_readmissao_pct'].median()
    tabela_esp['Destaque'] = tabela_esp['taxa_readmissao_pct'].apply(lambda x: 'Acima da Mediana' if x >= mediana_taxa else 'Abaixo da Mediana')
    
    col_h4a, col_h4b = st.columns(2)
    
    with col_h4a:
        st.subheader("Taxa de readmissão < 30 dias por especialidade")        
        fig4a = px.bar(
            tabela_esp,
            x='taxa_readmissao_pct',
            y='medical_specialty',
            orientation='h',
            title=f"Readmissão < 30 dias pelas Top {n_esp} Especialidades",
            labels={
                'taxa_readmissao_pct': 'Taxa de Readmissão < 30d (%)', 
                'medical_specialty': 'Especialidade',
                'total_pacientes': 'Total de pacientes',
                'total_readmitidos': 'Total readmitidos (<30d)'
            },
            color='Destaque',
            color_discrete_map={'Acima da Mediana': '#E05C5C', 'Abaixo da Mediana': '#4C9BE8'},
            hover_data={
                'Destaque': False,
                'medical_specialty': True,
                'total_pacientes': ":,",
                'total_readmitidos': ":,",
                'taxa_readmissao_pct': ":.2f"
            }
        )
        fig4a.add_vline(x=mediana_taxa, line_dash="dash", line_color="gray", annotation_text=f"Mediana: {mediana_taxa:.1f}%")
        
        fig4a.update_layout(
            margin=dict(l=20, r=20, t=70, b=20),
            hoverlabel=dict(bgcolor="black", font_color="white", font_size=13)
        )
        st.plotly_chart(fig4a, use_container_width=True)
        
    with col_h4b:
        st.subheader("Distribuição de todas as categorias de readmissão")        
        dist_esp = dados_esp.groupby(['medical_specialty', 'readmitted'], observed=True).size().unstack(fill_value=0)
        dist_esp_pct = dist_esp.div(dist_esp.sum(axis=1), axis=0).mul(100).reset_index()        
        dist_long = dist_esp_pct.melt(id_vars='medical_specialty', var_name='Readmissao', value_name='Porcentagem')
        
        fig4b = px.bar(
            dist_long,
            x='Porcentagem',
            y='medical_specialty',
            color='Readmissao',
            orientation='h',
            title="Distribuição percentual completa do desfecho",
            labels={
                'Porcentagem': '% de Pacientes', 
                'medical_specialty': 'Especialidade', 
                'Readmissao': 'Desfecho'
            },
            color_discrete_map={'NO': '#4C9BE8', '>30': '#F0A500', '<30': '#E05C5C'},
            hover_data={
                'medical_specialty': True,
                'Readmissao': True,
                'Porcentagem': ':.2f'
            }
        )
        
        fig4b.update_layout(
            margin=dict(l=20, r=20, t=70, b=20),
            hoverlabel=dict(bgcolor="black", font_color="white", font_size=13)
        )
        st.plotly_chart(fig4b, use_container_width=True)

    with st.expander("📋 Tabela resumo – Volume e taxas por especialidade"):
        resumo_tabela_entrega = (
            dados_esp.groupby('medical_specialty')
            .agg(
                total_pacientes=('readmitted', 'count'),
                total_readmitidos_30d=('readmit_30d', 'sum'),
            )
            .reset_index()
        )
        resumo_tabela_entrega['taxa_readmissao_pct'] = (resumo_tabela_entrega['total_readmitidos_30d'] / resumo_tabela_entrega['total_pacientes'] * 100).round(2)
        resumo_tabela_entrega = resumo_tabela_entrega.sort_values('total_pacientes', ascending=False).rename(
            columns={
                'medical_specialty': 'Especialidade Médica',
                'total_pacientes': 'Total de Pacientes Atendidos',
                'total_readmitidos_30d': 'Total de Readmissões (<30d)',
                'taxa_readmissao_pct': 'Taxa de Readmissão < 30 dias (%)'
            }
        )
        st.dataframe(resumo_tabela_entrega, use_container_width=True)

    with st.expander("📝 Conclusão – Hipótese 4"):
        esp_maior = tabela_esp.iloc[-1]['medical_specialty']
        taxa_max = tabela_esp.iloc[-1]['taxa_readmissao_pct']
        esp_menor = tabela_esp.iloc[0]['medical_specialty']
        taxa_min = tabela_esp.iloc[0]['taxa_readmissao_pct']
        
        st.markdown(f"""
        **Resultado:** Há uma flutuação estatística importante dependendo do nicho de atendimento:  
        - **Maior taxa de readmissão rápida (<30 dias):** {esp_maior} ({taxa_max:.2f}%)  
        - **Menor taxa de readmissão rápida (<30 dias):** {esp_menor} ({taxa_min:.2f}%)
        
        Especialidades voltadas ao acompanhamento continuado de patologias crônicas de alta complexidade (como Cardiologia ou Nefrologia) rotineiramente se posicionam acima da mediana geral.
        
        **Conclusão:** A hipótese é **confirmada**. A especialidade médica do atendimento reflete a complexidade clínica de base e a severidade da doença crônica do paciente, o que possui impacto direto sobre as chances de uma reinternação rápida.
        """)


#5. ABA GENERO (Lavínia)
with aba_genero:
    mostrar_analise(
        dados_df=dados,
        pergunta="Existe diferença na taxa de reinternação entre homens e mulheres?",
        hipotese="As taxas de reinternação variam entre os sexos.",
        coluna_grupo="genero",
        eixo_x="Gênero",
        titulo_grafico="Genero x taxa de Reinternação",
        cores={"Feminino": "#b44c43", "Masculino": "#2f4f4f"}
   )
    taxa_sexo = dados.groupby('genero')['readmitido'].mean() * 100

    taxa_fem = taxa_sexo.get('Feminino', 0.0)
    taxa_masc = taxa_sexo.get('Masculino', 0.0)

    st.info(f"Taxa de sobrevivência Feminina: **{taxa_fem:.1f}%** | Masculina: **{taxa_masc:.1f}%**\n\n Ao verificar as taxas de reintenação de homens e mulheres, nota-se uma variação pequena, confirmando a hipótese de que há variação, mas não é uma diferença significativa.")
    


#5. ABA DIAGNOSTICO (Lavínia)
with aba_diagnosticos:
    pergunta = "Pacientes com mais diagnósticos registrados apresentam maior risco de reinternação?"
    hipotese = "Quanto maior o número de diagnósticos, maior a probabilidade de reinternação."
    st.subheader("Pergunta e hipótese")
    st.markdown(f"**Pergunta:** {pergunta}")
    st.markdown(f"**Hipótese:** {hipotese}")

    fig = px.bar(
        dados.groupby(['quantidade_diagnostico', 'readmitted'])
             .size() # conta pacientes por combinação faixa+readmissão
             .unstack(fill_value=0) # transforma 'readmitted' em colunas (vira tabela)
             .pipe(lambda d: d.div(d.sum(axis=1), axis=0)).mul(100) #calcula a porcentagem de readmissão (ou a não readmissão)
             .reset_index()# tira 'quantidade_diagnostico' do índice, vira coluna normal
             .melt( 
                 # "desempilha" as colunas de readmissão em linhas (formato longo), formando a coluna de readmissão, que guarda o "No", ">30" e "<30" e a coluna que guarda a procentagem de cada classificação por faixa etária
                 id_vars='quantidade_diagnostico',  
                 var_name='Readmissao',      
                 value_name='Porcentagem'   
        ),
        x='quantidade_diagnostico', y='Porcentagem', color='Readmissao',
        barmode='stack', #o stack permite que as barras fiquem uma em cima da outra, formando uma única barra dividade pela % de cada classificação de readmissão
        text='Porcentagem',#coloca o valor da % na barra/cor correspondente
        color_discrete_map={'NO': '#61AFB6', '>30': '#C1B312', '<30': '#B83F39'},
        title="Distribuição completa de readmissão por faixa de diagnósticos"
    )

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside',textfont=dict(color='white'))
    st.plotly_chart(fig)
    
    st.info(
    "Pacientes na faixa de **1-5 diagnósticos** têm aproximadamente **36%** de chance de "
    "algum tipo de readmissão, enquanto pacientes na faixa de **11-15 diagnósticos** "
    "apresentam aproximadamente **56%** — um aumento de cerca de **20 pontos percentuais**.\n\n  "
    "Pacientes da faixa intermediária, **6-10**, apresentam **49%** de chance de readmissão"
    )
    
    st.write("Ao observar o gráfico vemos que a hipótese se confirma. Conforme o grupo de diagnósticos anteriores aumenta, nota-se um aumento no número das reinternações >30 e <30 dias. Pode-se perceber com base nos dois extremos (1-5) e (11-15), onde há uma diferença de 20 pontos na taxa de não readmissão")
    

# 7. ABA INTERNAÇÕES (Gabriel)
with aba_internacoes:
    mostrar_analise(
        dados_df=dados,
        pergunta="Pacientes que já tiveram internações anteriores são mais propensos a serem reinternados?",
        hipotese="Pacientes com histórico de internações anteriores apresentam maior risco de readmissão.",
        coluna_grupo="teve_internacao",
        eixo_x="Teve internação anterior?",
        titulo_grafico="Taxa de readmissão por histórico de internação",
        cores={"Nao": "#2563EB", "Sim": "#DC2626"}
    )

# 8. ABA URGÊNCIAS (Gabriel)
with aba_urgencias:
    mostrar_analise(
        dados_df=dados,
        pergunta="Pacientes que utilizaram mais vezes o pronto-socorro apresentam maior taxa de reinternação?",
        hipotese="Um maior número de atendimentos de urgencia está associado a maior risco de reinternação.",
        coluna_grupo="teve_urgencia",
        eixo_x="Teve urgência anterior?",
        titulo_grafico="Taxa de readmissão por histórico de urgência",
        cores={"Nao": "#059669", "Sim": "#D97706"}
    )

st.divider()



cores_customizadas = ["#ffaa00", "#7c8dbf", "#520052"] 
cmap_marilinda = mcolors.LinearSegmentedColormap.from_list("marilinda_dark", cores_customizadas)

#  9. ABA GLICEMIA (Marilinda s2)

with aba_glicemia:
    st.header("🔬 Cruzamento: Glicemia Média vs Reinternação")
    txt_tabela = analisar_glicemia_reinternacao(dados)
    pergunta = "O controle glicêmico influencia a taxa de reinternação? "
    hipotese = "Pacientes com resultados alterados de HbA1c possuem maior probabilidade de reinternação."
    st.subheader("Pergunta e hipótese")
    st.markdown(f"**Pergunta:** {pergunta}")
    st.markdown(f"**Hipótese:** {hipotese}")
    
    plt.style.use('dark_background')
    fig_m1, ax_m1 = plt.subplots(figsize=(7, 4))
    
    anotacoes_pct = txt_tabela.map(lambda x: f"{x:.1f}%").values
    
    sns.heatmap(
        txt_tabela, 
        annot=anotacoes_pct, 
        fmt="", 
        cmap=cmap_marilinda, 
        linewidths=1.5,
        linecolor="#111111",
        annot_kws={'color': '#ffffff', 'weight': 'bold', 'size': 11},
        cbar_kws={'label': 'Porcentagem (%)'}, 
        ax=ax_m1
    )
    
    ax_m1.set_title("Influência da Glicemia Média na Taxa de Reinternação", color='white', pad=15, weight='bold')
    ax_m1.set_xlabel("Paciente Reinternado?", color='white', labelpad=10)
    ax_m1.set_ylabel("Estimativa de Glicemia Média Diária", color='white', labelpad=10)
    
    plt.tight_layout()
    st.pyplot(fig_m1)
    plt.close(fig_m1)
    
    st.markdown("---")
    st.subheader("🔬 Entendendo os Números da HbA1c")
    df_conversao = calcular_dados_conversao_hba1c()
    st.table(df_conversao)
    
    st.write("Ao observar o gráfico vemos que a hipótese se confirma. O gráfico mostra que mais de 45% dos pacientes com glicemia média acima de 197 mg/dL (HbA1c > 8) foram readmitidos. Isso sugere que o controle glicêmico é um fator importante na probabilidade de reinternação hospitalar.")


 # 10. ABA CORRELAÇÃO (Marilinda s2)

with aba_correlacao:
    st.header("🧮 Matriz de Correlação Multivariada")
    matriz_corr = calcular_matriz_correlacao_multivariada(dados)
    pergunta = "Alterações na medicação para diabetes influenciam a ocorrência de reinternação? "
    hipotese = "Pacientes que tiveram mudanças na medicação apresentam comportamento diferente em relação à reinternação."
    st.subheader("Pergunta e hipótese")
    st.markdown(f"**Pergunta:** {pergunta}")
    st.markdown(f"**Hipótese:** {hipotese}")
    
    plt.style.use('dark_background')
    fig_m2, ax_m2 = plt.subplots(figsize=(9, 7))
   
    sns.heatmap(
        matriz_corr, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap_marilinda, 
        vmin=-1, 
        vmax=1, 
        center=0, 
        linewidths=1.5,
        linecolor="#111111",
        annot_kws={'color': '#ffffff', 'weight': 'bold', 'size': 10},
        ax=ax_m2
    )
    
    ax_m2.set_title("Matriz de Correlação Multivariada", color='white', pad=15, weight='bold')
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(color='white')
    
    plt.tight_layout()
    st.pyplot(fig_m2)
    plt.close(fig_m2)

    st.write("Ao observar o gráfico vemos que a hipótese não se confirma. A matriz de correlação mostra que a alteração na medicação tem uma correlação muito baixa com a readmição, sugerindo que mudanças na medicação para diabetes não influenciam significativamente a ocorrência de reinternação hospitalar.Mas também analisamos que um paciente que já possui uma rotina de internações recorrentes tem uma probabilidade significativamente maior de ser reinternado do que um paciente de primeira viagem, indicando uma possível cronicidade ou severidade da doença que o hospital não está conseguindo estabilizar a longo prazo.")

    