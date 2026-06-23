import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from analise_arquivo import carregar_e_preparar_dados_unificados, top_especialidades

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
aba_medicamentos, aba_especialidades,aba_genero,aba_diagnosticos, aba_internacoes, aba_urgencias = st.tabs([
    "3. Medicamentos (H3)",
    "4. Especialidades Médicas (H4)",
    "5. Internação por gênero",
    "6. Diagnósticos anteriores",
    "7. Internações anteriores", 
    "8. Urgências anteriores"
])

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
    
    fig4a.update_layout(height=600)  # aumenta a altura
    st.plotly_chart(fig4a, use_container_width=True)

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
    st.info(f"Taxa de sobrevivência Feminina: **{taxa_sexo['Feminino']:.1f}%** | Masculina: **{taxa_sexo['Masculino']:.1f}%**\n\n Ao verificar as taxas de reintenação de homens e mulheres, nota-se uma variação pequena, confirmando a hipótese de que há variação, mas não é uma diferença significativa.")
    


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

