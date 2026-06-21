import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Analise de Readmissao Hospitalar",
    layout="wide",
)


@st.cache_data
def carregar_dados():
    return pd.read_csv("diabetic_data.csv")


def preparar_dados(df):
    dados = df[["number_inpatient", "number_emergency", "readmitted"]].copy()

    dados["readmitido"] = dados["readmitted"].apply(
        lambda valor: 0 if valor == "NO" else 1
    )

    dados["teve_internacao"] = dados["number_inpatient"].apply(
        lambda valor: "Sim" if valor > 0 else "Nao"
    )
    dados["teve_urgencia"] = dados["number_emergency"].apply(
        lambda valor: "Sim" if valor > 0 else "Nao"
    )

    return dados


def criar_tabela_resumo(dados, coluna_grupo):
    tabela = (
        dados.groupby(coluna_grupo, observed=False)
        .agg(
            total_pacientes=("readmitido", "count"),
            total_readmitidos=("readmitido", "sum"),
            taxa_readmissao=("readmitido", "mean"),
        )
        .reset_index()
    )

    tabela["taxa_readmissao_pct"] = tabela["taxa_readmissao"] * 100
    return tabela


def criar_grafico(tabela, coluna_grupo, eixo_x, titulo, cores):
    fig = px.bar(
        tabela,
        x=coluna_grupo,
        y="taxa_readmissao_pct",
        text=None,
        title=titulo,
        labels={
            coluna_grupo: eixo_x,
            "taxa_readmissao_pct": "Taxa de readmissao (%)",
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
        yaxis_title="Taxa de readmissao (%)",
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=13),
        margin=dict(l=20, r=20, t=70, b=20),
    )

    fig.update_yaxes(range=[0, 100])
    return fig


def mostrar_analise(
    dados,
    pergunta,
    hipotese,
    coluna_grupo,
    eixo_x,
    titulo_grafico,
    cores,
):
    st.subheader("Pergunta e hipotese")
    st.markdown(f"**Pergunta:** {pergunta}")
    st.markdown(f"**Hipotese:** {hipotese}")

    tabela = criar_tabela_resumo(dados, coluna_grupo)

    st.subheader("Tabela de resultados")
    st.dataframe(
        tabela.rename(
            columns={
                coluna_grupo: "grupo",
                "total_pacientes": "total de pacientes",
                "total_readmitidos": "total readmitidos",
                "taxa_readmissao_pct": "taxa de readmissao (%)",
            }
        )[
            [
                "grupo",
                "total de pacientes",
                "total readmitidos",
                "taxa de readmissao (%)",
            ]
        ].round(2),
        width="stretch",
    )

    st.subheader("Grafico interativo")
    st.caption("Passe o mouse sobre as barras para ver os valores exatos.")

    fig = criar_grafico(tabela, coluna_grupo, eixo_x, titulo_grafico, cores)
    st.plotly_chart(fig, width="stretch")

st.title("Analise de Readmissao Hospitalar")

st.markdown(
    """
    Esta analise investiga se o historico de internacoes e de atendimentos de
    urgencia esta associado a maior taxa de readmissao hospitalar.
    """
)

df = carregar_dados()
dados = preparar_dados(df)

taxa_geral = dados["readmitido"].mean() * 100
total_pacientes = len(dados)
total_readmitidos = dados["readmitido"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total de pacientes", f"{total_pacientes:,}".replace(",", "."))
col2.metric("Total readmitidos", f"{total_readmitidos:,}".replace(",", "."))
col3.metric("Taxa geral de readmissao", f"{taxa_geral:.2f}%")

with st.expander("Ver primeiras linhas da base"):
    st.dataframe(df.head(), width="stretch")

with st.expander("Ver dados usados nesta analise"):
    st.dataframe(dados.head(), width="stretch")

st.divider()

aba_internacoes, aba_urgencias = st.tabs(
    ["7. Internacoes anteriores", "8. Urgencias anteriores"]
)

with aba_internacoes:
    mostrar_analise(
        dados=dados,
        pergunta=(
            "Pacientes que ja tiveram internacoes anteriores sao mais propensos "
            "a serem reinternados?"
        ),
        hipotese=(
            "Pacientes com historico de internacoes anteriores apresentam maior "
            "risco de readmissao."
        ),
        coluna_grupo="teve_internacao",
        eixo_x="Teve internacao anterior?",
        titulo_grafico="Taxa de readmissao por historico de internacao",
        cores={"Nao": "#2563EB", "Sim": "#DC2626"},
    )

with aba_urgencias:
    mostrar_analise(
        dados=dados,
        pergunta=(
            "Pacientes que utilizaram mais vezes o pronto-socorro apresentam "
            "maior taxa de reinternacao?"
        ),
        hipotese=(
            "Um maior numero de atendimentos de urgencia esta associado a maior "
            "risco de reinternacao."
        ),
        coluna_grupo="teve_urgencia",
        eixo_x="Teve urgencia anterior?",
        titulo_grafico="Taxa de readmissao por historico de urgencia",
        cores={"Nao": "#059669", "Sim": "#D97706"},
    )
