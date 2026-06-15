import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_FILE = 'hospital_readmission_risk_dataset.csv'


def carregar_dados():
    df = pd.read_csv(DATA_FILE)
    df.drop_duplicates(inplace=True)
    return df


def criar_tabela_taxa(df, coluna_grupo):
    tabela = (
        df.groupby(coluna_grupo, observed=False)['Readmitted_Within_30_Days']
        .agg(['count', 'sum', 'mean'])
    )

    tabela.rename(columns={
        'count': 'Total de pacientes',
        'sum': 'Total reinternados',
        'mean': 'Taxa de reinternação'
    }, inplace=True)

    tabela['Taxa de reinternação (%)'] = tabela['Taxa de reinternação'] * 100

    return tabela


def grafico_barras_detalhado(tabela, titulo, cor='steelblue'):
    coluna_taxa = 'Taxa de reinternação (%)'

    fig, ax = plt.subplots(figsize=(9, 5))

    barras = ax.bar(
        tabela.index.astype(str),
        tabela[coluna_taxa],
        color=cor
    )

    media_geral = df['Readmitted_Within_30_Days'].mean() * 100
    ax.axhline(
        media_geral,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'Média geral: {media_geral:.2f}%'
    )

    menor = tabela[coluna_taxa].min()
    maior = tabela[coluna_taxa].max()

    ax.set_ylim(menor - 2, maior + 2)

    ax.set_title(titulo)
    ax.set_ylabel('Taxa de reinternação (%)')
    ax.set_xlabel('')
    ax.legend()

    for barra in barras:
        altura = barra.get_height()
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            altura,
            f'{altura:.2f}%',
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.xticks(rotation=25)
    st.pyplot(fig)

    st.caption(
        "Observação: o eixo Y foi aproximado para destacar diferenças pequenas entre os grupos."
    )


st.title("Análise de Dados de Readmissão Hospitalar")

st.write(
    "Este aplicativo analisa os dados de readmissão hospitalar para identificar padrões "
    "relacionados à idade, risco socioeconômico, plano de saúde, diagnóstico primário "
    "e comorbidades."
)

df = carregar_dados()

st.subheader("Visualização inicial da base")
st.dataframe(df.head(10))

taxa_geral = df['Readmitted_Within_30_Days'].mean() * 100
st.metric("Taxa geral de reinternação em até 30 dias", f"{taxa_geral:.2f}%")

st.markdown("""
### Perguntas e hipóteses do estudo

**Pergunta 1:** Tem alguma faixa etária com maior taxa de reinternação?  
**Hipótese 1:** Pessoas de 46 a 60 anos e 60+ apresentam maior taxa de reinternação.

**Pergunta 2:** Quais diagnósticos primários possuem maior taxa de reinternação?

**Hipótese 2:** Pacientes com ao menos uma comorbidade são mais propensos à reinternação em até 30 dias.

**Pergunta 3:** Quanto maior o número de comorbidades, maior é a chance de reinternação?

**Pergunta 4:** A taxa de reinternação varia conforme risco socioeconômico e tipo de plano de saúde?
""")

df['Faixa_Etaria'] = pd.cut(
    df['Age'],
    bins=[19, 45, 60, 100],
    labels=['20-45', '46-60', '60+']
)

df['Nivel_Socioeconomico'] = pd.cut(
    df['Socioeconomic_Risk_Score'],
    bins=[0, 3, 6, 9],
    labels=['1-3: Baixo risco', '4-6: Médio risco', '7-9: Alto risco']
)

df['Tem_Comorbidade'] = np.where(
    df['Comorbidity_Index'] >= 1,
    'Com ao menos uma comorbidade',
    'Sem comorbidades'
)

st.subheader("1. Faixa Etária x Reinternação")

st.write("""
Nesta análise, os pacientes foram separados em três grupos etários: 20-45, 46-60 e 60+.
O objetivo é verificar se os grupos mais velhos apresentam maior taxa de reinternação.
""")

taxa_idade = criar_tabela_taxa(df, 'Faixa_Etaria')
taxa_idade = taxa_idade.sort_values('Taxa de reinternação (%)', ascending=False)

st.dataframe(taxa_idade.round(2))
grafico_barras_detalhado(
    taxa_idade,
    "Taxa de reinternação por faixa etária",
    cor='cornflowerblue'
)

maior_faixa = taxa_idade.index[0]
maior_taxa_faixa = taxa_idade.iloc[0]['Taxa de reinternação (%)']

st.write(
    f"**Resultado:** a faixa etária com maior taxa de reinternação foi "
    f"**{maior_faixa}**, com **{maior_taxa_faixa:.2f}%**."
)

st.subheader("2. Risco Socioeconômico e Plano de Saúde")

st.write("""
Aqui cruzamos o nível de risco socioeconômico com o tipo de plano de saúde.
Como a coluna se chama `Socioeconomic_Risk_Score`, a interpretação mais adequada é:
quanto maior a pontuação, maior o risco socioeconômico.
""")

taxa_socio_plano = (
    df.groupby(['Nivel_Socioeconomico', 'Insurance_Type'], observed=False)
    ['Readmitted_Within_30_Days']
    .mean() * 100
)

tabela_socio_plano = taxa_socio_plano.unstack()

st.write("Taxa de reinternação (%) por risco socioeconômico e plano de saúde:")
st.dataframe(tabela_socio_plano.round(2))

fig, ax = plt.subplots(figsize=(10, 5))
tabela_socio_plano.plot(kind='bar', ax=ax)

menor = tabela_socio_plano.min().min()
maior = tabela_socio_plano.max().max()

ax.set_ylim(menor - 2, maior + 2)
ax.axhline(taxa_geral, color='red', linestyle='--', label=f'Média geral: {taxa_geral:.2f}%')

ax.set_title("Taxa de reinternação por risco socioeconômico e plano")
ax.set_ylabel("Taxa de reinternação (%)")
ax.set_xlabel("")
ax.legend()
plt.xticks(rotation=20)

st.pyplot(fig)
st.caption("O eixo Y foi aproximado porque as diferenças entre os grupos são pequenas.")

st.subheader("3. Diagnóstico Primário x Reinternação")

st.write("""
Esta análise responde quais grupos de diagnóstico primário possuem maior taxa de reinternação.
A base trabalha com grupos de diagnóstico, como infecção, cardíaco, respiratório, diabetes e outros.
""")

taxa_diagnostico = criar_tabela_taxa(df, 'Primary_Diagnosis_Group')
taxa_diagnostico = taxa_diagnostico.sort_values('Taxa de reinternação (%)', ascending=False)

st.dataframe(taxa_diagnostico.round(2))
grafico_barras_detalhado(
    taxa_diagnostico,
    "Taxa de reinternação por diagnóstico primário",
    cor='seagreen'
)

maior_diagnostico = taxa_diagnostico.index[0]
maior_taxa_diagnostico = taxa_diagnostico.iloc[0]['Taxa de reinternação (%)']

st.write(
    f"**Resultado:** o diagnóstico primário com maior taxa de reinternação foi "
    f"**{maior_diagnostico}**, com **{maior_taxa_diagnostico:.2f}%**."
)

st.subheader("4. Comorbidades x Reinternação")

st.write("""
Esta análise compara pacientes sem comorbidades com pacientes que possuem ao menos uma comorbidade.
A hipótese é que pacientes com comorbidades apresentam maior chance de reinternação.
""")

taxa_comorbidade = criar_tabela_taxa(df, 'Tem_Comorbidade')
taxa_comorbidade = taxa_comorbidade.sort_values('Taxa de reinternação (%)', ascending=False)

st.dataframe(taxa_comorbidade.round(2))
grafico_barras_detalhado(
    taxa_comorbidade,
    "Taxa de reinternação por presença de comorbidade",
    cor='darkorange'
)

st.subheader("5. Quantidade de Comorbidades x Reinternação")

st.write("""
Nesta parte, analisamos se a taxa de reinternação aumenta conforme o índice de comorbidade cresce.
Se houver tendência de aumento, a hipótese de relação positiva entre comorbidades e reinternação ganha força.
""")

taxa_indice_comorbidade = criar_tabela_taxa(df, 'Comorbidity_Index')
taxa_indice_comorbidade = taxa_indice_comorbidade.sort_index()

st.dataframe(taxa_indice_comorbidade.round(2))

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    taxa_indice_comorbidade.index,
    taxa_indice_comorbidade['Taxa de reinternação (%)'],
    marker='o',
    linewidth=2,
    color='purple'
)

ax.axhline(
    taxa_geral,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label=f'Média geral: {taxa_geral:.2f}%'
)

menor = taxa_indice_comorbidade['Taxa de reinternação (%)'].min()
maior = taxa_indice_comorbidade['Taxa de reinternação (%)'].max()

ax.set_ylim(menor - 2, maior + 2)

for x, y in zip(
    taxa_indice_comorbidade.index,
    taxa_indice_comorbidade['Taxa de reinternação (%)']
):
    ax.text(x, y, f'{y:.2f}%', ha='center', va='bottom', fontsize=9)

ax.set_title("Taxa de reinternação por índice de comorbidade")
ax.set_xlabel("Índice de comorbidade")
ax.set_ylabel("Taxa de reinternação (%)")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
st.caption("O gráfico usa linha com marcadores para facilitar a visualização da tendência.")
