import pandas as pd
import numpy as np

def carregar_e_preparar_dados_unificados(): 
    df = pd.read_csv("diabetic_data.csv")
    
    # Seleção de todas as colunas necessárias para ambas as partes

    colunas_foco = [
    "encounter_id", 
    "readmitted", 
    "change", 
    "diabetesMed", 
    "num_medications", 
    "time_in_hospital", 
    "num_procedures",
    "number_diagnoses", 
    "number_inpatient", 
    "number_emergency", 
    "number_outpatient", 
    "medical_specialty", 
    "A1Cresult",
    "gender",
    "age",
    "race"
]

    dados = df[colunas_foco].copy()
    
    # --- Limpeza e Padronização ---
    dados['medical_specialty'] = dados['medical_specialty'].replace('?', np.nan)
    dados = dados.drop_duplicates(subset='encounter_id')
    dados['num_medications'] = pd.to_numeric(dados['num_medications'], errors='coerce')
    index=dados[dados['gender']=='Unknown/Invalid'].index
    dados=dados.drop(index)
    
    #Marilinda
    dados['race'] = dados['race'].replace('?', np.nan)
    dados['age'] = dados['age'].replace('?', np.nan)
    dados = dados.dropna(subset=['age']) 

    mapeamento_racas = {
        "Caucasian": "Caucasiano",
        "AfricanAmerican": "Afro-americano",
        "Hispanic": "Hispânico",
        "Asian": "Asiático",
        "Other": "Outros"
    }
    dados['race'] = dados['race'].map(mapeamento_racas).fillna("Não Informado")

    # --- Engenharia de Recursos (Parte Gabriel) ---
    dados["readmitido"] = dados["readmitted"].apply(
        lambda valor: 0 if valor == "NO" else 1
    )
    dados["teve_internacao"] = dados["number_inpatient"].apply(
        lambda valor: "Sim" if valor > 0 else "Nao"
    )
    dados["teve_urgencia"] = dados["number_emergency"].apply(
        lambda valor: "Sim" if valor > 0 else "Nao"
    )
    
    # --- Engenharia de Recursos (Parte Lucas) ---
    bins = [0, 5, 10, 15, 20, 30, 100]
    labels = ['1-5', '6-10', '11-15', '16-20', '21-30', '31+']
    dados['faixa_medicamentos'] = pd.cut(
        dados['num_medications'], bins=bins, labels=labels, right=True
    )
    dados['readmit_30d'] = (dados['readmitted'] == '<30').astype(int)
    
    # --- Engenharia de Recursos (Parte Lavínia) ---
    dados["genero"] = dados["gender"].apply(
        lambda valor: "Feminino" if valor == "Female" else "Masculino"
    )
    
    bins = [0, 5, 10, 15]
    qtde_diag = ['1-5', '6-10', '11-15']
    
    dados['quantidade_diagnostico'] = pd.cut(dados['number_diagnoses'], bins=bins, labels=qtde_diag)
    
    # --- Engenharia de Recursos (Parte Bianca) ---
    #print(dados.columns.tolist())
    #print(dados[['age']].head())
    # Agrupamento das faixas etárias
    # --- Engenharia de Recursos (Parte Bianca) ---

# Agrupamento das faixas etárias
    dados['faixa_etaria'] = dados['age'].replace({
        '[0-10)': 'Até 30 anos',
        '[10-20)': 'Até 30 anos',
        '[20-30)': 'Até 30 anos',

        '[30-40)': '31-60 anos',
        '[40-50)': '31-60 anos',
        '[50-60)': '31-60 anos',

        '[60-70)': 'Acima de 60 anos',
        '[70-80)': 'Acima de 60 anos',
        '[80-90)': 'Acima de 60 anos',
        '[90-100)': 'Acima de 60 anos'
    })

    # Agrupamento do tempo de internação
    dados['faixa_internacao'] = pd.cut(
        dados['time_in_hospital'],
        bins=[0, 3, 7, 14],
        labels=[
            '1-3 dias',
            '4-7 dias',
            '8-14 dias'
        ]
    )

    # Verificação
    print(dados.columns.tolist())
    print(dados[['age', 'faixa_etaria']].head())
    
    return df, dados

def top_especialidades(df, n=10):
    """Retorna as N especialidades com mais registros (excluindo valores ausentes)."""
    return (
        df['medical_specialty']
        .dropna()
        .value_counts()
        .head(n)
        .index
        .tolist()
    )
#Parte Marilinda
def analisar_glicemia_reinternacao(df):
    """Realiza o cruzamento estatístico entre Glicemia e Reinternação."""
    df_local = df.copy()
    
    col_a1c = 'A1cresult' if 'A1cresult' in df_local.columns else 'A1Cresult'
    
    df_local[col_a1c] = df_local[col_a1c].astype(str).str.strip()
    df_local['readmitted'] = df_local['readmitted'].astype(str).str.strip()
    
    txt_tabela = pd.crosstab(df_local[col_a1c], df_local['readmitted'], normalize='index') * 100
    
    mapeamento_glicemia = {
        "Norm": "Norm (~111 mg/dL)",
        ">7": ">7 (~169 mg/dL)",
        ">8": ">8 (~197 mg/dL)"
    }
    txt_tabela = txt_tabela.rename(index=mapeamento_glicemia)
    
    if all(col in txt_tabela.columns for col in ['<30', '>30', 'NO']):
        txt_tabela = txt_tabela[['<30', '>30', 'NO']]
        
    return txt_tabela

def calcular_dados_conversao_hba1c():
    """Gera a tabela analítica de conversão baseada na fórmula da HbA1c."""
    dados_conversao = {
        "Grupo": ["Norm", ">7", ">8"],
        "HbA1c Estimada (%)": [5.5, 7.5, 8.5],
        "Glicemia Média Diária (mg/dL)": [
            round((5.5 * 28.7) - 46.7),
            round((7.5 * 28.7) - 46.7),
            round((8.5 * 28.7) - 46.7)
        ],
        "Status Clínico": ["Glicemia Controlada", "Início de Descontrole", "Descontrole Clínico Elevado"]
    }
    return pd.DataFrame(dados_conversao)

def calcular_matriz_correlacao_multivariada(df):
    """Trata as variáveis qualitativas em numéricas e calcula a matriz de correlação atualizada."""
    df_corr = df.copy()

    for col in ['readmitted', 'change', 'diabetesMed']:
        if col in df_corr.columns:
            df_corr[col] = df_corr[col].astype(str).str.strip()
    
    df_corr['readmitted_num'] = df_corr['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})
    df_corr['change_num'] = df_corr['change'].map({'No': 0, 'Ch': 1})
    df_corr['diabetesMed_num'] = df_corr['diabetesMed'].map({'No': 0, 'Yes': 1})
    
    colunas_numericas_alvo = [
        'num_medications', 'time_in_hospital', 'num_procedures',
        'number_diagnoses', 'number_emergency', 'number_inpatient', 'number_outpatient'
    ]
    
    for col in colunas_numericas_alvo:
        if col in df_corr.columns:
            df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
        else:
            df_corr[col] = np.nan
            
    colunas_correlacao = [
        'readmitted_num', 'change_num', 'diabetesMed_num', 
        'num_medications', 'time_in_hospital', 'num_procedures',
        'number_diagnoses', 'number_emergency', 'number_inpatient', 'number_outpatient'
    ]
    
    matriz_corr = df_corr[colunas_correlacao].corr()
    
    nomes_exibicao = {
        'readmitted_num': 'Reinternação',
        'change_num': 'Mudança Medicação',
        'diabetesMed_num': 'Remédio Diab.',
        'num_medications': 'Qtd. Medicamentos',
        'time_in_hospital': 'Dias de Internação',
        'num_procedures': 'Qtd. Procedimentos',
        'number_diagnoses': 'Total Diagnósticos',
        'number_emergency': 'Idas Emergência',
        'number_inpatient': 'Internações Ant.',
        'number_outpatient': 'Cons. Ambulatório'
    }
    
    return matriz_corr.rename(index=nomes_exibicao, columns=nomes_exibicao)

