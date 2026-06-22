import pandas as pd
import numpy as np

def carregar_e_preparar_dados_unificados(): 
    df = pd.read_csv("diabetic_data.csv")
    
    # Seleção de todas as colunas necessárias para ambas as partes
    colunas_foco = ["encounter_id", "number_inpatient", "number_emergency", "num_medications", "medical_specialty", "readmitted","gender","number_diagnoses"]
    dados = df[colunas_foco].copy()
    
    # --- Limpeza e Padronização ---
    dados['medical_specialty'] = dados['medical_specialty'].replace('?', np.nan)
    dados = dados.drop_duplicates(subset='encounter_id')
    dados['num_medications'] = pd.to_numeric(dados['num_medications'], errors='coerce')
    index=dados[dados['gender']=='Unknown/Invalid'].index
    dados=dados.drop(index)
    
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
