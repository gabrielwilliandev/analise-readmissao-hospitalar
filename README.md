# Analise de Readmissao Hospitalar

Aplicativo Streamlit para explorar uma base de risco de readmissao hospitalar em ate 30 dias.

## O que o app analisa

- Taxa geral de readmissao hospitalar.
- Readmissao por faixa etaria.
- Relacao entre risco socioeconomico, plano de saude e readmissao.
- Diagnostico primario com maior taxa de readmissao.
- Presenca e quantidade de comorbidades.

## Como executar

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Execute o app:

```bash
streamlit run app.py
```

## Arquivos principais

- `app.py`: aplicacao Streamlit.
- `hospital_readmission_risk_dataset.csv`: base de dados usada na analise.
