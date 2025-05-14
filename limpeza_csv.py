import pandas as pd
from src.config.blocos_ahsd import blocos

# Carregar o arquivo original
df = pd.read_csv("data/respostas_completas.csv")

# Extrair todas as perguntas dos blocos (exceto 'Descritivo')
colunas_objetivas = []
for bloco, perguntas in blocos.items():
    if bloco != "Descritivo":
        colunas_existentes = [p for p in perguntas if p in df.columns]
        colunas_objetivas.extend(colunas_existentes)

# Adicione manualmente a coluna do nome do aluno e rótulo (se houver)
colunas_adicionais = [col for col in df.columns if "nome" in col.lower() and "aluno" in col.lower()]
if "nivel_indicativo_ahsd" in df.columns:
    colunas_adicionais.append("nivel_indicativo_ahsd")

# Gerar novo DataFrame com colunas úteis
df_limpo = df[colunas_adicionais + colunas_objetivas]

# Salvar para novo arquivo
df_limpo.to_csv("data/dados_filtrados.csv", index=False)
print("✅ Arquivo limpo salvo em: data/dados_filtrados.csv")
