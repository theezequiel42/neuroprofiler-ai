import pandas as pd

def carregar_dados(caminho):
    df = pd.read_csv(caminho)
    return df

if __name__ == "__main__":
    df = carregar_dados("data/dados_simulados.csv")
    print("🔍 Dados carregados:")
    print(df.head())
