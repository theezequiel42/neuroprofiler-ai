import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Mapeamento das respostas objetivas
mapa_respostas = {
    "Nunca": 0,
    "Raramente": 1,
    "Às vezes": 2,
    "Frequentemente": 3,
    "Sempre": 4
}

def preparar_dados(df):
    """
    Converte respostas objetivas em valores numéricos e codifica os rótulos.
    """
    colunas_objetivas = ['resposta_1', 'resposta_2', 'resposta_3']

    # Converte as respostas usando o mapa
    for col in colunas_objetivas:
        df[col] = df[col].map(mapa_respostas)

    # Codifica o nível indicativo de AH/SD
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['nivel_indicativo_ahsd'])

    X = df[colunas_objetivas]
    y = df['label']
    return X, y, le

def treinar_modelo(X, y):
    """
    Treina um modelo RandomForestClassifier simples e imprime o relatório.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print("📊 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

    return modelo

def salvar_modelo(modelo, caminho="modelos/modelo_ahsd.pkl"):
    joblib.dump(modelo, caminho)
    print(f"✅ Modelo salvo em {caminho}")

def carregar_modelo(caminho="modelos/modelo_ahsd.pkl"):
    return joblib.load(caminho)

