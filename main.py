# ✅ main.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
import joblib

from src.config.blocos_ahsd import blocos
from src.preprocessing.normalizacao import mapa_respostas

CAMINHO_DADOS_TREINAMENTO = "data/dados_treinamento.csv"
CAMINHO_MODELO = "modelos/modelo_ahsd.pkl"
CAMINHO_COLUNAS = "modelos/X_treinamento_colunas.csv"

# 1. Carrega os dados
df = pd.read_csv(CAMINHO_DADOS_TREINAMENTO)

# 2. Detecta colunas objetivas
colunas_objetivas = []
for bloco, perguntas in blocos.items():
    if bloco != "Descritivo":
        colunas_existentes = [p for p in perguntas if p in df.columns]
        colunas_objetivas.extend(colunas_existentes)

if not colunas_objetivas:
    raise ValueError("❌ Nenhuma pergunta objetiva encontrada no dataset!")

# 3. Mapeia as respostas
df_modelo = df.copy()
for col in colunas_objetivas:
    df_modelo[col] = df_modelo[col].map(mapa_respostas)

X = df_modelo[colunas_objetivas]
y_texto = df_modelo["nivel_indicativo_ahsd"]

# 4. Codifica os rótulos
le = LabelEncoder()
y = le.fit_transform(y_texto)

# 5. Divide com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 6. Treina o modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 7. Avalia
y_pred = modelo.predict(X_test)
classes_presentes = unique_labels(y_test, y_pred)
nomes_presentes = le.inverse_transform(classes_presentes)
print("\U0001F4CA Relatório de Classificação:")
print(classification_report(y_test, y_pred, labels=classes_presentes, target_names=nomes_presentes))

# 8. Salva modelo e colunas
joblib.dump(modelo, CAMINHO_MODELO)
X.to_csv(CAMINHO_COLUNAS, index=False)
print(f"\u2705 Modelo salvo em: {CAMINHO_MODELO}")
print(f"\u2705 Colunas salvas em: {CAMINHO_COLUNAS}")
