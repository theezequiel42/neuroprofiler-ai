# ✅ app.py (corrigido com col_nome_padronizado)
import streamlit as st
import pandas as pd
import joblib
import unicodedata

from src.preprocessing.normalizacao import normalizar_respostas, mapa_respostas
from src.dashboard.visualizacao import plot_radar
from src.models.classificador import carregar_modelo
from src.config.blocos_ahsd import blocos

def padronizar_coluna(texto):
    if not isinstance(texto, str):
        return texto
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    texto = texto.strip().lower()
    return texto

def run_dashboard():
    st.set_page_config(page_title="NeuroProfiler - AH/SD", layout="centered")
    st.title("🧠 NeuroProfiler - Triagem de Altas Habilidades/Superdotacao")

    arquivo = st.file_uploader("📂 Envie o arquivo de respostas (.csv)", type=["csv"])

    if arquivo:
        df = pd.read_csv(arquivo)
        colunas_originais = df.columns.tolist()

        # 🧠 Localiza coluna do nome original
        col_nome_original = None
        for c in colunas_originais:
            if "nome" in c.lower() and "aluno" in c.lower():
                col_nome_original = c
                break

        if not col_nome_original:
            st.error("❌ Nenhuma coluna com o nome do(a) aluno(a) foi encontrada.")
            return

        # Padroniza colunas do DataFrame
        col_nome_padronizado = padronizar_coluna(col_nome_original)
        df.columns = [padronizar_coluna(c) for c in df.columns]
        df_modelo = df.copy()

        # Carrega modelo treinado
        modelo = carregar_modelo()

        # Carrega colunas do treino
        try:
            X_treino_cols = pd.read_csv("modelos/X_treinamento_colunas.csv", nrows=0).columns.tolist()
        except FileNotFoundError:
            st.error("❌ Arquivo 'X_treinamento_colunas.csv' não encontrado.")
            return

        # Reconstrói o X com colunas padronizadas
        X_padronizado = pd.DataFrame()
        for col in X_treino_cols:
            col_pad = padronizar_coluna(col)
            if col_pad in df_modelo.columns:
                X_padronizado[col] = df_modelo[col_pad].map(mapa_respostas)
            else:
                X_padronizado[col] = 0

        try:
            y_pred = modelo.predict(X_padronizado)
            rotulos = modelo.classes_
            df["nivel_predito"] = [rotulos[i] for i in y_pred]
        except Exception as e:
            st.error(f"Erro ao fazer predição: {e}")
            return

        aluno = st.selectbox("👤 Selecione um aluno", df[col_nome_padronizado].unique())
        df_aluno = df[df[col_nome_padronizado] == aluno]
        nivel_predito = df_aluno['nivel_predito'].values[0]
        st.markdown(f"### 🔮 Nivel de Indicativo de AH/SD (modelo): **{nivel_predito}**")

        medias_blocos = []
        for bloco, perguntas in blocos.items():
            st.subheader(f"📚 {bloco}")
            perguntas_existentes = [p for p in perguntas if padronizar_coluna(p) in df.columns]
            if not perguntas_existentes:
                st.warning("⚠️ Nenhuma pergunta deste bloco foi encontrada.")
                continue

            respostas = df_aluno[[padronizar_coluna(p) for p in perguntas_existentes]].iloc[0]
            respostas_numericas = normalizar_respostas(respostas).map(mapa_respostas)
            media = pd.to_numeric(respostas_numericas, errors="coerce").mean()

            if pd.notna(media):
                medias_blocos.append(media)
                st.markdown(f"**Média:** `{media:.2f}`")
                st.progress(media / 4)
            else:
                st.warning("⚠️ Dados insuficientes ou inválidos.")

        if medias_blocos:
            media_geral = sum(medias_blocos) / len(medias_blocos)
            percentual = (media_geral / 4) * 100
            st.subheader("📈 Indicador Geral por Blocos")
            st.metric("Pontuação Geral", f"{percentual:.1f}%")
            st.progress(percentual / 100)

        st.subheader("📊 Radar por Bloco")
        escala = st.slider("Escala do gráfico", 3, 8, 5)
        labels = [bloco for bloco in blocos if bloco != "Descritivo"]
        valores = []

        for bloco in labels:
            perguntas_existentes = [p for p in blocos[bloco] if padronizar_coluna(p) in df.columns]
            respostas = df_aluno[[padronizar_coluna(p) for p in perguntas_existentes]].iloc[0]
            respostas_numericas = normalizar_respostas(respostas).map(mapa_respostas)
            media = pd.to_numeric(respostas_numericas, errors="coerce").mean()
            valores.append(media if pd.notna(media) else 0)

        fig, buf = plot_radar(aluno, labels, valores, escala)
        st.pyplot(fig)

        nome_arquivo = f"radar_{aluno.replace(' ', '_').lower()}.png"
        st.download_button("📥 Baixar gráfico como imagem", data=buf, file_name=nome_arquivo, mime="image/png")

    else:
        st.info("Envie um arquivo .csv com as respostas estruturadas conforme o modelo.")
