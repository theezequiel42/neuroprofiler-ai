import unicodedata

# Mapeamento de respostas objetivas
mapa_respostas = {
    "Nunca": 0,
    "Raramente": 1,
    "As vezes": 2,
    "Às vezes": 2,
    "Frequentemente": 3,
    "Sempre": 4
}

# Mapeamento para perguntas descritivas
mapa_diagnostico = {
    "Sim": 4,
    "Não": 0,
    "Altas": 4,
    "Alta": 4,
    "Média": 2,
    "Medias": 2,
    "Médias": 2,
    "Baixa": 0,
    "Baixas": 0
}

def normalizar_respostas(respostas):
    return respostas.astype(str).str.strip().apply(
        lambda x: unicodedata.normalize("NFKD", x).encode("ASCII", "ignore").decode("utf-8")
    )
