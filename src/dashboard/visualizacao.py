import matplotlib.pyplot as plt
import numpy as np
import io

def plot_radar(aluno, labels, valores, escala=5):
    labels_plot = labels + [labels[0]]
    valores_plot = valores + [valores[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(escala, escala * 0.8), subplot_kw=dict(polar=True))
    ax.plot(angles, valores_plot, linewidth=2, linestyle='solid', marker='o')
    ax.fill(angles, valores_plot, alpha=0.25)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['0', '1', '2', '3', '4'], fontsize=8)
    ax.set_ylim(0, 4)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels_plot, fontsize=9)
    ax.set_title(f'Radar das Pontuações – {aluno}', size=13, pad=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return fig, buf
