import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Caminho de saída gerado pelo seu script de geração de respostas
OUTPUT_BASE = "analyze-llm-code-efficiency/data"

def get_all_modelos_linguagens(base_path):
    modelos = []
    linguagens = set()
    for modelo in os.listdir(base_path):
        path = os.path.join(base_path, modelo)
        if os.path.isdir(path):
            modelos.append(modelo)
            for lang in os.listdir(path):
                lang_path = os.path.join(path, lang)
                if os.path.isdir(lang_path):
                    linguagens.add(lang)
    return sorted(modelos), sorted(list(linguagens))

def load_resultados_ok(base_path, modelos, linguagens):
    """Conta apenas as respostas OK de cada modelo por linguagem."""
    resultados = defaultdict(lambda: defaultdict(int))
    totais = defaultdict(lambda: defaultdict(int))
    for modelo in tqdm(modelos, desc="Modelos"):
        for linguagem in linguagens:
            pasta = os.path.join(base_path, modelo, linguagem)
            if not os.path.isdir(pasta):
                continue
            for fname in os.listdir(pasta):
                if fname.endswith(".json"):
                    path = os.path.join(pasta, fname)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            llm_resp = data.get("llm_response", {})
                            totais[modelo][linguagem] += 1
                            if llm_resp.get("valid", False):
                                resultados[modelo][linguagem] += 1
                    except Exception as e:
                        print(f"Erro lendo {path}: {e}")
    return resultados, totais

def plot_bar_ok(resultados_ok, totais, modelos, linguagens, export_path=None):
    linguagens = sorted(linguagens)
    n_models = len(modelos)
    bar_width = 0.15
    x_gap = 0.42
    x = np.arange(len(linguagens)) * (n_models * bar_width + x_gap)

    cmap = plt.get_cmap("tab10") if n_models <= 10 else plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(n_models)]

    fig, ax = plt.subplots(figsize=(2 + len(linguagens)*2.6, 7))

    for i, modelo in enumerate(modelos):
        ok_vals = [resultados_ok[modelo][ling] for ling in linguagens]
        tot_vals = [totais[modelo][ling] for ling in linguagens]
        pos = x + i * bar_width
        bars = ax.bar(pos, ok_vals, width=bar_width, color=colors[i], label=modelo)
        # Anota valor absoluto e taxa de acerto no topo de cada barra
        for j, bar in enumerate(bars):
            ok = ok_vals[j]
            total = tot_vals[j]
            taxa = 100.0 * ok / total if total > 0 else 0.0
            ax.text(bar.get_x() + bar.get_width()/2, ok + 0.8,
                    f"{ok}\n({taxa:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Centraliza o nome das linguagens
    centers = x + (n_models-1)*bar_width/2
    ax.set_xticks(centers)
    ax.set_xticklabels(linguagens, fontsize=13)
    ax.set_ylabel("Quantidade de respostas OK", fontsize=14)
    ax.set_title("Respostas corretas por modelo e linguagem", fontsize=16)

    ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1,1))
    plt.tight_layout()
    if export_path:
        plt.savefig(export_path)
    plt.show()

def main():
    modelos, linguagens = get_all_modelos_linguagens(OUTPUT_BASE)
    print("Modelos encontrados:", modelos)
    print("Linguagens encontradas:", linguagens)
    resultados_ok, totais = load_resultados_ok(OUTPUT_BASE, modelos, linguagens)
    plot_bar_ok(resultados_ok, totais, modelos, linguagens)

if __name__ == "__main__":
    main()
