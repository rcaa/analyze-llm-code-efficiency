import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from matplotlib.patches import Patch

# === CONFIGURAÇÕES ===
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "../data"))
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

MODELOS_OFICIAIS = [
    "allam-2-7b", "deepseek-r1-distill-llama-70b", "gemma2-9b-it",
    "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "llama3-70b-8192", "llama3-8b-8192",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "moonshotai/kimi-k2-instruct",
    "qwen/qwen3-32b"
]
LINGUAGENS = ["C++", "Java", "Python3"]
TOP_N_REASONS = 5

# Mapeia nome de pasta (onde "/" virou "_") para ID oficial
MODEL_DIR_TO_ID = {m.replace("/", "_"): m for m in MODELOS_OFICIAIS}

# === CARREGAR DADOS DIRETO DA PASTA data/ ===
def load_data_from_files(data_dir):
    """
    Itera por cada modelo e linguagem na pasta data/,
    Carrega cada JSON e agrupa em:
      data[qid][model_id][lang] = info_dict
    """
    data = defaultdict(lambda: defaultdict(dict))

    for dir_name in os.listdir(data_dir):
        model_path = os.path.join(data_dir, dir_name)
        if not os.path.isdir(model_path):
            continue
        model_id = MODEL_DIR_TO_ID.get(dir_name)
        if model_id not in MODELOS_OFICIAIS:
            continue

        for lang in LINGUAGENS:
            lang_path = os.path.join(model_path, lang)
            if not os.path.isdir(lang_path):
                continue
            for fname in os.listdir(lang_path):
                if not fname.endswith(".json"):
                    continue
                qid = os.path.splitext(fname)[0]
                try:
                    with open(os.path.join(lang_path, fname), encoding="utf-8") as f:
                        info = json.load(f)
                    data[qid][model_id][lang] = info
                except Exception:
                    # pular arquivos inválidos
                    continue
    return data

# === AGREGAR MÉTRICAS DE ACURÁCIA ===
def compute_metrics(data):
    acc = defaultdict(lambda: defaultdict(lambda: [0,0]))
    for qid, modelos in data.items():
        for modelo, langs in modelos.items():
            for lang, info in langs.items():
                is_correct = 1 if info.get('categoria') == 'correct' else 0
                acc[modelo][lang][0] += is_correct
                acc[modelo][lang][1] += 1
    models = [m for m in MODELOS_OFICIAIS if any(acc[m][l][1] > 0 for l in LINGUAGENS)]
    heat = np.zeros((len(models), len(LINGUAGENS)))
    for i, m in enumerate(models):
        for j, l in enumerate(LINGUAGENS):
            c, t = acc[m][l]
            heat[i, j] = c / t if t > 0 else np.nan
    return models, heat

# === AGREGAR CLASSIFICAÇÕES E MOTIVOS ===
def aggregate_classification(data):
    contagem = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    motivos_lang = defaultdict(Counter)
    motivos_model = defaultdict(Counter)
    for modelos in data.values():
        for modelo, langs in modelos.items():
            for lang, info in langs.items():
                cat = info.get('categoria', 'unknown')
                mot = info.get('motivo', '-')
                contagem[modelo][lang][cat] += 1
                motivos_lang[lang][mot] += 1
                motivos_model[modelo][mot] += 1
    return contagem, motivos_lang, motivos_model

# === PLOT FUNCTIONS ===
def plot_accuracy_heatmap(models, heat, export_path=None):
    fig, ax = plt.subplots(figsize=(8, max(4, len(models)*0.4)))
    im = ax.imshow(heat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(LINGUAGENS)))
    ax.set_xticklabels(LINGUAGENS)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([m.split('/')[-1] for m in models])
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(len(models)):
        for j in range(len(LINGUAGENS)):
            val = heat[i, j]
            label = f"{val*100:.1f}%" if not np.isnan(val) else '-'
            ax.text(j, i, label, ha='center', va='center',
                    color='white' if not np.isnan(val) and val >= 0.5 else 'black')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Accuracy')
    ax.set_title('Heatmap de Acurácia por Modelo e Linguagem')
    plt.tight_layout()
    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        fig.savefig(export_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_distribution(data, export_path=None):
    categorias = ['correct', 'plausible', 'invalid']
    colors = {'correct': '#2ca02c', 'plausible': '#ff7f0e', 'invalid': '#d62728'}
    dist = defaultdict(lambda: defaultdict(int))
    for modelos in data.values():
        for modelo, langs in modelos.items():
            for info in langs.values():
                dist[modelo][info.get('categoria','unknown')] += 1
    models = [m for m in MODELOS_OFICIAIS if sum(dist[m].values()) > 0]
    x = np.arange(len(models))
    bottom = np.zeros(len(models))
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat in categorias:
        vals = [dist[m].get(cat, 0) for m in models]
        ax.bar(x, vals, bottom=bottom, color=colors[cat], label=cat.title())
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([m.split('/')[-1] for m in models], rotation=45, ha='right')
    ax.set_ylabel('Total de Respostas')
    ax.set_title('Distribuição de Classificações por Modelo')
    ax.legend(title='Categoria')
    plt.tight_layout()
    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        fig.savefig(export_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_reason_chart(motivos_lang, export_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_reasons = set()
    top_by_lang = {}
    for lang in LINGUAGENS:
        ctr = motivos_lang.get(lang, Counter())
        top = ctr.most_common(TOP_N_REASONS)
        top_by_lang[lang] = dict(top)
        all_reasons.update(dict(top))
    all_reasons = sorted(all_reasons, key=lambda r: sum(top_by_lang[l].get(r,0) for l in LINGUAGENS), reverse=True)
    y = np.arange(len(all_reasons))
    width = 0.8 / len(LINGUAGENS)
    offsets = np.linspace(-0.4+width/2, 0.4-width/2, len(LINGUAGENS))
    for idx, lang in enumerate(LINGUAGENS):
        vals = [top_by_lang[lang].get(r,0) for r in all_reasons]
        ax.barh(y+offsets[idx], vals, height=width, label=lang)
        for i, v in enumerate(vals):
            if v>0:
                ax.text(v+0.3, y[i]+offsets[idx], str(v), va='center', fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(all_reasons)
    ax.invert_yaxis()
    ax.set_xlabel('Frequência')
    ax.set_title(f'Classificações por Linguagem')
    ax.legend(title='Linguagem', loc='lower center', bbox_to_anchor=(0.5,-0.2), ncol=len(LINGUAGENS))
    plt.tight_layout(rect=[0,0.05,1,1])
    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        fig.savefig(export_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_reason_by_model(motivos_model, export_path=None):
    models = [m for m in MODELOS_OFICIAIS]
    all_reasons = set()
    top_by_model = {}
    for m in models:
        ctr = motivos_model.get(m, Counter())
        top = ctr.most_common(TOP_N_REASONS)
        top_by_model[m] = dict(top)
        all_reasons.update(dict(top))
    all_reasons = sorted(all_reasons, key=lambda r: sum(top_by_model[m].get(r,0) for m in models), reverse=True)
    x = np.arange(len(models))
    width = 0.8 / len(all_reasons) if all_reasons else 0.8
    fig, ax = plt.subplots(figsize=(max(8,len(models)*0.4), len(all_reasons)*0.4+2))
    for idx, reason in enumerate(all_reasons):
        vals = [top_by_model[m].get(reason,0) for m in models]
        ax.bar(x + (idx - len(all_reasons)/2)*width, vals, width, label=reason)
        for xi,v in enumerate(vals):
            if v>0:
                ax.text(xi + (idx - len(all_reasons)/2)*width, v+0.2, str(v), ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split('/')[-1] for m in models], rotation=45, ha='right')
    ax.set_ylabel('Frequência')
    ax.set_title(f'Classificações por Modelo')
    ax.legend(title='Motivo', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        fig.savefig(export_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_classification_by_lang(contagem, export_path=None):
    modelos = [m for m in MODELOS_OFICIAIS if any(contagem[m][l] for l in LINGUAGENS)]
    categorias = ["correct","plausible","invalid"]
    colors = {"correct":"#2ca02c","plausible":"#ff7f0e","invalid":"#d62728","unknown":"#888888"}
    x = np.arange(len(modelos))
    fig, axes = plt.subplots(1,len(LINGUAGENS), figsize=(6*len(LINGUAGENS),6), sharey=True)
    if len(LINGUAGENS)==1: axes=[axes]
    for ax, lang in zip(axes, LINGUAGENS):
        bottom = np.zeros(len(modelos))
        ax.set_title(f"{lang}", fontsize=14)
        for cat in categorias:
            vals = np.array([contagem[m][lang].get(cat,0) for m in modelos])
            ax.bar(x, vals, bottom=bottom, color=colors[cat], label=cat.title())
            bottom+=vals
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1] for m in modelos], rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].set_ylabel("Quantidade de Respostas")
    handles=[Patch(color=colors[c],label=c.title()) for c in categorias]
    fig.legend(handles=handles,loc='upper center',ncol=len(categorias),bbox_to_anchor=(0.5,1.05),title="Categoria")
    fig.suptitle("Classificação por Modelo e Linguagem", y=1.12)
    plt.tight_layout()
    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.gcf().savefig(export_path, dpi=300, bbox_inches='tight')
    plt.show()

# === MAIN ===
if __name__ == '__main__':
    data = load_data_from_files(DATA_DIR)

    # accuracy heatmap
    models, heat = compute_metrics(data)
    plot_accuracy_heatmap(models, heat, export_path=os.path.join(PLOTS_DIR,'accuracy_heatmap.png'))

    # distribution by model
    plot_model_distribution(data, export_path=os.path.join(PLOTS_DIR,'distribution_by_model.png'))

    # classification & reasons
    contagem, motivos_lang, motivos_model = aggregate_classification(data)
    plot_classification_by_lang(contagem, export_path=os.path.join(PLOTS_DIR,'classificacao_por_linguagem.png'))
    plot_reason_chart(motivos_lang, export_path=os.path.join(PLOTS_DIR,'motivos_por_linguagem.png'))
    plot_reason_by_model(motivos_model, export_path=os.path.join(PLOTS_DIR,'motivos_por_modelo.png'))
