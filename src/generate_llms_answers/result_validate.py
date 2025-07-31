import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from matplotlib.patches import Patch

# Define o diretório base onde os resultados estão salvos
OUTPUT_BASE = "../../data"
# Diretório para salvar os gráficos gerados
PLOTS_DIR = "plots"

def listar_modelos_oficiais():
    """
    Retorna a lista de nomes de modelos 'oficiais', como vêm da API.
    Esta é a fonte da verdade para os nomes que serão exibidos no gráfico.
    """
    return [
        "allam-2-7b", "deepseek-r1-distill-llama-70b", "gemma2-9b-it",
        "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "llama3-70b-8192", "llama3-8b-8192",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "moonshotai/kimi-k2-instruct",
        "qwen/qwen3-32b"
    ]

def get_all_modelos_linguagens(base_path):
    """
    Descobre os modelos verificando se os diretórios (incluindo aninhados)
    correspondentes aos nomes oficiais existem no disco.
    """
    LINGUAGENS_VALIDAS = ["C++", "Java", "Python3"]
    modelos_oficiais = listar_modelos_oficiais()
    modelos_encontrados = []

    if not os.path.isdir(base_path):
        print(f"Diretório base '{base_path}' não encontrado.")
        # Criando diretório e alguns arquivos de exemplo para demonstração
        print("Criando diretórios e arquivos de exemplo para demonstração...")
        os.makedirs(base_path, exist_ok=True)
        for modelo in modelos_oficiais[:4]: # Criando para os 4 primeiros modelos
            for linguagem in LINGUAGENS_VALIDAS:
                path = os.path.join(base_path, modelo, linguagem)
                os.makedirs(path, exist_ok=True)
                for i in range(np.random.randint(15, 25)): # Quantidade variada de arquivos
                    status = np.random.choice(['ok', 'formato_invalido', 'erro'], p=[0.7, 0.2, 0.1])
                    with open(os.path.join(path, f'resultado_{i}.json'), 'w') as f:
                        json.dump({'status': status}, f)

    print("Verificando diretórios de modelos existentes (incluindo aninhados)...")
    for modelo in modelos_oficiais:
        caminho_modelo = os.path.join(base_path, modelo)
        if os.path.isdir(caminho_modelo):
            modelos_encontrados.append(modelo)
            print(f"  [OK] Encontrado: {caminho_modelo}")
        else:
            print(f"  [--] Não encontrado: {caminho_modelo}")

    return sorted(modelos_encontrados), LINGUAGENS_VALIDAS

def load_all_resultados(base_path, modelos_para_plotar, linguagens):
    """
    Carrega os resultados dos arquivos JSON, categorizando por status:
    'ok', 'formato_invalido' e 'erro'.
    """
    resultados = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    totais = defaultdict(lambda: defaultdict(int))

    for modelo in tqdm(modelos_para_plotar, desc="Lendo resultados dos modelos"):
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
                        totais[modelo][linguagem] += 1
                        status = data.get("status", "erro")
                        resultados[modelo][linguagem][status] += 1
                    except Exception as e:
                        print(f"Erro ao ler o arquivo {path}: {e}")
                        resultados[modelo][linguagem]['erro'] += 1

    return resultados, totais

# =============== FUNÇÃO DE PLOTAGEM ATUALIZADA ===============
def plot_stacked_bar_chart(resultados, totais, modelos, linguagens, export_path=None):
    """
    Gera um painel com um gráfico para cada linguagem. Em cada gráfico,
    os modelos estão no eixo X e as barras empilhadas representam os status.
    """
    modelos_ativos = sorted([m for m in modelos if sum(totais[m].values()) > 0])
    if not modelos_ativos:
        print("Nenhum modelo com dados encontrado para plotar.")
        return

    n_linguagens = len(linguagens)
    fig, axes = plt.subplots(1, n_linguagens, figsize=(7 * n_linguagens, 8), sharey=True)
    if n_linguagens == 1:
        axes = [axes]

    status_order = ['ok', 'formato_invalido', 'erro']
    status_colors = {
        'ok': '#2ca02c',
        'formato_invalido': '#ff7f0e',
        'erro': '#d62728'
    }

    for ax, linguagem in zip(axes, linguagens):
        ax.set_title(f"Linguagem: {linguagem}", fontsize=16, fontweight='bold')
        
        bottoms = np.zeros(len(modelos_ativos))
        data_by_status = {status: np.array([resultados[model][linguagem].get(status, 0) for model in modelos_ativos]) for status in status_order}

        x = np.arange(len(modelos_ativos))
        
        for status in status_order:
            valores = data_by_status[status]
            bars = ax.bar(x, valores, label=status, color=status_colors.get(status, '#888888'), bottom=bottoms, zorder=3)
            for bar, valor in zip(bars, valores):
                if valor > 0:
                    y_pos = bar.get_y() + bar.get_height() / 2
                    ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f'{int(valor)}', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
            bottoms += valores

        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1] for m in modelos_ativos], rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        #ax.set_xlabel("Modelos", fontsize=12)

    axes[0].set_ylabel("Quantidade de Respostas", fontsize=14)
    
    # A forma correta é passar apenas os 'handles'. O Matplotlib extrai os labels de dentro deles.
    handles = [Patch(color=color, label=label.replace('_', ' ').title()) for label, color in status_colors.items()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, title="Status da Resposta")
    # ---- FIM DA CORREÇÃO ----

    modelos_sem_dados = [m for m in modelos if m not in modelos_ativos]
    titulo_figura = "Desempenho dos Modelos por Linguagem e Status da Resposta"
    if modelos_sem_dados:
        # Quebra a lista de modelos para não ficar muito longa no título
        nomes_modelos_sem_dados = ", ".join([m.split('/')[-1] for m in modelos_sem_dados])
        titulo_figura += f"\nModelos sem dados: {nomes_modelos_sem_dados}"
    
    fig.suptitle(titulo_figura, fontsize=18, y=1.05)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {export_path}")

    plt.show()
def main():
    """
    Função principal para orquestrar a geração do gráfico.
    """
    modelos, linguagens = get_all_modelos_linguagens(OUTPUT_BASE)

    if not modelos:
        print("Nenhum diretório de modelo válido foi encontrado.")
        return

    print("\nModelos que serão analisados:", modelos)
    print("Linguagens a serem analisadas:", linguagens)

    resultados, totais = load_all_resultados(OUTPUT_BASE, modelos, linguagens)

    plot_stacked_bar_chart(
        resultados, totais, modelos, linguagens,
        export_path=os.path.join(PLOTS_DIR, "desempenho_por_linguagem.png")
    )

if __name__ == "__main__":
    main()