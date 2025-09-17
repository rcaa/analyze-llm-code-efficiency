import os
import pandas as pd
import json
import numpy as np

# Definir a semente para garantir reprodutibilidade
seed = 2025
np.random.seed(seed)

# Abre o arquivo JSON com todas as questões públicas
with open("datasets/teste_new_dataset/public_problems_without_images.json", "r", encoding="utf-8") as f:
    data = json.load(f)
# --- FILTRAR APENAS AS COLUNAS DE INTERESSE ---
cols_to_keep = [
    "id", "slug", "url", "titulo", "enunciado",
    "temas", "dificuldade", "tema_principal",
    "has_image", "starter_code", "stats"
]
cleaned = [
    { k: item.get(k, None) for k in cols_to_keep }
    for item in data
]
df = pd.DataFrame(cleaned)

# Limpeza das colunas de texto
df["enunciado"] = df["enunciado"].str.replace(r'[\n\r]+', ' ', regex=True).str.strip()
df["titulo"] = df["titulo"].str.replace(r'[\n\r]+', ' ', regex=True).str.strip()

# Filtra as questões sem erro
if "error" in df.columns:
    df = df[df["error"].isna() | (df["error"] == "")]

# Processa a coluna de temas
df["tema_principal"] = df["tema_principal"].str.strip()

# Contagem total de questões por dificuldade
N_total = len(df)
N_easy = len(df[df["dificuldade"] == "Fácil"])
N_med = len(df[df["dificuldade"] == "Média"])
N_hard = len(df[df["dificuldade"] == "Difícil"])

n_total = 336  # Total de questões que queremos amostrar

# Calculando a distribuição proporcional por dificuldade
n_easy = round(N_easy / N_total * n_total)
n_med = round(N_med / N_total * n_total)
n_hard = n_total - n_easy - n_med  # O restante para "Difícil"

# Ajusta a distribuição para garantir que a soma seja 336
total_calculado = n_easy + n_med + n_hard
if total_calculado != n_total:
    diff = n_total - total_calculado
    n_hard += diff  # Ajusta para a dificuldade "Difícil" caso o total não bata

print(f"Distribuição de questões por dificuldade:")
print(f"Fácil: {n_easy}, Média: {n_med}, Difícil: {n_hard}")

# Separar as questões por dificuldade
sample_easy = df[df["dificuldade"] == "Fácil"].copy()
sample_med = df[df["dificuldade"] == "Média"].copy()
sample_hard = df[df["dificuldade"] == "Difícil"].copy()

# Lista de todos os temas únicos
all_themes = df["tema_principal"].unique()
print(f"\nTotal de temas: {len(all_themes)}")

# Inicializar dicionário para controle de alocação por tema
theme_allocations = {theme: 0 for theme in all_themes}

# Lista para armazenar as questões selecionadas
selected_questions = []

# Função para selecionar questões de um tema/dificuldade
def select_questions(df_subset, theme, num_needed):
    theme_df = df_subset[df_subset["tema_principal"] == theme]
    if len(theme_df) == 0:
        return pd.DataFrame()
    
    # Selecionar no máximo o disponível
    n = min(num_needed, len(theme_df))
    return theme_df.sample(n, random_state=seed)

# 1. Garantir pelo menos 1 questão por tema em qualquer dificuldade
for theme in all_themes:
    # Tentar encontrar pelo menos 1 questão em qualquer dificuldade
    theme_quests = df[df["tema_principal"] == theme]
    if len(theme_quests) > 0:
        selected = theme_quests.sample(1, random_state=seed)
        selected_questions.append(selected)
        theme_allocations[theme] += 1
        # Remover a questão selecionada dos pools
        selected_id = selected["id"].values[0]
        sample_easy = sample_easy[sample_easy["id"] != selected_id]
        sample_med = sample_med[sample_med["id"] != selected_id]
        sample_hard = sample_hard[sample_hard["id"] != selected_id]

# 2. Atualizar contadores de questões restantes por dificuldade
# Contar quantas questões de cada dificuldade foram selecionadas na etapa 1
temp_df = pd.concat(selected_questions)
count_easy_step1 = len(temp_df[temp_df["dificuldade"] == "Fácil"])
count_med_step1 = len(temp_df[temp_df["dificuldade"] == "Média"])
count_hard_step1 = len(temp_df[temp_df["dificuldade"] == "Difícil"])

remaining_easy = n_easy - count_easy_step1
remaining_med = n_med - count_med_step1
remaining_hard = n_hard - count_hard_step1

print(f"\nQuestões restantes: Fácil={remaining_easy}, Média={remaining_med}, Difícil={remaining_hard}")

# 3. Distribuir as questões restantes por dificuldade de forma cíclica
print("\nDistribuindo questões Fáceis restantes de forma cíclica...")
def distribute_cyclic(difficulty_df, remaining_count, difficulty_name):
    global selected_questions, theme_allocations
    
    # Criar lista de temas com questões disponíveis
    available_themes = []
    for theme in all_themes:
        theme_available = difficulty_df[difficulty_df["tema_principal"] == theme]
        if len(theme_available) > 0:
            available_themes.append(theme)
    
    # Se não há temas disponíveis, retornar
    if not available_themes:
        print(f"  ATENÇÃO: Nenhum tema disponível para {difficulty_name}")
        return remaining_count, difficulty_df
    
    # Distribuir as questões de forma cíclica pelos temas
    while remaining_count > 0:
        progress = False
        for theme in available_themes:
            if remaining_count <= 0:
                break
                
            # Verificar disponibilidade no tema
            theme_available = difficulty_df[difficulty_df["tema_principal"] == theme]
            if len(theme_available) == 0:
                continue
                
            # Selecionar 1 questão
            selected = select_questions(theme_available, theme, 1)
            if len(selected) > 0:
                selected_questions.append(selected)
                theme_allocations[theme] += 1
                remaining_count -= 1
                # Remover questão selecionada do pool disponível
                selected_id = selected["id"].values[0]
                difficulty_df = difficulty_df[difficulty_df["id"] != selected_id]
                progress = True
                
        # Se não houve progresso, sair do loop
        if not progress:
            break
            
    return remaining_count, difficulty_df

# Distribuir para cada dificuldade
remaining_easy, sample_easy = distribute_cyclic(sample_easy, remaining_easy, "Fácil")
remaining_med, sample_med = distribute_cyclic(sample_med, remaining_med, "Média")
remaining_hard, sample_hard = distribute_cyclic(sample_hard, remaining_hard, "Difícil")

# 4. Completar com seleção aleatória se ainda faltarem questões
total_selected = len(pd.concat(selected_questions)) if selected_questions else 0
remaining_total = n_total - total_selected

if remaining_total > 0:
    print(f"\nCompletando {remaining_total} questões faltantes com seleção aleatória...")
    
    # Juntar todos os dados restantes
    all_remaining = pd.concat([sample_easy, sample_med, sample_hard])
    
    # Remover questões já selecionadas
    if selected_questions:
        selected_ids = pd.concat(selected_questions)["id"]
        all_remaining = all_remaining[~all_remaining["id"].isin(selected_ids)]
    
    # Selecionar questões aleatoriamente
    if len(all_remaining) < remaining_total:
        print(f"  ATENÇÃO: Apenas {len(all_remaining)} questões disponíveis para completar {remaining_total} faltantes")
        remaining_total = len(all_remaining)
    
    if remaining_total > 0:
        selected = all_remaining.sample(remaining_total, random_state=seed)
        selected_questions.append(selected)
        # Atualizar alocações por tema
        for theme in selected["tema_principal"].unique():
            count = len(selected[selected["tema_principal"] == theme])
            theme_allocations[theme] += count

# 5. Juntar todas as questões selecionadas
if selected_questions:
    final_sample_df = pd.concat(selected_questions)
    # Remover duplicatas baseado no ID da questão
    final_sample_df = final_sample_df.drop_duplicates(subset='id')
else:
    final_sample_df = pd.DataFrame()

# Garantir que o diretório de destino exista
output_dir = "datasets/teste_new_dataset/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ordenação e exportação da amostra final
if not final_sample_df.empty:
    final_sample_df = final_sample_df.sort_values(["tema_principal", "dificuldade", "id"]).reset_index(drop=True)
    final_sample_df.to_csv(f"{output_dir}sample.csv", index=False, encoding="utf-8")
else:
    print("Nenhuma questão foi selecionada!")

# Diagnóstico final
if not final_sample_df.empty:
    print("\nDistribuição final de temas na amostra:")
    final_counts = final_sample_df.groupby('tema_principal').size().reset_index(name='quantidade_total')
    final_counts = final_counts.sort_values(by='quantidade_total', ascending=False)
    print(final_counts)

    print("\nDistribuição por dificuldade:")
    difficulty_counts = final_sample_df.groupby('dificuldade').size()
    print(difficulty_counts)

    total_selected = len(final_sample_df)
    print(f"\nNúmero total de questões selecionadas: {total_selected}")
    print(f"Total esperado: {n_total}")
    print(f"Diferença: {n_total - total_selected}")

    # Verificar se todos os temas estão representados
    missing_themes = set(all_themes) - set(final_sample_df['tema_principal'].unique())
    if missing_themes:
        print(f"ATENÇÃO: Temas não representados: {missing_themes}")
    else:
        print("Todos os temas estão representados na amostra.")
else:
    print("Nenhuma questão foi selecionada para análise.")