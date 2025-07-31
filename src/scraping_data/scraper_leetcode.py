import requests, json, time
from collections import defaultdict
from bs4 import BeautifulSoup
import os

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/113.0.0.0 Safari/537.36"
}

url_list = "https://leetcode.com/api/problems/algorithms/"
response = requests.get(url_list, headers=headers)
data = response.json()
problems = data.get("stat_status_pairs", [])
print(f"Total de problemas encontrados (incluindo premium): {len(problems)}")

easy_questions, medium_questions, hard_questions = [], [], []
all_public_questions = []
questions_with_images = []
questions_without_images = []
errors = []
count_easy = count_medium = count_hard = 0
count_easy_premium = count_medium_premium = count_hard_premium = 0
count_public = count_premium = 0
count_erro = 0
count_with_image = count_without_image = 0

tema_dificuldade = defaultdict(lambda: {"Fácil": 0, "Média": 0, "Difícil": 0, "total": 0, "ids": []})

for idx, item in enumerate(problems, 1):
    stat = item.get("stat", {})
    paid = item.get("paid_only", True)
    difficulty_level = item.get("difficulty", {}).get("level", None)
    q_number = stat.get("frontend_question_id")
    q_title = stat.get("question__title")
    q_slug = stat.get("question__title_slug")
    
    if difficulty_level == 1:
        difficulty_text = "Fácil"
    elif difficulty_level == 2:
        difficulty_text = "Média"
    elif difficulty_level == 3:
        difficulty_text = "Difícil"
    else:
        difficulty_text = "Desconhecida"
    
    if paid:
        count_premium += 1
        if difficulty_level == 1: count_easy_premium += 1
        elif difficulty_level == 2: count_medium_premium += 1
        elif difficulty_level == 3: count_hard_premium += 1
        continue

    count_public += 1

    query = {
        "operationName": "questionData",
        "variables": {"titleSlug": q_slug},
        "query": """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    content
    topicTags { name }
  }
}"""
    }
    try:
        graphql_resp = requests.post("https://leetcode.com/graphql", json=query, headers=headers)
        result = graphql_resp.json()
    except Exception as e:
        count_erro += 1
        errors.append({"numero": q_number, "slug": q_slug, "titulo": q_title, "erro": str(e)})
        print(f"[Erro] {q_number} - {q_title}: {e}")
        continue

    data_q = result.get("data", {}).get("question", {})
    content_html = data_q.get("content", "")
    tags = [tag["name"] for tag in data_q.get("topicTags", [])]

    if not content_html:
        count_erro += 1
        errors.append({"numero": q_number, "slug": q_slug, "titulo": q_title, "erro": "Enunciado vazio"})
        print(f"[Aviso] Enunciado vazio para {q_slug}, pulando essa questão.")
        continue

    soup = BeautifulSoup(content_html, "html.parser")
    content_text = soup.get_text("\n").strip()
    tema_principal = tags[0] if tags else ""

    # Detecta imagens
    has_image = bool(soup.find_all("img"))

    print(f"{q_number} - {q_title} [{difficulty_text}] | Temas: {', '.join(tags) or 'Nenhum'} | Tema principal: {tema_principal} | Imagem: {'SIM' if has_image else 'NÃO'}")

    question_entry = {
        "id": q_number,
        "slug": q_slug,
        "url": f"https://leetcode.com/problems/{q_slug}/",
        "titulo": q_title,
        "enunciado": content_text,
        "temas": tags,
        "dificuldade": difficulty_text,
        "tema_principal": tema_principal,
        "has_image": has_image
    }
    if difficulty_level == 1:
        easy_questions.append(question_entry); count_easy += 1
    elif difficulty_level == 2:
        medium_questions.append(question_entry); count_medium += 1
    elif difficulty_level == 3:
        hard_questions.append(question_entry); count_hard += 1

    # Salva na lista geral de públicas
    all_public_questions.append(question_entry)

    # Salva em listas separadas com/sem imagem
    if has_image:
        questions_with_images.append(question_entry)
        count_with_image += 1
    else:
        questions_without_images.append(question_entry)
        count_without_image += 1

    # Lógica de contagem por tema
    for tema in tags:
        tema_dificuldade[tema][difficulty_text] += 1
        tema_dificuldade[tema]["total"] += 1
        tema_dificuldade[tema]["ids"].append(q_number)

    time.sleep(0.5)  # Ajuste conforme necessário

total_public = count_easy + count_medium + count_hard
total_premium = count_easy_premium + count_medium_premium + count_hard_premium

report = {
    "total_questoes": len(problems),
    "publicas": count_public,
    "premium": count_premium,
    "faceis_publicas": count_easy,
    "medias_publicas": count_medium,
    "dificeis_publicas": count_hard,
    "faceis_premium": count_easy_premium,
    "medias_premium": count_medium_premium,
    "dificeis_premium": count_hard_premium,
    "total_publicas_coletadas": total_public,
    "total_premium": total_premium,
    "questoes_com_erro": count_erro,
    "total_com_imagem": count_with_image,
    "total_sem_imagem": count_without_image,
    "porcentagem_publica": round(100 * count_public / len(problems), 2),
    "porcentagem_premium": round(100 * count_premium / len(problems), 2),
    "porcentagem_com_imagem": round(100 * count_with_image / total_public, 2) if total_public else 0,
    "porcentagem_sem_imagem": round(100 * count_without_image / total_public, 2) if total_public else 0,
    "erros": errors,
    "temas": {k: dict(v) for k, v in tema_dificuldade.items()},
}

os.makedirs("datasets/leetcode", exist_ok=True)
# Salva relatórios json como antes
with open("datasets/leetcode/easy.json", "w", encoding="utf-8") as fe:
    json.dump(easy_questions, fe, ensure_ascii=False, indent=4)
with open("datasets/leetcode/medium.json", "w", encoding="utf-8") as fm:
    json.dump(medium_questions, fm, ensure_ascii=False, indent=4)
with open("datasets/leetcode/hard.json", "w", encoding="utf-8") as fh:
    json.dump(hard_questions, fh, ensure_ascii=False, indent=4)
with open("datasets/leetcode/report.json", "w", encoding="utf-8") as fr:
    json.dump(report, fr, ensure_ascii=False, indent=4)

# Novo: salva todas as públicas (com tema principal) em JSON
with open("datasets/leetcode/public_problems.json", "w", encoding="utf-8") as f:
    json.dump(all_public_questions, f, ensure_ascii=False, indent=2)

# Salva as listas de questões com e sem imagem
with open("datasets/leetcode/public_problems_with_images.json", "w", encoding="utf-8") as f_img:
    json.dump(questions_with_images, f_img, ensure_ascii=False, indent=2)
with open("datasets/leetcode/public_problems_without_images.json", "w", encoding="utf-8") as f_noimg:
    json.dump(questions_without_images, f_noimg, ensure_ascii=False, indent=2)

print("\n===== RELATÓRIO TEMAS =====")
for tema, stats in tema_dificuldade.items():
    print(f"{tema}: Total={stats['total']} | Fácil={stats['Fácil']}, Média={stats['Média']}, Difícil={stats['Difícil']}")

print(f"\n===== RESUMO QUESTÕES COM E SEM IMAGEM =====")
print(f"Total com imagem: {count_with_image}")
print(f"Total sem imagem: {count_without_image}")
print(f"Porcentagem com imagem: {report['porcentagem_com_imagem']}%")
print(f"Porcentagem sem imagem: {report['porcentagem_sem_imagem']}%")
