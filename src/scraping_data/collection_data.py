import requests, json, time, re
from collections import defaultdict
from bs4 import BeautifulSoup
import os

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/113.0.0.0 Safari/537.36",
}

def get_starter_code(snippets, langs=["cpp", "java", "python3"]):
    starter = {}
    for snippet in snippets:
        langslug = snippet.get("langSlug")
        if langslug in langs:
            starter[langslug] = snippet["code"]
    return starter

def get_discuss_top(q_slug):
    url = "https://leetcode.com/graphql"
    discuss_query = {
        "operationName": "questionDiscussTopicList",
        "variables": {
            "titleSlug": q_slug,
            "orderBy": "most_votes",
            "skip": 0,
            "first": 1
        },
        "query": """
        query questionDiscussTopicList($titleSlug: String!, $orderBy: DiscussTopicListOrderBy, $skip: Int, $first: Int) {
          questionDiscussTopicList(titleSlug: $titleSlug, orderBy: $orderBy, skip: $skip, first: $first) {
            topics {
              id
              title
              url
              viewCount
              creationDate
              post {
                content
                voteCount
              }
              author {
                username
              }
              pinned
              solutionTags
              topLevelCommentCount
            }
          }
        }
        """
    }
    try:
        resp = requests.post(url, json=discuss_query, headers=headers)
        r = resp.json()
        topics = r.get("data", {}).get("questionDiscussTopicList", {}).get("topics", [])
        if topics:
            topic = topics[0]
            return {
                "title": topic.get("title"),
                "author": topic.get("author", {}).get("username"),
                "url": f"https://leetcode.com{topic.get('url')}",
                "votes": topic.get("post", {}).get("voteCount"),
                "content_html": topic.get("post", {}).get("content")
            }
        else:
            return None
    except Exception as e:
        return None

def get_top_solution_code(q_slug, lang="Python3"):
    url = "https://leetcode.com/graphql"
    query = {
        "operationName": "communitySolutions",
        "variables": {
            "questionSlug": q_slug,
            "orderBy": "most_votes",
            "languageTags": [lang],
            "skip": 0,
            "first": 1
        },
        "query": """
        query communitySolutions($questionSlug: String!, $orderBy: CommunitySolutionOrderBy, $languageTags: [String!], $skip: Int, $first: Int) {
          communitySolutions(
            questionSlug: $questionSlug
            orderBy: $orderBy
            languageTags: $languageTags
            skip: $skip
            first: $first
          ) {
            nodes {
              id
              title
              url
              post {
                content
                voteCount
                author {
                  username
                }
              }
              language {
                name
                verboseName
              }
            }
          }
        }
        """
    }
    try:
        resp = requests.post(url, json=query, headers=headers)
        r = resp.json()
        nodes = r.get("data", {}).get("communitySolutions", {}).get("nodes", [])
        if nodes:
            node = nodes[0]
            content_html = node["post"]["content"]
            soup = BeautifulSoup(content_html, "html.parser")
            code_blocks = soup.find_all("pre")
            code = "\n\n".join(cb.get_text() for cb in code_blocks) if code_blocks else ""
            return {
                "title": node["title"],
                "author": node["post"]["author"]["username"],
                "url": f"https://leetcode.com{node['url']}",
                "votes": node["post"]["voteCount"],
                "language": node["language"]["name"],
                "code": code
            }
        else:
            return None
    except Exception as e:
        print(f"[DiscussSolutionError] {q_slug} | {lang}: {e}")
        return None

def get_top_solution_post_scraping(q_slug):
    """
    Scrape da aba Solutions da interface web, pega o primeiro post mais votado.
    """
    url = f"https://leetcode.com/problems/{q_slug}/solutions/?orderBy=most_votes"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Tenta encontrar via data-e2e, senão pega o primeiro item da lista
        card = soup.find("div", {"data-e2e": "solution-card"})
        if not card:
            card = soup.find("div", {"role": "listitem"})
        if not card:
            return None
        # Extrai título, link, votos, autor
        title_tag = card.find("a", {"data-e2e": "solution-title"}) or card.find("a", href=True)
        title = title_tag.get_text(strip=True) if title_tag else ""
        post_url = "https://leetcode.com" + title_tag["href"] if title_tag and title_tag.has_attr("href") else url
        author_tag = card.find("a", {"data-e2e": "user-link"})
        author = author_tag.get_text(strip=True) if author_tag else ""
        votes_tag = card.find("span", {"data-e2e": "vote-count"})
        # Fallback: pega a primeira tag numérica que aparecer, se não houver votes_tag
        if not votes_tag:
            possible_votes = re.findall(r"(\d+)", card.get_text())
            votes = int(possible_votes[0]) if possible_votes else 0
        else:
            votes = int(votes_tag.get_text(strip=True)) if votes_tag and votes_tag.get_text(strip=True).isdigit() else 0

        # Pega o conteúdo detalhado da solução
        post_resp = requests.get(post_url, headers=headers, timeout=30)
        post_soup = BeautifulSoup(post_resp.text, "html.parser")
        content = post_soup.find("div", attrs={"data-track-load": "description_content"})
        content_html = str(content) if content else ""
        content_text = content.get_text("\n", strip=True) if content else ""
        found_langs = []
        # Heurística para tentar detectar a linguagem
        if content_html:
            if "```python" in content_html.lower(): found_langs.append("python")
            if "```java" in content_html.lower(): found_langs.append("java")
            if "```cpp" in content_html.lower() or "```c++" in content_html.lower(): found_langs.append("cpp")
        code_blocks = post_soup.find_all("pre")
        code = "\n\n".join(cb.get_text() for cb in code_blocks) if code_blocks else ""
        # Também tenta pegar blocos <code> isolados, caso não haja <pre>
        if not code:
            code_tags = post_soup.find_all("code")
            code = "\n\n".join(ct.get_text() for ct in code_tags) if code_tags else ""
        return {
            "title": title,
            "author": author,
            "votes": votes,
            "url": post_url,
            "content_html": content_html,
            "content_text": content_text,
            "langs": found_langs,
            "code": code
        }
    except Exception as e:
        print(f"[TopSolutionScrapeError] {q_slug}: {e}")
        return None

def safe_json_loads(text):
    try:
        return json.loads(text)
    except:
        return {}

# Diretório incremental
save_dir = "datasets/teste_new_dataset/questions"
os.makedirs(save_dir, exist_ok=True)

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

    filename = os.path.join(save_dir, f"{q_number}_{q_slug}.json")
    if os.path.exists(filename):
        print(f"[{idx}] {q_number} - {q_title} já salvo, pulando.")
        continue

    query = {
        "operationName": "questionData",
        "variables": {"titleSlug": q_slug},
        "query": """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    questionId
    title
    titleSlug
    content
    difficulty
    likes
    dislikes
    stats
    exampleTestcases
    sampleTestCase
    codeSnippets { lang langSlug code }
    topicTags { name }
    companyTagStats
    hints
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
    company_tags = safe_json_loads(data_q.get("companyTagStats") or "[]")
    company_tags_list = []
    if isinstance(company_tags, list):
        company_tags_list = [tag.get("name") for tag in company_tags if tag.get("name")]

    if not content_html:
        count_erro += 1
        errors.append({"numero": q_number, "slug": q_slug, "titulo": q_title, "erro": "Enunciado vazio"})
        print(f"[Aviso] Enunciado vazio para {q_slug}, pulando essa questão.")
        continue

    soup = BeautifulSoup(content_html, "html.parser")
    content_text = soup.get_text("\n").strip()
    tema_principal = tags[0] if tags else ""

    has_image = bool(soup.find_all("img"))
    stats = safe_json_loads(data_q.get("stats") or "{}")
    code_snippets = data_q.get("codeSnippets", [])
    starter_code = get_starter_code(code_snippets)
    hints = data_q.get("hints", [])
    likes = data_q.get("likes", 0)
    dislikes = data_q.get("dislikes", 0)
    example_testcases = data_q.get("exampleTestcases", "")
    sample_testcase = data_q.get("sampleTestCase", "")

    discuss_top = get_discuss_top(q_slug)
    discuss_top_entry = {}
    if discuss_top:
        discuss_content_html = discuss_top.get("content_html", "")
        discuss_content_text = BeautifulSoup(discuss_content_html or "", "html.parser").get_text("\n").strip() if discuss_content_html else ""
        discuss_top_entry = {
            "title": discuss_top.get("title"),
            "author": discuss_top.get("author"),
            "url": discuss_top.get("url"),
            "votes": discuss_top.get("votes"),
            "content": discuss_content_text
        }

    # Soluções mais votadas de Discuss por linguagem
    top_solutions = {}
    for langslug, tag in [("python3", "Python3"), ("cpp", "C++"), ("java", "Java")]:
        top_solutions[langslug] = get_top_solution_code(q_slug, tag)

    # Scraping da aba Solutions (primeiro post mais votado igual ao do front)
    top_solution_post = get_top_solution_post_scraping(q_slug)

    question_entry = {
        "id": q_number,
        "slug": q_slug,
        "url": f"https://leetcode.com/problems/{q_slug}/",
        "titulo": q_title,
        "enunciado": content_text,
        "temas": tags,
        "company_tags": company_tags_list,
        "dificuldade": difficulty_text,
        "tema_principal": tema_principal,
        "has_image": has_image,
        "likes": likes,
        "dislikes": dislikes,
        "stats": stats,
        "code_snippets": code_snippets,
        "starter_code": starter_code,
        "hints": hints,
        "example_testcases": example_testcases,
        "sample_testcase": sample_testcase,
        "discuss_top": discuss_top_entry,
        "top_solutions": top_solutions,
        "top_solution_post": top_solution_post,
    }

    with open(filename, "w", encoding="utf-8") as fq:
        json.dump(question_entry, fq, ensure_ascii=False, indent=2)
    print(f"[{idx}] {q_number} - {q_title} [{difficulty_text}] salvo em {filename}")

    if difficulty_level == 1:
        easy_questions.append(question_entry); count_easy += 1
    elif difficulty_level == 2:
        medium_questions.append(question_entry); count_medium += 1
    elif difficulty_level == 3:
        hard_questions.append(question_entry); count_hard += 1

    all_public_questions.append(question_entry)
    if has_image:
        questions_with_images.append(question_entry)
        count_with_image += 1
    else:
        questions_without_images.append(question_entry)
        count_without_image += 1

    for tema in tags:
        tema_dificuldade[tema][difficulty_text] += 1
        tema_dificuldade[tema]["total"] += 1
        tema_dificuldade[tema]["ids"].append(q_number)

    time.sleep(0.9)  # Evita bloqueio do LeetCode

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

os.makedirs("datasets/teste_new_dataset", exist_ok=True)
with open("datasets/teste_new_dataset/easy.json", "w", encoding="utf-8") as fe:
    json.dump(easy_questions, fe, ensure_ascii=False, indent=4)
with open("datasets/teste_new_dataset/medium.json", "w", encoding="utf-8") as fm:
    json.dump(medium_questions, fm, ensure_ascii=False, indent=4)
with open("datasets/teste_new_dataset/hard.json", "w", encoding="utf-8") as fh:
    json.dump(hard_questions, fh, ensure_ascii=False, indent=4)
with open("datasets/teste_new_dataset/report.json", "w", encoding="utf-8") as fr:
    json.dump(report, fr, ensure_ascii=False, indent=4)
with open("datasets/teste_new_dataset/public_problems.json", "w", encoding="utf-8") as f:
    json.dump(all_public_questions, f, ensure_ascii=False, indent=2)
with open("datasets/teste_new_dataset/public_problems_with_images.json", "w", encoding="utf-8") as f_img:
    json.dump(questions_with_images, f_img, ensure_ascii=False, indent=2)
with open("datasets/teste_new_dataset/public_problems_without_images.json", "w", encoding="utf-8") as f_noimg:
    json.dump(questions_without_images, f_noimg, ensure_ascii=False, indent=2)

print("\n===== RELATÓRIO TEMAS =====")
for tema, stats in tema_dificuldade.items():
    print(f"{tema}: Total={stats['total']} | Fácil={stats['Fácil']}, Média={stats['Média']}, Difícil={stats['Difícil']}")

print(f"\n===== RESUMO QUESTÕES COM E SEM IMAGEM =====")
print(f"Total com imagem: {count_with_image}")
print(f"Total sem imagem: {count_without_image}")
print(f"Porcentagem com imagem: {report['porcentagem_com_imagem']}%")
print(f"Porcentagem sem imagem: {report['porcentagem_sem_imagem']}%")
