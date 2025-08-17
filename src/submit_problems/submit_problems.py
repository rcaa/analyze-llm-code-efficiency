import os
import time
import json
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ==============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# ==============================================================================
BASE_PATH = "../data"
CSV_PATH = "resultado_leetcode.csv"
JSON_PATH = "resultado_leetcode.json"
GROUND_TRUTH_MODEL = "ground_truth"

LINGUAGEM_MAP = {
    "python3": "python3", "python": "python3",
    "c++": "cpp", "cpp": "cpp",
    "java": "java",
}

# ATENÇÃO: Seus cookies de sessão expiram. Talvez seja necessário atualizá-los.
COOKIES = {
    "LEETCODE_SESSION": "{SUA_LEETCODE_SESSION}",
    "csrftoken": "{sua_csrftoken}"
}
HEADERS = {
    "user-agent": "Mozilla/5.0",
    "referer": "https://leetcode.com",
    "origin": "https://leetcode.com",
    "x-csrftoken": COOKIES["csrftoken"],
}

# ==============================================================================
# 2. FUNÇÕES AUXILIARES E DE EXTRAÇÃO
# ==============================================================================

# A FUNÇÃO extract_code FOI REMOVIDA, POIS NÃO É MAIS NECESSÁRIA.

def is_valid_code(code: str) -> bool:
    """Verifica se o código extraído não é vazio ou composto apenas por espaços."""
    if not code or not code.strip():
        print("[WARN] Código vazio ou inválido detectado.")
        return False
    return True

# ==============================================================================
# 3. FUNÇÕES DE INTERAÇÃO COM A API DO LEETCODE
# ==============================================================================
# (Todas as funções de API como get_question_id, submit_code, etc., permanecem iguais)

def get_question_id(slug: str) -> str | None:
    # ... (código inalterado)
    try:
        resp = requests.get("https://leetcode.com/api/problems/all/", headers=HEADERS, cookies=COOKIES)
        resp.raise_for_status()
        for q in resp.json()["stat_status_pairs"]:
            if q["stat"]["question__title_slug"] == slug:
                return str(q["stat"]["question_id"])
    except Exception as e:
        print(f"[ERRO] ID questão: {e}")
    return None

def submit_code(slug: str, code: str, lang: str) -> bool:
    # ... (código inalterado)
    qid = get_question_id(slug)
    if not qid:
        print(f"[FAIL] Não encontrou o question_id para o slug: {slug}")
        return False
    payload = {"lang": lang, "question_id": qid, "typed_code": code}
    headers = HEADERS.copy()
    headers["content-type"] = "application/json"
    try:
        resp = requests.post(f"https://leetcode.com/problems/{slug}/submit/", headers=headers, cookies=COOKIES, json=payload)
        if resp.status_code == 200:
            print(f"[OK] Submetido: {slug} ({lang})")
            time.sleep(8)
            return True
        print(f"[FAIL] Submissão: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"[ERRO] Submissão: {e}")
    return False

def get_last_submission_id(slug: str, lang: str) -> int | None:
    # ... (código inalterado)
    url = f"https://leetcode.com/api/submissions/{slug}/"
    for attempt in range(5):
        try:
            resp = requests.get(url, headers=HEADERS, cookies=COOKIES)
            resp.raise_for_status()
            data = resp.json().get("submissions_dump", [])
            for sub in data:
                if sub.get("lang") == lang:
                    return sub.get("id")
        except Exception as e:
            print(f"[ERRO] get_last_submission_id (tentativa {attempt+1}): {e}")
        print(f"[INFO] Aguardando API... (tentativa {attempt+1}/5)")
        time.sleep(3)
    return None

def get_submission_result(slug: str, sub_id: int) -> dict:
    # ... (código inalterado)
    url = f"https://leetcode.com/submissions/detail/{sub_id}/check/"
    print(f"[DEBUG] Consultando resultado: slug={slug}, sub_id={sub_id}")
    for _ in range(30):
        try:
            resp = requests.get(url, headers=HEADERS, cookies=COOKIES)
            if not resp.headers.get("content-type", "").startswith("application/json"):
                print(f"[ERRO] Resposta não JSON (status={resp.status_code}): {resp.text[:200]}...")
                break
            data = resp.json()
            if data.get("state") == "SUCCESS":
                is_accepted = data.get('status_display') == 'Accepted'
                result = {
                    'status': data.get('status_display'), 'aceito': is_accepted,
                    'tempo_ms': data.get('status_runtime'), 'memoria_mb': data.get('status_memory'),
                    'compilou': data.get('run_success'), 'testes_passados': data.get('total_correct'),
                    'total_testes': data.get('total_testcases'),
                }
                if not is_accepted:
                    result['erro_compilacao'] = data.get('full_compile_error') or data.get('compile_error')
                    result['erro_runtime'] = data.get('full_runtime_error') or data.get('runtime_error')
                    result['ultimo_caso_teste'] = data.get('last_testcase')
                    result['saida_esperada'] = data.get('expected_output')
                    result['sua_saida'] = data.get('code_output')
                return result
            if data.get("state") in ["STARTED", "PENDING"]:
                time.sleep(2)
            else:
                print(f"[WARN] Estado inesperado da submissão: {data.get('state')}")
                return {'status': data.get('state'), 'aceito': False, 'tempo_ms': None, 'memoria_mb': None, 'compilou': False, 'testes_passados': 0, 'total_testes': None}
        except Exception as e:
            print(f"[ERRO] get_submission_result: {e}")
            time.sleep(2)
    print("[WARN] Timeout ao buscar resultado da submissão!")
    return {'status': 'Timeout', 'aceito': False, 'tempo_ms': None, 'memoria_mb': None, 'compilou': None, 'testes_passados': None, 'total_testes': None}

def get_top_community_code(slug: str, lang: str) -> str | None:
    # ... (código inalterado)
    try:
        url = f"https://leetcode.com/problems/{slug}/solutions/?orderBy=most_votes&languageTags={lang}"
        resp = requests.get(url, headers=HEADERS, cookies=COOKIES)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        blocks = [b.get_text() for b in soup.find_all('code') if len(b.get_text(strip=True).splitlines()) > 2]
        return blocks[0] if blocks else None
    except Exception as e:
        print(f"[ERRO] get_top_community_code: {e}")
    return None

# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    """Função principal que orquestra todo o processo de submissão e avaliação."""
    linhas_resultado = []
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            linhas_resultado = json.load(f)
        print(f"Carregados {len(linhas_resultado)} resultados existentes.")

    for modelo in os.listdir(BASE_PATH):
        modelo_path = os.path.join(BASE_PATH, modelo)
        if not os.path.isdir(modelo_path): continue

        print(f"\n{'='*20} Modelo: {modelo} {'='*20}")
        
        try:
            # ... (código para encontrar linguagens e questões permanece o mesmo) ...
            linguagens_disponiveis = [d for d in os.listdir(modelo_path) if os.path.isdir(os.path.join(modelo_path, d))]
            if not linguagens_disponiveis:
                print(f"[WARN] Nenhuma pasta de linguagem encontrada para o modelo {modelo}.")
                continue
            primeira_linguagem_path = os.path.join(modelo_path, linguagens_disponiveis[0])
            arquivos_questoes = sorted([f for f in os.listdir(primeira_linguagem_path) if f.endswith('.json')])
        except FileNotFoundError:
            print(f"[ERRO] Não foi possível encontrar pastas/arquivos para o modelo {modelo}.")
            continue

        for arquivo in arquivos_questoes:
            for linguagem in linguagens_disponiveis:
                lang_path = os.path.join(modelo_path, linguagem, arquivo)
                if not os.path.exists(lang_path): continue

                try:
                    with open(lang_path, encoding='utf-8') as f:
                        dados = json.load(f)
                except Exception as e:
                    print(f"[ERRO] Abrindo arquivo JSON: {lang_path} - {e}")
                    continue

                slug = dados.get('url', '').rstrip('/').split('/')[-1]
                if not slug: continue
                
                lang_api = LINGUAGEM_MAP.get(linguagem.lower())
                if not lang_api: continue

                ja_executado = any(r.get('modelo') == modelo and r.get('slug') == slug and r.get('linguagem', '').lower() == linguagem.lower() for r in linhas_resultado)
                if ja_executado:
                    print(f"[INFO] Já existe resultado para {modelo}/{linguagem}/{slug}. Pulando.")
                    continue
                
                # --- LÓGICA DE EXTRAÇÃO E CATEGORIZAÇÃO SIMPLIFICADA ---
                # Sempre usamos o campo "code", pois ele sempre está presente.
                codigo = dados.get("code")
                
                # Ainda categorizamos a resposta para fins de análise.
                status_original = dados.get("status", "desconhecido")
                formato_original = "ok" if status_original == "ok" else f"formato_invalido ({status_original})"
                
                if not is_valid_code(codigo): continue

                if lang_api == "python3" and "class Solution" not in codigo:
                    codigo = f"class Solution:\n" + "\n".join(["    " + line for line in codigo.split('\n')])
                    print(f"[INFO] Código Python para '{slug}' foi envelopado em 'class Solution'.")

                # Submissão e coleta de resultados
                print(f"\n--- Processando: {modelo} | {linguagem} | {slug} (Formato: {formato_original}) ---")
                if not submit_code(slug, codigo, lang_api): continue
                
                sub_id = get_last_submission_id(slug, lang_api)
                if not sub_id:
                    print(f"[FAIL] Não encontrou submission_id para {slug} ({lang_api}) após submissão.")
                    continue
                
                result = get_submission_result(slug, sub_id)
                
                # Salva o resultado
                linhas_resultado.append({
                    'modelo': modelo, 'linguagem': linguagem, 'id_questao': dados.get('id'),
                    'titulo': dados.get('titulo'), 'slug': slug, 'formato_original': formato_original,
                    **result, 'ground_truth': False
                })
                
                pd.DataFrame(linhas_resultado).to_csv(CSV_PATH, index=False)
                with open(JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(linhas_resultado, f, ensure_ascii=False, indent=2)

                # Processa o Ground Truth (lógica inalterada)
                gt_ja_executado = any(r.get('modelo') == GROUND_TRUTH_MODEL and r.get('slug') == slug and r.get('linguagem', '').lower() == linguagem.lower() for r in linhas_resultado)
                if not gt_ja_executado:
                    print(f"\n--- Processando: Ground Truth | {linguagem} | {slug} ---")
                    gt_code = get_top_community_code(slug, lang_api)
                    if gt_code and is_valid_code(gt_code) and submit_code(slug, gt_code, lang_api):
                        sub_id_gt = get_last_submission_id(slug, lang_api)
                        if sub_id_gt:
                            result_gt = get_submission_result(slug, sub_id_gt)
                            linhas_resultado.append({
                                'modelo': GROUND_TRUTH_MODEL, 'linguagem': linguagem, 'id_questao': dados.get('id'),
                                'titulo': dados.get('titulo'), 'slug': slug, 'formato_original': 'ok',
                                **result_gt, 'ground_truth': True
                            })
                            pd.DataFrame(linhas_resultado).to_csv(CSV_PATH, index=False)
                            with open(JSON_PATH, "w", encoding="utf-8") as f:
                                json.dump(linhas_resultado, f, ensure_ascii=False, indent=2)

    print("\n🏁 Resultados finais salvos.")

if __name__ == "__main__":
    main()