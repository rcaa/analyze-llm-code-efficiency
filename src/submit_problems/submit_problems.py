# -*- coding: utf-8 -*-
"""
Submete respostas de LLM ao LeetCode e coleta resultados.

Mudanças principais desta versão:
- NÃO há mais "um JSON gigante".
- Saída particionada por {modelo}/{linguagem}/{slug}.json (último veredito).
- Histórico por partição em NDJSON (um resultado por linha).
- CSV global por append para planilhas/analytics rápidos.
- Carregamento inicial reconstrói memória a partir das partições (e mantém compat com JSON legado).

Extras de proteção:
- Timeout por request (REQUEST_TIMEOUT)
- Retry com backoff para 429/5xx
- Pacing entre requests (REQUEST_COOLDOWN) e entre submissões (SUBMIT_COOLDOWN)
- Limite de tempo total no polling (MAX_POLL_TIME)

Ordem de submissão:
- Questão -> Modelo -> Linguagens

Novidades preservadas:
- Normalização de categorias declaradas
- Classificação automática pelo resultado do juiz
- Auditoria categoria_declarada vs categoria_calculada (categoria_ok)
- Sumário de divergências ao final
"""

import os
import re
import json
import time
import unicodedata
from typing import Optional, Tuple, Dict, List

import requests
from bs4 import BeautifulSoup
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==============================================================================
# 1) CONFIG (caminhos robustos + limites)
# ==============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent                    # .../src/submit_problems
PROJECT_ROOT = SCRIPT_DIR.parent.parent                           # raiz do projeto

# Onde estão as respostas das LLMs (padrão: <raiz>/data)
BASE_PATH  = Path(os.getenv("LLM_BASE_PATH", PROJECT_ROOT / "data"))

# Para onde salvar CSV/NDJSON/partições (padrão: <raiz>/out)
OUTPUT_DIR = Path(os.getenv("LLM_OUTPUT_DIR", PROJECT_ROOT / "out"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH  = Path(os.getenv("LLM_CSV_PATH", OUTPUT_DIR / "results.csv"))
# Mantido apenas para compatibilidade de leitura (não será mais escrito)
JSON_PATH = Path(os.getenv("LLM_JSON_PATH", OUTPUT_DIR / "resultado_leetcode.json"))

GROUND_TRUTH_MODEL = "ground_truth"

# Mapeia pasta de linguagem -> API do LeetCode
LINGUAGEM_MAP = {
    "python3": "python3", "python": "python3",
    "c++": "cpp", "cpp": "cpp",
    "java": "java",
}

# >>> AJUSTE ESTES LIMITES CONFORME NECESSÁRIO <<<
REQUEST_TIMEOUT   = float(os.getenv("LC_REQUEST_TIMEOUT",   15))  # seg por request
REQUEST_COOLDOWN  = float(os.getenv("LC_REQUEST_COOLDOWN",  0.5)) # seg entre chamadas HTTP
SUBMIT_COOLDOWN   = float(os.getenv("LC_SUBMIT_COOLDOWN",   8.0)) # seg entre submissões
MAX_POLL_TIME     = float(os.getenv("LC_MAX_POLL_TIME",     80))  # seg totais no /check/
RETRY_TOTAL       = int(os.getenv("LC_RETRY_TOTAL",         5))
RETRY_BACKOFF     = float(os.getenv("LC_RETRY_BACKOFF",     1.0))

# >>> ATUALIZE SEUS COOKIES <<<
COOKIES = {
    "LEETCODE_SESSION": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfYXV0aF91c2VyX2lkIjoiMTc2OTQyNzQiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJkamFuZ28uY29udHJpYi5hdXRoLmJhY2tlbmRzLk1vZGVsQmFja2VuZCIsIl9hdXRoX3VzZXJfaGFzaCI6ImJmNTJmZTgzNzNjNjcxYTJhMWEyZmNhNWNiMzYyOWZiNTI3MjU0YTY3ZGJkZWFkNTczMWE0N2Y4ZGQ1MDZmNTgiLCJzZXNzaW9uX3V1aWQiOiJiMTNlMjRkNSIsImlkIjoxNzY5NDI3NCwiZW1haWwiOiJhZGVuaWxzb24ucmFtb3NAdWZhcGUuZWR1LmJyIiwidXNlcm5hbWUiOiJBZGVuaWxzb25SYW1vcyIsInVzZXJfc2x1ZyI6IkFkZW5pbHNvblJhbW9zIiwiYXZhdGFyIjoiaHR0cHM6Ly9hc3NldHMubGVldGNvZGUuY29tL3VzZXJzL0FkZW5pbHNvblJhbW9zL2F2YXRhcl8xNzQ3MzEzNDc1LnBuZyIsInJlZnJlc2hlZF9hdCI6MTc1ODIwNjIzNiwiaXAiOiIyODA0OjE4OjU4MzY6MWY0NzplNDdkOjZhYzA6ZWIxMzpkNTNiIiwiaWRlbnRpdHkiOiJhM2Y1N2JiZTIxYzRlMzAzNzkyMjhhZDc3ODhmMjI0ZCIsImRldmljZV93aXRoX2lwIjpbImIxN2FlNzJjNzM2ZGE2NjkzZTUzNTc2MzhlNTAxODFlIiwiMjgwNDoxODo1ODM2OjFmNDc6ZTQ3ZDo2YWMwOmViMTM6ZDUzYiJdLCJfc2Vzc2lvbl9leHBpcnkiOjEyMDk2MDB9.1qKcPVtMceBBMejb5m_cAja7L5ZzSiFWRjbOdFttu9E",  # substitua
    "csrftoken": "sOuxMozWTNNz2S66vgexCBCgxUMPOLYCUmzion3KREQvIZMODGBSclZWOii3runT"          # substitua
}

HEADERS = {
    "user-agent": "Mozilla/5.0",
    "referer": "https://leetcode.com",
    "origin": "https://leetcode.com",
    "x-csrftoken": COOKIES.get("csrftoken", ""),
}

# Sessão com retry/backoff
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION.cookies.update(COOKIES)

retry = Retry(
    total=RETRY_TOTAL,
    backoff_factor=RETRY_BACKOFF,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
    raise_on_status=False,
    respect_retry_after_header=True,
)
adapter = HTTPAdapter(max_retries=retry)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)

# Pace simples para não bombardear o site
_last_http_ts = 0.0
def _pace(min_interval: float):
    global _last_http_ts
    now = time.monotonic()
    wait = min_interval - (now - _last_http_ts)
    if wait > 0:
        time.sleep(wait)
    _last_http_ts = time.monotonic()

def paced_get(url: str, **kwargs):
    _pace(REQUEST_COOLDOWN)
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    return SESSION.get(url, **kwargs)

def paced_post(url: str, **kwargs):
    _pace(REQUEST_COOLDOWN)
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    return SESSION.post(url, **kwargs)

# índice carregado de /api/problems/all/
_PROBLEMS_IDX: Dict[str, Dict] = {
    "by_slug": {},
    "by_frontend": {},
    "by_qid": {},
}

# ==============================================================================
# 2) CATEGORIAS: normalização, aceitação e classificação
# ==============================================================================
# Aliases amigáveis para normalização
CATS = {
    "invalid": {"invalid", "invalida", "inválida"},
    "semantic_incorrect": {"semantic_incorrect", "semantically_incorrect", "semantica_incorreta", "semântica_incorreta"},
    "plausible": {"plausible", "plausivel", "plausível"},
    "actual_correct": {"actual_correct", "correct", "correto", "correta"},
}

# Quais categorias do JSON você quer processar (mantém o filtro original: plausible/correct)
CATEGORIAS_ACEITAS_CANON = {"plausible", "actual_correct"}

def _norm_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.casefold()

def _norm_cat(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = _norm_text(s)
    for k, aliases in CATS.items():
        if t in { _norm_text(a) for a in aliases }:
            return k
    return None

def categoria_aceita(categoria: Optional[str]) -> bool:
    canon = _norm_cat(categoria)
    return canon in CATEGORIAS_ACEITAS_CANON

def classificar_por_resultado(result: dict) -> str:
    """
    Regras:
      - invalid: não compilou OU tem erro de compilação
      - actual_correct: aceito pelo juiz (Accepted)
      - semantic_incorrect: compilou mas NÃO aceito
      - plausible: opcional, se passou em testes locais mas falhou no juiz
    """
    compilou = result.get("compilou")
    aceito = result.get("aceito")
    has_ce = bool(result.get("erro_compilacao"))
    status = (result.get("status") or "").lower()

    if has_ce or compilou is False:
        return "invalid"

    if aceito:
        return "actual_correct"

    if result.get("passou_testes_locais"):
        return "plausible"

    if compilou or "wrong answer" in status or "time limit" in status or "memory limit" in status:
        return "semantic_incorrect"

    return "semantic_incorrect"

# ==============================================================================
# 3) AUXILIARES
# ==============================================================================
def linguagem_api(nome_dir: str) -> Optional[str]:
    return LINGUAGEM_MAP.get(nome_dir.strip().lower())

def extract_slug_from_filename(path) -> str:
    p = Path(path)
    slug = p.stem.strip().replace(" ", "-")
    return slug

def _parse_number(s):
    if s is None:
        return None
    m = re.search(r'[\d.]+', str(s))
    return float(m.group(0)) if m else None

def _parse_ms(s):
    v = _parse_number(s)
    return int(round(v)) if v is not None else None

def is_valid_code(code: str) -> bool:
    return bool(code and code.strip())

def ensure_python_solution_wrapper(slug: str, codigo: str) -> str:
    if "class Solution" not in codigo:
        print(f"[INFO] Python '{slug}': envelopando em 'class Solution'.")
        body = "\n".join(("    " + line if line.strip() else line) for line in codigo.splitlines())
        return "class Solution:\n" + body + ("\n" if not body.endswith("\n") else "")
    return codigo

def try_extract_code_from_resposta(resposta: str) -> Optional[str]:
    if not resposta:
        return None
    m = re.search(r"```(?:[a-zA-Z0-9+#]*)\s*([\s\S]*?)```", resposta)
    if m:
        return m.group(1).strip()
    return None

# ==============================================================================
# 3.1) NOVO: Saídas particionadas + NDJSON + CSV
# ==============================================================================
PARTITION_DIR = OUTPUT_DIR / "by_model"
NDJSON_DIR    = OUTPUT_DIR / "ndjson"
PARTITION_DIR.mkdir(parents=True, exist_ok=True)
NDJSON_DIR.mkdir(parents=True, exist_ok=True)

def _san(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._+-]+", "_", str(s or "").strip())

def _partition_path(modelo: str, linguagem: str, slug: str) -> Path:
    return PARTITION_DIR / _san(modelo) / _san(linguagem) / f"{_san(slug)}.json"

def save_result_partitioned(row: dict):
    """Grava o último veredito em by_model/<modelo>/<linguagem>/<slug>.json."""
    p = _partition_path(row.get("modelo"), row.get("linguagem"), row.get("slug"))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)

def append_ndjson(row: dict):
    """Acumula histórico por partição em NDJSON."""
    fname = f"{_san(row.get('modelo'))}__{_san(row.get('linguagem'))}.ndjson"
    path = NDJSON_DIR / fname
    row_with_ts = dict(row)
    row_with_ts.setdefault("ts_epoch", int(time.time()))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row_with_ts, ensure_ascii=False) + "\n")

_CSV_COLS = [
    "modelo","linguagem","slug","id_questao","categoria_declarada",
    "categoria_calculada","categoria_ok","status","aceito",
    "tempo_ms","memoria_mb","submission_id","ground_truth"
]
def append_csv(row: dict):
    import csv
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)

def save_result(row: dict):
    """Grava particionado + histórico NDJSON + CSV."""
    save_result_partitioned(row)
    append_ndjson(row)
    append_csv(row)

def load_existing_results() -> List[dict]:
    """Reconstrói memória a partir das partições. Mantém compat com JSON legado."""
    # 1) legado: se existir o JSON antigo, carrega (compat).
    if JSON_PATH.exists():
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # 2) Varre partições para montar o 'último' de cada slug.
    linhas: List[dict] = []
    if PARTITION_DIR.exists():
        for model_dir in PARTITION_DIR.iterdir():
            if not model_dir.is_dir(): 
                continue
            for lang_dir in model_dir.iterdir():
                if not lang_dir.is_dir(): 
                    continue
                for jf in lang_dir.glob("*.json"):
                    try:
                        with open(jf, "r", encoding="utf-8") as f:
                            linhas.append(json.load(f))
                    except Exception:
                        continue
    return linhas

# ==============================================================================
# 4) ÍNDICE DE QUESTÕES & RESOLUÇÃO DE SLUG/ID
# ==============================================================================
def ensure_problems_index():
    """Carrega / atualiza o índice de questões em memória."""
    if _PROBLEMS_IDX["by_slug"]:  # já carregado
        return
    try:
        resp = paced_get("https://leetcode.com/api/problems/all/")
        resp.raise_for_status()
        pairs = resp.json().get("stat_status_pairs", [])
        by_slug, by_frontend, by_qid = {}, {}, {}
        for q in pairs:
            stat = q.get("stat", {}) or {}
            slug = stat.get("question__title_slug")
            qid = stat.get("question_id")             # id interno
            fid = stat.get("frontend_question_id")    # id "visível" (ex.: 1006)
            if slug:
                by_slug[str(slug)] = {"qid": qid, "fid": fid}
            if fid is not None:
                by_frontend[str(fid)] = {"slug": slug, "qid": qid}
            if qid is not None:
                by_qid[str(qid)] = {"slug": slug, "fid": fid}
        _PROBLEMS_IDX["by_slug"] = by_slug
        _PROBLEMS_IDX["by_frontend"] = by_frontend
        _PROBLEMS_IDX["by_qid"] = by_qid
        print(f"[OK] Índice carregado: {len(by_slug)} questões.")
    except Exception as e:
        print(f"[ERRO] Carregando índice de questões: {e}")

def resolve_question(slugish: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Aceita:
      - slug puro (ex.: 'clumsy-factorial')
      - frontend id (ex.: '1006')
      - formato '1006-clumsy-factorial'
    Retorna: (slug_canonico, question_id_interno)
    """
    ensure_problems_index()
    if not slugish:
        return None, None
    s = slugish.strip().lower()

    m = re.match(r"^(\d+)[-_].+$", s)  # "1006-foo"
    if m:
        fid = m.group(1)
        hit = _PROBLEMS_IDX["by_frontend"].get(fid)
        if hit:
            return hit["slug"], str(hit["qid"])

    if s.isdigit():  # "1006"
        hit = _PROBLEMS_IDX["by_frontend"].get(s)
        if hit:
            return hit["slug"], str(hit["qid"])
        hit2 = _PROBLEMS_IDX["by_qid"].get(s)  # raro
        if hit2:
            return hit2["slug"], s
        return None, None

    hit = _PROBLEMS_IDX["by_slug"].get(s)  # slug
    if hit:
        return s, str(hit["qid"])

    return None, None

def get_question_id(slug_or_id: str) -> Optional[str]:
    _, qid = resolve_question(slug_or_id)
    return qid

# ==============================================================================
# 5) API DE SUBMISSÕES (com pacing e timeout)
# ==============================================================================
def submit_code(slug_or_id: str, code: str, lang: str) -> Optional[int]:
    slug, qid = resolve_question(slug_or_id)
    if not slug or not qid:
        print(f"[FAIL] Não encontrou question_id/slug para: {slug_or_id}")
        return None

    payload = {"lang": lang, "question_id": qid, "typed_code": code}
    headers = SESSION.headers.copy()
    headers["content-type"] = "application/json"

    try:
        resp = paced_post(f"https://leetcode.com/problems/{slug}/submit/",
                          headers=headers, json=payload)
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                data = {}
            sub_id = data.get("submission_id") or data.get("submissionId")
            if sub_id:
                print(f"[OK] Submetido: {slug} (qid={qid}, {lang}) -> submission_id={sub_id}")
                return int(sub_id)
            print("[WARN] submit sem submission_id; tentando fallback...")
            return get_last_submission_id(slug, lang)
        print(f"[FAIL] Submissão: {resp.status_code} - {resp.text[:200]}")
    except Exception as e:
        print(f"[ERRO] Submissão: {e}")
    return None

def get_last_submission_id(slug: str, lang: str) -> Optional[int]:
    url = f"https://leetcode.com/api/submissions/{slug}/?offset=0&limit=20"
    for attempt in range(5):
        try:
            resp = paced_get(url)
            resp.raise_for_status()
            data = resp.json().get("submissions_dump", [])
            data = sorted(data, key=lambda x: x.get("id", 0), reverse=True)
            for sub in data:
                if sub.get("lang") == lang:
                    return sub.get("id")
        except Exception as e:
            print(f"[ERRO] get_last_submission_id (tentativa {attempt+1}): {e}")
        print(f"[INFO] Aguardando API... (tentativa {attempt+1}/5)")
        time.sleep(3)
    return None

def get_submission_result(slug: str, sub_id: int) -> dict:
    url = f"https://leetcode.com/submissions/detail/{sub_id}/check/"
    print(f"[DEBUG] Consultando resultado: sub_id={sub_id}")
    backoff = 1.0
    start = time.monotonic()

    while True:
        if time.monotonic() - start > MAX_POLL_TIME:
            print("[WARN] Polling excedeu MAX_POLL_TIME.")
            break
        try:
            resp = paced_get(url)
            if not resp.headers.get("content-type", "").startswith("application/json"):
                print(f"[ERRO] Resposta não JSON (status={resp.status_code}): {resp.text[:200]}...")
                time.sleep(2)
                continue
            data = resp.json()
            state = data.get("state") or data.get("state_name")
            if state == "SUCCESS":
                status_display = (
                    data.get("status_msg")
                    or data.get("statusDisplay")
                    or data.get("status_display")
                    or data.get("status")
                )
                is_accepted = (status_display or "").lower().startswith("accepted")
                runtime_ms = _parse_ms(data.get("status_runtime") or data.get("runtime"))
                memory_mb = _parse_number(data.get("status_memory") or data.get("memory"))

                result = {
                    "status": status_display,
                    "aceito": bool(is_accepted),
                    "tempo_ms": runtime_ms,
                    "memoria_mb": memory_mb,
                    "compilou": bool(data.get("run_success")),
                    "testes_passados": data.get("total_correct"),
                    "total_testes": data.get("total_testcases"),
                    "percentil_tempo": data.get("runtime_percentile"),
                    "percentil_memoria": data.get("memory_percentile"),
                    "submission_id": data.get("submission_id") or data.get("submissionId") or sub_id,
                    "lang_pretty": data.get("pretty_lang"),
                }
                if not is_accepted:
                    result.update({
                        "erro_compilacao": data.get("full_compile_error") or data.get("compile_error"),
                        "erro_runtime": data.get("full_runtime_error") or data.get("runtime_error"),
                        "ultimo_caso_teste": data.get("last_testcase"),
                        "saida_esperada": data.get("expected_output"),
                        "sua_saida": data.get("code_output"),
                        "input": data.get("input"),
                    })
                return result

            if state in {"STARTED", "PENDING"}:
                time.sleep(backoff)
                backoff = min(5.0, backoff * 1.5)
                continue

            print(f"[WARN] Estado inesperado da submissão: {state} (sub_id={sub_id})")
            break

        except Exception as e:
            print(f"[ERRO] get_submission_result: {e}")
            time.sleep(2)

    return {
        "status": "Timeout", "aceito": False, "tempo_ms": None, "memoria_mb": None,
        "compilou": None, "testes_passados": None, "total_testes": None, "submission_id": sub_id
    }

def get_top_community_code(slug: str, lang: str) -> Optional[str]:
    try:
        url = f"https://leetcode.com/problems/{slug}/solutions/?orderBy=most_votes&languageTags={lang}"
        resp = paced_get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        blocks = [b.get_text() for b in soup.find_all('code') if len(b.get_text(strip=True).splitlines()) > 2]
        return blocks[0] if blocks else None
    except Exception as e:
        print(f"[ERRO] get_top_community_code: {e}")
    return None

# ==============================================================================
# 6) TESTES LOCAIS (stub opcional)
# ==============================================================================
def avaliar_testes_locais(codigo: str, linguagem_dir: str, slug: str) -> bool:
    """
    Conecte aqui seu runner local se quiser usar 'plausible' de verdade
    (passou nos testes do projeto, mas ainda não foi aceito pelo juiz).
    """
    return False

# ==============================================================================
# 7) EXECUÇÃO — Questão -> Modelo -> Linguagens
# ==============================================================================
def _sort_slugish_key(s: str):
    s = s.strip().lower()
    if s.isdigit():
        try:
            return (0, int(s))
        except ValueError:
            pass
    return (1, s)

def resumo_divergencias(linhas: List[dict]):
    diffs = [r for r in linhas if r.get("categoria_ok") is False]
    if not diffs:
        print("[QA] Sem divergências de categoria. KPI verde.")
        return
    from collections import Counter
    tipos = Counter((r.get("categoria_declarada"), r.get("categoria_calculada")) for r in diffs)
    print("[QA] Divergências de categoria:")
    for (decl, calc), cnt in tipos.most_common():
        print(f"  - {cnt}x: declarado={decl} -> calculado={calc}")

def main():
    linhas_resultado: List[dict] = load_existing_results()
    print(f"[OK] Carregados {len(linhas_resultado)} resultados existentes (particionados ou legado).")

    # ---------- PRÉ-SCAN: montar índice {slugish -> {modelo -> {linguagem -> path_json}}} ----------
    if not BASE_PATH.exists():
        print(f"[ERRO] BASE_PATH não existe: {BASE_PATH}")
        return

    modelos_dirs = sorted([p for p in BASE_PATH.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    idx: Dict[str, Dict[str, Dict[str, Path]]] = {}  # slugish -> model -> language -> path

    for modelo_dir in modelos_dirs:
        for lang_dir in sorted([p for p in modelo_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            for path_json in lang_dir.glob("*.json"):
                slugish = extract_slug_from_filename(path_json)
                idx.setdefault(slugish, {}).setdefault(modelo_dir.name, {})[lang_dir.name] = path_json

    if not idx:
        print("[WARN] Nenhuma questão encontrada no diretório base.")
        return

    print(f"[INFO] Total de questões detectadas: {len(idx)}")

    # ---------- LOOP PRINCIPAL: Questão -> Modelo -> Linguagens ----------
    for slugish in sorted(idx.keys(), key=_sort_slugish_key):
        slug_resolvido, qid = resolve_question(slugish)
        print(f"\n{'='*20} Questão: {slugish} -> {slug_resolvido or '??'} {'='*20}")
        if not slug_resolvido or not qid:
            print(f"[FAIL] Não encontrou question_id para o slug/id: {slugish}. Pulando questão.")
            continue

        # union de linguagens presentes em QUALQUER modelo para esta questão (para GT depois)
        linguagens_union = set()
        for model_name, langs_map in idx[slugish].items():
            linguagens_union.update(langs_map.keys())

        # ---- Modelo -> Linguagens para esta questão ----
        for modelo_dir in modelos_dirs:
            model_name = modelo_dir.name
            langs_map = idx[slugish].get(model_name, {})
            if not langs_map:
                continue  # este modelo não tem essa questão em nenhuma linguagem

            print(f"\n--- Modelo: {model_name} ---")

            for linguagem_dir in sorted(langs_map.keys(), key=lambda s: s.lower()):
                lang_api = linguagem_api(linguagem_dir)
                if not lang_api:
                    print(f"[WARN] Linguagem '{linguagem_dir}' não mapeada. Pulando.")
                    continue

                path_json = langs_map[linguagem_dir]

                # carrega JSON da resposta
                try:
                    with open(path_json, encoding="utf-8") as f:
                        dados = json.load(f)
                except Exception as e:
                    print(f"[ERRO] Abrindo JSON {path_json}: {e}")
                    continue

                # filtro por categoria declarada (mantém política original: só processa plausible/correct)
                cat_declarada_raw = dados.get("categoria")
                if not categoria_aceita(cat_declarada_raw):
                    print(f"[INFO] {model_name}/{linguagem_dir}/{slugish}: categoria '{cat_declarada_raw}' não aceita. Pulando.")
                    continue
                cat_declarada = _norm_cat(cat_declarada_raw)

                # evita duplicar
                ja_executado = any(
                    r.get("modelo") == model_name and
                    r.get("slug") == slug_resolvido and
                    _norm_text(r.get("linguagem", "")) == _norm_text(linguagem_dir)
                    for r in linhas_resultado
                )
                if ja_executado:
                    print(f"[INFO] Já existe resultado para {model_name}/{linguagem_dir}/{slug_resolvido}. Pulando.")
                    continue

                # extrai código
                codigo = dados.get("code")
                if not is_valid_code(codigo):
                    codigo = try_extract_code_from_resposta(dados.get("resposta", ""))
                if not is_valid_code(codigo):
                    print(f"[WARN] Sem código válido em {path_json}. Pulando.")
                    continue

                if lang_api == "python3":
                    codigo = ensure_python_solution_wrapper(slug_resolvido, codigo)

                status_original = dados.get("status", "desconhecido")
                formato_original = "ok" if status_original == "ok" else f"formato_invalido ({status_original})"

                # testes locais opcionais
                passou_locais = avaliar_testes_locais(codigo, linguagem_dir, slug_resolvido)

                print(f"Submetendo: {model_name} | {linguagem_dir}({lang_api}) | {slugish} -> {slug_resolvido} | cat_decl={cat_declarada}")
                sub_id = submit_code(slug_resolvido, codigo, lang_api)
                if not sub_id:
                    continue

                result = get_submission_result(slug_resolvido, sub_id)
                result["passou_testes_locais"] = bool(passou_locais)

                cat_calculada = classificar_por_resultado(result)
                categoria_ok = (cat_declarada == cat_calculada) if cat_declarada else False

                novo = {
                    "modelo": model_name,
                    "linguagem": linguagem_dir,
                    "id_questao": dados.get("id"),
                    "slug": slug_resolvido,
                    "categoria_declarada": cat_declarada or cat_declarada_raw,
                    "categoria_calculada": cat_calculada,
                    "categoria_ok": bool(categoria_ok),
                    "formato_original": formato_original,
                    **result,
                    "ground_truth": False,
                }
                linhas_resultado.append(novo)   # mantém em memória para QA
                save_result(novo)               # grava particionado + ndjson + csv

                # pausa entre submissões
                time.sleep(SUBMIT_COOLDOWN)

        # ---- Ground Truth por (questão, linguagem) após processar todos os modelos ----
        for linguagem_dir in sorted(linguagens_union, key=lambda s: s.lower()):
            lang_api = linguagem_api(linguagem_dir)
            if not lang_api:
                continue

            gt_ja_executado = any(
                r.get("modelo") == GROUND_TRUTH_MODEL and
                r.get("slug") == slug_resolvido and
                _norm_text(r.get("linguagem", "")) == _norm_text(linguagem_dir)
                for r in linhas_resultado
            )
            if gt_ja_executado:
                continue

            print(f"\n--- Ground Truth: {linguagem_dir} | {slug_resolvido} ---")
            gt_code = get_top_community_code(slug_resolvido, lang_api)
            if gt_code and is_valid_code(gt_code):
                sub_id_gt = submit_code(slug_resolvido, gt_code, lang_api)
                if sub_id_gt:
                    result_gt = get_submission_result(slug_resolvido, sub_id_gt)
                    cat_gt = classificar_por_resultado(result_gt)
                    novo_gt = {
                        "modelo": GROUND_TRUTH_MODEL,
                        "linguagem": linguagem_dir,
                        "id_questao": None,
                        "slug": slug_resolvido,
                        "categoria_declarada": "ground_truth",
                        "categoria_calculada": cat_gt,
                        "categoria_ok": None,
                        "formato_original": "ok",
                        **result_gt,
                        "ground_truth": True,
                    }
                    linhas_resultado.append(novo_gt)
                    save_result(novo_gt)
                    time.sleep(SUBMIT_COOLDOWN)

    print("\n🏁 Resultados finais salvos.")
    resumo_divergencias(linhas_resultado)

if __name__ == "__main__":
    main()
