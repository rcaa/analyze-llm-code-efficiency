# -*- coding: utf-8 -*-
"""
Submete respostas de LLM ao LeetCode e coleta resultados.

Versão anti-403 com login automático:
- Login automático com usuário/senha:
    * Primeiro tenta Playwright (simula navegador e vence WAF)
    * Se indisponível, tenta login por requests
- Coleta e injeta automaticamente o document.cookie no Session
- Handshake em /problems/<slug>/ (sem GET em /submit/ para evitar 405)
- Alternância automática de Referer
- Fallback 1: POST application/x-www-form-urlencoded com csrfmiddlewaretoken no corpo
- Fallback 2 (opcional): Playwright "como navegador" com fetch a partir da página base
- UA e cabeçalhos de navegador
- Cookies via env (LEETCODE_SESSION, csrftoken, cf_clearance, __cf_bm) e/ou LC_COOKIE_JAR/LC_COOKIE_FILE

Ambiente esperado para login automático:
  LC_LOGIN_USER=<seu_email_ou_username>
  LC_LOGIN_PASS=<sua_senha>
  (opcional) LC_AUTO_LOGIN=1 | 0           (default: 1 se user+pass existirem)
  (opcional) LC_LOGIN_MODE=auto|playwright|requests   (default: auto)
  (opcional) LC_PW_BROWSER=chromium|firefox|webkit    (default: chromium)
  (opcional) LC_PW_HEADLESS=1|0                       (default: 1)
  (opcional) LC_PW_WS_ENDPOINT=<ws://...>             (default: vazio)
  (opcional) LC_SAVE_COOKIE_FILE=./cookies.txt        (salva cookies após login)
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
# 1) CONFIG
# ==============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

BASE_PATH  = Path(os.getenv("LLM_BASE_PATH", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.getenv("LLM_OUTPUT_DIR", PROJECT_ROOT / "out"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH  = Path(os.getenv("LLM_CSV_PATH", OUTPUT_DIR / "results.csv"))
JSON_PATH = Path(os.getenv("LLM_JSON_PATH", OUTPUT_DIR / "resultado_leetcode.json"))

GROUND_TRUTH_MODEL = "ground_truth"

LINGUAGEM_MAP = {"python3": "python3", "python": "python3", "c++": "cpp", "cpp": "cpp", "java": "java"}

REQUEST_TIMEOUT   = float(os.getenv("LC_REQUEST_TIMEOUT",   15))
REQUEST_COOLDOWN  = float(os.getenv("LC_REQUEST_COOLDOWN",  0.5))
SUBMIT_COOLDOWN   = float(os.getenv("LC_SUBMIT_COOLDOWN",   8.0))
MAX_POLL_TIME     = float(os.getenv("LC_MAX_POLL_TIME",     80))
RETRY_TOTAL       = int(os.getenv("LC_RETRY_TOTAL",         5))
RETRY_BACKOFF     = float(os.getenv("LC_RETRY_BACKOFF",     1.0))
SUBMIT_RETRIES    = int(os.getenv("LC_SUBMIT_RETRIES",      6))

DEFAULT_UA = os.getenv(
    "LC_UA",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)

BASE_HEADERS = {
    "user-agent": DEFAULT_UA,
    "accept": "application/json, text/javascript, */*; q=0.01",
    "accept-language": "en-US,en;q=0.9,pt-BR;q=0.8",
    "origin": "https://leetcode.com",
    "x-requested-with": "XMLHttpRequest",
    "content-type": "application/json; charset=UTF-8",
    "sec-fetch-site": "same-origin",
    "sec-fetch-mode": "cors",
    "sec-fetch-dest": "empty",
    "sec-ch-ua": '"Chromium";v="141", "Not=A?Brand";v="99"',
    "sec-ch-ua-platform": '"Windows"',
    "sec-ch-ua-mobile": "?0",
}

# --- Login automático / Playwright ---
def _env_truthy(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}

# Mantidos fictícios como você pediu
LC_LOGIN_USER    = "analize"
LC_LOGIN_PASS    = "Khy33d99@Ade"
LC_AUTO_LOGIN    = _env_truthy("LC_AUTO_LOGIN") or (bool(LC_LOGIN_USER) and bool(LC_LOGIN_PASS) and os.getenv("LC_AUTO_LOGIN") is None)
LC_LOGIN_MODE    = (os.getenv("LC_LOGIN_MODE", "auto").strip().lower() or "auto")  # auto|playwright|requests
SAVE_COOKIE_FILE = os.getenv("LC_SAVE_COOKIE_FILE", "").strip()

USE_PLAYWRIGHT   = _env_truthy("LC_USE_PLAYWRIGHT")  # para fallback de submissão
PW_BROWSER       = os.getenv("LC_PW_BROWSER", "chromium").strip().lower()  # chromium|firefox|webkit
PW_HEADLESS      = _env_truthy("LC_PW_HEADLESS") or os.getenv("LC_PW_HEADLESS") is None  # default True
PW_WS_ENDPOINT   = os.getenv("LC_PW_WS_ENDPOINT", "").strip()  # opcional: conectar a browser existente

SESSION = requests.Session()
SESSION.headers.update(BASE_HEADERS)

# ------------------------------------------------------------------------------
# Cookies helpers
# ------------------------------------------------------------------------------
def _set_cookie(name: str, value: str, domain: str = ".leetcode.com", path: str = "/"):
    if value:
        SESSION.cookies.set(name, value, domain=domain, path=path)

def get_cookie_value(name: str, domain_hint: str = "leetcode.com") -> Optional[str]:
    cands = [c for c in SESSION.cookies if c.name == name]
    if not cands:
        return None
    def score(c):
        s = 0
        d = (c.domain or "")
        if d.endswith(domain_hint): s += 2
        if d.startswith("."): s += 1
        if (c.path or "/") == "/": s += 1
        return s
    cands.sort(key=score, reverse=True)
    return cands[0].value

def has_cookie(name: str) -> bool:
    return any(c.name == name for c in SESSION.cookies)

def _parse_cookie_header(header: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in header.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k].strip()
        out[k.strip()] = v.strip()
    return out

def _cookies_to_netscape_lines() -> List[str]:
    lines = ["# Netscape HTTP Cookie File"]
    for c in SESSION.cookies:
        domain = c.domain or ".leetcode.com"
        include_sub = "TRUE" if domain.startswith(".") else "FALSE"
        path = c.path or "/"
        secure = "TRUE" if getattr(c, "secure", True) else "FALSE"
        expiry = str(int(time.time()) + 86400 * 30)
        lines.append("\t".join([domain, include_sub, path, secure, expiry, c.name, c.value]))
    return lines

def save_cookies_to_file(path: str):
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(_cookies_to_netscape_lines()), encoding="utf-8")
        print(f"[COOKIES] Salvos em: {p}")
    except Exception as e:
        print(f"[COOKIES] Falha ao salvar cookies: {e}")

def load_cookies_from_env():
    env_cookies = {
        "LEETCODE_SESSION": os.getenv("LEETCODE_SESSION", ""),
        "csrftoken": os.getenv("LC_CSRF", ""),
        "cf_clearance": os.getenv("CF_CLEARANCE", ""),
        "__cf_bm": os.getenv("__CF_BM", "") or os.getenv("CF_BM", ""),
    }
    for k, v in env_cookies.items():
        _set_cookie(k, v)

    jar = os.getenv("LC_COOKIE_JAR", "").strip()
    if jar:
        try:
            for k, v in _parse_cookie_header(jar).items():
                _set_cookie(k, v)
        except Exception:
            pass

    cookie_file = os.getenv("LC_COOKIE_FILE", "").strip()
    if cookie_file:
        try:
            p = Path(cookie_file)
            text = p.read_text(encoding="utf-8") if p.exists() else cookie_file
            if "\t" not in text:
                for k, v in _parse_cookie_header(text).items():
                    _set_cookie(k, v)
            else:
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    cols = line.split("\t")
                    if len(cols) >= 7:
                        domain, _flag, path, _secure, _exp, name, value = cols[:7]
                        _set_cookie(name, value, domain=domain, path=(path or "/"))
        except Exception:
            pass

# ---- Login automático ---------------------------------------------------------
def _merge_cookies_from_list(cookies: List[dict]):
    for ck in cookies:
        name = ck.get("name"); value = ck.get("value")
        domain = ck.get("domain") or ".leetcode.com"
        if not domain.startswith("."):
            domain = "." + domain
        path = ck.get("path") or "/"
        if name and value:
            _set_cookie(name, value, domain=domain, path=path)

def login_with_playwright(user: str, password: str) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print(f"[LOGIN/PW] Playwright indisponível: {e}")
        return False

    try:
        print(f"[LOGIN/PW] Iniciando navegador ({PW_BROWSER}, headless={PW_HEADLESS})…")
        with sync_playwright() as p:
            if PW_WS_ENDPOINT:
                browser = p.chromium.connect_over_cdp(PW_WS_ENDPOINT)
            else:
                btype = {"chromium": p.chromium, "firefox": p.firefox, "webkit": p.webkit}.get(PW_BROWSER, p.chromium)
                browser = btype.launch(headless=PW_HEADLESS, args=["--disable-blink-features=AutomationControlled"])
            context = browser.new_context()
            page = context.new_page()

            login_url = "https://leetcode.com/accounts/login/"
            page.goto(login_url, wait_until="domcontentloaded", timeout=60000)

            for sel in [
                'button:has-text("Accept All")',
                'button:has-text("Accept all")',
                'button:has-text("Aceitar todos")',
                'button:has-text("Agree")',
            ]:
                try:
                    if page.locator(sel).first.is_visible():
                        page.locator(sel).first.click(timeout=2000)
                        break
                except Exception:
                    pass

            user_selectors = ['input[name="login"]', 'input[name="username"]', 'input[type="email"]']
            pass_selectors = ['input[name="password"]', 'input[type="password"]']
            submit_selectors = ['button[type="submit"]', 'button:has-text("Sign in")', 'button:has-text("Log in")']

            filled = False
            for us in user_selectors:
                try:
                    page.fill(us, user, timeout=8000)
                    filled = True
                    break
                except Exception:
                    continue
            if not filled:
                print("[LOGIN/PW] Campo de usuário não encontrado.")
                context.close(); 
                if not PW_WS_ENDPOINT: browser.close()
                return False

            filled = False
            for ps in pass_selectors:
                try:
                    page.fill(ps, password, timeout=8000)
                    filled = True
                    break
                except Exception:
                    continue
            if not filled:
                print("[LOGIN/PW] Campo de senha não encontrado.")
                context.close(); 
                if not PW_WS_ENDPOINT: browser.close()
                return False

            clicked = False
            for bs in submit_selectors:
                try:
                    page.click(bs, timeout=8000)
                    clicked = True
                    break
                except Exception:
                    continue
            if not clicked:
                print("[LOGIN/PW] Botão de submit não encontrado.")
                context.close(); 
                if not PW_WS_ENDPOINT: browser.close()
                return False

            page.wait_for_load_state("domcontentloaded", timeout=45000)
            time.sleep(2.0)

            ck_after = context.cookies()
            _merge_cookies_from_list(ck_after)

            has_session = has_cookie("LEETCODE_SESSION")
            has_csrf = has_cookie("csrftoken")
            print(f"[LOGIN/PW] Cookies coletados. SESSION={has_session}, CSRF={has_csrf}")

            try:
                ua = page.evaluate("navigator.userAgent")
                if ua:
                    SESSION.headers["user-agent"] = ua
            except Exception:
                pass

            if SAVE_COOKIE_FILE:
                save_cookies_to_file(SAVE_COOKIE_FILE)

            context.close()
            if not PW_WS_ENDPOINT:
                browser.close()
            return bool(has_session and has_csrf)
    except Exception as e:
        print(f"[LOGIN/PW] Erro: {e}")
        return False

def login_with_requests(user: str, password: str) -> bool:
    try:
        login_url = "https://leetcode.com/accounts/login/"
        r1 = requests.get(login_url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        for c in r1.cookies:
            if c.domain.endswith("leetcode.com"):
                _set_cookie(c.name, c.value, domain=c.domain, path=c.path or "/")

        csrf = get_cookie_value("csrftoken")
        if not csrf:
            print("[LOGIN/REQ] Sem csrftoken após GET inicial.")
            return False

        headers = BASE_HEADERS.copy()
        headers["referer"] = login_url
        headers["content-type"] = "application/x-www-form-urlencoded"
        headers["x-csrftoken"] = csrf
        headers["X-CSRFToken"] = csrf

        form = {
            "login": user,
            "password": password,
            "csrfmiddlewaretoken": csrf,
            "next": "/",
        }
        r2 = SESSION.post(login_url, data=form, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        for c in r2.cookies:
            if c.domain.endswith("leetcode.com"):
                _set_cookie(c.name, c.value, domain=c.domain, path=c.path or "/")

        has_session = has_cookie("LEETCODE_SESSION")
        has_csrf = has_cookie("csrftoken")
        print(f"[LOGIN/REQ] STATUS={r2.status_code} SESSION={has_session}, CSRF={has_csrf}")

        if SAVE_COOKIE_FILE:
            save_cookies_to_file(SAVE_COOKIE_FILE)

        return bool(has_session and has_csrf)
    except Exception as e:
        print(f"[LOGIN/REQ] Erro: {e}")
        return False

def ensure_login_cookies():
    if has_cookie("LEETCODE_SESSION") and has_cookie("csrftoken"):
        return
    if not LC_AUTO_LOGIN:
        print("[LOGIN] Auto-login desabilitado ou credenciais ausentes.")
        return
    if not (LC_LOGIN_USER and LC_LOGIN_PASS):
        print("[LOGIN] Credenciais não fornecidas (LC_LOGIN_USER/LC_LOGIN_PASS).")
        return

    mode = LC_LOGIN_MODE
    ok = False
    if mode in {"auto", "playwright"}:
        ok = login_with_playwright(LC_LOGIN_USER, LC_LOGIN_PASS)
        if not ok and mode == "auto":
            print("[LOGIN] Playwright falhou; tentando login via requests…")
            ok = login_with_requests(LC_LOGIN_USER, LC_LOGIN_PASS)
    elif mode == "requests":
        ok = login_with_requests(LC_LOGIN_USER, LC_LOGIN_PASS)

    if not ok:
        print("[LOGIN] Falha ao autenticar. Verifique usuário, senha e WAF.")

# ------------------------------------------------------------------------------
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

# Pace HTTP
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

def refresh_csrf_from_session():
    tok = get_cookie_value("csrftoken")
    if tok:
        SESSION.headers["x-csrftoken"] = tok
        SESSION.headers["X-CSRFToken"] = tok

def preflight_handshake(slug: str, tries: int = 2) -> bool:
    """
    Só abre a página base do problema. Nada de GET em /submit/ (gera 405).
    """
    base = f"https://leetcode.com/problems/{slug}/"
    ok = False
    for _ in range(tries):
        try:
            r1 = paced_get(base)
            refresh_csrf_from_session()
            ok = r1.status_code in (200, 302, 403)
            SESSION.headers["referer"] = base
            if ok:
                break
        except Exception:
            time.sleep(0.6)
    SESSION.headers["_lc_ref_base"] = base
    SESSION.headers["_lc_ref_submit"] = f"{base}submit/"
    return ok

# ------------------------------------------------------------------------------
# Índice de questões
# ------------------------------------------------------------------------------
_PROBLEMS_IDX: Dict[str, Dict] = {"by_slug": {}, "by_frontend": {}, "by_qid": {}}

def ensure_problems_index():
    if _PROBLEMS_IDX["by_slug"]:
        return
    try:
        resp = paced_get("https://leetcode.com/api/problems/all/")
        resp.raise_for_status()
        pairs = resp.json().get("stat_status_pairs", [])
        by_slug, by_frontend, by_qid = {}, {}, {}
        for q in pairs:
            stat = q.get("stat", {}) or {}
            slug = stat.get("question__title_slug")
            qid = stat.get("question_id")
            fid = stat.get("frontend_question_id")
            if slug:
                by_slug[str(slug)] = {"qid": qid, "fid": fid}
            if fid is not None:
                by_frontend[str(fid)] = {"slug": slug, "qid": qid}
            if qid is not None:
                by_qid[str(qid)] = {"slug": slug, "fid": fid}
        _PROBLEMS_IDX.update(by_slug=by_slug, by_frontend=by_frontend, by_qid=by_qid)
        print(f"[OK] Índice carregado: {len(by_slug)} questões.")
    except Exception as e:
        print(f"[ERRO] Carregando índice de questões: {e}")

def resolve_question(slugish: str) -> Tuple[Optional[str], Optional[str]]:
    ensure_problems_index()
    if not slugish:
        return None, None
    s = slugish.strip().lower()
    m = re.match(r"^(\d+)[-_].+$", s)
    if m:
        fid = m.group(1)
        hit = _PROBLEMS_IDX["by_frontend"].get(fid)
        if hit:
            return hit["slug"], str(hit["qid"])
    if s.isdigit():
        hit = _PROBLEMS_IDX["by_frontend"].get(s)
        if hit:
            return hit["slug"], str(hit["qid"])
        hit2 = _PROBLEMS_IDX["by_qid"].get(s)
        if hit2:
            return hit2["slug"], s
        return None, None
    hit = _PROBLEMS_IDX["by_slug"].get(s)
    if hit:
        return s, str(hit["qid"])
    return None, None

def get_question_id(slug_or_id: str) -> Optional[str]:
    _, qid = resolve_question(slug_or_id)
    return qid

# ==============================================================================
# 2) CATEGORIAS
# ==============================================================================
CATS = {
    "invalid": {"invalid", "invalida", "inválida"},
    "semantic_incorrect": {"semantic_incorrect", "semantically_incorrect", "semantica_incorreta", "semântica_incorreta"},
    "plausible": {"plausible", "plausivel", "plausível"},
    "actual_correct": {"actual_correct", "correct", "correto", "correta"},
}
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
        if t in {_norm_text(a) for a in aliases}:
            return k
    return None

def categoria_aceita(categoria: Optional[str]) -> bool:
    canon = _norm_cat(categoria)
    return canon in CATEGORIAS_ACEITAS_CANON

def classificar_por_resultado(result: dict) -> str:
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
    p = Path(path); return p.stem.strip().replace(" ", "-")

def _parse_number(s):
    if s is None: return None
    m = re.search(r'[\d.]+', str(s))
    return float(m.group(0)) if m else None

def _parse_ms(s):
    v = _parse_number(s)
    return int(round(v)) if v is not None else None

def is_valid_code(code: str) -> bool:
    return bool(code and code.strip())

def _ensure_indented_body(code: str) -> str:
    """
    Se existir 'def ...:' sem bloco, injeta 'pass' para evitar IndentationError.
    """
    lines = code.splitlines()
    out = []
    pending = False
    for i, line in enumerate(lines):
        out.append(line)
        if re.match(r"^\s*def\s+\w+\(.*\):\s*$", line):
            pending = True
            continue
        if pending:
            if line.strip() == "" or not line.startswith((" ", "\t")):
                # não veio bloco indentado logo depois
                out.insert(len(out)-1, "    pass")
                pending = False
            else:
                pending = False
    if pending:
        out.append("    pass")
    return "\n".join(out) + ("\n" if not code.endswith("\n") else "")

def ensure_python_solution_wrapper(slug: str, codigo: str) -> str:
    code = codigo.rstrip("\n") + "\n"
    if "class Solution" not in code:
        body = "\n".join(("    " + line if line.strip() else "") for line in code.splitlines())
        code = "class Solution:\n" + body + ("\n" if not body.endswith("\n") else "")
    # reforço contra 'def ...:' sem corpo
    code = _ensure_indented_body(code)
    return code

def try_extract_code_from_resposta(resposta: str) -> Optional[str]:
    if not resposta: return None
    m = re.search(r"```(?:[a-zA-Z0-9+#]*)\s*([\s\S]*?)```", resposta)
    return m.group(1).strip() if m else None

# ==============================================================================
# 3.1) Saída particionada + NDJSON + CSV
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
    p = _partition_path(row.get("modelo"), row.get("linguagem"), row.get("slug"))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)

def append_ndjson(row: dict):
    fname = f"{_san(row.get('modelo'))}__{_san(row.get('linguagem'))}.ndjson"
    path = NDJSON_DIR / fname
    row_with_ts = dict(row); row_with_ts.setdefault("ts_epoch", int(time.time()))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row_with_ts, ensure_ascii=False) + "\n")

_CSV_COLS = ["modelo","linguagem","slug","id_questao","categoria_declarada",
             "categoria_calculada","categoria_ok","status","aceito",
             "tempo_ms","memoria_mb","submission_id","ground_truth"]
def append_csv(row: dict):
    import csv
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        if write_header: w.writeheader()
        w.writerow(row)

def save_result(row: dict):
    save_result_partitioned(row); append_ndjson(row); append_csv(row)

def load_existing_results() -> List[dict]:
    if JSON_PATH.exists():
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    linhas: List[dict] = []
    if PARTITION_DIR.exists():
        for model_dir in PARTITION_DIR.iterdir():
            if not model_dir.is_dir(): continue
            for lang_dir in model_dir.iterdir():
                if not lang_dir.is_dir(): continue
                for jf in lang_dir.glob("*.json"):
                    try:
                        with open(jf, "r", encoding="utf-8") as f:
                            linhas.append(json.load(f))
                    except Exception:
                        continue
    return linhas

# ==============================================================================
# 4) Submissões via requests + fallbacks
# ==============================================================================
def _submit_json(url: str, payload: dict, headers: dict):
    headers = headers.copy()
    headers["content-type"] = "application/json; charset=UTF-8"
    return paced_post(url, headers=headers, json=payload)

def _submit_form(url: str, payload: dict, headers: dict):
    headers = headers.copy()
    csrf = get_cookie_value("csrftoken") or headers.get("x-csrftoken") or headers.get("X-CSRFToken")
    form = {
        "lang": payload.get("lang"),
        "question_id": str(payload.get("question_id")),
        "typed_code": payload.get("typed_code"),
        "csrfmiddlewaretoken": csrf or "",
    }
    headers["content-type"] = "application/x-www-form-urlencoded; charset=UTF-8"
    return paced_post(url, headers=headers, data=form)

# ---------- Playwright helpers ----------
def _cookies_for_playwright() -> List[dict]:
    cookies_map: Dict[str, dict] = {}
    for c in SESSION.cookies:
        if not (c.domain and c.value):
            continue
        if not c.domain.endswith("leetcode.com"):
            continue
        key = (c.name, c.domain, c.path or "/")
        cookies_map[key] = {
            "name": c.name,
            "value": c.value,
            "domain": c.domain if c.domain.startswith(".") else f".{c.domain}",
            "path": c.path or "/",
            "httpOnly": False,
            "secure": True,
            "sameSite": "Lax",
        }
    jar = os.getenv("LC_COOKIE_JAR", "").strip()
    if jar:
        for k, v in _parse_cookie_header(jar).items():
            key = (k, ".leetcode.com", "/")
            cookies_map[key] = {
                "name": k, "value": v,
                "domain": ".leetcode.com", "path": "/",
                "httpOnly": False, "secure": True, "sameSite": "Lax",
            }
    return list(cookies_map.values())

def playwright_submit_code(slug_or_id: str, code: str, lang: str) -> Optional[int]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print(f"[PW] Playwright indisponível: {e}")
        return None

    slug, qid = resolve_question(slug_or_id)
    if not slug or not qid:
        print(f"[PW] Não encontrou question_id/slug para: {slug_or_id}")
        return None

    cookies = _cookies_for_playwright()
    if not cookies:
        print("[PW] Sem cookies válidos para contexto. Faça login automático ou configure LC_COOKIE_JAR.")
        return None

    print(f"[PW] Abrindo navegador ({PW_BROWSER}, headless={PW_HEADLESS}) e carregando cookies…")
    try:
        with sync_playwright() as p:
            if PW_WS_ENDPOINT:
                browser = p.chromium.connect_over_cdp(PW_WS_ENDPOINT)
            else:
                btype = {"chromium": p.chromium, "firefox": p.firefox, "webkit": p.webkit}.get(PW_BROWSER, p.chromium)
                browser = btype.launch(headless=PW_HEADLESS, args=["--disable-blink-features=AutomationControlled"])
            context = browser.new_context()
            context.add_cookies(cookies)
            page = context.new_page()

            # Ir para a PÁGINA BASE do problema (evita 405)
            base_url = f"https://leetcode.com/problems/{slug}/"
            page.goto(base_url, wait_until="domcontentloaded", timeout=45000)

            token = None
            try:
                ck = page.evaluate("document.cookie")
                token = re.search(r"csrftoken=([^;]+)", ck).group(1) if "csrftoken=" in ck else None
            except Exception:
                token = None
            if not token:
                token = next((c["value"] for c in cookies if c["name"] == "csrftoken"), None)

            if not token:
                print("[PW] Não conseguiu determinar csrftoken. Abortando fallback.")
                context.close()
                if not PW_WS_ENDPOINT: browser.close()
                return None

            print("[PW] Enviando fetch('/submit/') a partir da página base…")
            payload = {"lang": lang, "question_id": qid, "typed_code": code}

            js = """
                async ({slug, payload, token}) => {
                  const url = `/problems/${slug}/submit/`;
                  const resp = await fetch(url, {
                    method: 'POST',
                    headers: {
                      'content-type': 'application/json; charset=UTF-8',
                      'x-csrftoken': token,
                      'X-CSRFToken': token,
                      'x-requested-with': 'XMLHttpRequest'
                    },
                    body: JSON.stringify(payload),
                    credentials: 'same-origin'
                  });
                  let text = await resp.text();
                  try { return {status: resp.status, json: JSON.parse(text)}; }
                  catch { return {status: resp.status, html: text.slice(0, 5000)}; }
                }
            """
            out = page.evaluate(js, {"slug": slug, "payload": payload, "token": token})
            sub_id = None
            if out and isinstance(out, dict) and "json" in out and out["json"]:
                sub_id = out["json"].get("submission_id") or out["json"].get("submissionId")

            if not sub_id:
                js2 = """
                    async ({slug, lang}) => {
                      const api = `/api/submissions/${slug}/?offset=0&limit=20`;
                      const r = await fetch(api, {credentials: 'same-origin'});
                      const j = await r.json();
                      const arr = (j.submissions_dump || []).sort((a,b)=>(b.id||0)-(a.id||0));
                      for (const s of arr) { if (s.lang === lang) return s.id; }
                      return null;
                    }
                """
                sub_id = page.evaluate(js2, {"slug": slug, "lang": lang})

            context.close()
            if not PW_WS_ENDPOINT:
                browser.close()

            if sub_id:
                print(f"[PW] Submetido com sucesso via Playwright: submission_id={sub_id}")
                return int(sub_id)

            print("[PW] Falha em obter submission_id mesmo via página.")
            return None
    except Exception as e:
        print(f"[PW] Erro no fallback Playwright: {e}")
        return None

def submit_code(slug_or_id: str, code: str, lang: str) -> Optional[int]:
    slug, qid = resolve_question(slug_or_id)
    if not slug or not qid:
        print(f"[FAIL] Não encontrou question_id/slug para: {slug_or_id}")
        return None

    ensure_login_cookies()
    preflight_handshake(slug)

    ua_env = os.getenv("LC_UA")
    if ua_env: SESSION.headers["user-agent"] = ua_env
    for ck in ("cf_clearance", "__cf_bm"):
        v = os.getenv("CF_CLEARANCE") if ck == "cf_clearance" else (os.getenv("__CF_BM") or os.getenv("CF_BM"))
        if v: _set_cookie(ck, v)

    payload = {"lang": lang, "question_id": qid, "typed_code": code}
    url = f"https://leetcode.com/problems/{slug}/submit/"

    refresh_csrf_from_session()
    headers = SESSION.headers.copy()

    dbg = {
        "ua": headers.get("user-agent"),
        "has_session": has_cookie("LEETCODE_SESSION"),
        "has_csrf": has_cookie("csrftoken"),
        "has_clearance": has_cookie("cf_clearance"),
        "has_cf_bm": has_cookie("__cf_bm"),
        "referer": headers.get("referer"),
    }
    print(f"[DEBUG] submit ctx {slug}: {dbg}")

    referers = [headers.get("_lc_ref_base")]  # usamos só a base para evitar 405
    if not any(referers):
        referers = [f"https://leetcode.com/problems/{slug}/"]

    max_tries = SUBMIT_RETRIES
    backoff = 2.0
    ref_idx = 0
    used_form_fallback = False
    waf_hits = 0

    for attempt in range(max_tries + 1):
        try:
            headers["referer"] = referers[ref_idx % len(referers)]
            resp = _submit_json(url, payload, headers)

            if resp.status_code == 200:
                data = {}
                try:
                    data = resp.json()
                except Exception:
                    pass
                sub_id = data.get("submission_id") or data.get("submissionId")
                if sub_id:
                    print(f"[OK] Submetido: {slug} (qid={qid}, {lang}) -> submission_id={sub_id}")
                    return int(sub_id)
                print("[WARN] submit sem submission_id; tentando fallback…")
                return get_last_submission_id(slug, lang)

            if resp.status_code == 403 and ("just a moment" in resp.text[:800].lower() or "cloudflare" in resp.text.lower() or "forbidden" in resp.text.lower()):
                waf_hits += 1
                print(f"[WAF] 403 para {slug}. Re-handshake. Backoff {backoff:.1f}s (tentativa {attempt+1}/{max_tries})")
                preflight_handshake(slug)
                refresh_csrf_from_session()
                time.sleep(backoff)
                backoff = min(30.0, backoff * 1.8)

                if attempt >= max_tries // 2 and not used_form_fallback:
                    used_form_fallback = True
                    headers["referer"] = referers[ref_idx % len(referers)]
                    print("[FALLBACK] Tentando submit como form x-www-form-urlencoded…")
                    resp2 = _submit_form(url, payload, headers)
                    if resp2.status_code == 200:
                        try:
                            data = resp2.json()
                        except Exception:
                            data = {}
                        sub_id = data.get("submission_id") or data.get("submissionId")
                        if sub_id:
                            print(f"[OK] Submetido (form): {slug} -> submission_id={sub_id}")
                            return int(sub_id)
                        sid = get_last_submission_id(slug, lang)
                        if sid:
                            print(f"[OK] Submetido (form, via listagem): sub_id={sid}")
                            return int(sid)
                        print(f"[WARN] Form retornou {resp2.status_code}, sem id. Continuando backoff.")

                if (USE_PLAYWRIGHT or LC_LOGIN_MODE in {"auto", "playwright"}) and waf_hits >= 2:
                    print("[WAF->PW] Ativando fallback Playwright por WAF persistente…")
                    sid = playwright_submit_code(slug, code, lang)
                    if sid:
                        return int(sid)
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                ra = resp.headers.get("Retry-After")
                wait_s = float(ra) if (ra and ra.isdigit()) else max(3.0, backoff)
                print(f"[RATE] {resp.status_code} no submit; aguardando {wait_s:.1f}s (tentativa {attempt+1}/{max_tries})")
                time.sleep(wait_s)
                backoff = min(30.0, backoff * 1.6)
                continue

            print(f"[FAIL] Submissão: {resp.status_code} - {resp.text[:240]}")
            break

        except Exception as e:
            print(f"[ERRO] Submissão (tentativa {attempt+1}): {e}")
            time.sleep(backoff)
            backoff = min(30.0, backoff * 1.6)

    if (USE_PLAYWRIGHT or LC_LOGIN_MODE in {"auto", "playwright"}):
        print("[FAIL->PW] Requests falhou. Tentando Playwright como último recurso…")
        sid = playwright_submit_code(slug, code, lang)
        if sid:
            return int(sid)

    print("[FAIL] Submissão: excedeu tentativas. Se continuar só em poucas slugs, garanta IP/UA iguais ao navegador "
          "e considere passar o document.cookie completo em LC_COOKIE_JAR.")
    return None

def get_last_submission_id(slug: str, lang: str) -> Optional[int]:
    preflight_handshake(slug)
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
                time.sleep(backoff); backoff = min(5.0, backoff * 1.5); continue

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
    if os.getenv("DISABLE_GROUND_TRUTH", "").strip().lower() in {"1", "true", "yes"}:
        print("[INFO] Ground Truth desabilitado por env (DISABLE_GROUND_TRUTH).")
        return None
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
# 6) TESTES LOCAIS (stub)
# ==============================================================================
def avaliar_testes_locais(codigo: str, linguagem_dir: str, slug: str) -> bool:
    return False

# ==============================================================================
# 7) Filtros: LC_ONLY
# ==============================================================================
def parse_only_targets() -> set[str]:
    raw = os.getenv("LC_ONLY", "").strip()
    if not raw: return set()
    parts = re.split(r"[,\s]+", raw)
    targets = set()
    for p in parts:
        p = p.strip()
        if not p: continue
        slug, _ = resolve_question(p)
        if slug: targets.add(slug)
    return targets

# ==============================================================================
# 8) EXECUÇÃO
# ==============================================================================
def _sort_slugish_key(s: str):
    s = s.strip().lower()
    if s.isdigit():
        try: return (0, int(s))
        except ValueError: pass
    return (1, s)

def resumo_divergencias(linhas: List[dict]):
    diffs = [r for r in linhas if r.get("categoria_ok") is False]
    if not diffs:
        print("[QA] Sem divergências de categoria. KPI verde."); return
    from collections import Counter
    tipos = Counter((r.get("categoria_declarada"), r.get("categoria_calculada")) for r in diffs)
    print("[QA] Divergências de categoria:")
    for (decl, calc), cnt in tipos.most_common():
        print(f"  - {cnt}x: declarado={decl} -> calculado={calc}")

def main():
    load_cookies_from_env()
    if LC_AUTO_LOGIN:
        ensure_login_cookies()

    linhas_resultado: List[dict] = load_existing_results()
    print(f"[OK] Carregados {len(linhas_resultado)} resultados existentes (particionados ou legado).")

    if not BASE_PATH.exists():
        print(f"[ERRO] BASE_PATH não existe: {BASE_PATH}"); return

    modelos_dirs = sorted([p for p in BASE_PATH.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    idx: Dict[str, Dict[str, Dict[str, Path]]] = {}

    for modelo_dir in modelos_dirs:
        for lang_dir in sorted([p for p in modelo_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            for path_json in lang_dir.glob("*.json"):
                slugish = extract_slug_from_filename(path_json)
                idx.setdefault(slugish, {}).setdefault(modelo_dir.name, {})[lang_dir.name] = path_json

    if not idx:
        print("[WARN] Nenhuma questão encontrada no diretório base."); return

    print(f"[INFO] Total de questões detectadas: {len(idx)}")

    only_targets = parse_only_targets()
    if only_targets: print(f"[INFO] Filtro LC_ONLY ativo: {sorted(only_targets)}")

    for slugish in sorted(idx.keys(), key=_sort_slugish_key):
        slug_resolvido, qid = resolve_question(slugish)
        if not slug_resolvido or not qid:
            print(f"[FAIL] Não encontrou question_id para o slug/id: {slugish}. Pulando questão.")
            continue
        if only_targets and slug_resolvido not in only_targets:
            continue

        print(f"\n{'='*20} Questão: {slugish} -> {slug_resolvido} {'='*20}")

        linguagens_union = set()
        for _, langs_map in idx[slugish].items():
            linguagens_union.update(langs_map.keys())

        for modelo_dir in modelos_dirs:
            model_name = modelo_dir.name
            langs_map = idx[slugish].get(model_name, {})
            if not langs_map: continue
            print(f"\n--- Modelo: {model_name} ---")

            for linguagem_dir in sorted(langs_map.keys(), key=lambda s: s.lower()):
                lang_api = linguagem_api(linguagem_dir)
                if not lang_api:
                    print(f"[WARN] Linguagem '{linguagem_dir}' não mapeada. Pulando."); continue

                path_json = langs_map[linguagem_dir]
                try:
                    with open(path_json, encoding="utf-8") as f:
                        dados = json.load(f)
                except Exception as e:
                    print(f"[ERRO] Abrindo JSON {path_json}: {e}"); continue

                cat_declarada_raw = dados.get("categoria")
                if not categoria_aceita(cat_declarada_raw):
                    print(f"[INFO] {model_name}/{linguagem_dir}/{slugish}: categoria '{cat_declarada_raw}' não aceita. Pulando.")
                    continue
                cat_declarada = _norm_cat(cat_declarada_raw)

                ja_executado = any(
                    r.get("modelo") == model_name and
                    r.get("slug") == slug_resolvido and
                    _norm_text(r.get("linguagem", "")) == _norm_text(linguagem_dir)
                    for r in linhas_resultado
                )
                if ja_executado:
                    print(f"[INFO] Já existe resultado para {model_name}/{linguagem_dir}/{slug_resolvido}. Pulando.")
                    continue

                codigo = dados.get("code") or try_extract_code_from_resposta(dados.get("resposta", ""))
                if not is_valid_code(codigo):
                    print(f"[WARN] Sem código válido em {path_json}. Pulando."); continue
                if lang_api == "python3":
                    codigo = ensure_python_solution_wrapper(slug_resolvido, codigo)

                status_original = dados.get("status", "desconhecido")
                formato_original = "ok" if status_original == "ok" else f"formato_invalido ({status_original})"

                passou_locais = avaliar_testes_locais(codigo, linguagem_dir, slug_resolvido)

                print(f"Submetendo: {model_name} | {linguagem_dir}({lang_api}) | {slugish} -> {slug_resolvido} | cat_decl={cat_declarada}")
                sub_id = submit_code(slug_resolvido, codigo, lang_api)
                if not sub_id: continue

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
                linhas_resultado.append(novo)
                save_result(novo)
                time.sleep(SUBMIT_COOLDOWN)

        for linguagem_dir in sorted(linguagens_union, key=lambda s: s.lower()):
            lang_api = linguagem_api(linguagem_dir)
            if not lang_api: continue
            gt_ja_executado = any(
                r.get("modelo") == GROUND_TRUTH_MODEL and
                r.get("slug") == slug_resolvido and
                _norm_text(r.get("linguagem", "")) == _norm_text(linguagem_dir)
                for r in linhas_resultado
            )
            if gt_ja_executado: continue

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
