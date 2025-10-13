# -*- coding: utf-8 -*-
"""
list_missing_submissions.py — lista APENAS o que falta submeter.

Regras:
- Universo esperado: data/<modelo>/<linguagem>/*.json
- Considera coberto se existir: out/by_model/<modelo>/<linguagem>/<slug>.json
  • Tenta tanto o slug canônico (resolvido via índice) quanto o slugish (nome do arquivo).
- NÃO lê ndjson. Foco em by_model/.
- NÃO depende da rede se existir results.csv ou cache local.
- Relatórios com caminhos RELATIVOS gravados em src/submit_problems/:
  • missing_submissions.csv
  • missing_submissions.txt

Flags por env (opcionais):
  LLM_BASE_PATH, LLM_OUTPUT_DIR
"""

from __future__ import annotations
import os, re, json, csv, sys, unicodedata
from pathlib import Path
from typing import Optional, List, Dict
import requests

# ───────────────────────────── Pastas ─────────────────────────────
SUBMIT_DIR    = Path(__file__).resolve().parent                       # .../src/submit_problems
PROJECT_ROOT  = SUBMIT_DIR.parent.parent
BASE_PATH     = Path(os.getenv("LLM_BASE_PATH", str(PROJECT_ROOT / "data")))
OUTPUT_DIR    = Path(os.getenv("LLM_OUTPUT_DIR", str(PROJECT_ROOT / "out")))
PARTITION_DIR = OUTPUT_DIR / "by_model"
PROBLEMS_CACHE = OUTPUT_DIR / ".cache_problems_index.json"            # cache de problemas
RESULTS_CSV    = OUTPUT_DIR / "results.csv"                           # mapeia id_questao↔slug

# ───────────────────────────── Saídas ─────────────────────────────
CSV_OUT = SUBMIT_DIR / "missing_submissions.csv"
TXT_OUT = SUBMIT_DIR / "missing_submissions.txt"

LANG_MAP = {"python3": "python3", "python": "python3", "c++": "cpp", "cpp": "cpp", "java": "java"}

# ─────────────────────── Utilidades de nome/slug ───────────────────────
def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.casefold().strip()

def linguagem_api(nome_dir: str) -> Optional[str]:
    return LANG_MAP.get(_norm(nome_dir))

def extract_slugish(p: Path) -> str:
    # nome do arquivo sem extensão, espaços viram '-'
    return p.stem.strip().replace(" ", "-")

def _san(s: str) -> str:
    # sanitize leve para nomes de arquivo/dir
    return re.sub(r"[^a-zA-Z0-9._+-]+", "_", s or "")

def relpath(p: Optional[Path]) -> str:
    if not p:
        return ""
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return os.path.relpath(str(p), start=str(PROJECT_ROOT))

# ────────────────────────── Índice de problemas ─────────────────────────
_PROBLEMS_IDX: Dict[str, Dict[str, dict]] = {"by_slug": {}, "by_frontend": {}, "by_qid": {}}

def _load_idx_cache() -> Optional[dict]:
    try:
        if PROBLEMS_CACHE.exists():
            return json.loads(PROBLEMS_CACHE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _save_idx_cache(idx: dict) -> None:
    try:
        PROBLEMS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PROBLEMS_CACHE.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _load_idx_from_results_csv() -> Optional[dict]:
    """
    Lê out/results.csv e monta índice mínimo:
      by_frontend[<id_questao>] -> {"slug": <slug>}
      by_slug[<slug>]            -> {"fid": <id_questao>}
    """
    if not RESULTS_CSV.exists():
        return None
    by_slug, by_frontend, by_qid = {}, {}, {}
    with open(RESULTS_CSV, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            slug = (row.get("slug") or "").strip()
            fid  = (row.get("id_questao") or "").strip()
            if not slug or not fid:
                continue
            by_slug[slug] = {"fid": fid}
            by_frontend[str(fid)] = {"slug": slug}
    return {"by_slug": by_slug, "by_frontend": by_frontend, "by_qid": by_qid}

def _fetch_and_cache_index() -> Optional[dict]:
    """
    Fallback: baixa /api/problems/all/ do LeetCode e grava cache.
    Usado apenas se não existir cache nem results.csv.
    """
    try:
        resp = requests.get(
            "https://leetcode.com/api/problems/all/",
            headers={"user-agent": "Mozilla/5.0"},
            timeout=30
        )
        resp.raise_for_status()
        pairs = resp.json().get("stat_status_pairs", [])
        by_slug, by_frontend, by_qid = {}, {}, {}
        for q in pairs:
            stat = q.get("stat", {}) or {}
            slug = stat.get("question__title_slug")
            qid  = stat.get("question_id")
            fid  = stat.get("frontend_question_id")
            if slug:
                by_slug[str(slug)] = {"qid": qid, "fid": fid}
            if fid is not None:
                by_frontend[str(fid)] = {"slug": slug, "qid": qid}
            if qid is not None:
                by_qid[str(qid)] = {"slug": slug, "fid": fid}
        idx = {"by_slug": by_slug, "by_frontend": by_frontend, "by_qid": by_qid}
        _save_idx_cache(idx)
        return idx
    except Exception:
        return None

def ensure_problems_index() -> None:
    """Carrega índice na ordem: cache -> results.csv -> rede."""
    if _PROBLEMS_IDX["by_slug"]:
        return
    cached = _load_idx_cache()
    if cached:
        _PROBLEMS_IDX.update(cached)
        return
    from_csv = _load_idx_from_results_csv()
    if from_csv:
        _PROBLEMS_IDX.update(from_csv)
        _save_idx_cache(_PROBLEMS_IDX)
        return
    fresh = _fetch_and_cache_index()
    if fresh:
        _PROBLEMS_IDX.update(fresh)

def resolve_canonical_slug(slugish: str) -> str:
    """
    Resolve usando índice local (cache/CSV). Heurística mínima:
    - '1234-algo' -> procura 1234 em by_frontend
    - '1234'      -> idem
    - senão tenta tirar prefixo numérico '1234-'
    """
    s = (slugish or "").strip().lower()
    ensure_problems_index()
    if not s:
        return s

    m = re.match(r"^(\d+)[-_].+$", s)
    if m and _PROBLEMS_IDX["by_frontend"]:
        fid = m.group(1)
        hit = _PROBLEMS_IDX["by_frontend"].get(fid)
        if hit and hit.get("slug"):
            return hit["slug"]

    if s.isdigit() and _PROBLEMS_IDX["by_frontend"]:
        hit = _PROBLEMS_IDX["by_frontend"].get(s)
        if hit and hit.get("slug"):
            return hit["slug"]

    m2 = re.match(r"^\d+-(.+)$", s)
    return m2.group(1) if m2 else s

# ─────────────────────────────── Check ───────────────────────────────
def by_model_exists(modelo: str, linguagem_dir: str, slug: str) -> bool:
    base = PARTITION_DIR / _san(modelo) / _san(linguagem_dir)
    p1 = base / f"{_san(slug)}.json"
    if p1.exists():
        return True
    # Se "slug" for numérico, tenta arquivos "1234-*.json"
    if slug.isdigit() and base.exists():
        try:
            for child in base.iterdir():
                if child.is_file() and child.suffix == ".json" and child.name.startswith(f"{_san(slug)}-"):
                    return True
        except FileNotFoundError:
            return False
    return False

# ─────────────────────────────── Main ───────────────────────────────
def main():
    if not BASE_PATH.exists():
        print(f"[ERRO] BASE_PATH não existe: {BASE_PATH}")
        sys.exit(2)

    missing_rows: List[dict] = []
    total_expected = 0

    modelos = sorted([p for p in BASE_PATH.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    for modelo_dir in modelos:
        for lang_dir in sorted([p for p in modelo_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            lang_api = linguagem_api(lang_dir.name)
            if not lang_api:
                # linguagem desconhecida nos dados? ignore
                continue
            for jf in sorted(lang_dir.glob("*.json"), key=lambda p: p.name):
                total_expected += 1
                slugish = extract_slugish(jf)
                slug_can = resolve_canonical_slug(slugish)

                # existe por canônico?
                ok = by_model_exists(modelo_dir.name, lang_dir.name, slug_can)
                # fallback: existe por slugish?
                if not ok and slugish != slug_can:
                    ok = by_model_exists(modelo_dir.name, lang_dir.name, slugish)

                if not ok:
                    missing_rows.append({
                        "modelo": modelo_dir.name,
                        "linguagem": lang_dir.name,
                        "slug": slug_can,
                        "slugish": slugish,
                        "data_json": relpath(jf),
                        "by_model_expected_can": relpath(PARTITION_DIR / _san(modelo_dir.name) / _san(lang_dir.name) / f"{_san(slug_can)}.json"),
                        "by_model_expected_alt": relpath(PARTITION_DIR / _san(modelo_dir.name) / _san(lang_dir.name) / f"{_san(slugish)}.json") if slugish != slug_can else "",
                    })

    # Ordena pra facilitar batch submit: por slug, depois modelo, linguagem
    missing_rows.sort(key=lambda r: (r["slug"], r["modelo"].lower(), r["linguagem"].lower()))

    # CSV
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    headers = ["modelo","linguagem","slug","slugish","data_json","by_model_expected_can","by_model_expected_alt"]
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in missing_rows:
            w.writerow({h: r.get(h, "") for h in headers})

    # TXT (uma linha por faltante: modelo;linguagem;slug;data_json)
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        for r in missing_rows:
            f.write(f"{r['modelo']};{r['linguagem']};{r['slug']};{r['data_json']}\n")

    # STDOUT resumido
    print(f"Esperados: {total_expected}")
    print(f"Faltando : {len(missing_rows)}")
    print(f"CSV: {relpath(CSV_OUT)}")
    print(f"TXT: {relpath(TXT_OUT)}")

    # Mostra uma pequena amostra no console
    for r in missing_rows[:25]:
        print(f"- {r['slug']} | {r['modelo']} | {r['linguagem']} -> {r['by_model_expected_can']}")

    sys.exit(0 if len(missing_rows)==0 else 1)

if __name__ == "__main__":
    main()
