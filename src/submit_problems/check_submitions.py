# -*- coding: utf-8 -*-
"""
check_submissions.py — v3 (slug canônico + CSV certo + modos de categoria)

Correções:
- Resolve slug canônico igual ao seu submit (id → slug oficial), com cache local.
- Tenta path por slug_canônico e, se não existir, tenta slugish do arquivo.
- Lê ambos legados: out/results.csv e out/resultado_leetcode.csv.
- Flag para aceitar qualquer categoria (contagem crua).
- Loga contagem de arquivos em out/by_model por modelo/linguagem antes de rodar.

Env/CLI:
  ONLY_ACCEPTED=true|false
  REQUIRE_GT=true|false
  ACCEPT_ANY_CATEGORY=true|false
  LLM_BASE_PATH, LLM_OUTPUT_DIR, LLM_CSV_PATH, LLM_JSON_PATH

  --only-accepted
  --require-gt
  --accept-any-category
  --log-level [DEBUG|INFO|WARNING|ERROR]
  --log-file PATH ("" desativa arquivo)
  --progress [auto|bar|none]
  --tick N
"""

from __future__ import annotations
import os, re, csv, json, sys, time, unicodedata, argparse, logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ───────────────────────── Caminhos ─────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BASE_PATH    = Path(os.getenv("LLM_BASE_PATH", PROJECT_ROOT / "data"))
OUTPUT_DIR   = Path(os.getenv("LLM_OUTPUT_DIR", PROJECT_ROOT / "out"))
# CSV legado: considerar ambos nomes
CSV_LEGADO_1 = OUTPUT_DIR / "results.csv"
CSV_LEGADO_2 = OUTPUT_DIR / "resultado_leetcode.csv"
JSON_LEGADO  = Path(os.getenv("LLM_JSON_PATH", OUTPUT_DIR / "resultado_leetcode.json"))

PARTITION_DIR = OUTPUT_DIR / "by_model"
NDJSON_DIR    = OUTPUT_DIR / "ndjson"
PROBLEMS_CACHE = OUTPUT_DIR / ".cache_problems_index.json"

ONLY_ACCEPTED_ENV      = str(os.getenv("ONLY_ACCEPTED", "false")).strip().lower() in {"1","true","yes"}
REQUIRE_GT_ENV         = str(os.getenv("REQUIRE_GT", "false")).strip().lower() in {"1","true","yes"}
ACCEPT_ANY_CATEGORY_ENV= str(os.getenv("ACCEPT_ANY_CATEGORY", "false")).strip().lower() in {"1","true","yes"}

# ───────────────────────── Util/log ─────────────────────────
def setup_logging(level: str, log_file: Optional[str]):
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-7s | %(message)s"
    datefmt = "%H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=lvl, format=fmt, datefmt=datefmt, handlers=handlers)

def human_pct(n, d): return f"{(100.0*n/d):.1f}%" if d else "0.0%"

def human_eta(start_ts, done, total):
    if done == 0: return "estimando…"
    sec_per_item = (time.time() - start_ts) / done
    remaining = int(sec_per_item * (total - done))
    if remaining < 60: return f"{remaining}s"
    return f"{remaining//60}m{remaining%60:02d}s"

def is_tty():
    try: return sys.stdout.isatty()
    except Exception: return False

class SimpleBar:
    def __init__(self, total, mode="auto"):
        self.total = max(1, total)
        self.mode = "bar" if (mode=="bar" or (mode=="auto" and is_tty())) else "none"
        self.width = 32
        self.start = time.time()
    def update(self, i, covered):
        if self.mode == "none": return
        frac = min(1.0, max(0.0, i / self.total))
        filled = int(frac * self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        pct = f"{frac*100:5.1f}%"
        eta = human_eta(self.start, i, self.total)
        msg = f"\r[{bar}] {pct} | OK {covered}/{self.total} | ETA {eta}"
        sys.stdout.write(msg); sys.stdout.flush()
        if i == self.total: sys.stdout.write("\n")

# ───────────────────────── Normalização ─────────────────────────
def _norm_text(s: Optional[str]) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.casefold().strip()

CATS_OK = {"plausible", "correct", "actual_correct"}
def _norm_cat(s: Optional[str]) -> Optional[str]:
    if not s: return None
    t = _norm_text(s)
    aliases = {
        "invalid": {"invalid", "invalida", "inválida"},
        "semantic_incorrect": {"semantic_incorrect", "semantically_incorrect", "semantica_incorreta", "semântica_incorreta"},
        "plausible": {"plausible", "plausivel", "plausível"},
        "correct": {"correct", "correto", "correta"},
        "actual_correct": {"actual_correct", "correct", "correto", "correta"},
    }
    for k, vs in aliases.items():
        if t in {_norm_text(v) for v in vs}: return k
    return t or None

def categoria_aceita(cat: Optional[str], accept_any: bool) -> bool:
    if accept_any: return True
    return (_norm_cat(cat) in CATS_OK)

def linguagem_api(nome_dir: str) -> Optional[str]:
    m = {"python3": "python3", "python": "python3", "c++": "cpp", "cpp": "cpp", "java": "java"}
    return m.get(_norm_text(nome_dir))

def extract_slug_from_filename(path: Path) -> str:
    return path.stem.strip().replace(" ", "-")

def _san_for_path(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._+-]+", "_", str(s or "").strip())

def _partition_path(modelo: str, linguagem: str, slug: str) -> Path:
    return PARTITION_DIR / _san_for_path(modelo) / _san_for_path(linguagem) / f"{_san_for_path(slug)}.json"

# ───────────────────────── Slug canônico (índice de problemas) ─────────────────────────
_PROBLEMS_IDX = {"by_slug": {}, "by_frontend": {}, "by_qid": {}}

def _load_cached_index() -> Optional[dict]:
    try:
        if PROBLEMS_CACHE.exists():
            data = json.loads(PROBLEMS_CACHE.read_text(encoding="utf-8"))
            # cache simples sem TTL chato
            return data
    except Exception:
        pass
    return None

def _save_cached_index(data: dict):
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PROBLEMS_CACHE.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass

def ensure_problems_index():
    if _PROBLEMS_IDX["by_slug"]: return
    cached = _load_cached_index()
    if cached:
        _PROBLEMS_IDX.update(cached)
        logging.info("Índice de problemas carregado do cache (%d slugs).", len(cached.get("by_slug",{})))
        return
    url = "https://leetcode.com/api/problems/all/"
    logging.info("Baixando índice de problemas do LeetCode (pode levar alguns segundos)...")
    try:
        req = Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urlopen(req, timeout=25) as r:
            data = json.loads(r.read().decode("utf-8"))
        pairs = data.get("stat_status_pairs", [])
        by_slug, by_frontend, by_qid = {}, {}, {}
        for q in pairs:
            stat = q.get("stat", {}) or {}
            slug = stat.get("question__title_slug")
            qid  = stat.get("question_id")
            fid  = stat.get("frontend_question_id")
            if slug: by_slug[str(slug)] = {"qid": qid, "fid": fid}
            if fid is not None: by_frontend[str(fid)] = {"slug": slug, "qid": qid}
            if qid is not None: by_qid[str(qid)] = {"slug": slug, "fid": fid}
        _PROBLEMS_IDX["by_slug"] = by_slug
        _PROBLEMS_IDX["by_frontend"] = by_frontend
        _PROBLEMS_IDX["by_qid"] = by_qid
        _save_cached_index(_PROBLEMS_IDX)
        logging.info("Índice carregado: %d slugs.", len(by_slug))
    except (URLError, HTTPError) as e:
        logging.warning("Falha ao baixar índice (%s). Continuando com heurística local.", e)

def resolve_canonical_slug(slugish: str) -> str:
    """
    Aceita formatos:
      '1006', '1006-qualquer-coisa', 'clumsy-factorial'
    Tenta resolver para slug oficial. Se não conseguir, devolve slugish normalizado.
    """
    s = slugish.strip().lower()
    ensure_problems_index()
    if not s: return s

    m = re.match(r"^(\d+)[-_].+$", s)  # "1006-foo"
    if m:
        fid = m.group(1)
        hit = _PROBLEMS_IDX["by_frontend"].get(fid)
        if hit and hit.get("slug"): return hit["slug"]

    if s.isdigit():  # "1006"
        hit = _PROBLEMS_IDX["by_frontend"].get(s)
        if hit and hit.get("slug"): return hit["slug"]
        hit2 = _PROBLEMS_IDX["by_qid"].get(s)
        if hit2 and hit2.get("slug"): return hit2["slug"]

    # pode já ser o slug
    if _PROBLEMS_IDX["by_slug"].get(s): return s

    # fallback: stripar prefixo numérico se existir
    m2 = re.match(r"^\d+-(.+)$", s)
    if m2: return m2.group(1)
    return s

# ───────────────────────── Estruturas ─────────────────────────
@dataclass(frozen=True)
class Task:
    slugish: str
    slug: str           # canônico
    modelo: str
    linguagem_dir: str
    linguagem_api: str

@dataclass
class Hit:
    exists: bool
    accepted: Optional[bool]
    status: Optional[str]
    path: Optional[Path]
    submission_id: Optional[str]
    ground_truth: bool = False

# ───────────────────────── Pré-scan de partições (sanidade) ─────────────────────────
def log_partitions_health():
    total_files = 0
    by_model = {}
    by_lang = {}
    if PARTITION_DIR.exists():
        for model_dir in PARTITION_DIR.iterdir():
            if not model_dir.is_dir(): continue
            for lang_dir in model_dir.iterdir():
                if not lang_dir.is_dir(): continue
                files = list(lang_dir.glob("*.json"))
                n = len(files)
                total_files += n
                by_model[model_dir.name] = by_model.get(model_dir.name, 0) + n
                by_lang[lang_dir.name] = by_lang.get(lang_dir.name, 0) + n
    logging.info("Partições: %d arquivos em %s", total_files, PARTITION_DIR)
    if total_files == 0:
        logging.warning("Nenhum arquivo em by_model/. Se você já submeteu, o OUTPUT_DIR pode estar errado.")
    else:
        top_models = sorted(by_model.items(), key=lambda x: x[1], reverse=True)[:5]
        top_langs  = sorted(by_lang.items(), key=lambda x: x[1], reverse=True)[:5]
        logging.info("Top modelos (arquivos): %s", ", ".join(f"{m}:{c}" for m,c in top_models))
        logging.info("Top linguagens (arquivos): %s", ", ".join(f"{l}:{c}" for l,c in top_langs))

# ───────────────────────── Leitura do esperado ─────────────────────────
def enumerate_expected(base_path: Path, accept_any: bool) -> List[Task]:
    if not base_path.exists():
        logging.error("BASE_PATH não existe: %s", base_path)
        return []
    modelos = sorted([p for p in base_path.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    expected: List[Task] = []
    for modelo_dir in modelos:
        for lang_dir in sorted([p for p in modelo_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            lang_api = linguagem_api(lang_dir.name)
            if not lang_api:
                logging.warning("Linguagem '%s' não mapeada. Pulando.", lang_dir.name)
                continue
            for jf in lang_dir.glob("*.json"):
                try:
                    dados = json.loads(jf.read_text(encoding="utf-8"))
                except Exception as e:
                    logging.warning("Falha lendo %s: %s", jf, e)
                    continue
                if not categoria_aceita(dados.get("categoria"), accept_any):
                    continue
                slugish = extract_slug_from_filename(jf)
                slug_can = resolve_canonical_slug(slugish)
                expected.append(Task(slugish=slugish, slug=slug_can,
                                     modelo=modelo_dir.name, linguagem_dir=lang_dir.name, linguagem_api=lang_api))
    logging.info("Pré-scan: %d modelos, %d tasks candidatas.", len(modelos), len(expected))
    return expected

# ───────────────────────── Carregadores de resultado ─────────────────────────
def load_partition_hit(t: Task) -> Hit:
    # tenta canônico
    p1 = _partition_path(t.modelo, t.linguagem_dir, t.slug)
    if p1.exists():
        try:
            data = json.loads(p1.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning("JSON corrompido em %s (%s)", p1, e)
            return Hit(True, None, "corrompido", p1, None)
        return Hit(True,
                   bool(data.get("aceito")) if data.get("aceito") is not None else None,
                   str(data.get("status")) if data.get("status") is not None else None,
                   p1,
                   str(data.get("submission_id")) if data.get("submission_id") is not None else None,
                   bool(data.get("ground_truth")))
    # tenta slugish do arquivo (compat)
    if t.slugish != t.slug:
        p2 = _partition_path(t.modelo, t.linguagem_dir, t.slugish)
        if p2.exists():
            try:
                data = json.loads(p2.read_text(encoding="utf-8"))
            except Exception as e:
                logging.warning("JSON corrompido em %s (%s)", p2, e)
                return Hit(True, None, "corrompido", p2, None)
            return Hit(True,
                       bool(data.get("aceito")) if data.get("aceito") is not None else None,
                       str(data.get("status")) if data.get("status") is not None else None,
                       p2,
                       str(data.get("submission_id")) if data.get("submission_id") is not None else None,
                       bool(data.get("ground_truth")))
    return Hit(False, None, None, None, None)

def _iter_json_legado() -> Iterable[dict]:
    if JSON_LEGADO.exists():
        try:
            return json.loads(JSON_LEGADO.read_text(encoding="utf-8"))
        except Exception:
            logging.warning("Falha lendo JSON legado: %s", JSON_LEGADO)
            return []
    return []

def _iter_csv_file(path: Path) -> Iterable[dict]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    yield row
        except Exception:
            logging.warning("Falha lendo CSV legado: %s", path)
            return []
    return []

def _iter_csv_legado() -> Iterable[dict]:
    # concatena os dois CSVs possíveis
    for row in _iter_csv_file(CSV_LEGADO_1):
        yield row
    for row in _iter_csv_file(CSV_LEGADO_2):
        yield row

def load_legacy_hit(t: Task) -> Hit:
    # JSON legado
    for r in _iter_json_legado():
        if _norm_text(r.get("modelo","")) == _norm_text(t.modelo) and \
           _norm_text(r.get("linguagem","")) == _norm_text(t.linguagem_dir) and \
           _norm_text(r.get("slug","")) in {_norm_text(t.slug), _norm_text(t.slugish)}:
            return Hit(True, _to_bool(r.get("aceito")), str(r.get("status")), JSON_LEGADO, str(r.get("submission_id")), _to_bool(r.get("ground_truth")))
    # CSV legado
    for r in _iter_csv_legado():
        if _norm_text(r.get("modelo","")) == _norm_text(t.modelo) and \
           _norm_text(r.get("linguagem","")) == _norm_text(t.linguagem_dir) and \
           _norm_text(r.get("slug","")) in {_norm_text(t.slug), _norm_text(t.slugish)}:
            return Hit(True, _to_bool(r.get("aceito")), str(r.get("status")), None, r.get("submission_id"), _to_bool(r.get("ground_truth")))
    return Hit(False, None, None, None, None)

def _to_bool(v) -> Optional[bool]:
    if v in (True, False): return bool(v)
    if v is None: return None
    s = _norm_text(str(v))
    if s in {"1","true","yes","sim"}: return True
    if s in {"0","false","no","nao","não"}: return False
    return None

# ───────────────────────── Avaliação ─────────────────────────
def evaluate(only_accepted: bool, require_gt: bool, accept_any: bool, progress_mode: str, tick_every: Optional[int]):
    log_partitions_health()
    start = time.time()
    expected = enumerate_expected(BASE_PATH, accept_any)
    total = len(expected)
    if not expected:
        logging.warning("Nada a avaliar. data/: %s | out/: %s", BASE_PATH, OUTPUT_DIR)
        return 2

    rows = []
    missing = []
    covered = 0
    bar = SimpleBar(total, mode=progress_mode)
    if tick_every is None:
        tick_every = max(25, total // 33)

    logging.info("Iniciando avaliação: %d itens | ONLY_ACCEPTED=%s | REQUIRE_GT=%s | ACCEPT_ANY_CATEGORY=%s",
                 total, only_accepted, require_gt, accept_any)
    for i, t in enumerate(expected, 1):
        if i == 1 or i % tick_every == 0:
            logging.info("Andamento: %d/%d (%s) | OK=%d | ETA ~%s",
                         i, total, human_pct(i, total), covered, human_eta(start, i, total))

        ph = load_partition_hit(t)
        lh = load_legacy_hit(t) if not ph.exists else Hit(False,None,None,None,None)
        hit = ph if ph.exists else lh

        ok = False
        reason = ""
        if not hit.exists:
            reason = "sem resultado"
            logging.debug("MISS %s | %s | %s -> %s", t.slug, t.modelo, t.linguagem_dir, reason)
        else:
            if only_accepted:
                ok = bool(hit.accepted)
                reason = "accepted" if ok else (hit.status or "não aceito")
            else:
                ok = bool(hit.submission_id or hit.status)
                reason = hit.status or ("submission_id" if hit.submission_id else "sem status")

        if require_gt:
            gt_p1 = _partition_path("ground_truth", t.linguagem_dir, t.slug)
            gt_p2 = _partition_path("ground_truth", t.linguagem_dir, t.slugish) if t.slugish != t.slug else None
            gt_ok = gt_p1.exists() or (gt_p2 and gt_p2.exists())
            if not gt_ok:
                ok = False
                reason = (reason + " | falta ground_truth").strip(" |")

        rows.append({
            "slug": t.slug,
            "slugish": t.slugish,
            "modelo": t.modelo,
            "linguagem": t.linguagem_dir,
            "found": hit.exists,
            "accepted": hit.accepted,
            "status": hit.status,
            "path": str(hit.path) if hit.path else "",
            "ok": ok,
            "reason": reason
        })
        if ok:
            covered += 1
            logging.debug("OK   %s | %s | %s -> %s", t.slug, t.modelo, t.linguagem_dir, reason)
        else:
            missing.append((t, reason))
            logging.debug("MISS %s | %s | %s -> %s", t.slug, t.modelo, t.linguagem_dir, reason)

        bar.update(i, covered)

    # Salva CSV e MD
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "coverage_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["slug","slugish","modelo","linguagem","found","accepted","status","path","ok","reason"])
        w.writeheader(); w.writerows(rows)

    md_path = OUTPUT_DIR / "coverage_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        pct = human_pct(covered, total)
        f.write(f"# Cobertura de Submissões\n\n")
        f.write(f"- Base: `{BASE_PATH}`\n- Resultados: `{OUTPUT_DIR}`\n")
        f.write(f"- ONLY_ACCEPTED={only_accepted} | REQUIRE_GT={require_gt} | ACCEPT_ANY_CATEGORY={accept_any}\n\n")
        f.write(f"**Coberto:** {covered}/{total} ({pct})\n\n")
        if missing:
            f.write("## Itens faltantes (amostra até 200)\n\n")
            for t, reason in missing[:200]:
                f.write(f"- {t.slug} ({t.slugish}) | {t.modelo} | {t.linguagem_dir} → {reason}\n")
        else:
            f.write("Sem faltas. Milagre estatístico.\n")

    dur = time.time() - start
    logging.info("Fim: %d/%d cobertos (%s) em %.1fs | CSV: %s | MD: %s",
                 covered, total, human_pct(covered, total), dur, csv_path, md_path)
    if missing:
        logging.warning("Faltando %d itens. Consulte o relatório MD.", len(missing))
    return 0 if covered == total else 1

# ───────────────────────── CLI ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Valida cobertura de submissões (slug canônico, CSV correto).")
    p.add_argument("--only-accepted", action="store_true", default=ONLY_ACCEPTED_ENV)
    p.add_argument("--require-gt", action="store_true", default=REQUIRE_GT_ENV)
    p.add_argument("--accept-any-category", action="store_true", default=ACCEPT_ANY_CATEGORY_ENV,
                   help="Não filtra categoria dos JSONs de data/. Conta tudo.")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL","INFO"),
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--log-file", default=os.getenv("LOG_FILE", str(OUTPUT_DIR / "check_submissions.log")))
    p.add_argument("--progress", default=os.getenv("PROGRESS","auto"),
                   choices=["auto","bar","none"])
    p.add_argument("--tick", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    log_file = args.log_file if (args.log_file is not None and args.log_file.strip()!="") else None
    setup_logging(args.log_level, log_file)
    logging.info("check_submissions iniciado. data=%s | out=%s", BASE_PATH, OUTPUT_DIR)
    try:
        code = evaluate(args.only_accepted, args.require_gt, args.accept_any_category, args.progress, args.tick)
    except KeyboardInterrupt:
        logging.error("Interrompido pelo usuário. Classe.")
        code = 130
    sys.exit(code)

if __name__ == "__main__":
    main()
