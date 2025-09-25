# -*- coding: utf-8 -*-
# dashboard.py
from __future__ import annotations
import json, re, textwrap, unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
import plotly.express as px
import requests
import streamlit as st

# ------------------------------ Page & Plotly ------------------------------
st.set_page_config(page_title="LeetCode Performance Dashboard", page_icon="🚀", layout="wide")
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = px.colors.qualitative.Set2

RESULTS_BASE = Path("out")
LLM_BASE = Path("data")
LEETCODE_GRAPHQL = "https://leetcode.com/graphql"

# ------------------------------ UI i18n ------------------------------
UI = {
    "en": {
        "lang": "English",
        "sidebar.language": "Output language",
        "sidebar.section": "Section",
        "sidebar.cache": "Cache",
        "sidebar.clear": "Clear cached data",

        "section.results": "📈 LeetCode Results",
        "section.compare": "🧠 Compare LLM Answers",

        "quality.title": "Data quality",
        "quality.noissues": "No relevant issues detected.",
        "quality.missing_model": "There are records without 'modelo'.",
        "quality.missing_lang": "There are records without 'linguagem'.",
        "quality.missing_qid": "There are records without 'qid' (question ID).",
        "quality.missing_time": "'tempo_s' is missing or invalid across all records.",
        "quality.missing_mem": "'memoria_mb' is missing or invalid across all records.",
        "quality.with_diff": "With difficulty (LC):",
        "quality.with_cat": "With category (LC):",
        "quality.with_ts": "Records with timestamp:",

        "filters.title": "Filters",
        "filters.models": "Models",
        "filters.langs": "Languages",
        "filters.questions": "Questions",
        "filters.status": "Status",
        "filters.hide_accepted": "Hide Accepted",
        "filters.search": "Search in title/slug",
        "filters.dedup": "Keep only most recent per (model, language, qid)",
        "filters.time": "Time (s)",
        "filters.memory": "Memory (MB)",
        "filters.diff": "Difficulty (LeetCode)",
        "filters.cat": "Category (LeetCode)",

        "warn.nodata": "No data for the selected filters.",
        "kpi.title": "Performance Overview",
        "kpi.total": "Total Submissions",
        "kpi.accept": "Overall Acceptance Rate",
        "kpi.time": "Avg Execution Time",
        "kpi.mem": "Avg Memory Usage",

        "chart.result_dist": "Result Distribution",
        "chart.accept_by_model": "Acceptance Rate by Model",
        "chart.time_vs_mem": "Time vs Memory",
        "chart.status_stacked": "Status by Model (stacked)",
        "chart.acc_by_diff": "Acceptance by Difficulty (LC)",
        "chart.hm_calc_vs_diff": "Heatmap: computed category × difficulty (LC)",
        "chart.acc_by_cat": "Acceptance by Category (LC)",
        "chart.hm_calc_vs_cat": "Heatmap: computed category × category (LC)",
        "chart.acc_by_question": "Acceptance by Question",

        "export.title": "Export",
        "export.csv": "Filtered CSV",
        "export.json": "Filtered JSON",

        "compare.title": "LLM Answers Comparator",
        "compare.join_diag": "Join diagnostics (by qid)",
        "compare.unmatched": "View unmatched pairs (by qid)",
        "compare.summary": "Summary per combination",
        "compare.hm": "Heatmap: computed category × LC category",
        "compare.acc_for_sel": "Acceptance by Model (LeetCode) for current selection",
        "compare.denom": "Denominator for acceptance rate",
        "compare.denom.only": "Only LLM answers that matched a LeetCode result",
        "compare.denom.all": "All selected LLM answers",
        "compare.details": "Computation details",
        "compare.code_size": "Code size by model",
        "compare.correlations": "Simple correlations",
        "compare.code_viewer": "Code viewer",
        "compare.code_a": "Code A",
        "compare.code_b": "Code B",
        "compare.meta_a": "LLM meta A",
        "compare.meta_b": "LLM meta B",
        "compare.download": "Download CSV (LLM vs LeetCode)",

        "labels.model": "Model",
        "labels.language": "Language",
        "labels.question": "Question",
        "labels.slug": "Slug",
        "labels.qid": "QID",
        "labels.lc_diff": "Difficulty (LC)",
        "labels.lc_cat": "Category (LC)",
        "labels.lc_topics": "LC Topics",
        "labels.final_status": "Final status",
        "labels.cat_calc": "Computed category",
        "labels.cat_decl": "Declared category",
        "labels.cat_lc": "LeetCode category",
        "labels.approval": "Approval rate",
        "labels.time_s": "Time (s)",
        "labels.mem_mb": "Memory (MB)",
        "labels.tests_passed": "Passed tests",
        "labels.tests_total": "Total tests",
        "labels.time_pct": "Time percentile",
        "labels.mem_pct": "Memory percentile",
        "labels.compiled": "Compiled",
        "labels.comp_err": "Compilation error",
        "labels.run_err": "Runtime error",
        "labels.last_case": "Last test case",
        "labels.exp_out": "Expected output",
        "labels.your_out": "Your output",
        "labels.input": "Input",
        "labels.timestamp": "Timestamp",

        "labels.cat_llm": "LLM category",
        "labels.status_llm": "LLM status",
        "labels.reason_llm": "LLM reason",
        "labels.efficiency": "Efficiency",
        "labels.time_complexity": "Time complexity",
        "labels.space_complexity": "Space complexity",
        "labels.energy": "Energy implications",
        "labels.explanation": "Explanation",
        "labels.starter_ok": "Starter OK",
        "labels.starter_reason": "Starter check reason",
        "labels.rechecked_at": "Rechecked at",

        "status.accepted": "Accepted",
        "status.aceito": "Aceito",

        "axis.model": "Model",
        "axis.acceptance": "Acceptance (%)",
        "axis.count": "Count",
        "axis.time": "Time (s)",
        "axis.memory": "Memory (MB)",
    },
    "pt": {
        "lang": "Português (Brasil)",
        "sidebar.language": "Idioma de saída",
        "sidebar.section": "Seção",
        "sidebar.cache": "Cache",
        "sidebar.clear": "Limpar cache",

        "section.results": "📈 Resultados (LeetCode)",
        "section.compare": "🧠 Comparar Respostas LLM",

        "quality.title": "Qualidade dos dados",
        "quality.noissues": "Sem problemas relevantes detectados.",
        "quality.missing_model": "Há registros sem 'modelo'.",
        "quality.missing_lang": "Há registros sem 'linguagem'.",
        "quality.missing_qid": "Há registros sem 'qid' (ID da questão).",
        "quality.missing_time": "Todos os 'tempo_s' estão ausentes ou inválidos.",
        "quality.missing_mem": "Todos os 'memoria_mb' estão ausentes ou inválidos.",
        "quality.with_diff": "Com dificuldade (LC):",
        "quality.with_cat": "Com categoria (LC):",
        "quality.with_ts": "Registros com timestamp:",

        "filters.title": "Filtros",
        "filters.models": "Modelos",
        "filters.langs": "Linguagens",
        "filters.questions": "Questões",
        "filters.status": "Status",
        "filters.hide_accepted": "Ocultar Aceitos",
        "filters.search": "Buscar em título/slug",
        "filters.dedup": "Manter só o mais recente por (modelo, linguagem, qid)",
        "filters.time": "Tempo (s)",
        "filters.memory": "Memória (MB)",
        "filters.diff": "Dificuldade (LeetCode)",
        "filters.cat": "Categoria (LeetCode)",

        "warn.nodata": "Nenhum dado encontrado para os filtros selecionados.",
        "kpi.title": "Visão Geral de Desempenho",
        "kpi.total": "Total de Submissões",
        "kpi.accept": "Taxa de Aceitação Geral",
        "kpi.time": "Tempo Médio de Execução",
        "kpi.mem": "Uso Médio de Memória",

        "chart.result_dist": "Distribuição de Resultados",
        "chart.accept_by_model": "Taxa de Aceitação por Modelo",
        "chart.time_vs_mem": "Tempo vs. Memória",
        "chart.status_stacked": "Status por Modelo (empilhado)",
        "chart.acc_by_diff": "Aceitação por Dificuldade (LC)",
        "chart.hm_calc_vs_diff": "Heatmap: categoria_calculada × dificuldade (LC)",
        "chart.acc_by_cat": "Aceitação por Categoria (LC)",
        "chart.hm_calc_vs_cat": "Heatmap: categoria_calculada × categoria (LC)",
        "chart.acc_by_question": "Taxa de Aceitação por Questão",

        "export.title": "Exportar",
        "export.csv": "CSV filtrado",
        "export.json": "JSON filtrado",

        "compare.title": "Comparador de Respostas das LLMs",
        "compare.join_diag": "Diagnóstico do cruzamento por ID",
        "compare.unmatched": "Ver pares sem correspondência (por qid)",
        "compare.summary": "Resumo por combinação",
        "compare.hm": "Heatmap: categoria_calculada × categoria (LC)",
        "compare.acc_for_sel": "Aceitação por Modelo (LeetCode) para a seleção",
        "compare.denom": "Como calcular a taxa?",
        "compare.denom.only": "Apenas respostas com match no LeetCode",
        "compare.denom.all": "Todas as respostas LLM da seleção",
        "compare.details": "Detalhes do cálculo",
        "compare.code_size": "Tamanho do código por modelo",
        "compare.correlations": "Correlações simples",
        "compare.code_viewer": "Visualizador de Código",
        "compare.code_a": "Código A",
        "compare.code_b": "Código B",
        "compare.meta_a": "LLM meta A",
        "compare.meta_b": "LLM meta B",
        "compare.download": "Baixar CSV (LLM vs LeetCode)",

        "labels.model": "Modelo",
        "labels.language": "Linguagem",
        "labels.question": "Questão",
        "labels.slug": "Slug",
        "labels.qid": "QID",
        "labels.lc_diff": "Dificuldade (LC)",
        "labels.lc_cat": "Categoria (LC)",
        "labels.lc_topics": "Tópicos (LC)",
        "labels.final_status": "Status final",
        "labels.cat_calc": "Categoria calculada",
        "labels.cat_decl": "Categoria declarada",
        "labels.cat_lc": "Categoria (LeetCode)",
        "labels.approval": "Taxa de aprovação",
        "labels.time_s": "Tempo (s)",
        "labels.mem_mb": "Memória (MB)",
        "labels.tests_passed": "Testes passados",
        "labels.tests_total": "Total de testes",
        "labels.time_pct": "Percentil de tempo",
        "labels.mem_pct": "Percentil de memória",
        "labels.compiled": "Compilou",
        "labels.comp_err": "Erro de compilação",
        "labels.run_err": "Erro de execução",
        "labels.last_case": "Último caso de teste",
        "labels.exp_out": "Saída esperada",
        "labels.your_out": "Sua saída",
        "labels.input": "Entrada",
        "labels.timestamp": "Timestamp",

        "labels.cat_llm": "Categoria LLM",
        "labels.status_llm": "Status LLM",
        "labels.reason_llm": "Motivo LLM",
        "labels.efficiency": "Eficiência",
        "labels.time_complexity": "Complexidade de tempo",
        "labels.space_complexity": "Complexidade de espaço",
        "labels.energy": "Implicações energéticas",
        "labels.explanation": "Explicação",
        "labels.starter_ok": "Starter OK",
        "labels.starter_reason": "Razão do starter",
        "labels.rechecked_at": "Reavaliado em",

        "status.accepted": "Aceito",
        "status.aceito": "Aceito",

        "axis.model": "Modelo",
        "axis.acceptance": "Aceitação (%)",
        "axis.count": "Qtde",
        "axis.time": "Tempo (s)",
        "axis.memory": "Memória (MB)",
    }
}
def t(lang: str, key: str) -> str:
    use = "pt" if lang.startswith("Portugu") else "en"
    return UI[use].get(key, key)

# ------------------------------ Data i18n (values) ------------------------------
def _nfkdc(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold().strip()

# ---- mapeamentos bidirecionais ----
PT_EN_RAW = {
    "invalida":"invalid","inválida":"invalid","semantica_incorreta":"semantically incorrect",
    "semântica_incorreta":"semantically incorrect","plausivel":"plausible","plausível":"plausible",
    "correto":"correct","correta":"correct","aceito":"Accepted","accepted":"Accepted",
    "resposta errada":"Wrong Answer","wrong answer":"Wrong Answer",
    "erro de execucao":"Runtime Error","erro de execução":"Runtime Error","runtime error":"Runtime Error",
    "erro de compilacao":"Compilation Error","erro de compilação":"Compilation Error","compilation error":"Compilation Error",
    "tempo limite excedido":"Time Limit Exceeded","time limit exceeded":"Time Limit Exceeded",
    "formato invalido":"Invalid Format","formato inválido":"Invalid Format","invalid format":"Invalid Format",
    "desconhecido":"Unknown Status","unknown":"Unknown Status",
    "todas as secoes presentes corretamente":"all sections present correctly",
    "todas as seções presentes corretamente":"all sections present correctly",
    "código extraído mas apenas 5/6 seções presentes":"code extracted but only 5/6 extracts present",
    "código extraído mas apenas 4/6 seções presentes":"code extracted but only 4/6 extracts present",
    "código extraído mas apenas 3/6 seções presentes":"code extracted but only 3/6 extracts present",
    "código extraído mas apenas 2/6 seções presentes":"code extracted but only 2/6 extracts present",
    "código extraído mas apenas 1/6 seções presentes":"code extracted but only 1/6 extracts present",
    "código extraído mas apenas 0/6 seções presentes":"code extracted but only 0/6 extracts present",
    "código não extraído":"code not extracted",
    "starter respeitado":"starter respected",
    "alta":"HIGH","baixa":"LOW","media":"MEDIUM","média":"MEDIUM","high":"HIGH","low":"LOW","medium":"MEDIUM",
    "fácil":"Easy","médio":"Medium","difícil":"Hard"
}
# normaliza TODAS as chaves PT para garantir match mesmo com/sem acentos/maiúsculas
PT_EN_NORM = { _nfkdc(k): v for k, v in PT_EN_RAW.items() }

EN_PT = {
    "invalid":"inválida","semantically incorrect":"semântica_incorreta","plausible":"plausível",
    "correct":"correto","Accepted":"Aceito","Wrong Answer":"Resposta Errada","Runtime Error":"Erro de Execução",
    "Compilation Error":"Erro de Compilação","Time Limit Exceeded":"Tempo Limite Excedido",
    "Invalid Format":"Formato Inválido","Unknown Status":"Desconhecido",
    "all sections present correctly":"todas as seções presentes corretamente",
    "code extracted but only 5/6 extracts present":"código extraído mas apenas 5/6 seções presentes",
    "code extracted but only 4/6 extracts present":"código extraído mas apenas 4/6 seções presentes",
    "code extracted but only 3/6 extracts present":"código extraído mas apenas 3/6 seções presentes",
    "code extracted but only 2/6 extracts present":"código extraído mas apenas 2/6 seções presentes",
    "code extracted but only 1/6 extracts present":"código extraído mas apenas 1/6 seções presentes",
    "code extracted but only 0/6 extracts present":"código extraído mas apenas 0/6 seções presentes",
    "code not extracted":"código não extraído",
    "starter respected":"starter respeitado","HIGH":"ALTA","LOW":"BAIXA","MEDIUM":"MÉDIA",
    "Easy":"Fácil","Medium":"Médio","Hard":"Difícil"
}

def tr_val(v, lang):
    if isinstance(v, (bool, np.bool_)) or v is None:
        return v
    s = str(v)
    if lang.startswith("Portugu"):
        return EN_PT.get(s, s)
    # PT -> EN usa o dicionário normalizado
    key = PT_EN_NORM.get(_nfkdc(s))
    return key if key is not None else s

def tr_df(df: pd.DataFrame, cols: List[str], lang: str):
    if df is None or df.empty: return
    for c in cols:
        if c in df.columns:
            s = df[c]
            if is_categorical_dtype(s): s = s.astype("object")
            df[c] = s.map(lambda x: tr_val(x, lang))

VAL_RES_COLS = [
    "status","formato_original","categoria_calculada","categoria_declarada","categoria_leetcode",
    "status_final","erro_compilacao","erro_runtime","ultimo_caso_teste","saida_esperada","sua_saida",
    "lc_category","lc_difficulty",
]
VAL_LLM_COLS = [
    "categoria_llm","status_llm","motivo_llm","efficiency","time_complexity","space_complexity",
    "energy_implications","explanation","starter_check_reason",
]

# ---- helpers i18n p/ filtros (display ↔ interno) ----
PT2EN_DIFF = {"Fácil": "Easy", "Médio": "Medium", "Difícil": "Hard"}
EN2PT_DIFF = {v: k for k, v in PT2EN_DIFF.items()}

def diff_display_list_from_df(df: pd.DataFrame, lang: str) -> List[str]:
    if "lc_difficulty" not in df.columns: return []
    raw = sorted([s for s in df["lc_difficulty"].dropna().astype(str).unique().tolist() if s])
    if lang.startswith("Portugu"):
        return [EN2PT_DIFF.get(x, x) for x in raw]
    return raw

def diff_to_internal(selected: List[str], lang: str) -> List[str]:
    if not lang.startswith("Portugu"): return selected
    return [PT2EN_DIFF.get(x, x) for x in selected]

def translate_list_for_display(values: List[str], lang: str) -> List[str]:
    return [tr_val(v, lang) for v in values]

def map_display_back(selected_display: List[str], original_values: List[str], lang: str) -> List[str]:
    display_list = translate_list_for_display(original_values, lang)
    rev = {d: o for d, o in zip(display_list, original_values)}
    return [rev.get(x, x) for x in selected_display]

# ------------------------------ Helpers & normalization ------------------------------
def slug_to_title(slug: str) -> str:
    if not isinstance(slug, str) or not slug: return ""
    m = re.match(r"^\d+[-_](.+)$", slug)
    if m: slug = m.group(1)
    return re.sub(r"[-_]+", " ", slug).strip().title()

def _coerce_num_from_str(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(-?\d+\.?\d*)", expand=False), errors="coerce")

def get_status_verdict(row: pd.Series) -> str:
    status = str(row.get("status") or "").lower()
    accepted = bool(row.get("aceito") or False)
    compiled = row.get("compilou")
    comp_err = str(row.get("erro_compilacao") or "")
    run_err = str(row.get("erro_runtime") or "")
    fmt = str(row.get("formato_original") or "")
    if accepted or status.startswith("accepted"): return "Accepted"
    if (compiled is False) or comp_err: return "Compilation Error"
    if run_err or "runtime error" in status: return "Runtime Error"
    if (compiled is True and not accepted) or "wrong answer" in status: return "Wrong Answer"
    if fmt and fmt != "ok": return "Invalid Format"
    return "Unknown Status"

def safe_pct(a,b): return (a/b*100.0) if (b and b!=0) else 0.0

def _extract_digits(val) -> Optional[str]:
    if val is None: return None
    m = re.search(r"\d+", str(val)); return m.group(0) if m else None

def norm_model(s: str) -> str:
    s = str(s or "").strip().lower(); return s.replace(" ", "").replace("_","-")

def norm_lang(s: str) -> str:
    s = str(s or "").strip().lower()
    aliases = {"py":"python3","py3":"python3","python":"python3","python3":"python3",
               "c++":"cpp","cpp":"cpp","c#":"csharp","csharp":"csharp",
               "js":"javascript","node":"javascript","javascript":"javascript",
               "ts":"typescript","typescript":"typescript",
               "java":"java","go":"go","rust":"rust"}
    return aliases.get(s, s)

def choose_qid_from_row(row: pd.Series, id_candidates: List[str], extras: List[str]=[]) -> Optional[str]:
    for c in id_candidates:
        if c in row and pd.notna(row[c]):
            q = _extract_digits(row[c]); 
            if q: return q
    for c in extras:
        if c in row and pd.notna(row[c]):
            q = _extract_digits(row[c]); 
            if q: return q
    return None

def add_normalized_keys(df: pd.DataFrame, is_llm: bool) -> pd.DataFrame:
    df = df.copy()
    df["modelo"] = df["modelo"] if "modelo" in df.columns else "unknown"
    df["linguagem"] = df["linguagem"] if "linguagem" in df.columns else "unknown"
    df["modelo_norm"] = df["modelo"].map(norm_model)
    df["lang_norm"]   = df["linguagem"].map(norm_lang)
    id_candidates = ["qid","id","question_id","id_questao","problem_id","id_question"]
    extras = ["slug","titulo"]; 
    if is_llm: extras = ["slugish","slug","titulo"]
    if "qid" not in df.columns or df["qid"].isna().any():
        df["qid"] = df.apply(lambda r: choose_qid_from_row(r,id_candidates,extras), axis=1)
    df["qid"] = df["qid"].astype(str).where(df["qid"].notna(), None)
    return df

def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns: df[c]=pd.NA
    return df

def _pick_options_col(df: pd.DataFrame, primary: str, aliases: List[str]):
    for c in [primary]+aliases:
        if c in df.columns:
            vals = sorted(pd.Series(df[c]).dropna().unique().tolist())
            return c, vals
    df[primary] = pd.Series(dtype=object); return primary, []

def fillna_dash(s: pd.Series, dash="—")->pd.Series:
    if is_categorical_dtype(s): s=s.astype("object")
    return s.fillna(dash).replace("",dash)

def wrap_labels(labels,width=18): return ["<br>".join(textwrap.wrap(str(x),width)) for x in labels]

def beautify(fig, title=None, xrotate=None, height=None):
    layout = dict(margin=dict(t=60,r=20,b=70,l=60), font=dict(size=13),
                  legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0))
    if title is not None: layout["title"]=title
    fig.update_layout(**layout)
    if xrotate is not None: fig.update_xaxes(tickangle=xrotate)
    if height is not None: fig.update_layout(height=height)
    return fig

# ------------------------------ Loaders & enrichment (sempre ativos) ------------------------------
def _read_ndjson(path: Path) -> pd.DataFrame:
    rows=[]
    try:
        for line in open(path,"r",encoding="utf-8"):
            line=line.strip(); 
            if line: rows.append(json.loads(line))
    except Exception: return pd.DataFrame()
    df=pd.DataFrame(rows)
    if "ts_epoch" in df.columns and ("timestamp" not in df.columns or df["timestamp"].isna().all()):
        try: df["timestamp"]=pd.to_datetime(df["ts_epoch"],unit="s",utc=False)
        except Exception: pass
    return df

def _scan_partitioned_dir(base: Path) -> pd.DataFrame:
    rows=[]
    if not base.exists(): return pd.DataFrame()
    for model_dir in base.iterdir():
        if not model_dir.is_dir(): continue
        for lang_dir in model_dir.iterdir():
            if not lang_dir.is_dir(): continue
            for jf in lang_dir.glob("*.json"):
                try:
                    obj=json.load(open(jf,"r",encoding="utf-8"))
                    if "timestamp" not in obj or not obj.get("timestamp"):
                        obj["timestamp"]=pd.to_datetime(jf.stat().st_mtime,unit="s",utc=False)
                    rows.append(obj)
                except Exception: continue
    df=pd.DataFrame(rows)
    if "timestamp" in df.columns: df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    return df

def _enrich_from_partitions(df: pd.DataFrame, out_dir: Path)->pd.DataFrame:
    by_model = out_dir/"by_model"
    if not by_model.exists() or df.empty: return df
    part=_scan_partitioned_dir(by_model)
    if part.empty: return df
    keys=["modelo","linguagem","slug"]
    for k in keys:
        if k not in df.columns: df[k]=pd.NA
    cols=["testes_passados","total_testes","percentil_tempo","percentil_memoria","compilou",
          "ultimo_caso_teste","saida_esperada","sua_saida","input","lang_pretty","passou_testes_locais",
          "tempo_ms","memoria_mb","status","aceito","categoria_calculada","categoria_declarada","formato_original",
          "submission_id","ground_truth","qid","id_questao"]
    cols=[c for c in cols if c in part.columns]
    part_small=part[keys+cols].drop_duplicates(subset=keys,keep="last")
    merged=df.merge(part_small,on=keys,how="left",suffixes=("","_p"))
    for c in cols:
        if c in merged.columns and f"{c}_p" in merged.columns:
            merged[c]=merged[c].where(merged[c].notna(),merged[f"{c}_p"] )
            merged=merged.drop(columns=[f"{c}_p"])
    if "timestamp" in merged.columns and "timestamp_p" in merged.columns:
        merged["timestamp"]=merged["timestamp"].where(merged["timestamp"].notna(),merged["timestamp_p"])
        merged=merged.drop(columns=["timestamp_p"])
    return merged

@st.cache_data(show_spinner=False)
def load_and_preprocess_data(base: Path)->pd.DataFrame:
    if base.is_dir():
        by=base/"by_model"
        if by.exists(): df=_scan_partitioned_dir(by)
        elif (base/"results.csv").exists():
            df=pd.read_csv(base/"results.csv",encoding="utf-8",low_memory=False); df=_enrich_from_partitions(df,base)
        else:
            parts=[]; nd=base/"ndjson"
            if nd.exists():
                for f in nd.glob("*.ndjson"): parts.append(_read_ndjson(f))
            df=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
            if not df.empty: df=_enrich_from_partitions(df,base)
    else:
        df=pd.DataFrame()
    if df.empty: return df

    opt=["modelo","linguagem","slug","titulo","aceito","compilou","tempo_ms","memoria_mb","testes_passados","total_testes",
         "status","formato_original","erro_compilacao","erro_runtime","percentil_tempo","percentil_memoria",
         "categoria_calculada","categoria_declarada","categoria_leetcode","categoria_ok","ultimo_caso_teste",
         "saida_esperada","sua_saida","input","ground_truth","validation_ok","validation_issues","timestamp",
         "id","question_id","qid","problem_id","id_questao","lang_pretty","passou_testes_locais","submission_id"]
    for c in opt:
        if c not in df.columns: df[c]=pd.NA
    if "linguagem" in df.columns and "lang_pretty" in df.columns:
        df["linguagem"]=df["linguagem"].fillna(df["lang_pretty"])

    df["tempo_s"]=_coerce_num_from_str(df["tempo_ms"])/1000.0
    df["memoria_mb"]=_coerce_num_from_str(df["memoria_mb"])
    for c in ["testes_passados","total_testes","percentil_tempo","percentil_memoria"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    for c in ["aceito","compilou","ground_truth","validation_ok","categoria_ok","passou_testes_locais"]:
        if c in df.columns: df[c]=df[c].fillna(False).astype(bool)
    for c in ["erro_compilacao","erro_runtime","status","formato_original","categoria_calculada","categoria_declarada",
              "categoria_leetcode","ultimo_caso_teste","saida_esperada","sua_saida","input","lang_pretty"]:
        df[c]=df[c].fillna("")
    if "titulo" not in df.columns or df["titulo"].isna().all():
        df["titulo"]=df["slug"].apply(slug_to_title)
    if "timestamp" in df.columns: df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    if ("timestamp" not in df.columns or df["timestamp"].isna().all()) and "ts_epoch" in df.columns:
        try: df["timestamp"]=pd.to_datetime(df["ts_epoch"],unit="s",errors="coerce")
        except Exception: pass
    df["status_final"]=df.apply(get_status_verdict,axis=1)
    df["approval_rate"]=df.apply(lambda r: safe_pct(r.get("testes_passados") or 0, r.get("total_testes") or 0),axis=1)
    for col in ["modelo","linguagem","status_final","titulo"]:
        df[col]=df[col].astype("category")
    for c in ["tempo_s","memoria_mb"]:
        df[c]=df[c].replace([np.inf,-np.inf],np.nan)
    df=add_normalized_keys(df,is_llm=False)
    return df

def _parse_llm_extra_fields(text:str)->Dict[str,Optional[str]]:
    def grab(key:str)->Optional[str]:
        m=re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+)$", text or "")
        return m.group(1).strip() if m else None
    return {
        "efficiency":grab("efficiency"),
        "time_complexity":grab("time complexity"),
        "space_complexity":grab("space complexity"),
        "energy_implications":grab("energy implications"),
        "explanation":grab("explanation"),
    }

@st.cache_data(show_spinner=False)
def load_llm_answers(base_dir=LLM_BASE)->pd.DataFrame:
    base=Path(base_dir); rows=[]
    if not base.exists(): return pd.DataFrame()
    for model_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        has_lang=any(p.is_dir() for p in model_dir.iterdir())
        dirs=[(model_dir,ld) for ld in model_dir.iterdir() if ld.is_dir()] if has_lang else [(model_dir,None)]
        for md, ld in dirs:
            files=(ld or md).glob("*.json")
            for js in files:
                try:
                    obj=json.load(open(js,"r",encoding="utf-8"))
                    slugish=js.stem; m=re.match(r"^\d+[-_](.+)$",slugish); slug=m.group(1) if m else slugish
                    resp=str(obj.get("resposta") or ""); code=obj.get("code")
                    if not code:
                        mcode=re.search(r"```(?:[a-zA-Z0-9+#]*)\s*([\s\S]*?)```", resp); code=mcode.group(1).strip() if mcode else ""
                    extras=_parse_llm_extra_fields(resp)
                    rows.append({
                        "modelo": md.name, "linguagem": (ld.name if ld else (obj.get("linguagem") or "Unknown")),
                        "slugish": slugish, "slug": slug, "titulo": slug_to_title(slug),
                        "categoria_llm": obj.get("categoria"), "status_llm": obj.get("status"), "motivo_llm": obj.get("motivo"),
                        "code": code, "code_len": len(code or ""), "id": obj.get("id"), "question_id": obj.get("question_id"),
                        "id_questao": obj.get("id_questao"), "problem_id": obj.get("problem_id"),
                        "timestamp_resposta": pd.to_datetime(obj.get("timestamp"), errors="coerce"),
                        "efficiency": extras.get("efficiency"), "time_complexity": extras.get("time_complexity"),
                        "space_complexity": extras.get("space_complexity"), "energy_implications": extras.get("energy_implications"),
                        "explanation": extras.get("explanation"),
                        "starter_ok": obj.get("starter_ok"), "starter_check_reason": obj.get("starter_check_reason"),
                        "code_extracted_previous": obj.get("code_extracted_previous"),
                        "reavaliado_em": pd.to_datetime(obj.get("reavaliado_em"), errors="coerce"),
                    })
                except Exception: continue
    df=pd.DataFrame(rows)
    if not df.empty:
        for c in ["modelo","linguagem","titulo"]: df[c]=df[c].astype("category")
        df=add_normalized_keys(df,is_llm=True)
        if "starter_ok" in df.columns: df["starter_ok"]=df["starter_ok"].fillna(False).astype(bool)
    return df

# ----------- LeetCode enrichment (sempre ativo) -----------
LC_QUERY = """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    questionId
    titleSlug
    difficulty
    categoryTitle
    topicTags { name slug }
  }
}
"""
def _lc_headers()->Dict[str,str]:
    return {"Content-Type":"application/json","Referer":"https://leetcode.com","Origin":"https://leetcode.com","User-Agent":"Mozilla/5.0"}

@st.cache_data(show_spinner=False)
def fetch_leetcode_meta_for_slugs(slugs: List[str])->Dict[str,Dict[str,Any]]:
    meta={}
    for slug in slugs:
        if not slug: continue
        try:
            resp=requests.post(LEETCODE_GRAPHQL,json={"query":LC_QUERY,"variables":{"titleSlug":slug}},headers=_lc_headers(),timeout=10)
            if resp.status_code!=200: continue
            q=resp.json().get("data",{}).get("question")
            if not q: continue
            meta[slug]={"questionId":str(q.get("questionId") or ""),"difficulty":q.get("difficulty") or "",
                        "category":q.get("categoryTitle") or "","topicTags":[t.get("name") for t in (q.get("topicTags") or []) if t and t.get("name")] }
        except Exception: continue
    return meta

def try_load_local_meta(path="meta/leetcode_meta.json")->Dict[str,Dict[str,Any]]:
    p=Path(path)
    try:
        return json.load(open(p,"r",encoding="utf-8")) if p.exists() else {}
    except Exception: return {}

def merge_leetcode_meta(df: pd.DataFrame, remote: Dict[str,Dict[str,Any]], local: Dict[str,Dict[str,Any]]=None)->pd.DataFrame:
    if df.empty: return df
    df2=df.copy()
    df2["lc_difficulty"]=df2["slug"].map(lambda s:(remote.get(s) or {}).get("difficulty") if isinstance(s,str) else None)
    df2["lc_category"]=df2["slug"].map(lambda s:(remote.get(s) or {}).get("category") if isinstance(s,str) else None)
    df2["lc_topics"]=df2["slug"].map(lambda s:(remote.get(s) or {}).get("topicTags") if isinstance(s,str) else None)
    df2["lc_questionId"]=df2["slug"].map(lambda s:(remote.get(s) or {}).get("questionId") if isinstance(s,str) else None)
    if local:
        def from_local(r):
            if r.get("lc_difficulty") or r.get("lc_category"): return r.get("lc_difficulty"),r.get("lc_category"),r.get("lc_topics"),r.get("lc_questionId")
            slug=r.get("slug"); qid=str(r.get("qid") or ""); hit=local.get(slug) or local.get(qid)
            if hit: return hit.get("difficulty"),hit.get("category"),hit.get("topicTags"),str(hit.get("questionId") or "")
            return r.get("lc_difficulty"),r.get("lc_category"),r.get("lc_topics"),r.get("lc_questionId")
        df2[["lc_difficulty","lc_category","lc_topics","lc_questionId"]]=df2.apply(lambda r: pd.Series(from_local(r)),axis=1)
    df2["lc_difficulty"]=df2["lc_difficulty"].fillna("").astype("category")
    df2["lc_category"]=df2["lc_category"].fillna("").astype("category")
    return df2

# ------------------------------ UI ------------------------------
st.title("🚀 LeetCode Performance Dashboard")
with st.sidebar:
    lang = st.selectbox(UI["en"]["sidebar.language"], [UI["en"]["lang"], UI["pt"]["lang"]], index=0)
    _lang_key = "pt" if lang.startswith("Portugu") else "en"
    def k(name: str) -> str:
        return f"{name}__{_lang_key}"

    section = st.radio(t(lang,"sidebar.section"), [t(lang,"section.results"), t(lang,"section.compare")], index=0)
    st.markdown("---")
    st.caption(t(lang,"sidebar.cache"))
    if st.button(t(lang,"sidebar.clear")):
        st.cache_data.clear(); st.experimental_rerun()

df = load_and_preprocess_data(RESULTS_BASE)
df_llm = load_llm_answers(LLM_BASE)

if not df.empty:
    with st.spinner("Fetching LeetCode difficulty/category/tags..." if lang=="English" else "Consultando dificuldade/categoria/tópicos no LeetCode..."):
        slugs = sorted(set([s for s in df["slug"].dropna().astype(str).tolist() if s]))
        meta_r = fetch_leetcode_meta_for_slugs(slugs)
        meta_l = try_load_local_meta()
        df = merge_leetcode_meta(df, meta_r, meta_l)

# >>> mantemos df/df_llm internos em EN (não traduzimos in-place)
# tr_df(df, VAL_RES_COLS, lang)
# tr_df(df_llm, VAL_LLM_COLS, lang)

def col_labels_results(lang):
    return {
        "modelo": t(lang,"labels.model"),
        "linguagem": t(lang,"labels.language"),
        "titulo": t(lang,"labels.question"),
        "slug": t(lang,"labels.slug"),
        "qid": t(lang,"labels.qid"),
        "lc_difficulty": t(lang,"labels.lc_diff"),
        "lc_category": t(lang,"labels.lc_cat"),
        "lc_topics": t(lang,"labels.lc_topics"),
        "status_final": t(lang,"labels.final_status"),
        "categoria_calculada": t(lang,"labels.cat_calc"),
        "categoria_declarada": t(lang,"labels.cat_decl"),
        "categoria_leetcode": t(lang,"labels.cat_lc"),
        "approval_rate": t(lang,"labels.approval"),
        "tempo_s": t(lang,"labels.time_s"),
        "memoria_mb": t(lang,"labels.mem_mb"),
        "testes_passados": t(lang,"labels.tests_passed"),
        "total_testes": t(lang,"labels.tests_total"),
        "percentil_tempo": t(lang,"labels.time_pct"),
        "percentil_memoria": t(lang,"labels.mem_pct"),
        "compilou": t(lang,"labels.compiled"),
        "erro_compilacao": t(lang,"labels.comp_err"),
        "erro_runtime": t(lang,"labels.run_err"),
        "ultimo_caso_teste": t(lang,"labels.last_case"),
        "saida_esperada": t(lang,"labels.exp_out"),
        "sua_saida": t(lang,"labels.your_out"),
        "input": t(lang,"labels.input"),
        "timestamp": t(lang,"labels.timestamp"),
    }

def col_labels_llm(lang):
    return {
        "modelo": t(lang,"labels.model"),
        "linguagem": t(lang,"labels.language"),
        "titulo": t(lang,"labels.question"),
        "qid": t(lang,"labels.qid"),
        "lc_difficulty": t(lang,"labels.lc_diff"),
        "lc_category": t(lang,"labels.lc_cat"),
        "categoria_llm": t(lang,"labels.cat_llm"),
        "status_llm": t(lang,"labels.status_llm"),
        "motivo_llm": t(lang,"labels.reason_llm"),
        "categoria_calculada": t(lang,"labels.cat_calc"),
        "categoria_declarada": t(lang,"labels.cat_decl"),
        "status_final_result": t(lang,"labels.final_status"),
        "aceito_result": t(lang,"status.accepted") if not lang.startswith("Portugu") else t(lang,"status.aceito"),
        "tempo_s_result": t(lang,"labels.time_s"),
        "memoria_mb_result": t(lang,"labels.mem_mb"),
        "code_len": "Code length" if not lang.startswith("Portugu") else "Tamanho do código",
        "efficiency": t(lang,"labels.efficiency"),
        "time_complexity": t(lang,"labels.time_complexity"),
        "space_complexity": t(lang,"labels.space_complexity"),
        "energy_implications": t(lang,"labels.energy"),
        "starter_ok": t(lang,"labels.starter_ok"),
        "starter_check_reason": t(lang,"labels.starter_reason"),
        "reavaliado_em": t(lang,"labels.rechecked_at"),
    }

# ------------------------------ SECTION: Results ------------------------------
if section == t(lang,"section.results"):
    if df.empty:
        st.warning("No results found in 'out/'." if not lang.startswith("Portugu") else "Nenhum resultado em 'out/'.")
    else:
        with st.expander(t(lang,"quality.title")):
            probs=[]
            if df["modelo"].isna().any(): probs.append(t(lang,"quality.missing_model"))
            if df["linguagem"].isna().any(): probs.append(t(lang,"quality.missing_lang"))
            if df["qid"].isna().any(): probs.append(t(lang,"quality.missing_qid"))
            if df["tempo_s"].isna().all(): probs.append(t(lang,"quality.missing_time"))
            if df["memoria_mb"].isna().all(): probs.append(t(lang,"quality.missing_mem"))
            if "timestamp" in df.columns and df["timestamp"].notna().any():
                st.write(f"{t(lang,'quality.with_ts')} {df['timestamp'].notna().sum()} / {len(df)}")
            if "lc_difficulty" in df.columns and df["lc_difficulty"].astype(str).str.len().gt(0).any():
                st.write(f"{t(lang,'quality.with_diff')} {df['lc_difficulty'].astype(str).str.len().gt(0).sum()} / {len(df)}")
            if "lc_category" in df.columns and df["lc_category"].astype(str).str.len().gt(0).any():
                st.write(f"{t(lang,'quality.with_cat')} {df['lc_category'].astype(str).str.len().gt(0).sum()} / {len(df)}")
            if probs: 
                for p in probs: st.write("• " + p)
            else:
                st.write(t(lang,"quality.noissues"))

        st.markdown("### " + t(lang,"filters.title"))
        f_top1, f_top2 = st.columns(2)
        with f_top1:
            models = sorted(df["modelo"].dropna().unique().tolist())
            sel_models = st.multiselect(t(lang,"filters.models"), options=models, default=models)
            langs = sorted(df["linguagem"].dropna().unique().tolist())
            sel_langs = st.multiselect(t(lang,"filters.langs"), options=langs, default=langs)
        with f_top2:
            quests = sorted(df["titulo"].dropna().unique().tolist())
            sel_quests = st.multiselect(t(lang,"filters.questions"), options=quests, default=quests)
            q_text = st.text_input(t(lang,"filters.search"), value="")

        f_mid1, f_mid2, f_mid3 = st.columns([1,1,1])
        with f_mid1:
            statuses_internal = sorted(df["status_final"].dropna().astype(str).unique().tolist())
            statuses_display  = translate_list_for_display(statuses_internal, lang)
            sel_statuses_display = st.multiselect(
                t(lang,"filters.status"),
                options=statuses_display,
                default=statuses_display,
                key=k("filters.status")
            )
            sel_statuses = map_display_back(sel_statuses_display, statuses_internal, lang)
        with f_mid2:
            hide_ok = st.checkbox(t(lang,"filters.hide_accepted"), value=False, key=k("filters.hide_accepted"))
            dedup_latest = st.checkbox(t(lang,"filters.dedup"), value=False)
        with f_mid3:
            tmax = float(np.nanmax(df["tempo_s"])) if df["tempo_s"].notna().any() else 0.0
            mmax = float(np.nanmax(df["memoria_mb"])) if df["memoria_mb"].notna().any() else 0.0
            sel_t = st.slider(t(lang,"filters.time"), 0.0, float(max(tmax,0.0)), (0.0,float(max(tmax,0.0))))
            sel_m = st.slider(t(lang,"filters.memory"), 0.0, float(max(mmax,0.0)), (0.0,float(max(mmax,0.0))))

        have_lc_diff = "lc_difficulty" in df.columns and df["lc_difficulty"].astype(str).str.len().gt(0).any()
        if have_lc_diff:
            diffs_display = diff_display_list_from_df(df, lang)
            sel_diffs_display = st.multiselect(
                t(lang,"filters.diff"),
                options=diffs_display, default=diffs_display,
                key=k("filters.diff")
            )
            sel_diffs = diff_to_internal(sel_diffs_display, lang)
        else:
            sel_diffs=None

        if "lc_category" in df.columns and df["lc_category"].astype(str).str.len().gt(0).any():
            cats = sorted([c for c in df["lc_category"].dropna().astype(str).unique().tolist() if c])
            sel_cats = st.multiselect(
                t(lang,"filters.cat"),
                options=cats, default=cats,
                key=k("filters.cat")
            )
        else:
            sel_cats=None

        work=df.copy()
        if dedup_latest and "timestamp" in work.columns and work["timestamp"].notna().any():
            work = work.sort_values("timestamp").drop_duplicates(subset=["modelo","linguagem","qid"], keep="last")

        mask_text = True
        if q_text:
            mask_text = work["titulo"].astype(str).str.contains(q_text,case=False,na=False) | work["slug"].astype(str).str.contains(q_text,case=False,na=False)
        mask_status = work["status_final"].isin(sel_statuses) if sel_statuses else True
        if hide_ok:
            mask_status = mask_status & (work["status_final"] != "Accepted")
        mask_ranges = work["tempo_s"].fillna(0).between(sel_t[0],sel_t[1]) & work["memoria_mb"].fillna(0).between(sel_m[0],sel_m[1])
        mask_diff = True if sel_diffs is None else (work["lc_difficulty"].astype(str).isin(sel_diffs) | (work["lc_difficulty"].astype(str)==""))
        mask_cat = True if sel_cats is None else (work["lc_category"].astype(str).isin(sel_cats) | (work["lc_category"].astype(str)==""))

        filtered=work[
            (work["modelo"].isin(sel_models) if sel_models else True) &
            (work["linguagem"].isin(sel_langs) if sel_langs else True) &
            (work["titulo"].isin(sel_quests) if sel_quests else True) &
            mask_status & mask_text & mask_ranges & mask_diff & mask_cat
        ]

        if filtered.empty:
            st.warning(t(lang,"warn.nodata"))
        else:
            st.subheader(t(lang,"kpi.title"))
            c1,c2,c3,c4=st.columns(4)
            total=len(filtered)
            acc_label = "Accepted"
            rate=(filtered["status_final"]==acc_label).mean() if total else 0
            c1.metric(t(lang,"kpi.total"), f"{total}")
            c2.metric(t(lang,"kpi.accept"), f"{rate:.2%}")
            c3.metric(t(lang,"kpi.time"), f"{(filtered['tempo_s'].dropna().mean() or 0):.3f} s")
            c4.metric(t(lang,"kpi.mem"), f"{(filtered['memoria_mb'].dropna().mean() or 0):.2f} MB")

            st.markdown("---")
            left,right=st.columns(2, gap="large")
            with left:
                st.subheader(t(lang,"chart.result_dist"))
                vc=filtered["status_final"].value_counts()
                names_disp=[tr_val(n,lang) for n in vc.index]
                fig=px.pie(names=names_disp, values=vc.values, hole=0.42)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                beautify(fig,height=420); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
            with right:
                st.subheader(t(lang,"chart.accept_by_model"))
                ok=(filtered["status_final"]==acc_label)
                acc = (ok.groupby(filtered["modelo"]).mean()*100).reindex(sel_models or filtered["modelo"].unique(), fill_value=0)
                fig=px.bar(acc, x=wrap_labels(acc.index,16), y=acc.values,
                           labels={"x":t(lang,"axis.model"),"y":t(lang,"axis.acceptance")},
                           text=[f"{v:.1f}" for v in acc.values])
                fig.update_traces(textposition="outside", cliponaxis=False)
                beautify(fig,height=460,xrotate=-30); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

            st.markdown("---")
            e1,e2=st.columns(2, gap="large")
            with e1:
                st.subheader(t(lang,"chart.time_vs_mem"))
                fig=px.scatter(filtered, x="tempo_s", y="memoria_mb", color="modelo",
                               hover_data=["titulo","linguagem","status_final","qid","lc_difficulty","lc_category"],
                               labels={"tempo_s":t(lang,"axis.time"),"memoria_mb":t(lang,"axis.memory")})
                beautify(fig,height=480); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
            with e2:
                st.subheader(t(lang,"chart.status_stacked"))
                ctab=(filtered.groupby(["modelo","status_final"]).size().reset_index(name="count"))
                ctab["status_final_display"]=ctab["status_final"].map(lambda x: tr_val(x,lang))
                fig=px.bar(ctab, x="modelo", y="count", color="status_final_display",
                           labels={"count":t(lang,"axis.count"),"modelo":t(lang,"axis.model"),"status_final_display":"Status"})
                beautify(fig,height=480,xrotate=-30); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

            if "lc_difficulty" in filtered.columns and filtered["lc_difficulty"].astype(str).str.len().gt(0).any():
                st.markdown("---")
                g1,g2=st.columns(2, gap="large")
                with g1:
                    st.subheader(t(lang,"chart.acc_by_diff"))
                    tmp=(filtered.assign(_ok=(filtered["status_final"]==acc_label)).groupby("lc_difficulty")["_ok"].mean().mul(100).sort_values(ascending=False))
                    idx_disp=[EN2PT_DIFF.get(i,i) if lang.startswith("Portugu") else i for i in tmp.index]
                    fig=px.bar(tmp, x=idx_disp, y=tmp.values, labels={"x":t(lang,"labels.lc_diff"),"y":t(lang,"axis.acceptance")},
                               text=[f"{v:.1f}" for v in tmp.values])
                    fig.update_traces(textposition="outside",cliponaxis=False)
                    beautify(fig,height=420); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
                with g2:
                    st.subheader(t(lang,"chart.hm_calc_vs_diff"))
                    hm=pd.crosstab(fillna_dash(filtered["categoria_calculada"]), fillna_dash(filtered["lc_difficulty"]))
                    if lang.startswith("Portugu"):
                        hm.columns=[EN2PT_DIFF.get(c,c) for c in hm.columns]
                    hm.index=[tr_val(i,lang) for i in hm.index]
                    fig=px.imshow(hm, text_auto=True, aspect="auto", labels=dict(x=t(lang,"labels.lc_diff"), y=t(lang,"labels.cat_calc"), color=t(lang,"axis.count")))
                    beautify(fig,height=420); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

            if "lc_category" in filtered.columns and filtered["lc_category"].astype(str).str.len().gt(0).any():
                st.markdown("---")
                c1_,c2_=st.columns(2, gap="large")
                with c1_:
                    st.subheader(t(lang,"chart.acc_by_cat"))
                    tmp=(filtered.assign(_ok=(filtered["status_final"]==acc_label)).groupby("lc_category")["_ok"].mean().mul(100).sort_values(ascending=False))
                    fig=px.bar(tmp, x=tmp.index, y=tmp.values, labels={"x":t(lang,"labels.lc_cat"),"y":t(lang,"axis.acceptance")},
                               text=[f"{v:.1f}" for v in tmp.values])
                    fig.update_traces(textposition="outside",cliponaxis=False)
                    beautify(fig,height=420,xrotate=-15); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
                with c2_:
                    st.subheader(t(lang,"chart.hm_calc_vs_cat"))
                    hm2=pd.crosstab(fillna_dash(filtered["categoria_calculada"]), fillna_dash(filtered["lc_category"]))
                    hm2.index=[tr_val(i,lang) for i in hm2.index]
                    fig=px.imshow(hm2, text_auto=True, aspect="auto", labels=dict(x=t(lang,"labels.lc_cat"), y=t(lang,"labels.cat_calc"), color=t(lang,"axis.count")))
                    beautify(fig,height=420); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

            st.markdown("---")
            st.subheader(t(lang,"chart.acc_by_question"))
            qperf=(filtered.groupby("titulo")["status_final"].apply(lambda s: (s==acc_label).mean()*100).sort_values(ascending=True))
            fig=px.bar(qperf, y=wrap_labels(qperf.index,38), x=qperf.values, orientation="h",
                       labels={"x":t(lang,"axis.acceptance"),"y":t(lang,"labels.question")})
            fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside", cliponaxis=False)
            beautify(fig,height=max(420, 28*len(qperf)+80)); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

            st.markdown("---")
            st.subheader(t(lang,"export.title"))
            show_cols=["modelo","linguagem","titulo","slug","qid","lc_difficulty","lc_category","lc_topics","status_final",
                       "categoria_calculada","categoria_declarada","categoria_leetcode","approval_rate","tempo_s","memoria_mb",
                       "testes_passados","total_testes","percentil_tempo","percentil_memoria","compilou","erro_compilacao","erro_runtime",
                       "ultimo_caso_teste","saida_esperada","sua_saida","input","timestamp"]
            show_cols=[c for c in show_cols if c in filtered.columns]
            display_df = filtered[show_cols].copy()
            tr_df(display_df, VAL_RES_COLS, lang)
            st.dataframe(display_df.rename(columns=col_labels_results(lang)), use_container_width=True)
            st.download_button(t(lang,"export.csv"), data=filtered.to_csv(index=False).encode("utf-8"),
                               file_name="filtered_results.csv", mime="text/csv")
            st.download_button(t(lang,"export.json"), data=filtered.to_json(orient="records",force_ascii=False).encode("utf-8"),
                               file_name="filtered_results.json", mime="application/json")

# ------------------------------ SECTION: Compare ------------------------------
elif section == t(lang,"section.compare"):
    st.subheader(t(lang,"compare.title"))
    if df_llm.empty:
        st.info("Nothing found in 'data/'." if not lang.startswith("Portugu") else "Nada encontrado em 'data/'.")
    else:
        df_norm = add_normalized_keys(df, is_llm=False) if not df.empty else pd.DataFrame()
        df_llm_norm = add_normalized_keys(df_llm, is_llm=True).copy()
        df_llm_norm["row_id"]=np.arange(len(df_llm_norm))

        meta_cols=[c for c in ["lc_difficulty","lc_category"] if (not df_norm.empty and c in df_norm.columns)]
        if meta_cols:
            df_llm_norm = df_llm_norm.merge(df_norm[["qid"]+meta_cols].drop_duplicates(), on="qid", how="left")

        res_side=pd.DataFrame()
        if not df_norm.empty:
            res_side = df_norm[["qid","modelo_norm","linguagem","lang_norm","timestamp","status_final","aceito","tempo_s","memoria_mb","categoria_calculada","categoria_declarada"]].rename(columns={
                "modelo_norm":"modelo_norm_res","lang_norm":"lang_norm_res","linguagem":"linguagem_res","timestamp":"timestamp_res",
                "status_final":"status_final_result","aceito":"aceito_result","tempo_s":"tempo_s_result","memoria_mb":"memoria_mb_result"})
        if not res_side.empty:
            cand = df_llm_norm.merge(res_side,on="qid",how="left")
            cand["score_match"]=(cand["modelo_norm"]==cand["modelo_norm_res"]).astype(int)
            cand["score_match"]+=(cand["lang_norm"]==cand["lang_norm_res"]).astype(int)
            cand=cand.sort_values(["row_id","score_match","timestamp_res"],ascending=[True,False,False])
            df_join=cand.drop_duplicates(subset=["row_id"],keep="first").copy()
        else:
            df_join=df_llm_norm.copy()
            for col in ["status_final_result","aceito_result","tempo_s_result","memoria_mb_result","modelo_norm_res","lang_norm_res","timestamp_res","linguagem_res","categoria_calculada","categoria_declarada"]:
                df_join[col]=pd.NA

        st.markdown("#### " + t(lang,"compare.join_diag"))
        missing_qid_llm=df_llm_norm["qid"].isna().sum()
        missing_qid_res=df_norm["qid"].isna().sum() if not df_norm.empty else 0
        st.write(f"LLM without qid: **{missing_qid_llm}** · Results without qid: **{missing_qid_res}**")
        matched=df_join["status_final_result"].notna()
        st.write(f"Matched: **{matched.sum()}** · Unmatched: **{(~matched).sum()}**")
        if (~matched).sum()>0:
            wanted=["modelo","linguagem","qid","titulo","slug","slugish"]
            df_diag=_ensure_columns(df_join,wanted)
            if "linguagem" not in df_diag.columns or df_diag["linguagem"].isna().all():
                for alt in ["linguagem_res","linguagem_x","linguagem_y","lang_norm"]:
                    if alt in df_diag.columns: df_diag["linguagem"]=df_diag["linguagem"].fillna(df_diag[alt])
            with st.expander(t(lang,"compare.unmatched")):
                st.dataframe(df_diag.loc[~matched,wanted].drop_duplicates().sort_values(["qid","modelo","linguagem"]), use_container_width=True)

        model_col, model_opts=_pick_options_col(df_join,"modelo",["modelo_norm"])
        lang_col, lang_opts=_pick_options_col(df_join,"linguagem",["linguagem_x","linguagem_y","linguagem_res","lang_norm"])
        quest_col, quest_opts=_pick_options_col(df_join,"titulo",["slug","slugish"])

        c1,c2,c3=st.columns(3)
        with c1: sel_models = st.multiselect(t(lang,"filters.models"), options=model_opts, default=model_opts)
        with c2: sel_langs = st.multiselect(t(lang,"filters.langs"), options=lang_opts, default=lang_opts)
        with c3: sel_quests = st.multiselect(t(lang,"filters.questions"), options=quest_opts, default=quest_opts)

        dfj=df_join[
            (df_join[model_col].isin(sel_models) if len(sel_models)>0 else True) &
            (df_join[lang_col].isin(sel_langs) if len(sel_langs)>0 else True) &
            (df_join[quest_col].isin(sel_quests) if len(sel_quests)>0 else True)
        ].copy()
        if dfj.empty:
            st.warning(t(lang,"warn.nodata"))
        else:
            if "linguagem" not in dfj.columns or dfj["linguagem"].isna().all(): dfj["linguagem"]=dfj[lang_col]
            if "titulo" not in dfj.columns or dfj["titulo"].isna().all(): dfj["titulo"]=dfj[quest_col]

            st.markdown("### " + t(lang,"compare.summary"))
            sum_cols=["modelo","linguagem","titulo","qid","lc_difficulty","lc_category","categoria_llm","status_llm","motivo_llm",
                      "categoria_calculada","categoria_declarada","status_final_result","aceito_result","tempo_s_result","memoria_mb_result",
                      "code_len","efficiency","time_complexity","space_complexity","energy_implications","starter_ok","starter_check_reason","reavaliado_em"]
            sum_cols=[c for c in sum_cols if c in dfj.columns]
            summary_df = dfj[sum_cols].copy()
            tr_df(summary_df, VAL_LLM_COLS+["lc_difficulty","lc_category","categoria_calculada","categoria_declarada","status_final_result"], lang)
            sort_keys = [c for c in ["titulo","modelo","linguagem"] if c in summary_df.columns]
            if sort_keys:
                summary_df = summary_df.sort_values(sort_keys)
            st.dataframe(summary_df.rename(columns=col_labels_llm(lang)), use_container_width=True, height=440)

            with st.expander(t(lang,"compare.hm")):
                if "categoria_calculada" in dfj.columns and "lc_category" in dfj.columns:
                    hm=pd.crosstab(fillna_dash(dfj["categoria_calculada"]), fillna_dash(dfj["lc_category"]))
                    hm.index=[tr_val(i,lang) for i in hm.index]
                    fig=px.imshow(hm, text_auto=True, aspect="auto",
                                  labels=dict(x=t(lang,"labels.lc_cat"), y=t(lang,"labels.cat_calc"), color=t(lang,"axis.count")))
                    beautify(fig,height=420); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

            st.markdown("### " + t(lang,"compare.acc_for_sel"))
            if "aceito_result" in dfj.columns:
                denom_choice = st.radio(t(lang,"compare.denom"),
                                        [t(lang,"compare.denom.only"), t(lang,"compare.denom.all")],
                                        index=0, horizontal=True)
                g=dfj.groupby("modelo").agg(total_llm=("modelo","size"),
                                            matched=("aceito_result",lambda s: s.notna().sum()),
                                            accepted=("aceito_result",lambda s: s.fillna(False).astype(int).sum()))
                order = sel_models if sel_models else g.index.tolist()
                denom = g["matched"].replace(0,np.nan) if denom_choice==t(lang,"compare.denom.only") else g["total_llm"]
                g["acceptance_%"]=(g["accepted"]/denom*100).fillna(0)
                g=g.reindex(order).fillna({"acceptance_%":0,"total_llm":0,"matched":0,"accepted":0})

                fig=px.bar(g.reset_index(), x="modelo", y="acceptance_%",
                           labels={"acceptance_%":t(lang,"axis.acceptance"),"modelo":t(lang,"labels.model")},
                           text=g["acceptance_%"].map(lambda v:f"{v:.1f}"))
                fig.update_traces(textposition="outside",cliponaxis=False)
                beautify(fig,height=max(420,60+28*len(g)),xrotate=-30); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

                with st.expander(t(lang,"compare.details")):
                    try:
                        gi = g.reset_index(names="modelo")
                    except TypeError:
                        gi = g.reset_index().rename(columns={g.index.name or "index":"modelo"})
                    st.dataframe(gi[["modelo","total_llm","matched","accepted","acceptance_%"]]
                                 .rename(columns={"modelo": t(lang,"labels.model")}), use_container_width=True)

            st.markdown("### " + t(lang,"compare.code_size"))
            if "code_len" in dfj.columns:
                fig=px.box(dfj, x="modelo", y="code_len", points="suspectedoutliers",
                           labels={"code_len":"Code size (chars)" if not lang.startswith("Portugu") else "Tamanho do código (chars)",
                                   "modelo":t(lang,"labels.model")})
                beautify(fig,height=460,xrotate=-30); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

            st.markdown("### " + t(lang,"compare.correlations"))
            corr=[("code_len","tempo_s_result"),("code_len","aceito_result")]
            for x,y in corr:
                if x in dfj.columns and y in dfj.columns:
                    sub=dfj[[x,y]].dropna()
                    if not sub.empty:
                        try:
                            cval=sub[x].corr(sub[y]); st.write(f"Correlation {x} × {y}: **{cval:.3f}** (Pearson)")
                        except Exception: pass

            st.markdown("### " + t(lang,"compare.code_viewer"))
            def code_picker(df_base: pd.DataFrame, side_label: str):
                col1, col2, col3 = st.columns(3)
                with col1:
                    m_opts = sorted(df_base["modelo"].unique().tolist())
                    m_sel = st.selectbox(f"{side_label} · {t(lang,'labels.model')}", m_opts, key=f"{side_label}_m")
                with col2:
                    l_opts = sorted(df_base[df_base["modelo"]==m_sel]["linguagem"].unique().tolist())
                    l_sel = st.selectbox(f"{side_label} · {t(lang,'labels.language')}", l_opts, key=f"{side_label}_l")
                with col3:
                    q_opts = sorted(df_base[(df_base["modelo"]==m_sel)&(df_base["linguagem"]==l_sel)]["titulo"].unique().tolist())
                    q_sel = st.selectbox(f"{side_label} · {t(lang,'labels.question')}", q_opts, key=f"{side_label}_q")
                hit = df_base[(df_base["modelo"]==m_sel)&(df_base["linguagem"]==l_sel)&(df_base["titulo"]==q_sel)]
                if hit.empty: return None
                return hit.index[0]

            col_pick_a, col_pick_b = st.columns(2)
            with col_pick_a:
                pick_left = code_picker(dfj, t(lang,"compare.code_a"))
            with col_pick_b:
                pick_right = code_picker(dfj, t(lang,"compare.code_b"))

            colA,colB = st.columns(2, gap="large")
            if pick_left is not None:
                with colA:
                    st.caption(f"🅰️ {dfj.loc[pick_left,'modelo']} · {dfj.loc[pick_left,'linguagem']} · {dfj.loc[pick_left,'titulo']} · ID {dfj.loc[pick_left,'qid']}")
                    langA=str(dfj.loc[pick_left,'linguagem']).lower().replace("c++","cpp")
                    st.code(dfj.loc[pick_left,'code'] or "", language=langA if langA else None)
                    st.markdown("**" + t(lang,"compare.meta_a") + "**")
                    metaA_cols=["categoria_llm","status_llm","motivo_llm","efficiency","time_complexity","space_complexity","energy_implications","starter_ok","starter_check_reason","reavaliado_em"]
                    metaA={k: dfj.loc[pick_left,k] for k in metaA_cols if k in dfj.columns}
                    st.json({col_labels_llm(lang).get(k,k): tr_val(v,lang) for k,v in metaA.items()})
            if pick_right is not None:
                with colB:
                    st.caption(f"🅱️ {dfj.loc[pick_right,'modelo']} · {dfj.loc[pick_right,'linguagem']} · {dfj.loc[pick_right,'titulo']} · ID {dfj.loc[pick_right,'qid']}")
                    langB=str(dfj.loc[pick_right,'linguagem']).lower().replace("c++","cpp")
                    st.code(dfj.loc[pick_right,'code'] or "", language=langB if langB else None)
                    st.markdown("**" + t(lang,"compare.meta_b") + "**")
                    metaB_cols=["categoria_llm","status_llm","motivo_llm","efficiency","time_complexity","space_complexity","energy_implications","starter_ok","starter_check_reason","reavaliado_em"]
                    metaB={k: dfj.loc[pick_right,k] for k in metaB_cols if k in dfj.columns}
                    st.json({col_labels_llm(lang).get(k,k): tr_val(v,lang) for k,v in metaB.items()})

            csv_llm=dfj.to_csv(index=False).encode("utf-8")
            st.download_button(t(lang,"compare.download"), data=csv_llm, file_name="llm_vs_leetcode.csv", mime="text/csv")
