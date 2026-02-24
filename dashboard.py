# -*- coding: utf-8 -*-
"""
LeetCode LLM Performance Dashboard - Comprehensive Research Companion
=====================================================================
Aligned with the paper's three research questions:
  RQ1: Acceptance Rate   (OAR, ARM, ARD, ARL + cross-tabs)
  RQ2: Execution Time    (OAET, ETM, ETD, ETL + cross-tabs)
  RQ3: Memory Usage      (OAMU, MUM, MUD, MUL + cross-tabs)
Plus: Statistical Analysis (ANOVA, Kruskal-Wallis, Cohen's d, Pearson/Spearman)
"""
from __future__ import annotations

import json, re, textwrap, warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ==================================================================
# CONFIG
# ==================================================================
st.set_page_config(
    page_title="LLM Code Efficiency - Research Dashboard",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)
px.defaults.template = "plotly_white"
COLORS_MODEL = px.colors.qualitative.Set3
COLORS_LANG = {"C++": "#636EFA", "Java": "#EF553B", "Python3": "#00CC96"}
COLORS_DIFF = {"Easy": "#2ecc71", "Medium": "#f39c12", "Hard": "#e74c3c"}
COLORS_STATUS = {"Accepted": "#2ecc71", "Compilation Error": "#e74c3c", "Wrong Answer": "#f39c12"}

RESULTS_CSV = Path("out/results.csv")
RESULTS_BY_MODEL = Path("out/by_model")
LLM_DATA = Path("data")
DATASETS = Path("datasets/leetcode")

EXPECTED_PER_MODEL = 1_008  # 336 problems x 3 languages
NUM_MODELS = 11
TOTAL_EXPECTED = EXPECTED_PER_MODEL * NUM_MODELS  # 11,088

MODEL_SHORT = {
    "allam-2-7b": "allam-2-7b",
    "deepseek-r1-distill-llama-70b": "deepseek-r1-70b",
    "gemma2-9b-it": "gemma2-9b-it",
    "llama-3.1-8b-instant": "llama-3.1-8b",
    "llama-3.3-70b-versatile": "llama-3.3-70b",
    "llama3-70b-8192": "llama3-70b",
    "llama3-8b-8192": "llama3-8b",
    "meta-llama_llama-4-maverick-17b-128e-instruct": "llama-4-maverick",
    "meta-llama_llama-4-scout-17b-16e-instruct": "llama-4-scout",
    "moonshotai_kimi-k2-instruct": "kimi-k2",
    "qwen_qwen3-32b": "qwen3-32b",
}

DIFF_PT_EN = {"Fácil": "Easy", "Média": "Medium", "Difícil": "Hard"}
DIFF_ORDER = ["Easy", "Medium", "Hard"]
LANG_ORDER = ["C++", "Java", "Python3"]


# ==================================================================
# i18n - BILINGUAL SUPPORT (EN / PT-BR)
# ==================================================================
UI: Dict[str, Dict[str, str]] = {
    "en": {
        "sidebar.title": "\U0001F4CA Global Filters",
        "sidebar.caption": "These filters affect all tabs",
        "sidebar.language": "Output language",
        "sidebar.models": "Models",
        "sidebar.langs": "Languages",
        "sidebar.difficulty": "Difficulty",
        "sidebar.cache": "Cache",
        "sidebar.clear_cache": "\U0001F5D1\uFE0F Clear cache",
        "main.title": "\U0001F4CA LLM Code Efficiency - Research Dashboard",
        "main.caption": "Interactive dashboard for consulting the research data on LLM-generated code efficiency",
        "main.nodata": "No data found. Check that `out/` and `datasets/leetcode/` directories exist.",
        "tab.overview": "\U0001F4CA Overview",
        "tab.rq1": "\u2705 RQ1: Acceptance",
        "tab.rq2": "\u23F1\uFE0F RQ2: Execution Time",
        "tab.rq3": "\U0001F4BE RQ3: Memory Usage",
        "tab.stats": "\U0001F4C8 Statistical Analysis",
        "tab.explorer": "\U0001F50D Code Explorer",
        "tab.export": "\U0001F4CB Export Data",
        "ov.header": "Study Overview",
        "ov.desc": "This dashboard presents the results of a large-scale evaluation of **11 LLMs** available on the Groq Cloud platform, evaluated on **336 LeetCode problems** in **3 programming languages** (C++, Java, Python3).",
        "ov.total_subs": "Total Submissions",
        "ov.oar": "Acceptance Rate (OAR)",
        "ov.avg_time": "Avg Time (Accepted)",
        "ov.avg_mem": "Avg Memory (Accepted)",
        "ov.incomplete_header": "Incomplete Generations",
        "ov.incomplete_desc": "Some LLMs produced truncated or incomplete outputs that were not submitted to LeetCode.",
        "ov.total_expected": "Total Expected",
        "ov.total_sub_missing": "Total Submitted / Missing",
        "ov.dist_header": "Overall Outcome Distribution (OAR)",
        "ov.pie_title": "Distribution of 3 Statuses (Paper)",
        "ov.table_3cat": "**Results Table (3 categories - paper)**",
        "ov.table_granular": "**Granular breakdown (original LeetCode status)**",
        "ov.sample_header": "Sample Composition",
        "ov.by_diff": "**By Difficulty**",
        "ov.models_eval": "**Models Evaluated**",
        "ov.by_lang": "**Languages**",
        "ov.topic_header": "Distribution by Topic (tema_principal)",
        "ov.topic_chart": "Number of problems per topic in the sample",
        "col.model": "Model",
        "col.expected": "Expected",
        "col.submitted": "Submitted",
        "col.missing": "Missing",
        "col.result": "Result",
        "col.count": "Count",
        "col.status": "Status",
        "col.problems": "Problems",
        "col.submissions": "Submissions",
        "col.topic": "Topic",
        "rq1.header": "RQ1: Acceptance Rates",
        "rq1.desc": "*How do the acceptance rates of different LLMs vary by difficulty and language?*",
        "rq1.nodata": "No data available for the selected filters.",
        "rq1.ard_header": "ARD - Acceptance Rate by Difficulty",
        "rq1.ard_chart": "Acceptance Rate by Difficulty (ARD)",
        "rq1.arm_header": "ARM - Acceptance Rate by Model",
        "rq1.arm_chart": "Acceptance Rate by Model (ARM)",
        "rq1.arl_header": "ARL - Acceptance Rate by Language",
        "rq1.arl_chart": "Acceptance Rate by Language (ARL)",
        "rq1.crosstabs": "Cross-Tabulations",
        "rq1.ct_md": "Model x Difficulty",
        "rq1.ct_ml": "Model x Language",
        "rq1.ct_ld": "Language x Difficulty",
        "rq1.ct_md_title": "**Acceptance Rate (%) by Model and Difficulty**",
        "rq1.ct_ml_title": "**Acceptance Rate (%) by Model and Language**",
        "rq1.ct_ld_title": "**Acceptance Rate (%) by Language and Difficulty**",
        "rq1.hm_md": "Heatmap: Acceptance (%) - Model x Difficulty",
        "rq1.hm_ml": "Heatmap: Acceptance (%) - Model x Language",
        "rq1.fail_header": "Failure Distribution by Model",
        "rq1.fail_stacked": "Status by Model (stacked)",
        "rq1.fail_granular": "Granular status breakdown (7 original categories)",
        "rq1.byq_header": "Acceptance Rate by Problem",
        "rq1.byq_chart": "Acceptance by Problem",
        "rq2.header": "RQ2: Execution Time",
        "rq2.desc": "*How does the average execution time of accepted solutions vary by model, difficulty and language?*\n\n**Note:** Metrics computed only on accepted submissions.",
        "rq2.nodata": "No accepted submissions for the selected filters.",
        "rq2.oaet_header": "OAET - Overall Execution Time Statistics",
        "rq2.mean": "Mean",
        "rq2.median": "Median",
        "rq2.std": "Std Dev",
        "rq2.p99": "P99",
        "rq2.max": "Maximum",
        "rq2.based_on": "Based on {n:,} accepted submissions",
        "rq2.hist_header": "Execution Time Distribution",
        "rq2.hist_title": "Execution Time Distribution (Accepted)",
        "rq2.etd_header": "ETD - Execution Time by Difficulty",
        "rq2.etd_chart": "Mean Time by Difficulty (ETD)",
        "rq2.etm_header": "ETM - Execution Time by Model",
        "rq2.etm_chart": "Mean Time by Model (ETM) - sorted",
        "rq2.etl_header": "ETL - Execution Time by Language",
        "rq2.etl_chart": "Mean Time by Language (ETL)",
        "rq2.ct_header": "Cross-Tabulations - Execution Time",
        "rq2.ct_md_title": "**Mean Time (ms) by Model and Difficulty**",
        "rq2.ct_ml_title": "**Mean Time (ms) by Model and Language**",
        "rq2.ct_ld_title": "**Mean Time (ms) by Language and Difficulty**",
        "rq2.hm_md": "Heatmap: Mean Time (ms) - Model x Difficulty",
        "rq2.hm_ml": "Heatmap: Mean Time (ms) - Model x Language",
        "rq2.box_header": "Distributions (Box Plots)",
        "rq2.box_model": "By Model",
        "rq2.box_lang": "By Language",
        "rq2.box_diff": "By Difficulty",
        "rq2.box_model_title": "Time Distribution by Model",
        "rq2.box_lang_title": "Time Distribution by Language",
        "rq2.box_diff_title": "Time Distribution by Difficulty",
        "rq3.header": "RQ3: Memory Usage",
        "rq3.desc": "*How does the average memory usage of accepted solutions vary by model, difficulty and language?*\n\n**Note:** Metrics computed only on accepted submissions.",
        "rq3.nodata": "No accepted submissions for the selected filters.",
        "rq3.oamu_header": "OAMU - Overall Memory Usage Statistics",
        "rq3.hist_header": "Memory Usage Distribution",
        "rq3.hist_title": "Memory Usage Distribution (Accepted)",
        "rq3.mud_header": "MUD - Memory Usage by Difficulty",
        "rq3.mud_chart": "Mean Memory by Difficulty (MUD)",
        "rq3.mum_header": "MUM - Memory Usage by Model",
        "rq3.mum_chart": "Mean Memory by Model (MUM) - sorted",
        "rq3.mul_header": "MUL - Memory Usage by Language",
        "rq3.mul_chart": "Mean Memory by Language (MUL)",
        "rq3.ct_header": "Cross-Tabulations - Memory Usage",
        "rq3.ct_md_title": "**Mean Memory (MB) by Model and Difficulty**",
        "rq3.ct_ml_title": "**Mean Memory (MB) by Model and Language**",
        "rq3.ct_ld_title": "**Mean Memory (MB) by Language and Difficulty**",
        "rq3.hm_md": "Heatmap: Mean Memory (MB) - Model x Difficulty",
        "rq3.hm_ml": "Heatmap: Mean Memory (MB) - Model x Language",
        "rq3.box_header": "Distributions (Box Plots)",
        "rq3.box_model_title": "Memory Distribution by Model",
        "rq3.box_lang_title": "Memory Distribution by Language",
        "rq3.box_diff_title": "Memory Distribution by Difficulty",
        "rq3.tradeoff_header": "Speed x Memory Trade-off",
        "rq3.tradeoff_chart": "Trade-off: Mean Time x Mean Memory by Language",
        "st.header": "Statistical Analysis",
        "st.desc": "Analyses inspired by Guimaraes et al. (2025):\nANOVA, Kruskal-Wallis, Cohen's d, Pearson and Spearman correlations.",
        "st.no_scipy": "\u26A0\uFE0F `scipy` library not found. Install with `pip install scipy` to enable statistical tests.",
        "st.nodata": "No data available.",
        "st.anova_header": "ANOVA and Effect Size (eta squared)",
        "st.anova_desc": "eta squared thresholds: **small** (< 0.06), **medium** (0.06-0.14), **large** (> 0.14)",
        "st.insufficient": "Insufficient data for statistical tests.",
        "st.cohens_lang_header": "Cohen's d - Language Comparisons",
        "st.cohens_desc": "Thresholds: **negligible** (|d| < 0.2), **small** (0.2-0.5), **medium** (0.5-0.8), **large** (> 0.8)",
        "st.cohens_diff_header": "Cohen's d - Difficulty Comparisons (Acceptance)",
        "st.corr_header": "Correlations Between Metrics (Model Level)",
        "st.corr_desc": "Pearson (r) and Spearman (rho) between acceptance rate, mean time and mean memory, aggregated by model.",
        "st.corr_viz_header": "Correlation Visualizations",
        "st.corr_acc_mem": "Acceptance vs Mean Memory (per model)",
        "st.corr_acc_time": "Acceptance vs Mean Time (per model)",
        "st.corr_need": "scipy required for correlation calculations.",
        "st.corr_insuf": "Insufficient data.",
        "st.composite_header": "Composite Score (Trade-off)",
        "st.composite_desc": "Normalized ranking combining acceptance (up), execution time (down) and memory (down), with equal weights. Higher values = better overall balance.",
        "st.composite_chart": "Composite Score by Model (higher = better)",
        "st.rec_header": "Practical Recommendations",
        "st.rec_body": """Based on the combined analysis of the three RQs:

| Priority | Recommendation |
|---|---|
| **Maximize acceptance** | `kimi-k2-instruct` (79.4%), `qwen3-32b` (72.6%), `deepseek-r1-70b` (67.0%). C++ and Java ~48%, both above Python3 (32.5%). |
| **Minimize time** | `llama-3.3-70b` (25.6ms), `llama-3.1-8b` (29.2ms). Java (24.5ms) ~ C++ (26.7ms) << Python3 (119.9ms). |
| **Minimize memory** | Python3 (20.9MB) < C++ (28.9MB) < Java (46.9MB). `llama3-8b` (29.2MB), `llama-3.1-8b` (29.3MB). |
| **Best balance** | `llama-3.1-8b-instant` (composite 0.73): moderate acceptance + low time and memory. |

**No model or language simultaneously dominates across all three metrics.**""",
        "stcol.factor": "Factor",
        "stcol.metric": "Metric",
        "stcol.acceptance": "Acceptance",
        "stcol.time": "Time (ms)",
        "stcol.memory": "Memory (MB)",
        "stcol.effect": "Effect",
        "stcol.pair": "Pair",
        "stcol.magnitude": "Magnitude",
        "stcol.trend": "Trend",
        "ax.acceptance_pct": "Acceptance (%)",
        "ax.model": "Model",
        "ax.language": "Language",
        "ax.difficulty": "Difficulty",
        "ax.time_ms": "Time (ms)",
        "ax.memory_mb": "Memory (MB)",
        "ax.count": "Count",
        "ax.frequency": "Frequency",
        "ax.mean_time_ms": "Mean Time (ms)",
        "ax.mean_memory_mb": "Mean Memory (MB)",
        "tc.difficulty": "Difficulty",
        "tc.accepted": "Accepted",
        "tc.total": "Total",
        "tc.rate_pct": "Rate%",
        "tc.language": "Language",
        "tc.mean_ms": "Mean (ms)",
        "tc.median_ms": "Median (ms)",
        "tc.std_ms": "SD (ms)",
        "tc.mean_mb": "Mean (MB)",
        "tc.median_mb": "Median (MB)",
        "tc.std_mb": "SD (MB)",
        "tc.n": "N",
        "ex.header": "Code Explorer",
        "ex.desc": "View the code generated by LLMs and the LeetCode submission results.",
        "ex.nodata": "No LLM responses found in `data/`.",
        "ex.model": "Model",
        "ex.language": "Language",
        "ex.problem": "Problem (ID)",
        "ex.code_title": "**Generated Code**",
        "ex.meta_title": "**LLM Metadata**",
        "ex.category": "Category",
        "ex.reason": "Reason",
        "ex.starter_ok": "Starter OK",
        "ex.result_title": "**LeetCode Result**",
        "ex.status": "Status",
        "ex.time": "Time",
        "ex.memory": "Memory",
        "ex.problem_info": "**Problem Info**",
        "ex.title_field": "Title",
        "ex.diff_field": "Difficulty",
        "ex.topic_field": "Topic",
        "ex.no_result": "No submission result found for this combination.",
        "ex.empty_code": "(empty)",
        "ex.compare_header": "Side-by-Side Comparison",
        "ex.model_a": "**Model A**",
        "ex.model_b": "**Model B**",
        "ex.compare_problem": "Problem to compare",
        "ex.no_common": "No common problems between the selections.",
        "exp.header": "Export Data",
        "exp.desc": "Download filtered data in various formats for external analyses.",
        "exp.nodata": "No data available.",
        "exp.sub_header": "Submission Data (Filtered)",
        "exp.record_count": "{n:,} records with current filters",
        "exp.showing_500": "Showing first 500 rows. Download to see all data.",
        "exp.dl_csv_all": "\U0001F4E5 Download CSV (All submissions)",
        "exp.dl_json_all": "\U0001F4E5 Download JSON (All submissions)",
        "exp.dl_csv_acc": "\U0001F4E5 Download CSV (Accepted only)",
        "exp.tables_header": "Summary Tables for the Paper",
        "exp.tables_desc": "Ready-to-use tables for copy/paste into the paper.",
        "exp.table_oar": "Table: Overall Distribution (OAR)",
        "exp.table_arm_diff": "Table: ARM x Difficulty",
        "exp.table_arm_lang": "Table: ARM x Language",
        "exp.table_etm_diff": "Table: ETM x Difficulty",
        "exp.table_etm_lang": "Table: ETM x Language",
        "exp.table_mum_diff": "Table: MUM x Difficulty",
        "exp.table_mum_lang": "Table: MUM x Language",
        "exp.meta_header": "Problem Metadata",
        "exp.meta_count": "{n} problems in metadata",
        "exp.dl_meta": "\U0001F4E5 Download CSV (Metadata)",
    },
    "pt": {
        "sidebar.title": "\U0001F4CA Filtros Globais",
        "sidebar.caption": "Estes filtros afetam todas as abas",
        "sidebar.language": "Idioma de saida",
        "sidebar.models": "Modelos",
        "sidebar.langs": "Linguagens",
        "sidebar.difficulty": "Dificuldade",
        "sidebar.cache": "Cache",
        "sidebar.clear_cache": "\U0001F5D1\uFE0F Limpar cache",
        "main.title": "\U0001F4CA LLM Code Efficiency - Research Dashboard",
        "main.caption": "Dashboard interativo para consulta dos dados da pesquisa sobre eficiencia de codigo gerado por LLMs",
        "main.nodata": "Nenhum dado encontrado. Verifique se os diretorios `out/` e `datasets/leetcode/` existem.",
        "tab.overview": "\U0001F4CA Visao Geral",
        "tab.rq1": "\u2705 RQ1: Aceitacao",
        "tab.rq2": "\u23F1\uFE0F RQ2: Tempo de Execucao",
        "tab.rq3": "\U0001F4BE RQ3: Uso de Memoria",
        "tab.stats": "\U0001F4C8 Analise Estatistica",
        "tab.explorer": "\U0001F50D Explorador de Codigo",
        "tab.export": "\U0001F4CB Exportar Dados",
        "ov.header": "Visao Geral do Estudo",
        "ov.desc": "Este dashboard apresenta os resultados de uma avaliacao em larga escala de **11 LLMs** disponiveis na plataforma Groq Cloud, avaliados em **336 problemas do LeetCode** em **3 linguagens de programacao** (C++, Java, Python3).",
        "ov.total_subs": "Total de Submissoes",
        "ov.oar": "Taxa de Aceitacao (OAR)",
        "ov.avg_time": "Tempo Medio (Aceitos)",
        "ov.avg_mem": "Memoria Media (Aceitos)",
        "ov.incomplete_header": "Geracoes Incompletas",
        "ov.incomplete_desc": "Algumas LLMs produziram saidas truncadas ou incompletas que nao foram submetidas ao LeetCode.",
        "ov.total_expected": "Total Esperado",
        "ov.total_sub_missing": "Total Submetido / Faltante",
        "ov.dist_header": "Distribuicao Geral dos Resultados (OAR)",
        "ov.pie_title": "Distribuicao dos 3 Status (Paper)",
        "ov.table_3cat": "**Tabela de Resultados (3 categorias - artigo)**",
        "ov.table_granular": "**Detalhamento granular (status original do LeetCode)**",
        "ov.sample_header": "Composicao da Amostra",
        "ov.by_diff": "**Por Dificuldade**",
        "ov.models_eval": "**Modelos Avaliados**",
        "ov.by_lang": "**Linguagens**",
        "ov.topic_header": "Distribuicao por Topico (tema_principal)",
        "ov.topic_chart": "Numero de problemas por topico na amostra",
        "col.model": "Modelo",
        "col.expected": "Esperado",
        "col.submitted": "Submetido",
        "col.missing": "Faltante",
        "col.result": "Resultado",
        "col.count": "Contagem",
        "col.status": "Status",
        "col.problems": "Problemas",
        "col.submissions": "Submissoes",
        "col.topic": "Topico",
        "rq1.header": "RQ1: Taxas de Aceitacao",
        "rq1.desc": "*Como as taxas de aceitacao dos diferentes LLMs comparados variam por dificuldade e linguagem?*",
        "rq1.nodata": "Nenhum dado disponivel para os filtros selecionados.",
        "rq1.ard_header": "ARD - Taxa de Aceitacao por Dificuldade",
        "rq1.ard_chart": "Taxa de Aceitacao por Dificuldade (ARD)",
        "rq1.arm_header": "ARM - Taxa de Aceitacao por Modelo",
        "rq1.arm_chart": "Taxa de Aceitacao por Modelo (ARM)",
        "rq1.arl_header": "ARL - Taxa de Aceitacao por Linguagem",
        "rq1.arl_chart": "Taxa de Aceitacao por Linguagem (ARL)",
        "rq1.crosstabs": "Tabulacoes Cruzadas",
        "rq1.ct_md": "Modelo x Dificuldade",
        "rq1.ct_ml": "Modelo x Linguagem",
        "rq1.ct_ld": "Linguagem x Dificuldade",
        "rq1.ct_md_title": "**Taxa de Aceitacao (%) por Modelo e Dificuldade**",
        "rq1.ct_ml_title": "**Taxa de Aceitacao (%) por Modelo e Linguagem**",
        "rq1.ct_ld_title": "**Taxa de Aceitacao (%) por Linguagem e Dificuldade**",
        "rq1.hm_md": "Heatmap: Aceitacao (%) - Modelo x Dificuldade",
        "rq1.hm_ml": "Heatmap: Aceitacao (%) - Modelo x Linguagem",
        "rq1.fail_header": "Distribuicao de Falhas por Modelo",
        "rq1.fail_stacked": "Status por Modelo (empilhado)",
        "rq1.fail_granular": "Detalhamento granular dos status (7 categorias originais)",
        "rq1.byq_header": "Taxa de Aceitacao por Problema",
        "rq1.byq_chart": "Aceitacao por Problema",
        "rq2.header": "RQ2: Tempo de Execucao",
        "rq2.desc": "*Como o tempo medio de execucao das solucoes aceitas varia por modelo, dificuldade e linguagem?*\n\n**Nota:** Metricas calculadas apenas sobre submissoes aceitas (Accepted).",
        "rq2.nodata": "Nenhuma submissao aceita para os filtros selecionados.",
        "rq2.oaet_header": "OAET - Estatisticas Gerais do Tempo de Execucao",
        "rq2.mean": "Media",
        "rq2.median": "Mediana",
        "rq2.std": "Desvio Padrao",
        "rq2.p99": "P99",
        "rq2.max": "Maximo",
        "rq2.based_on": "Baseado em {n:,} submissoes aceitas",
        "rq2.hist_header": "Distribuicao do Tempo de Execucao",
        "rq2.hist_title": "Distribuicao do Tempo de Execucao (Aceitos)",
        "rq2.etd_header": "ETD - Tempo de Execucao por Dificuldade",
        "rq2.etd_chart": "Tempo Medio por Dificuldade (ETD)",
        "rq2.etm_header": "ETM - Tempo de Execucao por Modelo",
        "rq2.etm_chart": "Tempo Medio por Modelo (ETM) - ordenado",
        "rq2.etl_header": "ETL - Tempo de Execucao por Linguagem",
        "rq2.etl_chart": "Tempo Medio por Linguagem (ETL)",
        "rq2.ct_header": "Tabulacoes Cruzadas - Tempo de Execucao",
        "rq2.ct_md_title": "**Tempo Medio (ms) por Modelo e Dificuldade**",
        "rq2.ct_ml_title": "**Tempo Medio (ms) por Modelo e Linguagem**",
        "rq2.ct_ld_title": "**Tempo Medio (ms) por Linguagem e Dificuldade**",
        "rq2.hm_md": "Heatmap: Tempo Medio (ms) - Modelo x Dificuldade",
        "rq2.hm_ml": "Heatmap: Tempo Medio (ms) - Modelo x Linguagem",
        "rq2.box_header": "Distribuicoes (Box Plots)",
        "rq2.box_model": "Por Modelo",
        "rq2.box_lang": "Por Linguagem",
        "rq2.box_diff": "Por Dificuldade",
        "rq2.box_model_title": "Distribuicao do Tempo por Modelo",
        "rq2.box_lang_title": "Distribuicao do Tempo por Linguagem",
        "rq2.box_diff_title": "Distribuicao do Tempo por Dificuldade",
        "rq3.header": "RQ3: Uso de Memoria",
        "rq3.desc": "*Como o uso medio de memoria das solucoes aceitas varia por modelo, dificuldade e linguagem?*\n\n**Nota:** Metricas calculadas apenas sobre submissoes aceitas (Accepted).",
        "rq3.nodata": "Nenhuma submissao aceita para os filtros selecionados.",
        "rq3.oamu_header": "OAMU - Estatisticas Gerais do Uso de Memoria",
        "rq3.hist_header": "Distribuicao do Uso de Memoria",
        "rq3.hist_title": "Distribuicao do Uso de Memoria (Aceitos)",
        "rq3.mud_header": "MUD - Uso de Memoria por Dificuldade",
        "rq3.mud_chart": "Memoria Media por Dificuldade (MUD)",
        "rq3.mum_header": "MUM - Uso de Memoria por Modelo",
        "rq3.mum_chart": "Memoria Media por Modelo (MUM) - ordenado",
        "rq3.mul_header": "MUL - Uso de Memoria por Linguagem",
        "rq3.mul_chart": "Memoria Media por Linguagem (MUL)",
        "rq3.ct_header": "Tabulacoes Cruzadas - Uso de Memoria",
        "rq3.ct_md_title": "**Memoria Media (MB) por Modelo e Dificuldade**",
        "rq3.ct_ml_title": "**Memoria Media (MB) por Modelo e Linguagem**",
        "rq3.ct_ld_title": "**Memoria Media (MB) por Linguagem e Dificuldade**",
        "rq3.hm_md": "Heatmap: Memoria Media (MB) - Modelo x Dificuldade",
        "rq3.hm_ml": "Heatmap: Memoria Media (MB) - Modelo x Linguagem",
        "rq3.box_header": "Distribuicoes (Box Plots)",
        "rq3.box_model_title": "Distribuicao da Memoria por Modelo",
        "rq3.box_lang_title": "Distribuicao da Memoria por Linguagem",
        "rq3.box_diff_title": "Distribuicao da Memoria por Dificuldade",
        "rq3.tradeoff_header": "Trade-off Velocidade x Memoria",
        "rq3.tradeoff_chart": "Trade-off: Tempo Medio x Memoria Media por Linguagem",
        "st.header": "Analise Estatistica",
        "st.desc": "Analises inspiradas na metodologia de Guimaraes et al. (2025):\nANOVA, Kruskal-Wallis, Cohen's d, correlacoes de Pearson e Spearman.",
        "st.no_scipy": "\u26A0\uFE0F Biblioteca `scipy` nao encontrada. Instale com `pip install scipy` para habilitar os testes estatisticos.",
        "st.nodata": "Nenhum dado disponivel.",
        "st.anova_header": "ANOVA e Tamanho de Efeito (eta squared)",
        "st.anova_desc": "Limiares de eta squared: **small** (< 0.06), **medium** (0.06-0.14), **large** (> 0.14)",
        "st.insufficient": "Dados insuficientes para realizar os testes.",
        "st.cohens_lang_header": "Cohen's d - Comparacoes entre Linguagens",
        "st.cohens_desc": "Limiares: **negligible** (|d| < 0.2), **small** (0.2-0.5), **medium** (0.5-0.8), **large** (> 0.8)",
        "st.cohens_diff_header": "Cohen's d - Comparacoes entre Dificuldades (Aceitacao)",
        "st.corr_header": "Correlacoes entre Metricas (Nivel de Modelo)",
        "st.corr_desc": "Pearson (r) e Spearman (rho) entre taxa de aceitacao, tempo medio e memoria media, agregados por modelo.",
        "st.corr_viz_header": "Visualizacoes de Correlacao",
        "st.corr_acc_mem": "Aceitacao vs Memoria Media (por modelo)",
        "st.corr_acc_time": "Aceitacao vs Tempo Medio (por modelo)",
        "st.corr_need": "scipy necessario para calculos de correlacao.",
        "st.corr_insuf": "Dados insuficientes.",
        "st.composite_header": "Escore Composto (Trade-off)",
        "st.composite_desc": "Ranking normalizado combinando aceitacao (cima), tempo de execucao (baixo) e memoria (baixo), com pesos iguais. Valores maiores = melhor equilibrio geral.",
        "st.composite_chart": "Escore Composto por Modelo (maior = melhor)",
        "st.rec_header": "Recomendacoes Praticas",
        "st.rec_body": """Com base na analise combinada das tres RQs:

| Prioridade | Recomendacao |
|---|---|
| **Maximizar aceitacao** | `kimi-k2-instruct` (79.4%), `qwen3-32b` (72.6%), `deepseek-r1-70b` (67.0%). C++ e Java ~48%, ambos acima de Python3 (32.5%). |
| **Minimizar tempo** | `llama-3.3-70b` (25.6ms), `llama-3.1-8b` (29.2ms). Java (24.5ms) ~ C++ (26.7ms) << Python3 (119.9ms). |
| **Minimizar memoria** | Python3 (20.9MB) < C++ (28.9MB) < Java (46.9MB). `llama3-8b` (29.2MB), `llama-3.1-8b` (29.3MB). |
| **Melhor equilibrio** | `llama-3.1-8b-instant` (composto 0.73): aceitacao moderada + baixo tempo e memoria. |

**Nenhum modelo ou linguagem domina simultaneamente em todas as tres metricas.**""",
        "stcol.factor": "Fator",
        "stcol.metric": "Metrica",
        "stcol.acceptance": "Aceitacao",
        "stcol.time": "Tempo (ms)",
        "stcol.memory": "Memoria (MB)",
        "stcol.effect": "Efeito",
        "stcol.pair": "Par",
        "stcol.magnitude": "Magnitude",
        "stcol.trend": "Tendencia",
        "ax.acceptance_pct": "Aceitacao (%)",
        "ax.model": "Modelo",
        "ax.language": "Linguagem",
        "ax.difficulty": "Dificuldade",
        "ax.time_ms": "Tempo (ms)",
        "ax.memory_mb": "Memoria (MB)",
        "ax.count": "Contagem",
        "ax.frequency": "Frequencia",
        "ax.mean_time_ms": "Tempo Medio (ms)",
        "ax.mean_memory_mb": "Memoria Media (MB)",
        "tc.difficulty": "Dificuldade",
        "tc.accepted": "Aceitos",
        "tc.total": "Total",
        "tc.rate_pct": "Taxa%",
        "tc.language": "Linguagem",
        "tc.mean_ms": "Media (ms)",
        "tc.median_ms": "Mediana (ms)",
        "tc.std_ms": "DP (ms)",
        "tc.mean_mb": "Media (MB)",
        "tc.median_mb": "Mediana (MB)",
        "tc.std_mb": "DP (MB)",
        "tc.n": "N",
        "ex.header": "Explorador de Codigo",
        "ex.desc": "Visualize o codigo gerado pelas LLMs e os resultados de submissao no LeetCode.",
        "ex.nodata": "Nenhuma resposta de LLM encontrada em `data/`.",
        "ex.model": "Modelo",
        "ex.language": "Linguagem",
        "ex.problem": "Problema (ID)",
        "ex.code_title": "**Codigo Gerado**",
        "ex.meta_title": "**Metadados da LLM**",
        "ex.category": "Categoria",
        "ex.reason": "Motivo",
        "ex.starter_ok": "Starter OK",
        "ex.result_title": "**Resultado no LeetCode**",
        "ex.status": "Status",
        "ex.time": "Tempo",
        "ex.memory": "Memoria",
        "ex.problem_info": "**Info do Problema**",
        "ex.title_field": "Titulo",
        "ex.diff_field": "Dificuldade",
        "ex.topic_field": "Topico",
        "ex.no_result": "Nenhum resultado de submissao encontrado para esta combinacao.",
        "ex.empty_code": "(vazio)",
        "ex.compare_header": "Comparacao Lado a Lado",
        "ex.model_a": "**Modelo A**",
        "ex.model_b": "**Modelo B**",
        "ex.compare_problem": "Problema para comparar",
        "ex.no_common": "Nenhum problema em comum entre as selecoes.",
        "exp.header": "Exportar Dados",
        "exp.desc": "Baixe os dados filtrados em diversos formatos para analises externas.",
        "exp.nodata": "Nenhum dado disponivel.",
        "exp.sub_header": "Dados de Submissao (Filtrados)",
        "exp.record_count": "{n:,} registros com os filtros atuais",
        "exp.showing_500": "Exibindo 500 primeiras linhas. Faca o download para ver todos os dados.",
        "exp.dl_csv_all": "\U0001F4E5 Download CSV (Todas submissoes)",
        "exp.dl_json_all": "\U0001F4E5 Download JSON (Todas submissoes)",
        "exp.dl_csv_acc": "\U0001F4E5 Download CSV (Apenas Aceitos)",
        "exp.tables_header": "Tabelas Resumo para o Artigo",
        "exp.tables_desc": "Tabelas prontas para copiar/colar no paper.",
        "exp.table_oar": "Tabela: Distribuicao geral (OAR)",
        "exp.table_arm_diff": "Tabela: ARM x Dificuldade",
        "exp.table_arm_lang": "Tabela: ARM x Linguagem",
        "exp.table_etm_diff": "Tabela: ETM x Dificuldade",
        "exp.table_etm_lang": "Tabela: ETM x Linguagem",
        "exp.table_mum_diff": "Tabela: MUM x Dificuldade",
        "exp.table_mum_lang": "Tabela: MUM x Linguagem",
        "exp.meta_header": "Metadados dos Problemas",
        "exp.meta_count": "{n} problemas na base de metadados",
        "exp.dl_meta": "\U0001F4E5 Download CSV (Metadados)",
    },
}


def t(key: str) -> str:
    """Get translated string for the current language."""
    lang = st.session_state.get("ui_lang", "pt")
    return UI[lang].get(key, key)


# ==================================================================
# DATA LOADING
# ==================================================================
@st.cache_data(show_spinner="Loading / Carregando...")
def load_results() -> pd.DataFrame:
    """Load submission results from out/results.csv (or out/by_model/ JSONs)."""
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV, encoding="utf-8", low_memory=False)
    elif RESULTS_BY_MODEL.exists():
        rows = []
        for model_dir in RESULTS_BY_MODEL.iterdir():
            if not model_dir.is_dir():
                continue
            for lang_dir in model_dir.iterdir():
                if not lang_dir.is_dir():
                    continue
                for jf in lang_dir.glob("*.json"):
                    try:
                        rows.append(json.load(open(jf, "r", encoding="utf-8")))
                    except Exception:
                        continue
        df = pd.DataFrame(rows)
    else:
        return pd.DataFrame()

    if df.empty:
        return df

    # Normalize columns
    df["tempo_ms"] = pd.to_numeric(df.get("tempo_ms"), errors="coerce")
    df["memoria_mb"] = pd.to_numeric(df.get("memoria_mb"), errors="coerce")
    if "aceito" in df.columns:
        df["aceito"] = df["aceito"].map(
            lambda v: True if str(v).strip().lower() in ("true", "1", "yes") else False
        )
    if "id_questao" in df.columns:
        df["id_questao"] = pd.to_numeric(df["id_questao"], errors="coerce")

    # Short model name
    df["model_short"] = df["modelo"].map(lambda m: MODEL_SHORT.get(str(m).strip(), str(m).strip()))

    # Paper 3-category status
    def paper_status(row):
        s = str(row.get("status", "")).strip()
        if s == "Accepted":
            return "Accepted"
        if "Compile" in s or "Compilation" in s:
            return "Compilation Error"
        return "Wrong Answer"

    df["status_paper"] = df.apply(paper_status, axis=1)
    df["status_detail"] = df["status"].astype(str).str.strip()

    return df


@st.cache_data(show_spinner="Loading / Carregando...")
def load_problem_metadata() -> pd.DataFrame:
    """Load problem metadata from datasets/leetcode/{easy,medium,hard}.json."""
    rows = []
    for fname in ["easy.json", "medium.json", "hard.json"]:
        fpath = DATASETS / fname
        if not fpath.exists():
            continue
        try:
            problems = json.load(open(fpath, "r", encoding="utf-8"))
            for p in problems:
                diff_pt = str(p.get("dificuldade", ""))
                diff_en = DIFF_PT_EN.get(diff_pt, diff_pt)
                rows.append({
                    "problem_id": p.get("id"),
                    "slug": p.get("slug", ""),
                    "titulo": p.get("titulo", ""),
                    "difficulty": diff_en,
                    "tema_principal": p.get("tema_principal", ""),
                    "temas": p.get("temas", []),
                    "has_image": p.get("has_image", False),
                })
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Loading / Carregando...")
def load_llm_answers() -> pd.DataFrame:
    """Load LLM-generated code from data/ directory."""
    rows = []
    if not LLM_DATA.exists():
        return pd.DataFrame()
    for model_dir in sorted(LLM_DATA.iterdir()):
        if not model_dir.is_dir():
            continue
        for lang_dir in model_dir.iterdir():
            if not lang_dir.is_dir():
                continue
            for jf in lang_dir.glob("*.json"):
                try:
                    obj = json.load(open(jf, "r", encoding="utf-8"))
                    rows.append({
                        "modelo": model_dir.name,
                        "linguagem": lang_dir.name,
                        "problem_id": obj.get("id"),
                        "code": obj.get("code", ""),
                        "resposta": obj.get("resposta", ""),
                        "categoria": obj.get("categoria", ""),
                        "motivo": obj.get("motivo", ""),
                        "timestamp": obj.get("timestamp", ""),
                        "starter_ok": obj.get("starter_ok"),
                    })
                except Exception:
                    continue
    return pd.DataFrame(rows)


def merge_difficulty(df_results: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """Merge difficulty info into results."""
    if df_meta.empty or df_results.empty:
        return df_results

    slug_diff = df_meta.set_index("slug")["difficulty"].to_dict()
    id_diff = df_meta.set_index("problem_id")["difficulty"].to_dict()
    slug_tema = df_meta.set_index("slug")["tema_principal"].to_dict()
    id_tema = df_meta.set_index("problem_id")["tema_principal"].to_dict()
    slug_titulo = df_meta.set_index("slug")["titulo"].to_dict()

    df = df_results.copy()
    df["difficulty"] = df["slug"].map(slug_diff)
    mask = df["difficulty"].isna()
    if mask.any() and "id_questao" in df.columns:
        df.loc[mask, "difficulty"] = df.loc[mask, "id_questao"].map(id_diff)

    df["tema_principal"] = df["slug"].map(slug_tema)
    mask2 = df["tema_principal"].isna()
    if mask2.any() and "id_questao" in df.columns:
        df.loc[mask2, "tema_principal"] = df.loc[mask2, "id_questao"].map(id_tema)

    df["titulo"] = df["slug"].map(slug_titulo)
    df["difficulty"] = pd.Categorical(df["difficulty"], categories=DIFF_ORDER, ordered=True)

    return df


# ==================================================================
# STATISTICAL HELPERS
# ==================================================================
def compute_eta_squared(f_stat, df_between, df_within):
    return (f_stat * df_between) / (f_stat * df_between + df_within)


def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled if pooled > 0 else np.nan


def run_anova(df, group_col, value_col):
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col) if len(g[value_col].dropna()) > 0]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    if HAS_SCIPY:
        f_stat, p_val = sp_stats.f_oneway(*groups)
        k = len(groups)
        n = sum(len(g) for g in groups)
        eta2 = compute_eta_squared(f_stat, k - 1, n - k)
        return f_stat, p_val, eta2
    return np.nan, np.nan, np.nan


def run_kruskal(df, group_col, value_col):
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col) if len(g[value_col].dropna()) > 0]
    if len(groups) < 2 or not HAS_SCIPY:
        return np.nan, np.nan
    h_stat, p_val = sp_stats.kruskal(*groups)
    return h_stat, p_val


def effect_label(eta2):
    if pd.isna(eta2):
        return "n.s."
    if eta2 < 0.06:
        return "small"
    if eta2 < 0.14:
        return "medium"
    return "large"


# ==================================================================
# LOAD DATA
# ==================================================================
df_raw = load_results()
df_meta = load_problem_metadata()
df_llm = load_llm_answers()

if not df_raw.empty and not df_meta.empty:
    df_all = merge_difficulty(df_raw, df_meta)
else:
    df_all = df_raw.copy() if not df_raw.empty else pd.DataFrame()

if not df_all.empty:
    df_accepted = df_all[df_all["status_paper"] == "Accepted"].copy()
else:
    df_accepted = pd.DataFrame()


# ==================================================================
# SIDEBAR - Language Selector + Global Filters
# ==================================================================
with st.sidebar:
    lang_choice = st.selectbox(
        "\U0001F310 Language / Idioma",
        ["Portugues (Brasil)", "English"],
        index=0,
        key="lang_select",
    )
    st.session_state["ui_lang"] = "pt" if lang_choice.startswith("Portugu") else "en"

    st.title(t("sidebar.title"))
    st.caption(t("sidebar.caption"))

    if not df_all.empty:
        all_models = sorted(df_all["model_short"].dropna().unique().tolist())
        sel_models = st.multiselect(t("sidebar.models"), all_models, default=all_models)

        sel_langs = st.multiselect(t("sidebar.langs"), LANG_ORDER,
                                   default=[l for l in LANG_ORDER if l in df_all["linguagem"].unique()])

        available_diffs = [d for d in DIFF_ORDER if d in df_all["difficulty"].dropna().unique()]
        sel_diffs = st.multiselect(t("sidebar.difficulty"), available_diffs, default=available_diffs)

        st.markdown("---")
        st.caption(t("sidebar.cache"))
        if st.button(t("sidebar.clear_cache")):
            st.cache_data.clear()
            st.rerun()
    else:
        sel_models, sel_langs, sel_diffs = [], [], []

# Apply filters
if not df_all.empty:
    mask = (
        df_all["model_short"].isin(sel_models) &
        df_all["linguagem"].isin(sel_langs) &
        (df_all["difficulty"].isin(sel_diffs) | df_all["difficulty"].isna())
    )
    df = df_all[mask].copy()
    df_acc = df[df["status_paper"] == "Accepted"].copy()
else:
    df = pd.DataFrame()
    df_acc = pd.DataFrame()


# ==================================================================
# MAIN UI
# ==================================================================
st.title(t("main.title"))
st.caption(t("main.caption"))

if df_all.empty:
    st.error(t("main.nodata"))
    st.stop()

tab_overview, tab_rq1, tab_rq2, tab_rq3, tab_stats, tab_explorer, tab_export = st.tabs([
    t("tab.overview"), t("tab.rq1"), t("tab.rq2"), t("tab.rq3"),
    t("tab.stats"), t("tab.explorer"), t("tab.export"),
])

# ==================================================================
# TAB 1: OVERVIEW
# ==================================================================
with tab_overview:
    st.header(t("ov.header"))
    st.markdown(t("ov.desc"))

    total_judged = len(df)
    total_accepted = len(df_acc)
    oar = (total_accepted / total_judged * 100) if total_judged > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("ov.total_subs"), f"{total_judged:,}")
    col2.metric(t("ov.oar"), f"{oar:.1f}%")
    col3.metric(t("ov.avg_time"), f"{df_acc['tempo_ms'].mean():.1f} ms" if not df_acc.empty else "-")
    col4.metric(t("ov.avg_mem"), f"{df_acc['memoria_mb'].mean():.1f} MB" if not df_acc.empty else "-")

    # Incomplete generations
    st.subheader(t("ov.incomplete_header"))
    st.markdown(t("ov.incomplete_desc"))

    incomplete_data = []
    for model_full, model_short in MODEL_SHORT.items():
        count = len(df_all[df_all["modelo"] == model_full])
        missing = EXPECTED_PER_MODEL - count
        incomplete_data.append({
            t("col.model"): model_short,
            t("col.expected"): EXPECTED_PER_MODEL,
            t("col.submitted"): count,
            t("col.missing"): max(0, missing),
        })
    df_incomplete = pd.DataFrame(incomplete_data).sort_values(t("col.missing"), ascending=False)
    total_submitted = df_incomplete[t("col.submitted")].sum()
    total_missing = TOTAL_EXPECTED - total_submitted

    c1, c2 = st.columns(2)
    c1.metric(t("ov.total_expected"), f"{TOTAL_EXPECTED:,}")
    c2.metric(t("ov.total_sub_missing"), f"{total_submitted:,} / {max(0, total_missing):,}")

    st.dataframe(df_incomplete, use_container_width=True, hide_index=True)

    # Overall outcome distribution
    st.subheader(t("ov.dist_header"))
    c1, c2 = st.columns([1, 1])
    with c1:
        status_counts = df["status_paper"].value_counts()
        fig = px.pie(
            names=status_counts.index,
            values=status_counts.values,
            hole=0.42,
            color=status_counts.index,
            color_discrete_map=COLORS_STATUS,
            title=t("ov.pie_title"),
        )
        fig.update_traces(textposition="inside", textinfo="percent+label+value")
        fig.update_layout(height=400, margin=dict(t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(t("ov.table_3cat"))
        summary_3cat = pd.DataFrame({
            t("col.result"): ["Accepted", "Compilation Error", "Wrong Answer"],
            t("col.count"): [
                len(df[df["status_paper"] == "Accepted"]),
                len(df[df["status_paper"] == "Compilation Error"]),
                len(df[df["status_paper"] == "Wrong Answer"]),
            ],
        })
        summary_3cat["%"] = (summary_3cat[t("col.count")] / summary_3cat[t("col.count")].sum() * 100).round(1)
        st.dataframe(summary_3cat, use_container_width=True, hide_index=True)

        st.markdown(t("ov.table_granular"))
        detail_counts = df["status_detail"].value_counts().reset_index()
        detail_counts.columns = [t("col.status"), t("col.count")]
        detail_counts["%"] = (detail_counts[t("col.count")] / detail_counts[t("col.count")].sum() * 100).round(1)
        st.dataframe(detail_counts, use_container_width=True, hide_index=True)

    # Sample composition
    st.subheader(t("ov.sample_header"))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(t("ov.by_diff"))
        diff_comp = df_all.drop_duplicates("slug").groupby("difficulty", observed=True).size().reset_index(name=t("col.problems"))
        st.dataframe(diff_comp, use_container_width=True, hide_index=True)
    with c2:
        st.markdown(t("ov.models_eval"))
        st.dataframe(
            pd.DataFrame({t("col.model"): sorted(MODEL_SHORT.values())}),
            use_container_width=True, hide_index=True,
        )
    with c3:
        st.markdown(t("ov.by_lang"))
        lang_comp = df_all.groupby("linguagem").size().reset_index(name=t("col.submissions"))
        st.dataframe(lang_comp, use_container_width=True, hide_index=True)

    # Topic distribution
    if "tema_principal" in df_all.columns and df_all["tema_principal"].notna().any():
        st.subheader(t("ov.topic_header"))
        topic_dist = (
            df_all.drop_duplicates("slug")
            .groupby("tema_principal").size()
            .sort_values(ascending=False)
            .reset_index(name=t("col.problems"))
        )
        topic_dist.columns = [t("col.topic"), t("col.problems")]
        fig = px.bar(topic_dist, x=t("col.topic"), y=t("col.problems"),
                     title=t("ov.topic_chart"), height=400)
        fig.update_xaxes(tickangle=-40)
        fig.update_layout(margin=dict(t=60, b=120))
        st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# TAB 2: RQ1 - ACCEPTANCE RATES
# ==================================================================
with tab_rq1:
    st.header(t("rq1.header"))
    st.markdown(t("rq1.desc"))

    if df.empty:
        st.warning(t("rq1.nodata"))
    else:
        df["is_accepted"] = (df["status_paper"] == "Accepted").astype(int)

        # ARD
        st.subheader(t("rq1.ard_header"))
        ard = df.groupby("difficulty", observed=True)["is_accepted"].agg(["mean", "sum", "count"]).reset_index()
        ard.columns = [t("tc.difficulty"), "Taxa", t("tc.accepted"), t("tc.total")]
        ard[t("tc.rate_pct")] = (ard["Taxa"] * 100).round(1)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(ard, x=t("tc.difficulty"), y=t("tc.rate_pct"), text=t("tc.rate_pct"),
                         color=t("tc.difficulty"), color_discrete_map=COLORS_DIFF,
                         title=t("rq1.ard_chart"),
                         labels={t("tc.rate_pct"): t("ax.acceptance_pct")}, height=400)
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis_range=[0, max(ard[t("tc.rate_pct")]) * 1.15])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(ard[[t("tc.difficulty"), t("tc.accepted"), t("tc.total"), t("tc.rate_pct")]],
                         use_container_width=True, hide_index=True)

        # ARM
        st.subheader(t("rq1.arm_header"))
        arm = df.groupby("model_short")["is_accepted"].agg(["mean", "sum", "count"]).reset_index()
        arm.columns = [t("col.model"), "Taxa", t("tc.accepted"), t("tc.total")]
        arm[t("tc.rate_pct")] = (arm["Taxa"] * 100).round(1)
        arm = arm.sort_values(t("tc.rate_pct"), ascending=False)

        fig = px.bar(arm, x=t("col.model"), y=t("tc.rate_pct"), text=t("tc.rate_pct"),
                     title=t("rq1.arm_chart"),
                     labels={t("tc.rate_pct"): t("ax.acceptance_pct"), t("col.model"): t("ax.model")},
                     height=480, color_discrete_sequence=["#636EFA"])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        fig.update_xaxes(tickangle=-35)
        fig.update_layout(yaxis_range=[0, max(arm[t("tc.rate_pct")]) * 1.12])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(arm[[t("col.model"), t("tc.accepted"), t("tc.total"), t("tc.rate_pct")]],
                     use_container_width=True, hide_index=True)

        # ARL
        st.subheader(t("rq1.arl_header"))
        arl = df.groupby("linguagem")["is_accepted"].agg(["mean", "sum", "count"]).reset_index()
        arl.columns = [t("tc.language"), "Taxa", t("tc.accepted"), t("tc.total")]
        arl[t("tc.rate_pct")] = (arl["Taxa"] * 100).round(1)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(arl, x=t("tc.language"), y=t("tc.rate_pct"), text=t("tc.rate_pct"),
                         color=t("tc.language"), color_discrete_map=COLORS_LANG,
                         title=t("rq1.arl_chart"),
                         labels={t("tc.rate_pct"): t("ax.acceptance_pct")}, height=400)
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis_range=[0, max(arl[t("tc.rate_pct")]) * 1.15])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(arl[[t("tc.language"), t("tc.accepted"), t("tc.total"), t("tc.rate_pct")]],
                         use_container_width=True, hide_index=True)

        # Cross-tabs
        st.subheader(t("rq1.crosstabs"))
        ct1, ct2, ct3 = st.tabs([t("rq1.ct_md"), t("rq1.ct_ml"), t("rq1.ct_ld")])

        with ct1:
            st.markdown(t("rq1.ct_md_title"))
            ct_md = df.pivot_table(index="model_short", columns="difficulty",
                                   values="is_accepted", aggfunc="mean").multiply(100)
            ct_md = ct_md.reindex(columns=[d for d in DIFF_ORDER if d in ct_md.columns])
            st.dataframe(ct_md.round(1).style.format("{:.1f}"), use_container_width=True)

            fig = px.imshow(ct_md.values, x=ct_md.columns.tolist(), y=ct_md.index.tolist(),
                            text_auto=".1f", aspect="auto",
                            color_continuous_scale="YlGn",
                            title=t("rq1.hm_md"),
                            labels=dict(x=t("ax.difficulty"), y=t("ax.model"), color=t("ax.acceptance_pct")))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with ct2:
            st.markdown(t("rq1.ct_ml_title"))
            ct_ml = df.pivot_table(index="model_short", columns="linguagem",
                                   values="is_accepted", aggfunc="mean").multiply(100)
            ct_ml = ct_ml.reindex(columns=[l for l in LANG_ORDER if l in ct_ml.columns])
            st.dataframe(ct_ml.round(1).style.format("{:.1f}"), use_container_width=True)

            fig = px.imshow(ct_ml.values, x=ct_ml.columns.tolist(), y=ct_ml.index.tolist(),
                            text_auto=".1f", aspect="auto",
                            color_continuous_scale="YlGn",
                            title=t("rq1.hm_ml"),
                            labels=dict(x=t("ax.language"), y=t("ax.model"), color=t("ax.acceptance_pct")))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with ct3:
            st.markdown(t("rq1.ct_ld_title"))
            ct_ld = df.pivot_table(index="linguagem", columns="difficulty",
                                   values="is_accepted", aggfunc="mean").multiply(100)
            ct_ld = ct_ld.reindex(columns=[d for d in DIFF_ORDER if d in ct_ld.columns])
            ct_ld = ct_ld.reindex([l for l in LANG_ORDER if l in ct_ld.index])
            st.dataframe(ct_ld.round(1).style.format("{:.1f}"), use_container_width=True)

        # Failure-mode distribution
        st.subheader(t("rq1.fail_header"))
        fail_dist = df.groupby(["model_short", "status_paper"]).size().reset_index(name="count")
        fail_piv = fail_dist.pivot_table(index="model_short", columns="status_paper", values="count", fill_value=0)
        fail_piv["Total"] = fail_piv.sum(axis=1)
        for col in ["Accepted", "Compilation Error", "Wrong Answer"]:
            if col in fail_piv.columns:
                fail_piv[f"{col} (%)"] = (fail_piv[col] / fail_piv["Total"] * 100).round(1)
        st.dataframe(fail_piv, use_container_width=True)

        fig = px.bar(fail_dist, x="model_short", y="count", color="status_paper",
                     color_discrete_map=COLORS_STATUS,
                     title=t("rq1.fail_stacked"),
                     labels={"count": t("ax.count"), "model_short": t("ax.model"), "status_paper": t("col.status")},
                     height=480)
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander(t("rq1.fail_granular")):
            detail_dist = df.groupby(["model_short", "status_detail"]).size().reset_index(name="count")
            detail_piv = detail_dist.pivot_table(index="model_short", columns="status_detail", values="count", fill_value=0)
            st.dataframe(detail_piv, use_container_width=True)

        with st.expander(t("rq1.byq_header")):
            qperf = df.groupby(["slug", "titulo"])["is_accepted"].mean().multiply(100).reset_index()
            qperf.columns = ["slug", "titulo", t("tc.rate_pct")]
            qperf = qperf.sort_values(t("tc.rate_pct"), ascending=True)
            fig = px.bar(qperf, y="titulo", x=t("tc.rate_pct"), orientation="h",
                         title=t("rq1.byq_chart"),
                         labels={t("tc.rate_pct"): t("ax.acceptance_pct"), "titulo": ""},
                         height=max(400, 22 * len(qperf) + 80))
            fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside", cliponaxis=False)
            fig.update_layout(margin=dict(l=250))
            st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# TAB 3: RQ2 - EXECUTION TIME
# ==================================================================
with tab_rq2:
    st.header(t("rq2.header"))
    st.markdown(t("rq2.desc"))

    if df_acc.empty:
        st.warning(t("rq2.nodata"))
    else:
        n_acc = len(df_acc)

        # OAET summary
        st.subheader(t("rq2.oaet_header"))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(t("rq2.mean"), f"{df_acc['tempo_ms'].mean():.1f} ms")
        c2.metric(t("rq2.median"), f"{df_acc['tempo_ms'].median():.1f} ms")
        c3.metric(t("rq2.std"), f"{df_acc['tempo_ms'].std():.1f} ms")
        c4.metric(t("rq2.p99"), f"{df_acc['tempo_ms'].quantile(0.99):.1f} ms")
        c5.metric(t("rq2.max"), f"{df_acc['tempo_ms'].max():.1f} ms")
        st.caption(t("rq2.based_on").format(n=n_acc))

        with st.expander(t("rq2.hist_header")):
            fig = px.histogram(df_acc, x="tempo_ms", nbins=100,
                               title=t("rq2.hist_title"),
                               labels={"tempo_ms": t("ax.time_ms"), "count": t("ax.frequency")},
                               height=350)
            fig.update_layout(margin=dict(t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)

        # ETD
        st.subheader(t("rq2.etd_header"))
        etd = df_acc.groupby("difficulty", observed=True)["tempo_ms"].agg(["mean", "median", "std", "count"]).reset_index()
        etd.columns = [t("tc.difficulty"), t("tc.mean_ms"), t("tc.median_ms"), t("tc.std_ms"), t("tc.n")]
        etd = etd.sort_values(t("tc.difficulty"))

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(etd, x=t("tc.difficulty"), y=t("tc.mean_ms"), text=t("tc.mean_ms"),
                         color=t("tc.difficulty"), color_discrete_map=COLORS_DIFF,
                         title=t("rq2.etd_chart"), height=400)
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis_range=[0, max(etd[t("tc.mean_ms")]) * 1.15])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(etd.round(1), use_container_width=True, hide_index=True)

        # ETM
        st.subheader(t("rq2.etm_header"))
        etm = df_acc.groupby("model_short")["tempo_ms"].agg(["mean", "median", "std", "count"]).reset_index()
        etm.columns = [t("col.model"), t("tc.mean_ms"), t("tc.median_ms"), t("tc.std_ms"), t("tc.n")]
        etm = etm.sort_values(t("tc.mean_ms"))

        fig = px.bar(etm, x=t("col.model"), y=t("tc.mean_ms"), text=t("tc.mean_ms"),
                     title=t("rq2.etm_chart"),
                     labels={t("tc.mean_ms"): t("ax.mean_time_ms")}, height=480)
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
        fig.update_xaxes(tickangle=-35)
        fig.update_layout(yaxis_range=[0, max(etm[t("tc.mean_ms")]) * 1.15])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(etm.round(1), use_container_width=True, hide_index=True)

        # ETL
        st.subheader(t("rq2.etl_header"))
        etl = df_acc.groupby("linguagem")["tempo_ms"].agg(["mean", "median", "std", "count"]).reset_index()
        etl.columns = [t("tc.language"), t("tc.mean_ms"), t("tc.median_ms"), t("tc.std_ms"), t("tc.n")]

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(etl, x=t("tc.language"), y=t("tc.mean_ms"), text=t("tc.mean_ms"),
                         color=t("tc.language"), color_discrete_map=COLORS_LANG,
                         title=t("rq2.etl_chart"), height=400)
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis_range=[0, max(etl[t("tc.mean_ms")]) * 1.15])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(etl.round(1), use_container_width=True, hide_index=True)

        # Cross-tabs
        st.subheader(t("rq2.ct_header"))
        ct1, ct2, ct3 = st.tabs([t("rq1.ct_md"), t("rq1.ct_ml"), t("rq1.ct_ld")])

        with ct1:
            st.markdown(t("rq2.ct_md_title"))
            ct = df_acc.pivot_table(index="model_short", columns="difficulty", values="tempo_ms", aggfunc="mean")
            ct = ct.reindex(columns=[d for d in DIFF_ORDER if d in ct.columns])
            st.dataframe(ct.round(1).style.format("{:.1f}"), use_container_width=True)
            fig = px.imshow(ct.values, x=ct.columns.tolist(), y=ct.index.tolist(),
                            text_auto=".1f", aspect="auto", color_continuous_scale="YlOrRd",
                            title=t("rq2.hm_md"),
                            labels=dict(x=t("ax.difficulty"), y=t("ax.model"), color=t("ax.time_ms")))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with ct2:
            st.markdown(t("rq2.ct_ml_title"))
            ct = df_acc.pivot_table(index="model_short", columns="linguagem", values="tempo_ms", aggfunc="mean")
            ct = ct.reindex(columns=[l for l in LANG_ORDER if l in ct.columns])
            st.dataframe(ct.round(1).style.format("{:.1f}"), use_container_width=True)
            fig = px.imshow(ct.values, x=ct.columns.tolist(), y=ct.index.tolist(),
                            text_auto=".1f", aspect="auto", color_continuous_scale="YlOrRd",
                            title=t("rq2.hm_ml"),
                            labels=dict(x=t("ax.language"), y=t("ax.model"), color=t("ax.time_ms")))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with ct3:
            st.markdown(t("rq2.ct_ld_title"))
            ct = df_acc.pivot_table(index="linguagem", columns="difficulty", values="tempo_ms", aggfunc="mean")
            ct = ct.reindex(columns=[d for d in DIFF_ORDER if d in ct.columns])
            ct = ct.reindex([l for l in LANG_ORDER if l in ct.index])
            st.dataframe(ct.round(1).style.format("{:.1f}"), use_container_width=True)

        # Box plots
        st.subheader(t("rq2.box_header"))
        bp1, bp2, bp3 = st.tabs([t("rq2.box_model"), t("rq2.box_lang"), t("rq2.box_diff")])

        with bp1:
            fig = px.box(df_acc, x="model_short", y="tempo_ms", points=False,
                         title=t("rq2.box_model_title"),
                         labels={"tempo_ms": t("ax.time_ms"), "model_short": t("ax.model")}, height=450)
            fig.update_xaxes(tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        with bp2:
            fig = px.box(df_acc, x="linguagem", y="tempo_ms", color="linguagem",
                         color_discrete_map=COLORS_LANG, points=False,
                         title=t("rq2.box_lang_title"),
                         labels={"tempo_ms": t("ax.time_ms"), "linguagem": t("ax.language")}, height=400)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with bp3:
            fig = px.box(df_acc, x="difficulty", y="tempo_ms", color="difficulty",
                         color_discrete_map=COLORS_DIFF, points=False,
                         title=t("rq2.box_diff_title"),
                         labels={"tempo_ms": t("ax.time_ms"), "difficulty": t("ax.difficulty")}, height=400)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# TAB 4: RQ3 - MEMORY USAGE
# ==================================================================
with tab_rq3:
    st.header(t("rq3.header"))
    st.markdown(t("rq3.desc"))

    if df_acc.empty:
        st.warning(t("rq3.nodata"))
    else:
        n_acc = len(df_acc)

        # OAMU summary
        st.subheader(t("rq3.oamu_header"))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(t("rq2.mean"), f"{df_acc['memoria_mb'].mean():.1f} MB")
        c2.metric(t("rq2.median"), f"{df_acc['memoria_mb'].median():.1f} MB")
        c3.metric(t("rq2.std"), f"{df_acc['memoria_mb'].std():.1f} MB")
        c4.metric(t("rq2.p99"), f"{df_acc['memoria_mb'].quantile(0.99):.1f} MB")
        c5.metric(t("rq2.max"), f"{df_acc['memoria_mb'].max():.1f} MB")
        st.caption(t("rq2.based_on").format(n=n_acc))

        with st.expander(t("rq3.hist_header")):
            fig = px.histogram(df_acc, x="memoria_mb", nbins=100,
                               title=t("rq3.hist_title"),
                               labels={"memoria_mb": t("ax.memory_mb"), "count": t("ax.frequency")},
                               height=350)
            fig.update_layout(margin=dict(t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)

        # MUD
        st.subheader(t("rq3.mud_header"))
        mud = df_acc.groupby("difficulty", observed=True)["memoria_mb"].agg(["mean", "median", "std", "count"]).reset_index()
        mud.columns = [t("tc.difficulty"), t("tc.mean_mb"), t("tc.median_mb"), t("tc.std_mb"), t("tc.n")]

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(mud, x=t("tc.difficulty"), y=t("tc.mean_mb"), text=t("tc.mean_mb"),
                         color=t("tc.difficulty"), color_discrete_map=COLORS_DIFF,
                         title=t("rq3.mud_chart"), height=400)
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis_range=[0, max(mud[t("tc.mean_mb")]) * 1.15])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(mud.round(1), use_container_width=True, hide_index=True)

        # MUM
        st.subheader(t("rq3.mum_header"))
        mum = df_acc.groupby("model_short")["memoria_mb"].agg(["mean", "median", "std", "count"]).reset_index()
        mum.columns = [t("col.model"), t("tc.mean_mb"), t("tc.median_mb"), t("tc.std_mb"), t("tc.n")]
        mum = mum.sort_values(t("tc.mean_mb"))

        fig = px.bar(mum, x=t("col.model"), y=t("tc.mean_mb"), text=t("tc.mean_mb"),
                     title=t("rq3.mum_chart"),
                     labels={t("tc.mean_mb"): t("ax.mean_memory_mb")}, height=480)
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
        fig.update_xaxes(tickangle=-35)
        fig.update_layout(yaxis_range=[0, max(mum[t("tc.mean_mb")]) * 1.15])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(mum.round(1), use_container_width=True, hide_index=True)

        # MUL
        st.subheader(t("rq3.mul_header"))
        mul_df = df_acc.groupby("linguagem")["memoria_mb"].agg(["mean", "median", "std", "count"]).reset_index()
        mul_df.columns = [t("tc.language"), t("tc.mean_mb"), t("tc.median_mb"), t("tc.std_mb"), t("tc.n")]

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(mul_df, x=t("tc.language"), y=t("tc.mean_mb"), text=t("tc.mean_mb"),
                         color=t("tc.language"), color_discrete_map=COLORS_LANG,
                         title=t("rq3.mul_chart"), height=400)
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis_range=[0, max(mul_df[t("tc.mean_mb")]) * 1.15])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(mul_df.round(1), use_container_width=True, hide_index=True)

        # Cross-tabs
        st.subheader(t("rq3.ct_header"))
        ct1, ct2, ct3 = st.tabs([t("rq1.ct_md"), t("rq1.ct_ml"), t("rq1.ct_ld")])

        with ct1:
            st.markdown(t("rq3.ct_md_title"))
            ct = df_acc.pivot_table(index="model_short", columns="difficulty", values="memoria_mb", aggfunc="mean")
            ct = ct.reindex(columns=[d for d in DIFF_ORDER if d in ct.columns])
            st.dataframe(ct.round(1).style.format("{:.1f}"), use_container_width=True)
            fig = px.imshow(ct.values, x=ct.columns.tolist(), y=ct.index.tolist(),
                            text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
                            title=t("rq3.hm_md"),
                            labels=dict(x=t("ax.difficulty"), y=t("ax.model"), color=t("ax.memory_mb")))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with ct2:
            st.markdown(t("rq3.ct_ml_title"))
            ct = df_acc.pivot_table(index="model_short", columns="linguagem", values="memoria_mb", aggfunc="mean")
            ct = ct.reindex(columns=[l for l in LANG_ORDER if l in ct.columns])
            st.dataframe(ct.round(1).style.format("{:.1f}"), use_container_width=True)
            fig = px.imshow(ct.values, x=ct.columns.tolist(), y=ct.index.tolist(),
                            text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
                            title=t("rq3.hm_ml"),
                            labels=dict(x=t("ax.language"), y=t("ax.model"), color=t("ax.memory_mb")))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with ct3:
            st.markdown(t("rq3.ct_ld_title"))
            ct = df_acc.pivot_table(index="linguagem", columns="difficulty", values="memoria_mb", aggfunc="mean")
            ct = ct.reindex(columns=[d for d in DIFF_ORDER if d in ct.columns])
            ct = ct.reindex([l for l in LANG_ORDER if l in ct.index])
            st.dataframe(ct.round(1).style.format("{:.1f}"), use_container_width=True)

        # Box plots
        st.subheader(t("rq3.box_header"))
        bp1, bp2, bp3 = st.tabs([t("rq2.box_model"), t("rq2.box_lang"), t("rq2.box_diff")])

        with bp1:
            fig = px.box(df_acc, x="model_short", y="memoria_mb", points=False,
                         title=t("rq3.box_model_title"),
                         labels={"memoria_mb": t("ax.memory_mb"), "model_short": t("ax.model")}, height=450)
            fig.update_xaxes(tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        with bp2:
            fig = px.box(df_acc, x="linguagem", y="memoria_mb", color="linguagem",
                         color_discrete_map=COLORS_LANG, points=False,
                         title=t("rq3.box_lang_title"),
                         labels={"memoria_mb": t("ax.memory_mb"), "linguagem": t("ax.language")}, height=400)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with bp3:
            fig = px.box(df_acc, x="difficulty", y="memoria_mb", color="difficulty",
                         color_discrete_map=COLORS_DIFF, points=False,
                         title=t("rq3.box_diff_title"),
                         labels={"memoria_mb": t("ax.memory_mb"), "difficulty": t("ax.difficulty")}, height=400)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Speed-memory trade-off
        st.subheader(t("rq3.tradeoff_header"))
        trade_lang = df_acc.groupby("linguagem").agg(
            tempo_medio=("tempo_ms", "mean"),
            memoria_media=("memoria_mb", "mean"),
            n=("memoria_mb", "count"),
        ).reset_index()
        fig = px.scatter(trade_lang, x="tempo_medio", y="memoria_media",
                         color="linguagem", color_discrete_map=COLORS_LANG,
                         size="n", text="linguagem",
                         title=t("rq3.tradeoff_chart"),
                         labels={"tempo_medio": t("ax.mean_time_ms"), "memoria_media": t("ax.mean_memory_mb")},
                         height=450)
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# TAB 5: STATISTICAL ANALYSIS
# ==================================================================
with tab_stats:
    st.header(t("st.header"))
    st.markdown(t("st.desc"))

    if not HAS_SCIPY:
        st.warning(t("st.no_scipy"))

    if df.empty:
        st.warning(t("st.nodata"))
    else:
        df_stat = df.copy()
        df_stat["is_accepted"] = (df_stat["status_paper"] == "Accepted").astype(int)

        # ANOVA
        st.subheader(t("st.anova_header"))
        st.markdown(t("st.anova_desc"))

        anova_rows = []
        tests = [
            (t("ax.model"), "model_short", t("stcol.acceptance"), "is_accepted", df_stat),
            (t("ax.language"), "linguagem", t("stcol.acceptance"), "is_accepted", df_stat),
            (t("ax.difficulty"), "difficulty", t("stcol.acceptance"), "is_accepted", df_stat),
            (t("ax.model"), "model_short", t("stcol.time"), "tempo_ms", df_acc),
            (t("ax.language"), "linguagem", t("stcol.time"), "tempo_ms", df_acc),
            (t("ax.model"), "model_short", t("stcol.memory"), "memoria_mb", df_acc),
            (t("ax.language"), "linguagem", t("stcol.memory"), "memoria_mb", df_acc),
        ]

        for factor_label, group_col, metric_label, value_col, data in tests:
            if data.empty or value_col not in data.columns:
                continue
            f_val, p_val, eta2 = run_anova(data, group_col, value_col)
            h_val, p_kw = run_kruskal(data, group_col, value_col)
            anova_rows.append({
                t("stcol.factor"): factor_label,
                t("stcol.metric"): metric_label,
                "F (ANOVA)": f"{f_val:.2f}" if not np.isnan(f_val) else "-",
                "p (ANOVA)": f"{p_val:.2e}" if not np.isnan(p_val) else "-",
                "eta2": f"{eta2:.3f}" if not np.isnan(eta2) else "-",
                t("stcol.effect"): effect_label(eta2),
                "H (K-W)": f"{h_val:.2f}" if not np.isnan(h_val) else "-",
                "p (K-W)": f"{p_kw:.4f}" if not np.isnan(p_kw) else "-",
            })

        if anova_rows:
            st.dataframe(pd.DataFrame(anova_rows), use_container_width=True, hide_index=True)
        else:
            st.info(t("st.insufficient"))

        # Cohen's d (languages)
        st.subheader(t("st.cohens_lang_header"))
        st.markdown(t("st.cohens_desc"))

        cohens_rows = []
        metrics_for_cohens = [
            (t("stcol.acceptance"), "is_accepted", df_stat),
            (t("stcol.time"), "tempo_ms", df_acc),
            (t("stcol.memory"), "memoria_mb", df_acc),
        ]
        lang_pairs = [("C++", "Java"), ("C++", "Python3"), ("Java", "Python3")]

        for metric_label, col, data in metrics_for_cohens:
            for l1, l2 in lang_pairs:
                g1 = data[data["linguagem"] == l1][col].dropna().values
                g2 = data[data["linguagem"] == l2][col].dropna().values
                d = compute_cohens_d(g1, g2)
                cohens_rows.append({
                    t("stcol.metric"): metric_label,
                    t("stcol.pair"): f"{l1} vs {l2}",
                    "Cohen's d": f"{d:.2f}" if not np.isnan(d) else "-",
                    t("stcol.magnitude"): (
                        "large" if abs(d) > 0.8 else
                        "medium" if abs(d) > 0.5 else
                        "small" if abs(d) > 0.2 else
                        "negligible"
                    ) if not np.isnan(d) else "-",
                })

        if cohens_rows:
            st.dataframe(pd.DataFrame(cohens_rows), use_container_width=True, hide_index=True)

        # Cohen's d (difficulty)
        st.subheader(t("st.cohens_diff_header"))
        diff_pairs = [("Easy", "Medium"), ("Easy", "Hard"), ("Medium", "Hard")]
        diff_cohens = []
        for d1, d2 in diff_pairs:
            g1 = df_stat[df_stat["difficulty"] == d1]["is_accepted"].dropna().values
            g2 = df_stat[df_stat["difficulty"] == d2]["is_accepted"].dropna().values
            d = compute_cohens_d(g1, g2)
            diff_cohens.append({
                t("stcol.pair"): f"{d1} vs {d2}",
                "Cohen's d": f"{d:.2f}" if not np.isnan(d) else "-",
                t("stcol.magnitude"): (
                    "large" if abs(d) > 0.8 else
                    "medium" if abs(d) > 0.5 else
                    "small" if abs(d) > 0.2 else
                    "negligible"
                ) if not np.isnan(d) else "-",
            })
        st.dataframe(pd.DataFrame(diff_cohens), use_container_width=True, hide_index=True)

        # Correlations
        st.subheader(t("st.corr_header"))
        st.markdown(t("st.corr_desc"))

        model_agg = df_stat.groupby("model_short").agg(
            acceptance=("is_accepted", "mean"),
        ).reset_index()

        if not df_acc.empty:
            model_eff = df_acc.groupby("model_short").agg(
                mean_time=("tempo_ms", "mean"),
                mean_memory=("memoria_mb", "mean"),
            ).reset_index()
            model_agg = model_agg.merge(model_eff, on="model_short", how="left")

        if HAS_SCIPY and "mean_time" in model_agg.columns and "mean_memory" in model_agg.columns:
            corr_pairs = [
                (t("stcol.acceptance"), "acceptance", t("stcol.time"), "mean_time"),
                (t("stcol.acceptance"), "acceptance", t("stcol.memory"), "mean_memory"),
                (t("stcol.time"), "mean_time", t("stcol.memory"), "mean_memory"),
            ]
            corr_rows = []
            for l1, c1, l2, c2 in corr_pairs:
                valid = model_agg[[c1, c2]].dropna()
                if len(valid) < 3:
                    continue
                r, p_r = sp_stats.pearsonr(valid[c1], valid[c2])
                rho, p_rho = sp_stats.spearmanr(valid[c1], valid[c2])
                corr_rows.append({
                    t("stcol.pair"): f"{l1} x {l2}",
                    "Pearson r": f"{r:.2f}",
                    "p (Pearson)": f"{p_r:.4f}",
                    "Spearman rho": f"{rho:.2f}",
                    "p (Spearman)": f"{p_rho:.4f}",
                })
            if corr_rows:
                st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

            st.subheader(t("st.corr_viz_header"))
            c1, c2 = st.columns(2)
            with c1:
                if "mean_memory" in model_agg.columns:
                    fig = px.scatter(model_agg, x="acceptance", y="mean_memory",
                                     text="model_short",
                                     title=t("st.corr_acc_mem"),
                                     labels={"acceptance": t("ax.acceptance_pct"), "mean_memory": t("ax.mean_memory_mb")},
                                     height=400)
                    fig.update_traces(textposition="top center", marker_size=10)
                    valid_data = model_agg.dropna(subset=["acceptance", "mean_memory"])
                    if len(valid_data) >= 2:
                        z = np.polyfit(valid_data["acceptance"], valid_data["mean_memory"], 1)
                        x_line = np.linspace(valid_data["acceptance"].min(), valid_data["acceptance"].max(), 50)
                        fig.add_trace(go.Scatter(x=x_line, y=np.polyval(z, x_line),
                                                 mode="lines", line=dict(dash="dash", color="red"),
                                                 name=t("stcol.trend")))
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                if "mean_time" in model_agg.columns:
                    fig = px.scatter(model_agg, x="acceptance", y="mean_time",
                                     text="model_short",
                                     title=t("st.corr_acc_time"),
                                     labels={"acceptance": t("ax.acceptance_pct"), "mean_time": t("ax.mean_time_ms")},
                                     height=400)
                    fig.update_traces(textposition="top center", marker_size=10)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(t("st.corr_need") if not HAS_SCIPY else t("st.corr_insuf"))

        # Composite score
        st.subheader(t("st.composite_header"))
        st.markdown(t("st.composite_desc"))

        if "mean_time" in model_agg.columns and "mean_memory" in model_agg.columns:
            ma = model_agg.dropna(subset=["acceptance", "mean_time", "mean_memory"]).copy()
            if len(ma) > 1:
                ma["norm_acc"] = (ma["acceptance"] - ma["acceptance"].min()) / (ma["acceptance"].max() - ma["acceptance"].min() + 1e-9)
                ma["norm_time"] = 1 - (ma["mean_time"] - ma["mean_time"].min()) / (ma["mean_time"].max() - ma["mean_time"].min() + 1e-9)
                ma["norm_mem"] = 1 - (ma["mean_memory"] - ma["mean_memory"].min()) / (ma["mean_memory"].max() - ma["mean_memory"].min() + 1e-9)
                ma["composite"] = (ma["norm_acc"] + ma["norm_time"] + ma["norm_mem"]) / 3
                ma = ma.sort_values("composite", ascending=False)

                composite_label = "Composite Score" if st.session_state.get("ui_lang") == "en" else "Escore Composto"
                display_cols = {
                    "model_short": t("col.model"),
                    "acceptance": t("stcol.acceptance"),
                    "mean_time": t("stcol.time"),
                    "mean_memory": t("stcol.memory"),
                    "composite": composite_label,
                }
                st.dataframe(
                    ma[list(display_cols.keys())].rename(columns=display_cols).style.format({
                        t("stcol.acceptance"): "{:.2%}",
                        t("stcol.time"): "{:.1f}",
                        t("stcol.memory"): "{:.1f}",
                        composite_label: "{:.3f}",
                    }),
                    use_container_width=True, hide_index=True,
                )

                fig = px.bar(ma, x="model_short", y="composite", text="composite",
                             title=t("st.composite_chart"),
                             labels={"composite": "Score", "model_short": t("ax.model")},
                             height=420)
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
                fig.update_xaxes(tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.subheader(t("st.rec_header"))
        st.markdown(t("st.rec_body"))


# ==================================================================
# TAB 6: CODE EXPLORER
# ==================================================================
with tab_explorer:
    st.header(t("ex.header"))
    st.markdown(t("ex.desc"))

    if df_llm.empty:
        st.warning(t("ex.nodata"))
    else:
        df_llm_view = df_llm.copy()
        df_llm_view["model_short"] = df_llm_view["modelo"].map(
            lambda m: MODEL_SHORT.get(str(m).strip(), str(m).strip())
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            llm_models = sorted(df_llm_view["model_short"].dropna().unique().tolist())
            sel_llm_model = st.selectbox(t("ex.model"), llm_models, key="explorer_model")
        with c2:
            llm_langs = sorted(
                df_llm_view[df_llm_view["model_short"] == sel_llm_model]["linguagem"].dropna().unique().tolist()
            )
            sel_llm_lang = st.selectbox(t("ex.language"), llm_langs, key="explorer_lang")
        with c3:
            subset = df_llm_view[
                (df_llm_view["model_short"] == sel_llm_model) &
                (df_llm_view["linguagem"] == sel_llm_lang)
            ]
            problem_ids = sorted(subset["problem_id"].dropna().unique().tolist())
            sel_problem = st.selectbox(t("ex.problem"), problem_ids, key="explorer_problem")

        hit = subset[subset["problem_id"] == sel_problem]
        if not hit.empty:
            row = hit.iloc[0]
            code = row.get("code", "")
            lang_map = {"C++": "cpp", "Java": "java", "Python3": "python"}

            result_row = None
            if not df_all.empty:
                result_match = df_all[
                    (df_all["modelo"] == row["modelo"]) &
                    (df_all["linguagem"] == row["linguagem"]) &
                    (df_all["id_questao"] == sel_problem)
                ]
                if not result_match.empty:
                    result_row = result_match.iloc[0]

            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"{t('ex.code_title')} - {sel_llm_model} / {sel_llm_lang} / Problem {sel_problem}")
                st.code(code or t("ex.empty_code"), language=lang_map.get(sel_llm_lang, "text"))

            with c2:
                st.markdown(t("ex.meta_title"))
                meta = {
                    t("ex.category"): row.get("categoria", "-"),
                    t("ex.reason"): row.get("motivo", "-"),
                    t("ex.starter_ok"): row.get("starter_ok", "-"),
                }
                for k, v in meta.items():
                    st.write(f"**{k}:** {v}")

                if result_row is not None:
                    st.markdown("---")
                    st.markdown(t("ex.result_title"))
                    status = result_row.get("status", "-")
                    is_acc = result_row.get("aceito", False)
                    color = "green" if is_acc else "red"
                    st.markdown(f"**{t('ex.status')}:** :{color}[{status}]")
                    if is_acc:
                        tempo = result_row.get("tempo_ms", "-")
                        mem = result_row.get("memoria_mb", "-")
                        st.write(f"**{t('ex.time')}:** {tempo} ms")
                        st.write(f"**{t('ex.memory')}:** {mem} MB")

                    if not df_meta.empty:
                        pmeta = df_meta[df_meta["problem_id"] == sel_problem]
                        if not pmeta.empty:
                            pm = pmeta.iloc[0]
                            st.markdown("---")
                            st.markdown(t("ex.problem_info"))
                            st.write(f"**{t('ex.title_field')}:** {pm.get('titulo', '-')}")
                            st.write(f"**{t('ex.diff_field')}:** {pm.get('difficulty', '-')}")
                            st.write(f"**{t('ex.topic_field')}:** {pm.get('tema_principal', '-')}")
                else:
                    st.info(t("ex.no_result"))

        # Side-by-side comparison
        st.markdown("---")
        st.subheader(t("ex.compare_header"))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(t("ex.model_a"))
            m_a = st.selectbox(t("ex.model") + " A", llm_models, key="cmp_model_a")
            l_a_opts = sorted(df_llm_view[df_llm_view["model_short"] == m_a]["linguagem"].dropna().unique().tolist())
            l_a = st.selectbox(t("ex.language") + " A", l_a_opts, key="cmp_lang_a")
        with c2:
            st.markdown(t("ex.model_b"))
            m_b = st.selectbox(t("ex.model") + " B", llm_models, index=min(1, len(llm_models) - 1), key="cmp_model_b")
            l_b_opts = sorted(df_llm_view[df_llm_view["model_short"] == m_b]["linguagem"].dropna().unique().tolist())
            l_b = st.selectbox(t("ex.language") + " B", l_b_opts, key="cmp_lang_b")

        ids_a = set(df_llm_view[(df_llm_view["model_short"] == m_a) & (df_llm_view["linguagem"] == l_a)]["problem_id"].dropna())
        ids_b = set(df_llm_view[(df_llm_view["model_short"] == m_b) & (df_llm_view["linguagem"] == l_b)]["problem_id"].dropna())
        common = sorted(ids_a & ids_b)

        if common:
            sel_cmp = st.selectbox(t("ex.compare_problem"), common, key="cmp_problem")

            row_a = df_llm_view[
                (df_llm_view["model_short"] == m_a) &
                (df_llm_view["linguagem"] == l_a) &
                (df_llm_view["problem_id"] == sel_cmp)
            ]
            row_b = df_llm_view[
                (df_llm_view["model_short"] == m_b) &
                (df_llm_view["linguagem"] == l_b) &
                (df_llm_view["problem_id"] == sel_cmp)
            ]

            ca, cb = st.columns(2)
            with ca:
                if not row_a.empty:
                    st.markdown(f"**{m_a} / {l_a}**")
                    st.code(row_a.iloc[0].get("code", ""), language=lang_map.get(l_a, "text"))
            with cb:
                if not row_b.empty:
                    st.markdown(f"**{m_b} / {l_b}**")
                    st.code(row_b.iloc[0].get("code", ""), language=lang_map.get(l_b, "text"))
        else:
            st.info(t("ex.no_common"))


# ==================================================================
# TAB 7: DATA EXPORT
# ==================================================================
with tab_export:
    st.header(t("exp.header"))
    st.markdown(t("exp.desc"))

    if df.empty:
        st.warning(t("exp.nodata"))
    else:
        st.subheader(t("exp.sub_header"))
        st.caption(t("exp.record_count").format(n=len(df)))

        show_cols = [c for c in [
            "modelo", "model_short", "linguagem", "slug", "titulo", "id_questao",
            "difficulty", "tema_principal", "status", "status_paper", "aceito",
            "tempo_ms", "memoria_mb", "submission_id",
        ] if c in df.columns]

        st.dataframe(df[show_cols].head(500), use_container_width=True, hide_index=True)
        if len(df) > 500:
            st.caption(t("exp.showing_500"))

        c1, c2, c3 = st.columns(3)
        with c1:
            csv_data = df[show_cols].to_csv(index=False).encode("utf-8")
            st.download_button(t("exp.dl_csv_all"), csv_data,
                               file_name="submissions_filtered.csv", mime="text/csv")
        with c2:
            json_data = df[show_cols].to_json(orient="records", force_ascii=False).encode("utf-8")
            st.download_button(t("exp.dl_json_all"), json_data,
                               file_name="submissions_filtered.json", mime="application/json")
        with c3:
            if not df_acc.empty:
                csv_acc = df_acc[show_cols].to_csv(index=False).encode("utf-8")
                st.download_button(t("exp.dl_csv_acc"), csv_acc,
                                   file_name="accepted_only.csv", mime="text/csv")

        # Summary tables
        st.markdown("---")
        st.subheader(t("exp.tables_header"))
        st.markdown(t("exp.tables_desc"))

        with st.expander(t("exp.table_oar")):
            summary = pd.DataFrame({
                t("col.result"): ["Accepted", "Compilation Error", "Wrong Answer", "Total"],
                t("col.count"): [
                    len(df[df["status_paper"] == "Accepted"]),
                    len(df[df["status_paper"] == "Compilation Error"]),
                    len(df[df["status_paper"] == "Wrong Answer"]),
                    len(df),
                ],
            })
            summary["%"] = (summary[t("col.count")] / len(df) * 100).round(1)
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.download_button("CSV", summary.to_csv(index=False).encode("utf-8"),
                               "oar_summary.csv", "text/csv", key="dl_oar")

        with st.expander(t("exp.table_arm_diff")):
            df_tmp = df.copy()
            df_tmp["acc"] = (df_tmp["status_paper"] == "Accepted").astype(int)
            ct = df_tmp.pivot_table(index="model_short", columns="difficulty", values="acc", aggfunc="mean").multiply(100)
            ct = ct.reindex(columns=[d for d in DIFF_ORDER if d in ct.columns])
            st.dataframe(ct.round(1), use_container_width=True)
            st.download_button("CSV", ct.round(1).to_csv().encode("utf-8"),
                               "arm_difficulty.csv", "text/csv", key="dl_arm_diff")

        with st.expander(t("exp.table_arm_lang")):
            df_tmp = df.copy()
            df_tmp["acc"] = (df_tmp["status_paper"] == "Accepted").astype(int)
            ct = df_tmp.pivot_table(index="model_short", columns="linguagem", values="acc", aggfunc="mean").multiply(100)
            ct = ct.reindex(columns=[l for l in LANG_ORDER if l in ct.columns])
            st.dataframe(ct.round(1), use_container_width=True)
            st.download_button("CSV", ct.round(1).to_csv().encode("utf-8"),
                               "arm_language.csv", "text/csv", key="dl_arm_lang")

        with st.expander(t("exp.table_etm_diff")):
            if not df_acc.empty:
                ct = df_acc.pivot_table(index="model_short", columns="difficulty", values="tempo_ms", aggfunc="mean")
                ct = ct.reindex(columns=[d for d in DIFF_ORDER if d in ct.columns])
                st.dataframe(ct.round(1), use_container_width=True)
                st.download_button("CSV", ct.round(1).to_csv().encode("utf-8"),
                                   "etm_difficulty.csv", "text/csv", key="dl_etm_diff")

        with st.expander(t("exp.table_etm_lang")):
            if not df_acc.empty:
                ct = df_acc.pivot_table(index="model_short", columns="linguagem", values="tempo_ms", aggfunc="mean")
                ct = ct.reindex(columns=[l for l in LANG_ORDER if l in ct.columns])
                st.dataframe(ct.round(1), use_container_width=True)
                st.download_button("CSV", ct.round(1).to_csv().encode("utf-8"),
                                   "etm_language.csv", "text/csv", key="dl_etm_lang")

        with st.expander(t("exp.table_mum_diff")):
            if not df_acc.empty:
                ct = df_acc.pivot_table(index="model_short", columns="difficulty", values="memoria_mb", aggfunc="mean")
                ct = ct.reindex(columns=[d for d in DIFF_ORDER if d in ct.columns])
                st.dataframe(ct.round(1), use_container_width=True)
                st.download_button("CSV", ct.round(1).to_csv().encode("utf-8"),
                                   "mum_difficulty.csv", "text/csv", key="dl_mum_diff")

        with st.expander(t("exp.table_mum_lang")):
            if not df_acc.empty:
                ct = df_acc.pivot_table(index="model_short", columns="linguagem", values="memoria_mb", aggfunc="mean")
                ct = ct.reindex(columns=[l for l in LANG_ORDER if l in ct.columns])
                st.dataframe(ct.round(1), use_container_width=True)
                st.download_button("CSV", ct.round(1).to_csv().encode("utf-8"),
                                   "mum_language.csv", "text/csv", key="dl_mum_lang")

        # Problem metadata
        if not df_meta.empty:
            st.markdown("---")
            st.subheader(t("exp.meta_header"))
            st.caption(t("exp.meta_count").format(n=len(df_meta)))
            st.dataframe(df_meta[["problem_id", "slug", "titulo", "difficulty", "tema_principal"]].head(336),
                         use_container_width=True, hide_index=True)
            st.download_button(t("exp.dl_meta"),
                               df_meta.to_csv(index=False).encode("utf-8"),
                               "problem_metadata.csv", "text/csv", key="dl_meta")
