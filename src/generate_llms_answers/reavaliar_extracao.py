#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, re
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
OUTPUT_BASE = os.path.join(project_root, "data")  # mude para "data" se quiser
DATASET_DIR = os.path.join(project_root, "datasets", "leetcode")
STARTER_FILES = [
    "public_problems_without_images.json",
    # Se quiser considerar mais fontes, descomente:
    # "public_problems.json",
    # "public_problems_with_images.json",
    # "easy.json", "medium.json", "hard.json",
]

# ──────────────────────────────────────────────────────────────────────────────
# Extração robusta (preserva "class Solution {" quando vier após ```class ...)
# ──────────────────────────────────────────────────────────────────────────────
_KNOWN_LANG_TAGS = {
    "c","cpp","c++","cc","h","hpp","java","python","python3","py","go","javascript","js","typescript","ts",
    "csharp","c#","cs","rust","rs","ruby","rb","php","swift","kotlin","scala","perl","pl","r","julia",
    "solidity","sol","dart","zig","lua","haskell","hs","ocaml","ml","shell","bash","sh","powershell","ps1",
    "sql","mysql","postgresql"
}
_FENCE_RE = re.compile(r"(?is)(?P<fence>`{3,}|~{3,})(?P<inner>.*?)(?P=fence)")

def _looks_like_lang_tag(line0: str) -> bool:
    t = line0.strip().lower()
    if " " in t:
        return False
    t = t.strip(" :;,.")
    return t in _KNOWN_LANG_TAGS

def extrair_codigo(conteudo: str) -> str:
    if not conteudo:
        return ""
    m_sol = re.search(r"(?i)solution\s*:\s*", conteudo)
    section = conteudo[m_sol.end():] if m_sol else conteudo
    m = _FENCE_RE.search(section)
    if m:
        inner = m.group("inner").strip("\n\r\t ")
        lines = inner.splitlines()
        if lines and _looks_like_lang_tag(lines[0]):
            lines = lines[1:]
        code = "\n".join(lines).strip()
        if code:
            return code
    labels_pattern = r"(?i)\n\s*(efficiency:|time complexity:|space complexity:|energy implications:|explanation:)"
    code = re.split(labels_pattern, section, maxsplit=1)[0].strip()
    return code

# ──────────────────────────────────────────────────────────────────────────────
# Remoção de imports/includes por linguagem
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_lang(lang: str) -> str:
    if not lang:
        return ""
    l = lang.strip().lower()
    if "c++" in l or l == "cpp":
        return "c++"
    if l.startswith("python"):
        return "python"
    if l == "java":
        return "java"
    return l

def remover_imports(code: str, lang: str) -> str:
    if not code:
        return code
    norm = _normalize_lang(lang)
    out = []
    for line in code.splitlines():
        if norm == "c++":
            if re.match(r'^\s*#\s*include\b', line):
                continue
        elif norm == "java":
            if re.match(r'^\s*import\s+.+;\s*$', line):
                continue
        elif norm == "python":
            if re.match(r'^\s*import\b', line):
                continue
            if re.match(r'^\s*from\b.+\bimport\b', line):
                continue
        out.append(line)
    cleaned = "\n".join(out).strip()
    return cleaned if cleaned else code

# ──────────────────────────────────────────────────────────────────────────────
# Carregar starters (id → {"cpp": "...", "java": "...", "python3": "..."} )
# ──────────────────────────────────────────────────────────────────────────────
def carregar_starters():
    starters = {}
    carregados = 0
    for fname in STARTER_FILES:
        fpath = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[starter] arquivo não encontrado: {fname}")
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if isinstance(items, dict):
                # alguns dumps podem ser dict com chave "items" etc.
                items = items.get("items") or items.get("data") or []
            for it in items:
                _id = it.get("id")
                sc = it.get("starter_code") or {}
                if _id is not None and isinstance(sc, dict):
                    starters[_id] = sc
            print(f"[starter] carregado: {fname} (itens: {len(items)})")
            carregados += 1
        except Exception as e:
            print(f"[starter][ERRO] {fname}: {e}")
    print(f"[starter] arquivos processados: {carregados}, starters únicos: {len(starters)}")
    return starters

# ──────────────────────────────────────────────────────────────────────────────
# Validação de aderência ao starter
# ──────────────────────────────────────────────────────────────────────────────
def _extract_method_from_cpp(starter: str):
    # tenta capturar nome e parâmetros do primeiro método dentro do class Solution
    m_cls = re.search(r"class\s+Solution\b.*?\{(.*)\};", starter, flags=re.S | re.M)
    body = m_cls.group(1) if m_cls else starter
    m = re.search(r"([A-Za-z_][\w:<>\s\*&\[\]]+)\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", body, flags=re.S)
    if not m:
        return None, []
    method = m.group(2)
    params_raw = m.group(3).strip()
    params = []
    if params_raw:
        for p in params_raw.split(","):
            # variável é o último identificador
            mname = re.search(r"([A-Za-z_]\w*)\s*(?:\[\s*\])?\s*$", p.strip())
            if mname:
                params.append(mname.group(1))
    return method, params

def _extract_method_from_java(starter: str):
    m_cls = re.search(r"class\s+Solution\b.*?\{(.*)\}", starter, flags=re.S | re.M)
    body = m_cls.group(1) if m_cls else starter
    m = re.search(r"(public|protected|private)?\s*(static\s+)?([\w\[\]<> ,]+)\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", body, flags=re.S)
    if not m:
        return None, []
    method = m.group(4)
    params_raw = m.group(5).strip()
    params = []
    if params_raw:
        for p in params_raw.split(","):
            mname = re.search(r"([A-Za-z_]\w*)\s*$", p.strip())
            if mname:
                params.append(mname.group(1))
    return method, params

def _extract_method_from_python(starter: str):
    # busca dentro da classe Solution
    m_cls = re.search(r"class\s+Solution\s*:\s*(.*)", starter, flags=re.S)
    body = m_cls.group(1) if m_cls else starter
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:", body)
    if not m:
        return None, []
    method = m.group(1)
    params_raw = m.group(2)
    # remove type hints e espaços
    params = []
    for p in [x.strip() for x in params_raw.split(",") if x.strip()]:
        name = p.split(":")[0].strip().split("=")[0].strip()
        params.append(name)
    return method, params

def obter_header_starter(starter: str, lang: str):
    langn = _normalize_lang(lang)
    if langn == "c++":
        return _extract_method_from_cpp(starter)
    if langn == "java":
        return _extract_method_from_java(starter)
    if langn == "python":
        return _extract_method_from_python(starter)
    return None, []

def check_starter_compliance(starter: str, code: str, lang: str):
    """
    Retorna (ok: bool, reason: str).
    Regra: deve existir 'class Solution' (C++/Java/Python) e o mesmo método
    com os mesmos nomes de parâmetros na mesma ordem.
    """
    if not starter:
        return False, "starter ausente no dataset"
    if not code.strip():
        return False, "code vazio"

    langn = _normalize_lang(lang)
    # precisa ter 'class Solution'
    if langn in ("c++", "java", "python") and "class Solution" not in code:
        return False, "ausente 'class Solution'"

    method, params = obter_header_starter(starter, lang)
    if not method:
        return False, "não consegui ler assinatura do starter"

    # monta regex para encontrar a assinatura no código final
    # exigimos os MESMOS nomes de parâmetros, em ordem (tipos/hints livres)
    param_parts = [r"[^,)]*\b" + re.escape(p) + r"\b" for p in params]
    joined = r"\s*,\s*".join(param_parts)
    pat = r"\b" + re.escape(method) + r"\s*\(\s*" + joined + r"\s*\)"
    if not re.search(pat, code, flags=re.S):
        return False, f"assinatura divergente: esperada '{method}({', '.join(params)})'"

    return True, "starter respeitado"

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    starters = carregar_starters()
    total = alterados = sem_resposta = 0

    for root, _, files in os.walk(OUTPUT_BASE):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            path = os.path.join(root, fn)
            total += 1

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[ERRO] ler {path}: {e}")
                continue

            _id = data.get("id")
            lang = data.get("linguagem", "")
            resposta = data.get("resposta", "")
            if not resposta:
                sem_resposta += 1
                print(f"[{path}] id={_id} lang={lang} -> sem 'resposta'")
                continue

            # 1) extrair & limpar imports
            novo_code = extrair_codigo(resposta)
            novo_code = remover_imports(novo_code, lang)

            antigo_code = data.get("code", "")
            if "code_extracted_previous" not in data:
                data["code_extracted_previous"] = antigo_code

            # 2) validar contra starter
            starter = None
            st_map = starters.get(_id)
            # mapear chave de linguagem do dataset
            if st_map:
                key = {"C++": "cpp", "Java": "java", "Python3": "python3"}.get(lang, None)
                starter = st_map.get(key) if key else None

            starter_ok, starter_reason = check_starter_compliance(starter or "", novo_code, lang)

            # 3) salvar
            if novo_code != antigo_code:
                alterados += 1
            data["code"] = novo_code
            data["starter_ok"] = starter_ok
            data["starter_check_reason"] = starter_reason
            data["reavaliado_em"] = datetime.now().isoformat()

            # limpeza de sobras antigas
            for k in ("categoria_reavaliada", "motivo_reavaliado"):
                if k in data:
                    del data[k]

            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[ERRO] salvar {path}: {e}")
                continue

            # 4) DEBUG
            print(f"[OK] {path}")
            print(f"     id={_id} lang={lang} code_len={len(novo_code)}")
            print(f"     starter_src={'ok' if starter else 'não encontrado'} | starter_ok={starter_ok} | motivo={starter_reason}")

    print("─" * 70)
    print(f"Arquivos processados: {total}")
    print(f"Atualizados (code mudou): {alterados}")
    print(f"Sem 'resposta': {sem_resposta}")

if __name__ == "__main__":
    main()
