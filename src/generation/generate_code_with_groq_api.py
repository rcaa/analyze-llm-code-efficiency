import os
import json
import pandas as pd
import time
import re
from datetime import datetime
from groq import Groq

# === CONFIGURAÇÕES ===
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

CSV_PATH = os.path.join(project_root, "datasets", "leetcode", "sample.csv")
OUTPUT_BASE = os.path.join(project_root, "data")
LINGUAGENS = ["C++", "Java", "Python3"]
CHECKPOINT_PATH = os.path.join(OUTPUT_BASE, "checkpoint.json")
ERROR_LOG_PATH = os.path.join(OUTPUT_BASE, "error_log.json")

# Carrega a API key
config_path = os.path.join(project_root, "config.json")
with open(config_path, "r") as config_file:
    config = json.load(config_file)
    api_key = config["api_key"]

client = Groq(api_key=api_key)

def listar_modelos():
    modelos_ok = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
        "deepseek-r1-distill-llama-70b",
        "llama-4-maverick-17b-128e-instruct",
        "llama-4-scout-17b-16e-instruct",
        "qwen-qwq-32b",
        "qwen3-32b",
    ]
    modelos_api = [m.id for m in client.models.list().data]
    return [m for m in modelos_ok if m in modelos_api]

def extrair_segundos_espera(mensagem):
    match = re.search(r"(\d{1,5}\.?\d*)s", mensagem)
    if match:
        return float(match.group(1))
    return 60

def filtrar_apenas_resposta(conteudo):
    match = re.search(r'solution:', conteudo)
    if match:
        return conteudo[match.start():].strip()
    return conteudo.strip()

def validar_resposta_formatada(conteudo):
    campos = [
        "solution:",
        "efficiency:",
        "time complexity:",
        "space complexity:",
        "energy implications:",
        "explanation:"
    ]
    return all(campo in conteudo for campo in campos)

def carregar_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def salvar_checkpoint(checkpoint):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

def registrar_erro(log_item):
    log = []
    if os.path.exists(ERROR_LOG_PATH):
        with open(ERROR_LOG_PATH, "r", encoding="utf-8") as f:
            log = json.load(f)
    log.append(log_item)
    with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

def carregar_erros():
    if os.path.exists(ERROR_LOG_PATH):
        with open(ERROR_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def obter_resposta_llm(client, modelo, prompt_system, prompt_user, max_tentativas=3, retry_correction=False):
    """Solicita resposta à LLM, com controle robusto de rate limit e tentativas."""
    tentativa = 0
    erros = []
    start = time.time()
    backoff = 60  # tempo inicial de espera em segundos para rate limit
    while tentativa < max_tentativas:
        try:
            resposta = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ],
                model=modelo,
                temperature=0
            )
            tempo_resposta = time.time() - start
            conteudo = resposta.choices[0].message.content
            conteudo_formatado = filtrar_apenas_resposta(conteudo)
            valid = validar_resposta_formatada(conteudo_formatado)
            # Se não for válida, tente corrigir prompt se permitido
            if not valid and retry_correction and tentativa < max_tentativas-1:
                print(f"[WARN] Resposta mal formatada, tentando autocorreção (tentativa {tentativa+1})...")
                prompt_user_corrigido = (
                    f"A resposta anterior não seguiu o formato. "
                    f"Responda estritamente usando apenas o seguinte template, sem comentários, sem <think>, sem raciocínio:\n"
                    f"{prompt_system}\n\nProblema: {prompt_user}"
                )
                prompt_user = prompt_user_corrigido
                tentativa += 1
                continue
            return {
                "resposta_bruta": conteudo,
                "resposta_formatada": conteudo_formatado,
                "valid": valid,
                "tempo_resposta": tempo_resposta,
                "tentativas": tentativa+1,
                "erro": None if valid else "mal formatado",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            msg = str(e)
            erros.append(msg)
            if "429" in msg and ("rate limit" in msg.lower() or "limit" in msg.lower()):
                espera = extrair_segundos_espera(msg)
                espera = max(espera, backoff)
                print(f"[RATE LIMIT] Aguardando {espera:.0f} segundos (tentativa {tentativa+1})...")
                time.sleep(espera)
                backoff = min(backoff * 2, 600)  # aumenta até 10 min, mas nunca para sempre!
                continue  # **nunca sai do loop por causa de rate limit**
            else:
                print(f"[ERRO] {msg}")
                break
        tentativa += 1
    # Se chegou aqui, deu erro!
    return {
        "resposta_bruta": None,
        "resposta_formatada": None,
        "valid": False,
        "tempo_resposta": None,
        "tentativas": tentativa,
        "erro": erros[-1] if erros else "Erro desconhecido",
        "timestamp": datetime.now().isoformat(),
    }

PROMPT_SYSTEM = (
    "You are a programming and algorithms specialist. "
    "When the user provides a problem statement and programming language, analyze and efficiently solve the problem. "
    "Respond strictly and ONLY in the following format, with no additional thoughts, commentary, or step-by-step reasoning. "
    "Do NOT write any intermediate thoughts, explanations of your process, or reasoning steps before the final answer. "
    "Do not include <think>, preambles, or anything outside this format:\n\n"
    "solution: <code or algorithmic description>\n"
    "efficiency: <HIGH | MEDIUM | LOW>\n"
    "time complexity: <BIG O notation>\n"
    "space complexity: <BIG O notation>\n"
    "energy implications: <LOW | MEDIUM | HIGH>\n"
    "explanation: <Concise explanation of the main approach and key strengths/weaknesses. Do not include anything outside of this format.>"
)

USER_PROMPT = (
    "How would you efficiently solve the following problem in {language}? {problem}"
)

def ler_questoes_csv(path, n=None):
    df = pd.read_csv(path)
    questoes = []
    for idx, row in df.iterrows():
        if n is not None and idx >= n:
            break
        questoes.append({
            "id": str(row["id"]),
            "titulo": str(row["titulo"]),
            "url": str(row["url"]),
            "enunciado": str(row["enunciado"])
        })
    return questoes

def main(retry_failed=False):
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Arquivo CSV não encontrado em: {CSV_PATH}")
    
    questoes = ler_questoes_csv(CSV_PATH, n=20)
    modelos = listar_modelos()
    checkpoint = carregar_checkpoint()
    error_log = carregar_erros()

    error_set = set()
    if retry_failed:
        for erro in error_log:
            error_set.add((erro["modelo"], erro["linguagem"], erro["id"]))

    for modelo in modelos:
        for linguagem in LINGUAGENS:
            pasta = os.path.join(OUTPUT_BASE, modelo, linguagem)
            os.makedirs(pasta, exist_ok=True)
            chave_ck = f"{modelo}|{linguagem}"
            if chave_ck not in checkpoint:
                checkpoint[chave_ck] = []
            for idx, questao in enumerate(questoes, start=1):
                id_questao = questao["id"]
                nome_arquivo_json = os.path.join(pasta, f"{idx:02d}.json")
                nome_arquivo_txt = os.path.join(pasta, f"{idx:02d}.txt")

                if retry_failed:
                    if (modelo, linguagem, id_questao) not in error_set:
                        continue
                    print(f"[RETRY] {modelo}, {linguagem}, Questão {id_questao} (idx {idx})")
                else:
                    if id_questao in checkpoint[chave_ck]:
                        print(f"[CHECKPOINT] {modelo}, {linguagem}, Questão {id_questao} já processada. Pulando.")
                        continue
                    if os.path.exists(nome_arquivo_json):
                        checkpoint[chave_ck].append(id_questao)
                        salvar_checkpoint(checkpoint)
                        print(f"[CHECKPOINT-ARQ] {modelo}, {linguagem}, Questão {id_questao} já salva em arquivo. Pulando.")
                        continue
                    print(f"[PROCESSANDO] {modelo}, {linguagem}, Questão {id_questao} (idx {idx})")

                user_prompt = USER_PROMPT.format(language=linguagem, problem=questao["enunciado"])
                while True:  # Mantém até sucesso ou erro não recuperável
                    resultado = obter_resposta_llm(
                        client, modelo, PROMPT_SYSTEM, user_prompt,
                        max_tentativas=3, retry_correction=True
                    )
                    if resultado["erro"] and "rate limit" in str(resultado["erro"]).lower():
                        print("[RATE LIMIT PERSISTENTE] Aguardando e tentando novamente...")
                        time.sleep(60)  # espera adicional
                        continue  # tenta de novo até não dar mais rate limit!
                    break

                # Salva resposta como JSON estruturado
                resposta_json = {
                    "id": questao["id"],
                    "titulo": questao["titulo"],
                    "url": questao["url"],
                    "llm_response": resultado
                }
                with open(nome_arquivo_json, "w", encoding="utf-8") as f_json:
                    json.dump(resposta_json, f_json, indent=2, ensure_ascii=False)

                # Salva também como TXT para leitura rápida/humana
                with open(nome_arquivo_txt, "w", encoding="utf-8") as f_txt:
                    f_txt.write(f"ID: {questao['id']}\n")
                    f_txt.write(f"Title: {questao['titulo']}\n")
                    f_txt.write(f"URL: {questao['url']}\n")
                    f_txt.write(f"Status: {'OK' if resultado['valid'] else 'FALHA'}\n")
                    if resultado["resposta_formatada"]:
                        f_txt.write("\n" + resultado["resposta_formatada"] + "\n")
                    elif resultado["resposta_bruta"]:
                        f_txt.write("\n[Resposta bruta]\n" + resultado["resposta_bruta"] + "\n")
                    else:
                        f_txt.write("\n[Sem resposta da LLM]\n")

                print(f"Salvo: {nome_arquivo_json} e {nome_arquivo_txt} (ok={resultado['valid']})")

                # Se erro ou inválido, registra no erro_log
                if not resultado["valid"]:
                    registrar_erro({
                        "id": id_questao,
                        "modelo": modelo,
                        "linguagem": linguagem,
                        "idx": idx,
                        "erro": resultado["erro"] or "Resposta vazia/formatada incorretamente",
                        "timestamp": resultado["timestamp"],
                        "prompt": user_prompt
                    })

                # Se foi tudo ok, checkpoint
                if resultado["valid"] and not retry_failed:
                    checkpoint[chave_ck].append(id_questao)
                    salvar_checkpoint(checkpoint)

if __name__ == "__main__":
    retry_failed = False  # True para processar só erros antigos
    main(retry_failed=retry_failed)
