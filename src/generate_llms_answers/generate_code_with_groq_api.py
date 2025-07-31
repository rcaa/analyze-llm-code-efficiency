import os
import json
import pandas as pd
import time
import re
import ast
from groq import Groq, RateLimitError
from datetime import datetime

# === CONFIGURAÇÕES GLOBAIS ===
# Caminhos
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
PROGRESS_FILE_PATH = os.path.join(project_root, "src/generate_llms_answers/progresso.json")
STATUS_FILE_PATH = os.path.join(project_root, "src/generate_llms_answers/status_respostas.json")
CSV_PATH = os.path.join(project_root, "datasets", "leetcode", "sample.csv")
OUTPUT_BASE = os.path.join(project_root, "data")

# Parâmetros de Coleta
BATCH_SIZE = 20
LINGUAGENS = ["C++", "Java", "Python3"]
LANG_KEY = { "C++": "cpp", "Java": "java", "Python3": "python3" }

# === INICIALIZAÇÃO DA API ===
try:
    config_path = os.path.join(project_root, "config.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
        api_key = config["api_key"]
    client = Groq(api_key=api_key)
except FileNotFoundError:
    print(f"ERRO: Arquivo de configuração 'config.json' não encontrado em {project_root}")
    exit()
except KeyError:
    print("ERRO: A chave 'api_key' não foi encontrada dentro de 'config.json'")
    exit()

# === PROMPTS (Sem alterações) ===
PROMPT_SYSTEM = (
    "You are a programming and algorithms specialist. "
    "When the user provides a problem statement and programming language, analyze and efficiently solve the problem. "
    "Respond strictly and ONLY in the following format, with no additional thoughts, commentary, or step-by-step reasoning. "
    "Do NOT write any intermediate thoughts, explanations of your process, or reasoning steps before the final answer. "
    "Do not include <think>, preambles, or anything outside this format:\n\n"
    "solution: <algorithm solution in code>\n"
    "efficiency: <HIGH | MEDIUM | LOW>\n"
    "time complexity: <BIG O notation>\n"
    "space complexity: <BIG O notation>\n"
    "energy implications: <LOW | MEDIUM | HIGH>\n"
    "explanation: <Concise explanation of the main approach and key strengths/weaknesses. Do not include anything outside of this format.>\n\n"
    "**Important**: Use the provided starter code **exactly as given**. Do not change the function/class signature or add any import statements. Only fill in the required logic within the starter code."
)
USER_PROMPT = (
    "How would you efficiently solve the following LeetCode problem in {language}?\n\n"
    "Problem statement:\n{problem}\n\n"
    "Starter code ({language}):\n```{starter}```\n\n"
    "Please **insert your solution into the starter code above without altering its structure**, and respond **only** in the prescribed format (no commentary):"
)

# === FUNÇÕES AUXILIARES (Sem alterações) ===
def carregar_progresso():
    if not os.path.exists(PROGRESS_FILE_PATH): return 0
    with open(PROGRESS_FILE_PATH, "r") as f:
        return json.load(f).get("ultimo_indice_processado", 0)

def salvar_progresso(indice):
    with open(PROGRESS_FILE_PATH, "w") as f:
        json.dump({"ultimo_indice_processado": indice}, f, indent=2)

def carregar_status_respostas():
    if os.path.exists(STATUS_FILE_PATH):
        with open(STATUS_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def salvar_status_respostas(status_respostas):
    with open(STATUS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(status_respostas, f, indent=2, ensure_ascii=False)

def listar_modelos():
    modelos_excluidos = ["whisper", "tts", "guard", "prompt-guard", "compound", "distil-whisper", "mistral"]
    todos = [m.id for m in client.models.list().data]
    return sorted({m for m in todos if not any(ex in m.lower() for ex in modelos_excluidos)})

def extrair_codigo(conteudo: str) -> str:
    parts = re.split(r'(?i)solution:', conteudo, maxsplit=1)
    if len(parts) < 2: return ""
    after = parts[1]
    before_label = re.split(r'(?i)\n\s*(efficiency:|time complexity:|space complexity:|energy implications:|explanation:)', after, maxsplit=1)[0]
    without_open = re.sub(r'```[^\n]*\n', "", before_label)
    without_close = re.sub(r'\n```', "", without_open)
    return without_close.strip()

def compare_signature(stub_line: str, code_line: str) -> bool:
    m1 = re.search(r'\b(\w+)\s*\(', stub_line)
    m2 = re.search(r'\b(\w+)\s*\(', code_line)
    if not (m1 and m2) or m1.group(1) != m2.group(1): return False
    def tipos(line):
        inside = line[line.find('(')+1 : line.rfind(')')]
        parts = [p.strip() for p in inside.split(',') if p.strip()]
        tipos_list = []
        for p in parts:
            tokens = p.split()
            tipos_list.append(" ".join(tokens[:-1]) if len(tokens) > 1 else tokens[0])
        return tipos_list
    return tipos(stub_line) == tipos(code_line)

def _encontrar_linha_assinatura(codigo: str) -> str:
    for linha in codigo.splitlines():
        linha_strip = linha.strip()
        if '(' in linha and ')' in linha and not linha_strip.startswith(('#', '//', '/*', '*')):
            return linha_strip
    return ""

def validar_resposta(conteudo: str, starter: str, language: str) -> bool:
    conteudo_lower = conteudo.lower()
    expected_labels = ["solution:", "efficiency:", "time complexity:", "space complexity:", "energy implications:", "explanation:"]
    if not all(label in conteudo_lower for label in expected_labels): return False
    codigo_gerado = extrair_codigo(conteudo)
    if not codigo_gerado: return False
    linha_assinatura_starter = _encontrar_linha_assinatura(starter)
    linha_assinatura_gerada = _encontrar_linha_assinatura(codigo_gerado)
    if not linha_assinatura_starter or not linha_assinatura_gerada: return False
    return compare_signature(linha_assinatura_starter, linha_assinatura_gerada)

def filtrar_apenas_resposta(conteudo):
    match = re.search(r'solution:', conteudo, re.IGNORECASE)
    return conteudo[match.start():].strip() if match else conteudo.strip()

def ler_questoes_csv(path, start_index=0, batch_size=20):
    try:
        df = pd.read_csv(path)
        fim_index = start_index + batch_size
        batch_df = df.iloc[start_index:fim_index]
        if batch_df.empty: return []
        questions = []
        for _, row in batch_df.iterrows():
            try:
                starter_dict = ast.literal_eval(row.get("starter_code", "{}"))
            except Exception:
                starter_dict = {}
            questions.append({"id": row["id"], "titulo": row["titulo"], "url": row["url"], "enunciado": row["enunciado"], "starter_code": starter_dict})
        return questions
    except FileNotFoundError: raise
    except Exception as e:
        print(f"Erro ao ler ou processar o CSV: {e}")
        return []

# === LÓGICA PRINCIPAL DE PROCESSAMENTO ===
def determinar_tarefas_de_retentativa(status_respostas, modelos, questoes_totais_df):
    tarefas_para_retentar = []
    mapa_questoes = {str(q["id"]): q for q in questoes_totais_df.to_dict('records')}
    for qid, modelos_status in status_respostas.items():
        for modelo, linguagens_status in modelos_status.items():
            for linguagem, status in linguagens_status.items():
                if status == "erro" and qid in mapa_questoes and modelo in modelos and linguagem in LINGUAGENS:
                    tarefas_para_retentar.append((mapa_questoes[qid], modelo, linguagem))
    return tarefas_para_retentar

# ======================= FUNÇÃO CORRIGIDA =======================
def processar_tarefa(questao, modelo, linguagem, status_respostas):
    """Executa a tarefa e retorna True se a API foi chamada com sucesso, False caso contrário."""
    pasta_modelo = modelo.replace("/", "_")
    pasta = os.path.join(OUTPUT_BASE, pasta_modelo, linguagem)
    os.makedirs(pasta, exist_ok=True)
    
    key = LANG_KEY[linguagem]
    qid = str(questao["id"])
    
    # Inicializa todas as variáveis de resultado
    status, error_message, conteudo_bruto, conteudo_formatado, codigo_extraido, tempo_resposta = None, None, "", "", "", None
    sucesso_api = False

    print(f"Executando tarefa para Questão {qid} com {modelo}/{linguagem}...")
    
    starter = questao["starter_code"].get(key, "")
    
    # CORREÇÃO: Se o starter code não for encontrado, registra um erro em vez de pular.
    if not starter:
        status = "erro"
        error_message = f"Starter code para a linguagem '{linguagem}' não encontrado no CSV."
        print(f"AVISO: {error_message}")
        sucesso_api = True # Consideramos "sucesso" para não colocar o modelo em quarentena por um erro de dados.
    else:
        try:
            start_time = time.time()
            resposta = client.chat.completions.create(messages=[{"role": "system", "content": PROMPT_SYSTEM}, {"role": "user", "content": USER_PROMPT.format(language=linguagem, problem=questao["enunciado"], starter=starter)}], model=modelo, temperature=0)
            tempo_resposta = time.time() - start_time
            conteudo_bruto = resposta.choices[0].message.content
            conteudo_formatado = filtrar_apenas_resposta(conteudo_bruto)
            codigo_extraido = extrair_codigo(conteudo_bruto)
            status = "ok" if validar_resposta(conteudo_bruto, starter, linguagem) else "formato_invalido"
            sucesso_api = True
        except RateLimitError as e:
            status, error_message = "erro", f"RateLimitError: {e}"
        except Exception as e:
            status, error_message = "erro", str(e)

    # O bloco de salvamento agora é sempre executado
    resposta_json = {
        "id": questao["id"], "titulo": questao["titulo"], "url": questao["url"], "modelo": modelo,
        "linguagem": linguagem, "status": status, "error_message": error_message,
        "tempo_resposta": tempo_resposta, "resposta_bruta": conteudo_bruto,
        "resposta_formatada": conteudo_formatado, "code": codigo_extraido,
        "timestamp": datetime.now().isoformat()
    }
    
    path_arquivo_json = os.path.join(pasta, f"{qid}.json")
    with open(path_arquivo_json, "w", encoding="utf-8") as f:
        json.dump(resposta_json, f, indent=2, ensure_ascii=False)
    
    if qid not in status_respostas: status_respostas[qid] = {}
    if modelo not in status_respostas[qid]: status_respostas[qid][modelo] = {}
    status_respostas[qid][modelo][linguagem] = status
    salvar_status_respostas(status_respostas)
    
    print(f"--> Concluído: {path_arquivo_json} com status '{status}'\n")
    return sucesso_api

# ======================= FUNÇÃO CORRIGIDA =======================
def main():
    """Função principal que orquestra o processo de coleta contínua."""
    print("Iniciando serviço de coleta de dados...")
    modelos = listar_modelos()
    status_respostas = carregar_status_respostas()
    try:
        questoes_totais_df = pd.read_csv(CSV_PATH)
        def parse_starter_code(code_str):
            try:
                if pd.isna(code_str): return {}
                return ast.literal_eval(str(code_str))
            except (ValueError, SyntaxError): return {}
        questoes_totais_df['starter_code'] = questoes_totais_df['starter_code'].apply(parse_starter_code)
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Arquivo de questões não encontrado em {CSV_PATH}")
        return

    while True:
        print("\n" + "="*50 + f"\nINICIANDO NOVO CICLO DE PROCESSAMENTO - {datetime.now()}\n" + "="*50)
        
        print("\n--- FASE 1: Verificando tarefas com erro para retentativa ---")
        tarefas_para_retentar = determinar_tarefas_de_retentativa(status_respostas, modelos, questoes_totais_df)
        if tarefas_para_retentar:
            print(f"Encontradas {len(tarefas_para_retentar)} tarefas para retentar.")
            # Laço simplificado: apenas processa a tarefa, sem quarentena.
            for q, m, l in tarefas_para_retentar:
                processar_tarefa(q, m, l, status_respostas)
            print("--- FASE 1 CONCLUÍDA ---")
        else:
            print("Nenhuma tarefa com erro encontrada para retentativa.")

        print("\n--- FASE 2: Buscando novo lote de questões para processar ---")
        ultimo_indice = carregar_progresso()
        novo_lote = ler_questoes_csv(CSV_PATH, start_index=ultimo_indice, batch_size=BATCH_SIZE)
        
        if not novo_lote:
            print("\nNão há mais questões novas no arquivo CSV. O script continuará apenas retentando erros.")
            # O 'break' foi removido para que o script continue rodando em modo de retentativa.
            # Se quiser que o script pare completamente, descomente a linha abaixo.
            # break
        else:
            print(f"Processando novo lote de {len(novo_lote)} questões (índices {ultimo_indice} a {ultimo_indice + len(novo_lote) - 1}).")
            # Laços simplificados: processa cada tarefa independentemente de falhas anteriores.
            for questao in novo_lote:
                for modelo in modelos:
                    for linguagem in LINGUAGENS:
                        processar_tarefa(questao, modelo, linguagem, status_respostas)
            
            novo_indice = ultimo_indice + len(novo_lote)
            salvar_progresso(novo_indice)
            print(f"--- FASE 2 CONCLUÍDA. Progresso salvo. Próximo ciclo começará no índice {novo_indice} ---")
            
        print("\nAguardando 60 segundos antes do próximo ciclo...")
        time.sleep(60)
# === PONTO DE ENTRADA DO SCRIPT ===
if __name__ == "__main__":
    try:
        modelos_disponiveis = listar_modelos()
        print("Modelos de código disponíveis:", *modelos_disponiveis, sep="\n- ")
        main()
    except Exception as e:
        print(f"\nOcorreu um erro fatal não tratado no script: {e}")