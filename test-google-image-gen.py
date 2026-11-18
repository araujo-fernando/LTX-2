import base64
import time
from google import genai

# Inicializa o cliente lendo automaticamente a variável GEMINI_API_KEY
client = genai.Client()

# Lista de prompts que você deseja gerar no lote
prompts = [
    "Um café expresso fumegante em uma mesa de madeira rústica, iluminação natural da manhã",
    "Um gato astronauta flutuando em uma estação espacial, estilo ilustração digital detalhada",
    "Uma paisagem futurista de Belo Horizonte com arquitetura sustentável, luz do pôr do sol",
]

# Monta as requisições no formato esperado pelo Batch API (inline, limite de até 20 MB)
requisicoes_batch = [
    {
        "contents": [{"parts": [{"text": prompt}], "role": "user"}],
        "config": {
            "response_modalities": ["IMAGE"],
            # Opcional: defina proporção (ex: "1:1", "16:9", "9:16")
            "image_config": {"aspect_ratio": "1:1"},
        },
    }
    for prompt in prompts
]

# 1. Cria e envia o job em lote para o Nano Banana 2
job = client.batches.create(
    model="gemini-3.1-flash-image",
    src=requisicoes_batch,
    config={"display_name": "lote-imagens-nano-banana-2"},
)

print(f"Lote enviado! ID do Job: {job.name}")

# 2. Aguarda a conclusão do processamento (polling)
estados_finais = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

while True:
    job_atual = client.batches.get(name=job.name)
    estado = job_atual.state.name
    print(f"Status atual: {estado}")

    if estado in estados_finais:
        break

    # Verifica a cada 30 segundos (em produção/cron, pode aumentar esse intervalo)
    time.sleep(30)

# 3. Extrai e salva as imagens geradas
if job_atual.state.name == "JOB_STATE_SUCCEEDED":
    respostas = job_atual.dest.inlined_responses

    for idx, item in enumerate(respostas):
        if item.response:
            for part in item.response.candidates[0].content.parts:
                if part.inline_data:
                    dados_img = part.inline_data.data
                    # Garante decodificação caso retorne em base64 string ou bytes brutos
                    if isinstance(dados_img, str):
                        dados_img = base64.b64decode(dados_img)

                    nome_arquivo = f"imagem_{idx + 1}.png"
                    with open(nome_arquivo, "wb") as f:
                        f.write(dados_img)
                    print(f"Salvo: {nome_arquivo}")
        elif item.error:
            print(f"Erro no prompt {idx + 1}: {item.error}")
else:
    print(f"O job não foi concluído com sucesso. Estado final: {job_atual.state.name}")
