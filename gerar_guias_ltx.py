import base64
import os
import time
from google import genai

client = genai.Client()

# Lê os prompts do arquivo (ignora linhas vazias)
with open("prompts.txt", "r", encoding="utf-8") as f:
    prompts = [linha.strip() for linha in f if linha.strip()]

if not prompts:
    raise ValueError("O arquivo prompts.txt está vazio.")

os.makedirs("guias_16_9", exist_ok=True)

# Monta as requisições otimizadas para LTX 720p (16:9 em 1K = ~1344x768)
requisicoes_batch = [
    {
        "contents": [{"parts": [{"text": prompt}], "role": "user"}],
        "config": {
            "response_modalities": ["IMAGE"],
            "image_config": {
                "aspect_ratio": "16:9",
                "image_size": "1K",
            },
        },
    }
    for prompt in prompts
]

print(f"Enviando lote com {len(prompts)} imagens em 16:9...")
job = client.batches.create(
    model="gemini-3.1-flash-image",
    src=requisicoes_batch,
    config={"display_name": "guias-ltx-720p-16-9"},
)

print(f"Job criado: {job.name}")

estados_finais = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

while True:
    job_atual = client.batches.get(name=job.name)
    estado = job_atual.state.name
    print(f"Status: {estado}")

    if estado in estados_finais:
        break

    time.sleep(30)

if job_atual.state.name == "JOB_STATE_SUCCEEDED":
    for idx, item in enumerate(job_atual.dest.inlined_responses):
        if item.response:
            for part in item.response.candidates[0].content.parts:
                if part.inline_data:
                    dados_img = part.inline_data.data
                    if isinstance(dados_img, str):
                        dados_img = base64.b64decode(dados_img)

                    caminho = os.path.join("guias_16_9", f"cena_{idx + 1:03d}.png")
                    with open(caminho, "wb") as img_file:
                        img_file.write(dados_img)
                    print(f"Salvo: {caminho}")
        elif item.error:
            print(f"Erro na cena {idx + 1}: {item.error}")
else:
    print(f"Falha no processamento. Estado final: {job_atual.state.name}")
