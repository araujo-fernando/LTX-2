
  ### 1. Modelos Baixados e Vinculados

  • LTX-2.5 Distilled LoRA (Refinamento do Estágio 2):
      • Baixado no cache do Hugging Face (Lightricks/LTX-2.5) e vinculado em models/ltx-2.5/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors.
  • Upscalers de Anime (Pós-processamento na RX 9070 XT):
      • 2x-AnimeSharpV4 (Fast RCAN PU e RCAN Full): models/upscalers/2x-AnimeSharpV4/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors e models/upscalers/2x-AnimeSharpV4/2x-AnimeSharpV4_RCAN.safetensors.
      • RealESRGAN_x4plus_anime_6B: models/upscalers/RealESRGAN_x4plus_anime_6B.pth.

  ──────
  ### 2. Novas Flags no uv run ltx-run (packages/ltx-pipelines/src/ltx_pipelines/cli_rocm.py) e Upscaler (packages/ltx-pipelines/src/ltx_pipelines/anime_upscaler.py)

  • --hq (Pipeline de 2 Estágios + Auto-Chunking):
      • Aciona automaticamente o pipeline de 2 estágios (packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py):
          1. Gemma 4 12B (split 50:50 entre cuda:0 e cuda:1) codifica o prompt → descarrega da VRAM.
          2. Estágio 1 (LTX-2.5 22B INT8 ConvRot, split 50:50) gera a estrutura e o movimento na metade da resolução → descarrega da VRAM.
          3. 2x Latent Spatial Upsampler roda na RX 9070 XT (cuda:0) → descarrega da VRAM.
          4. Estágio 2 (LTX-2.5 22B INT8 ConvRot + Distilled LoRA fundido em tempo de carga nos pesos INT8 com rotação de Hadamard, split 50:50) refina os detalhes em 3 passos rápidos → descarrega da VRAM.
          5. Video Diffusion VAE + Audio VAE decodificam na RX 9070 XT (cuda:0) → descarregam da VRAM.
      • Padrões automáticos do --hq: resolução 720p (1280x704), 97 frames (~4s), 25 passos e saída em output_hq.mp4.
      • Janelas deslizantes automáticas para cenas de 10s a 30s: sempre que --num-frames for maior que 97 (ex.: 241 para 10s ou 721 para 30s), o ltx-run injeta automaticamente --chunk-pixel-frames 97 --chunk-carry-frames 25, mantendo o uso de VRAM constante.
  • --animate <caminho_do_frame.png> [força] (Animação de Frame Inicial):
      • Permite animar qualquer imagem/frame inicial diretamente (atalho para --image <caminho> 0 <força>, com força padrão 1.0).
      • Além disso, a flag --image <caminho.png> também foi simplificada para aceitar apenas 1 argumento (assumindo frame 0 e força 1.0), continuando compatível com a sintaxe completa (--image <caminho> <frame_idx> <força> [guide|replace]) caso queira fixar keyframes intermediários ou finais.
  • --anime-upscale [animesharp|animesharp-hq|realesrgan] e --upscale-res {1080p,720p,WxH}:
      • Após salvar o vídeo (e com a VRAM totalmente livre), carrega o upscaler de anime na RX 9070 XT (cuda:0), faz o upscale dos frames limpando o traço e as cores estilo Ghibli/cel-shading (por padrão para 1920x1080 / 1080p quando usado com --hq), remuxa o áudio original em estéreo 48 kHz e descarrega o upscaler.

  ──────
  ### 3. Exemplos Práticos de Uso

  1. Gerar cena de 10 segundos em 720p HQ e fazer upscale final para 1080p (YouTube):

    uv run ltx-run --hq --anime-upscale \
      --prompt "Hand-drawn Studio Ghibli anime style, a young girl with a straw hat walking along a sunlit country path beside a lush green forest, cumulus clouds in a deep blue summer sky, gentle breeze rustling the leaves, soft acoustic guitar and wind sound" \
      --num-frames 241 \
      --output-path cena_01_1080p.mp4

  2. Animar um frame existente (frame_inicial.png) por 10 segundos em alta qualidade (720p → 1080p):

    uv run ltx-run --hq --animate frame_inicial.png --anime-upscale \
      --prompt "Hand-drawn Studio Ghibli anime scene, camera slowly pans right as cherry blossom petals blow in the wind, warm sunset lighting" \
      --num-frames 241 \
      --output-path cena_animada_1080p.mp4

  3. Usar o RealESRGAN_x4plus_anime_6B em vez do 2x-AnimeSharpV4:

    uv run ltx-run --hq --animate frame_inicial.png --anime-upscale realesrgan \
      --prompt "Studio Ghibli anime style, rain falling on a quiet train station at dusk" \
      --output-path cena_chuva_1080p.mp4
