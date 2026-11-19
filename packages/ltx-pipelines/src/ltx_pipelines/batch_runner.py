"""Phase-by-phase (model-by-model) multi-scene batch pipeline for LTX-2.5 on ROCm.

Loads each model exactly once across all scenes in the batch, processes every scene
and temporal chunk for that phase, and unloads the model before advancing to the next phase:
  Phase 1: Gemma 4 12B (50:50 split) + EmbeddingsProcessor -> encodes all scene prompts
  Phase 2: VideoEncoder (ImageConditioner on RX 9070 XT) -> encodes all initial/guide frames
  Phase 3: Stage 1 Transformer (LTX-2.5 22B INT8 ConvRot, 50:50 split) -> denoises all scenes
  Phase 4: (--hq only) VideoEncoder + 2x Latent Spatial Upsampler (RX 9070 XT) -> upsamples all latents
  Phase 5: (--hq only) Stage 2 Transformer (+ Distilled LoRA, 50:50 split) -> refines all scenes
  Phase 6: Video Diffusion VAE Decoder (RX 9070 XT) -> decodes all video latents to CPU frames
  Phase 7: Audio VAE Decoder + Vocoder (RX 9070 XT) -> decodes all audio latents and saves MP4s
  Phase 8: (--anime-upscale only) 2x-AnimeSharpV4 / RealESRGAN (RX 9070 XT) -> upscales all MP4s
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path

import torch

from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import ConditioningItem
from ltx_core.model.video_vae import AUTO_TILING, TilingConfig
from ltx_core.types import VideoPixelShape
from ltx_pipelines.anime_upscaler import upscale_videos_anime_batch
from ltx_pipelines.chunks import (
    Chunk,
    ChunkConfig,
    PlaybackSegment,
    decode_chunks,
    denoise_chunks,
    generate_uniform_chunks,
    spatially_upsample_chunks,
)
from ltx_pipelines.chunks.operators import _decode_chunk_video
from ltx_pipelines.chunks.conditionings import (
    assert_image_frames_in_clip,
    image_conditionings_for_chunk,
    images_for_chunk,
)
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.blocks import resolve_num_frames
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    ensure_tiling_config,
    image_conditionings,
    snap_frames_to_grid,
)
from ltx_pipelines.utils.media_io import encode_media_segments
from ltx_pipelines.utils.types import ImageConditioningInput, LoadedImageConditioning, VideoAudio

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchSceneSpec:
    """Specification for a single scene in a multi-scene batch."""

    prompt: str
    output_path: Path
    images: tuple[ImageConditioningInput, ...] = ()
    seed: int | None = None
    num_frames: int | None = None
    negative_prompt: str | None = None


@dataclass
class _PreparedScene:
    spec: BatchSceneSpec
    seed: int
    num_frames: int
    generator: torch.Generator
    noiser: GaussianNoiser
    loaded_images: list[LoadedImageConditioning]
    v_context_p: torch.Tensor
    a_context_p: torch.Tensor
    v_context_n: torch.Tensor
    a_context_n: torch.Tensor
    tiling_config: TilingConfig
    chunks: list[Chunk]


def load_batch_specs(
    *,
    batch_file: Path | None = None,
    prompts_file: Path | None = None,
    images_dir: Path | None = None,
    output_dir: Path,
    default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> list[BatchSceneSpec]:
    """Build a list of BatchSceneSpec from either a JSON/JSONL file or a prompts.txt + optional images folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    specs: list[BatchSceneSpec] = []

    if batch_file is not None:
        raw_text = batch_file.read_text(encoding="utf-8").strip()
        if raw_text.startswith("["):
            entries = json.loads(raw_text)
        else:
            entries = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        for idx, item in enumerate(entries):
            prompt = item["prompt"]
            out = Path(item.get("output_path") or (output_dir / f"cena_{idx + 1:03d}.mp4"))
            images: list[ImageConditioningInput] = []
            if "animate" in item and item["animate"]:
                img_p = str(Path(item["animate"]).expanduser().resolve())
                strength = float(item.get("strength", 1.0))
                images.append(ImageConditioningInput.at(img_p, 0, strength, None))
            elif "image" in item and item["image"]:
                img_p = str(Path(item["image"]).expanduser().resolve())
                strength = float(item.get("strength", 1.0))
                images.append(ImageConditioningInput.at(img_p, 0, strength, None))
            specs.append(
                BatchSceneSpec(
                    prompt=prompt,
                    output_path=out,
                    images=tuple(images),
                    seed=item.get("seed"),
                    num_frames=item.get("num_frames"),
                    negative_prompt=item.get("negative_prompt", default_negative_prompt),
                )
            )
        return specs

    if prompts_file is None:
        raise ValueError("Either batch_file or prompts_file must be provided for batch execution.")

    prompts = [line.strip() for line in prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"Prompts file {prompts_file} is empty.")

    image_files: list[Path] = []
    if images_dir is not None:
        if images_dir.is_file():
            image_files = [images_dir]
        elif images_dir.is_dir():
            image_files = sorted(
                p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            )
        if image_files and len(image_files) != len(prompts):
            logger.warning(
                "Found %d images in %s for %d prompts; pairing up to the minimum count.",
                len(image_files),
                images_dir,
                len(prompts),
            )

    count = min(len(prompts), len(image_files)) if image_files else len(prompts)
    for idx in range(count):
        stem = image_files[idx].stem if image_files else f"cena_{idx + 1:03d}"
        out_path = output_dir / f"{stem}.mp4"
        imgs: tuple[ImageConditioningInput, ...] = ()
        if image_files:
            imgs = (ImageConditioningInput.at(str(image_files[idx].resolve()), 0, 1.0, None),)
        specs.append(
            BatchSceneSpec(
                prompt=prompts[idx],
                output_path=out_path,
                images=imgs,
                negative_prompt=default_negative_prompt,
            )
        )
    return specs


@torch.inference_mode()
def run_phased_batch(  # noqa: PLR0912, PLR0913, PLR0915
    pipeline: TI2VidTwoStagesPipeline | TI2VidOneStagePipeline,
    scenes: Sequence[BatchSceneSpec],
    *,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    seed: int,
    video_guider_params: MultiModalGuiderParams,
    audio_guider_params: MultiModalGuiderParams,
    chunk_config: ChunkConfig | None = None,
    max_batch_size: int = 1,
    refine_audio: bool = False,
    anime_upscaler_path: Path | None = None,
    upscale_res: tuple[int, int] | None = None,
) -> None:
    """Execute a batch of scenes phase-by-phase (one model loaded at a time across all scenes)."""
    if not scenes:
        return
    is_two_stage = isinstance(pipeline, TI2VidTwoStagesPipeline)
    assert_resolution(height=height, width=width, is_two_stage=is_two_stage)
    device = pipeline.device
    dtype = pipeline.dtype
    vae_dtype = torch.bfloat16

    stage_1 = pipeline.stage_1 if is_two_stage else pipeline.stage
    stage_1_width = width // 2 if is_two_stage else width
    stage_1_height = height // 2 if is_two_stage else height

    # -------------------------------------------------------------------------
    # PHASE 1: Prompt Encoding (Gemma 4 12B 50:50 + EmbeddingsProcessor once!)
    # -------------------------------------------------------------------------
    logger.info("=== [PHASE 1] Encoding %d scene prompts with Gemma 4 12B (loaded once) ===", len(scenes))
    flat_prompts: list[str] = []
    for scene in scenes:
        flat_prompts.append(scene.prompt)
        flat_prompts.append(scene.negative_prompt or DEFAULT_NEGATIVE_PROMPT)
    encoded_contexts = pipeline.prompt_encoder(flat_prompts)

    prepared_scenes: list[_PreparedScene] = []
    for idx, scene in enumerate(scenes):
        ctx_p = encoded_contexts[2 * idx]
        ctx_n = encoded_contexts[2 * idx + 1]
        scene_seed = scene.seed if scene.seed is not None else seed
        scene_raw_frames = scene.num_frames if scene.num_frames is not None else num_frames
        resolved_frames = snap_frames_to_grid(
            resolve_num_frames(
                scene_raw_frames,
                pipeline.duration_predictor,
                video_encoding=ctx_p.video_encoding,
                audio_encoding=ctx_p.audio_encoding,
                frame_rate=frame_rate,
            )
        )
        scene_chunk_cfg = (
            chunk_config
            if chunk_config is not None
            else (
                ChunkConfig(chunk_pixel_frames=97, next_video_carry_frames=25)
                if resolved_frames > 121
                else ChunkConfig(chunk_pixel_frames=resolved_frames, next_video_carry_frames=0)
            )
        )
        tiling_cfg = ensure_tiling_config(
            AUTO_TILING,
            vae_checkpoint_path=pipeline.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(
                batch=1,
                frames=min(resolved_frames, scene_chunk_cfg.chunk_pixel_frames),
                height=height,
                width=width,
                fps=frame_rate,
            ),
            diffvae_optimization=pipeline.video_decoder.diffvae_optimization,
            device=device,
            keyframes=False,
            vae_dtype=vae_dtype,
        )
        loaded_imgs = pipeline.image_conditioner.load_conditionings(scene.images, None)
        assert_image_frames_in_clip(loaded_imgs, resolved_frames)

        gen = torch.Generator(device=device).manual_seed(scene_seed)
        noiser = GaussianNoiser(generator=gen)
        target_s1 = VideoPixelShape(
            batch=1,
            frames=resolved_frames,
            height=stage_1_height,
            width=stage_1_width,
            fps=frame_rate,
        )
        raw_chunks = list(
            generate_uniform_chunks(
                target=target_s1,
                context=VideoAudio(video=ctx_p.video_encoding, audio=ctx_p.audio_encoding),
                device=device,
                dtype=dtype,
                config=scene_chunk_cfg,
                video_scale_factors=stage_1.video_scale_factors,
            )
        )
        prepared_scenes.append(
            _PreparedScene(
                spec=scene,
                seed=scene_seed,
                num_frames=resolved_frames,
                generator=gen,
                noiser=noiser,
                loaded_images=loaded_imgs,
                v_context_p=ctx_p.video_encoding,
                a_context_p=ctx_p.audio_encoding,
                v_context_n=ctx_n.video_encoding,
                a_context_n=ctx_n.audio_encoding,
                tiling_config=tiling_cfg,
                chunks=raw_chunks,
            )
        )

    # -------------------------------------------------------------------------
    # PHASE 2: Image Conditioning (VideoEncoder loaded once on RX 9070 XT!)
    # -------------------------------------------------------------------------
    if any(ps.loaded_images for ps in prepared_scenes):
        logger.info("=== [PHASE 2] Encoding initial/guide frames with VideoEncoder (loaded once) ===")
        with pipeline.image_conditioner.resident() as video_encoder:
            for ps in prepared_scenes:
                if not ps.loaded_images:
                    continue
                updated_chunks: list[Chunk] = []
                for chunk in ps.chunks:
                    local_imgs = images_for_chunk(chunk, ps.loaded_images, ps.num_frames)
                    if chunk.video is not None and local_imgs:
                        s1_items = image_conditionings(
                            images=list(local_imgs),
                            height=stage_1_height,
                            width=stage_1_width,
                            video_encoder=video_encoder,
                            dtype=chunk.video.dtype,
                            device=chunk.video.device,
                        )
                        s2_items = (
                            image_conditionings(
                                images=list(local_imgs),
                                height=height,
                                width=width,
                                video_encoder=video_encoder,
                                dtype=chunk.video.dtype,
                                device=chunk.video.device,
                            )
                            if is_two_stage
                            else s1_items
                        )
                        s1_h = stage_1_height // stage_1.video_scale_factors.height
                        updated_chunks.append(
                            replace(
                                chunk,
                                make_video_conditionings=lambda c, f1=s1_items, f2=s2_items, h1=s1_h: (
                                    list(f1) if c.video is not None and c.video.shape[-2] == h1 else list(f2)
                                ),
                            )
                        )
                    else:
                        updated_chunks.append(chunk)
                ps.chunks = updated_chunks

    # -------------------------------------------------------------------------
    # PHASE 3: Stage 1 Transformer Denoising (LTX-2.5 22B 50:50 loaded once!)
    # -------------------------------------------------------------------------
    logger.info("=== [PHASE 3] Running Stage 1 Transformer across all %d scenes (loaded once) ===", len(prepared_scenes))
    sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=device)
    with stage_1.resident():
        for idx, ps in enumerate(prepared_scenes):
            logger.info("--- Stage 1 Denoising scene %d/%d (%s) ---", idx + 1, len(prepared_scenes), ps.spec.output_path.name)
            denoiser = FactoryGuidedDenoiser(
                video_guider_factory=create_multimodal_guider_factory(
                    params=video_guider_params, negative_context=ps.v_context_n
                ),
                audio_guider_factory=create_multimodal_guider_factory(
                    params=audio_guider_params, negative_context=ps.a_context_n
                ),
            )
            ps.chunks = list(
                denoise_chunks(
                    iter(ps.chunks),
                    stage_1,
                    sigmas=sigmas,
                    noiser=ps.noiser,
                    fps=frame_rate,
                    max_batch_size=max_batch_size,
                    denoiser=denoiser,
                )
            )

    if is_two_stage:
        assert isinstance(pipeline, TI2VidTwoStagesPipeline)
        # ---------------------------------------------------------------------
        # PHASE 4: 2x Latent Spatial Upsampler (loaded once on RX 9070 XT!)
        # ---------------------------------------------------------------------
        logger.info("=== [PHASE 4] Running 2x Latent Spatial Upsampler across all %d scenes (loaded once) ===", len(prepared_scenes))
        with pipeline.upsampler.resident():
            for ps in prepared_scenes:
                ps.chunks = list(spatially_upsample_chunks(iter(ps.chunks), pipeline.upsampler))

        # ---------------------------------------------------------------------
        # PHASE 5: Stage 2 Transformer + Distilled LoRA (50:50 loaded once!)
        # ---------------------------------------------------------------------
        logger.info("=== [PHASE 5] Running Stage 2 Transformer (Distilled LoRA) across all %d scenes (loaded once) ===", len(prepared_scenes))
        s2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
        with pipeline.stage_2.resident():
            for idx, ps in enumerate(prepared_scenes):
                logger.info("--- Stage 2 Refining scene %d/%d (%s) ---", idx + 1, len(prepared_scenes), ps.spec.output_path.name)
                ps.chunks = list(
                    denoise_chunks(
                        iter(ps.chunks),
                        pipeline.stage_2,
                        sigmas=s2_sigmas,
                        noiser=ps.noiser,
                        fps=frame_rate,
                        freeze_audio=not refine_audio,
                    )
                )

    # -------------------------------------------------------------------------
    # PHASE 6 & 7: Video Diffusion VAE Decode (loaded once) + Audio VAE Decode (loaded once)
    # -------------------------------------------------------------------------
    logger.info("=== [PHASE 6] Decoding video latents for all %d scenes with Video Diffusion VAE (loaded once) ===", len(prepared_scenes))
    scene_raw_decoded_video: list[list[torch.Tensor | None]] = []
    with pipeline.video_decoder.resident(dtype=vae_dtype):
        for idx, ps in enumerate(prepared_scenes):
            logger.info("--- Decoding video for scene %d/%d (%s) ---", idx + 1, len(prepared_scenes), ps.spec.output_path.name)
            per_chunk_video: list[torch.Tensor | None] = []
            for chunk in ps.chunks:
                decoded = _decode_chunk_video(
                    chunk,
                    pipeline.video_decoder,
                    ps.tiling_config,
                    ps.generator,
                    dtype=vae_dtype,
                    keyframes=False,
                )
                per_chunk_video.append(decoded.cpu() if decoded is not None else None)
            scene_raw_decoded_video.append(per_chunk_video)

    logger.info("=== [PHASE 7] Decoding audio latents with Audio VAE + Vocoder (loaded once) and encoding MP4s ===")
    with pipeline.audio_decoder.resident():
        for ps, predecoded_video in zip(prepared_scenes, scene_raw_decoded_video, strict=True):
            segments = decode_chunks(
                iter(ps.chunks),
                pipeline.video_decoder,
                pipeline.audio_decoder,
                fps=frame_rate,
                tiling_config=ps.tiling_config,
                generator=ps.generator,
                dtype=vae_dtype,
                keyframes=False,
                predecoded_video=predecoded_video,
            )
            encode_media_segments(
                segments,
                fps=frame_rate,
                output_path=ps.spec.output_path,
                video_chunks_number=None,
                color_space=None,
            )

    # -------------------------------------------------------------------------
    # PHASE 8: Post-Generation Anime Upscaler (loaded once on RX 9070 XT!)
    # -------------------------------------------------------------------------
    if anime_upscaler_path is not None:
        logger.info("=== [PHASE 8] Upscaling all %d videos with %s (loaded once) ===", len(prepared_scenes), anime_upscaler_path.name)
        target_w, target_h = upscale_res if upscale_res is not None else ((1920, 1080) if is_two_stage else (None, None))
        pairs = [(ps.spec.output_path, ps.spec.output_path) for ps in prepared_scenes]
        upscale_videos_anime_batch(
            pairs,
            model_path=anime_upscaler_path,
            target_width=target_w,
            target_height=target_h,
        )
