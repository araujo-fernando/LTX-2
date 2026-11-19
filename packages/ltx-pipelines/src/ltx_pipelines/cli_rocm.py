"""Simplified CLI entrypoint for LTX-2.5 on ROCm with pre-configured model paths, --hq, --animate, --batch, and anime upscalers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys

# Ensure only the two discrete RDNA4 GPUs (RX 9070 XT and RX 9070) are visible to ROCm
# before torch is imported, excluding the integrated GPU (gfx1036).
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0,1")

_COMFY_SITE = Path("/home/fernando/workspace/ComfyUI/.venv/lib/python3.12/site-packages")
try:
    import comfy_kitchen  # noqa: F401
except ImportError:
    if (_COMFY_SITE / "comfy_kitchen").is_dir():
        sys.path.append(str(_COMFY_SITE))

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODELS_DIR = _REPO_ROOT / "models" / "ltx-2.5"
_UPSCALERS_DIR = _REPO_ROOT / "models" / "upscalers"

_DEFAULT_FLAGS: tuple[tuple[str, Path], ...] = (
    (
        "--transformer-path",
        _MODELS_DIR / "diffusion_models" / "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
    ),
    (
        "--text-encoder-path",
        _MODELS_DIR / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    ),
    (
        "--video-vae-path",
        _MODELS_DIR / "vae" / "ltx-2.5-video-vae-bf16.safetensors",
    ),
    (
        "--audio-vae-path",
        _MODELS_DIR / "vae" / "ltx-2.5-audio-vae-bf16.safetensors",
    ),
    (
        "--duration-head-path",
        _MODELS_DIR / "model_patches" / "ltx-2.5-duration-head-bf16.safetensors",
    ),
)

_FAST_DEFAULT_PARAMS: tuple[tuple[str, str], ...] = (
    ("--width", "512"),
    ("--height", "320"),
    ("--num-frames", "25"),
    ("--num-inference-steps", "15"),
    ("--output-path", "output.mp4"),
)

# Two-stage (--hq) requires width and height divisible by 64 (1280x704 is 16:9 720p on the 64-pixel grid).
_HQ_DEFAULT_PARAMS: tuple[tuple[str, str], ...] = (
    ("--width", "1280"),
    ("--height", "704"),
    ("--num-frames", "97"),
    ("--num-inference-steps", "25"),
    ("--output-path", "output_hq.mp4"),
)

_HQ_SPATIAL_UPSAMPLER = _MODELS_DIR / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
_HQ_DISTILLED_LORA = _MODELS_DIR / "loras" / "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"

_ANIME_UPSCALERS: dict[str, Path] = {
    "animesharp": _UPSCALERS_DIR / "2x-AnimeSharpV4" / "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",
    "animesharp-hq": _UPSCALERS_DIR / "2x-AnimeSharpV4" / "2x-AnimeSharpV4_RCAN.safetensors",
    "realesrgan": _UPSCALERS_DIR / "RealESRGAN_x4plus_anime_6B.pth",
}


@dataclass
class _RocmCliConfig:
    cleaned_args: list[str]
    use_hq: bool = False
    anime_upscaler_choice: str | None = None
    upscale_res: tuple[int, int] | None = None
    batch_mode: bool = False
    batch_file: Path | None = None
    prompts_file: Path | None = None
    images_dir: Path | None = None
    output_dir: Path | None = None


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _get_flag_value(argv: list[str], flag: str, default: str | None = None) -> str | None:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return default


def _resolve_upscaler_path(choice: str | None) -> Path | None:
    if choice is None:
        return None
    if choice in _ANIME_UPSCALERS:
        path = _ANIME_UPSCALERS[choice]
    else:
        path = Path(choice).expanduser().resolve()
    if not path.exists():
        raise SystemExit(
            f"Anime upscaler not found: {path}. "
            f"Valid presets: {', '.join(_ANIME_UPSCALERS.keys())}"
        )
    return path


def _preprocess_rocm_args(argv: list[str]) -> _RocmCliConfig:  # noqa: PLR0912, PLR0915
    """Extract custom convenience flags (--hq, --animate, --batch, --anime-upscale, --upscale-res) and expand shortcuts."""
    cfg = _RocmCliConfig(cleaned_args=[])
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--hq":
            cfg.use_hq = True
            i += 1
            continue

        if arg == "--batch" or arg.startswith("--batch="):
            cfg.batch_mode = True
            if arg.startswith("--batch="):
                val = arg.split("=", 1)[1]
                p = Path(val).expanduser().resolve()
                if p.suffix.lower() in {".json", ".jsonl"}:
                    cfg.batch_file = p
                else:
                    cfg.prompts_file = p
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                p = Path(argv[i + 1]).expanduser().resolve()
                if p.suffix.lower() in {".json", ".jsonl"}:
                    cfg.batch_file = p
                else:
                    cfg.prompts_file = p
                i += 2
            else:
                i += 1
            continue

        if arg == "--prompts-file" or arg.startswith("--prompts-file="):
            cfg.batch_mode = True
            if arg.startswith("--prompts-file="):
                cfg.prompts_file = Path(arg.split("=", 1)[1]).expanduser().resolve()
                i += 1
            elif i + 1 < len(argv):
                cfg.prompts_file = Path(argv[i + 1]).expanduser().resolve()
                i += 2
            else:
                raise SystemExit("--prompts-file requires a path to a text file with one prompt per line.")
            continue

        if arg in ("--images-dir", "--animate-dir") or arg.startswith(("--images-dir=", "--animate-dir=")):
            cfg.batch_mode = True
            if "=" in arg:
                cfg.images_dir = Path(arg.split("=", 1)[1]).expanduser().resolve()
                i += 1
            elif i + 1 < len(argv):
                cfg.images_dir = Path(argv[i + 1]).expanduser().resolve()
                i += 2
            else:
                raise SystemExit(f"{arg} requires a directory path containing guide frames.")
            continue

        if arg == "--output-dir" or arg.startswith("--output-dir="):
            if arg.startswith("--output-dir="):
                cfg.output_dir = Path(arg.split("=", 1)[1]).expanduser().resolve()
                i += 1
            elif i + 1 < len(argv):
                cfg.output_dir = Path(argv[i + 1]).expanduser().resolve()
                i += 2
            else:
                raise SystemExit("--output-dir requires a destination directory path.")
            continue

        if arg == "--animate" or arg.startswith("--animate="):
            if arg.startswith("--animate="):
                img_path = arg.split("=", 1)[1]
                strength = "1.0"
                i += 1
            else:
                if i + 1 >= len(argv):
                    raise SystemExit("--animate requires an image path (and optional strength, default 1.0).")
                img_path = argv[i + 1]
                if i + 2 < len(argv) and not argv[i + 2].startswith("-"):
                    try:
                        float(argv[i + 2])
                        strength = argv[i + 2]
                        i += 3
                    except ValueError:
                        strength = "1.0"
                        i += 2
                else:
                    strength = "1.0"
                    i += 2
            p_cand = Path(img_path).expanduser().resolve()
            if p_cand.is_dir():
                cfg.batch_mode = True
                cfg.images_dir = p_cand
            else:
                cfg.cleaned_args.extend(["--image", img_path, "0", strength])
            continue

        if arg == "--image":
            pos_args: list[str] = []
            j = i + 1
            while j < len(argv) and not argv[j].startswith("-"):
                pos_args.append(argv[j])
                j += 1
            if len(pos_args) == 1:
                cfg.cleaned_args.extend(["--image", pos_args[0], "0", "1.0"])
                i = j
                continue
            if len(pos_args) == 2:
                if "." in pos_args[1]:
                    cfg.cleaned_args.extend(["--image", pos_args[0], "0", pos_args[1]])
                else:
                    cfg.cleaned_args.extend(["--image", pos_args[0], pos_args[1], "1.0"])
                i = j
                continue

        if arg == "--anime-upscale" or arg.startswith("--anime-upscale="):
            if arg.startswith("--anime-upscale="):
                cfg.anime_upscaler_choice = arg.split("=", 1)[1].lower()
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                cfg.anime_upscaler_choice = argv[i + 1].lower()
                i += 2
            else:
                cfg.anime_upscaler_choice = "animesharp"
                i += 1
            continue

        if arg == "--upscale-res" or arg.startswith("--upscale-res="):
            if arg.startswith("--upscale-res="):
                raw_res = arg.split("=", 1)[1].lower()
                i += 1
            elif i + 1 < len(argv):
                raw_res = argv[i + 1].lower()
                i += 2
            else:
                raise SystemExit("--upscale-res expects 1080p, 720p, or WIDTHxHEIGHT (e.g. 1920x1080).")
            if raw_res == "1080p":
                cfg.upscale_res = (1920, 1080)
            elif raw_res == "720p":
                cfg.upscale_res = (1280, 720)
            elif "x" in raw_res:
                w_str, h_str = raw_res.split("x", 1)
                cfg.upscale_res = (int(w_str), int(h_str))
            else:
                raise SystemExit(f"Invalid --upscale-res {raw_res!r}; use 1080p, 720p, or WIDTHxHEIGHT.")
            continue

        cfg.cleaned_args.append(arg)
        i += 1

    return cfg


def _run_batch_mode(cfg: _RocmCliConfig) -> None:
    """Parse pipeline args and execute phase-by-phase (model-by-model) multi-scene batch generation."""
    logging.basicConfig(level=logging.INFO)
    from ltx_core.components.guiders import MultiModalGuiderParams  # noqa: PLC0415
    from ltx_pipelines.batch_runner import load_batch_specs, run_phased_batch  # noqa: PLC0415
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline  # noqa: PLC0415
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline  # noqa: PLC0415
    from ltx_pipelines.utils.args import (  # noqa: PLC0415
        add_chunk_layout_args,
        add_generated_keyframes_arg,
        add_keyframe_decode_arg,
        chunk_config_from_args,
        default_1_stage_arg_parser,
        default_2_stage_arg_parser,
        resolve_cli_params,
    )

    if not _has_flag(sys.argv[1:], "--prompt"):
        sys.argv.extend(["--prompt", "__batch__"])

    params = resolve_cli_params()
    base_parser = (
        default_2_stage_arg_parser(params=params, supports_auto_duration=True)
        if cfg.use_hq
        else default_1_stage_arg_parser(params=params, supports_auto_duration=True)
    )
    parser = add_chunk_layout_args(add_keyframe_decode_arg(add_generated_keyframes_arg(base_parser)))
    parser.add_argument("--refine-audio", action="store_true")
    args = parser.parse_args()

    prompts_file = cfg.prompts_file
    images_dir = cfg.images_dir
    if cfg.batch_file is None and prompts_file is None:
        default_prompts = Path.cwd() / "prompts.txt"
        if default_prompts.is_file():
            prompts_file = default_prompts
        else:
            raise SystemExit(
                "--batch requires a prompts file (e.g. --prompts-file prompts.txt or --batch prompts.txt) "
                "or a JSON/JSONL manifest (--batch scenes.json)."
            )
    if images_dir is None and (Path.cwd() / "guias_16_9").is_dir():
        images_dir = Path.cwd() / "guias_16_9"

    output_dir = cfg.output_dir or (Path.cwd() / ("videos_hq" if cfg.use_hq else "videos_batch"))
    scenes = load_batch_specs(
        batch_file=cfg.batch_file,
        prompts_file=prompts_file,
        images_dir=images_dir,
        output_dir=output_dir,
        default_negative_prompt=args.negative_prompt,
    )

    if cfg.use_hq:
        pipeline: TI2VidTwoStagesPipeline | TI2VidOneStagePipeline = TI2VidTwoStagesPipeline(
            model_paths=args.model_paths,
            distilled_lora=args.distilled_lora,
            spatial_upsampler_path=args.spatial_upsampler_path,
            loras=tuple(args.lora) if args.lora else (),
            quantization=args.quantization,
            compilation_config=args.compile,
            offload_mode=args.offload_mode,
            prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
            diffvae_optimization=args.diffvae_optimization,
        )
    else:
        pipeline = TI2VidOneStagePipeline(
            model_paths=args.model_paths,
            loras=tuple(args.lora) if args.lora else (),
            quantization=args.quantization,
            compilation_config=args.compile,
            offload_mode=args.offload_mode,
            prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
            diffvae_optimization=args.diffvae_optimization,
        )

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=args.video_cfg_guidance_scale,
        stg_scale=args.video_stg_guidance_scale,
        rescale_scale=args.video_rescale_scale,
        modality_scale=args.a2v_guidance_scale,
        skip_step=args.video_skip_step,
        stg_blocks=args.video_stg_blocks,
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=args.audio_cfg_guidance_scale,
        stg_scale=args.audio_stg_guidance_scale,
        rescale_scale=args.audio_rescale_scale,
        modality_scale=args.v2a_guidance_scale,
        skip_step=args.audio_skip_step,
        stg_blocks=args.audio_stg_blocks,
    )
    raw_frames = args.num_frames if isinstance(args.num_frames, int) else (97 if cfg.use_hq else 25)
    run_phased_batch(
        pipeline=pipeline,
        scenes=scenes,
        height=args.height,
        width=args.width,
        num_frames=raw_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        chunk_config=chunk_config_from_args(args),
        max_batch_size=args.max_batch_size,
        refine_audio=getattr(args, "refine_audio", False),
        anime_upscaler_path=_resolve_upscaler_path(cfg.anime_upscaler_choice),
        upscale_res=cfg.upscale_res,
    )


def main() -> None:
    raw_args = sys.argv[1:]
    if "-h" in raw_args or "--help" in raw_args:
        print(
            "LTX-2.5 ROCm Shortcut Flags (ltx-run):\n"
            "  --hq                              Run 2-stage HQ pipeline (Stage 1 + 2x Latent Upsampler + Stage 2 Distilled LoRA)\n"
            "                                    Defaults: 1280x704 (720p), 97 frames (~4s), 25 steps, auto-chunking for >121 frames.\n"
            "  --animate PATH [STRENGTH]         Animate an initial frame (or pass a directory of frames to enable --batch).\n"
            "  --batch [PROMPTS_OR_JSON]         Phase-by-phase multi-scene batch mode: loads each model ONCE across all scenes\n"
            "                                    (Gemma -> VideoEncoder -> Stage 1 -> Upsampler -> Stage 2 -> VideoVAE -> AudioVAE -> AnimeUpscaler).\n"
            "  --prompts-file PATH               Text file with one prompt per line for --batch (default: prompts.txt if present).\n"
            "  --images-dir PATH                 Directory of initial frames paired in sorted order with --prompts-file (default: guias_16_9 if present).\n"
            "  --output-dir PATH                 Output directory for --batch videos (default: videos_hq/ with --hq, or videos_batch/).\n"
            "  --anime-upscale [MODEL]           Post-upscale output MP4(s) with anime upscaler on RX 9070 XT (cuda:0).\n"
            "                                    MODEL choices: animesharp (default, 2x-AnimeSharpV4 Fast), animesharp-hq, realesrgan.\n"
            "  --upscale-res {1080p,720p,WxH}    Target resolution for --anime-upscale (default: 1080p when --hq is used).\n"
            "--------------------------------------------------------------------------------"
        )

    cfg = _preprocess_rocm_args(raw_args)
    user_args = cfg.cleaned_args
    injected: list[str] = []

    for flag, path in _DEFAULT_FLAGS:
        if not _has_flag(user_args, flag) and path.exists():
            injected.extend([flag, str(path)])

    default_params = _HQ_DEFAULT_PARAMS if cfg.use_hq else _FAST_DEFAULT_PARAMS
    for flag, default_val in default_params:
        if not _has_flag(user_args, flag):
            injected.extend([flag, default_val])

    if cfg.use_hq:
        if not _has_flag(user_args, "--spatial-upsampler-path"):
            injected.extend(["--spatial-upsampler-path", str(_HQ_SPATIAL_UPSAMPLER)])
        if not _has_flag(user_args, "--distilled-lora"):
            injected.extend(["--distilled-lora", str(_HQ_DISTILLED_LORA), "0.8"])

    # Auto-enable sliding-window chunking for long clips (>121 frames, e.g. 10s=241 or 30s=721)
    # so VRAM usage stays constant on the two 16GB RDNA4 GPUs while 5s clips (121 frames) run in a single pass.
    combined_preview = [*injected, *user_args]
    num_frames_str = _get_flag_value(combined_preview, "--num-frames")
    if (
        num_frames_str is not None
        and num_frames_str.isdigit()
        and int(num_frames_str) > 121
        and not _has_flag(user_args, "--chunked")
        and not _has_flag(user_args, "--chunk-pixel-frames")
    ):
        injected.extend(["--chunk-pixel-frames", "97", "--chunk-carry-frames", "25"])

    sys.argv = [sys.argv[0], *injected, *user_args]

    if cfg.batch_mode and "-h" not in raw_args and "--help" not in raw_args:
        _run_batch_mode(cfg)
        return

    if cfg.use_hq:
        from ltx_pipelines.ti2vid_two_stages import main as run_pipeline  # noqa: PLC0415
    else:
        from ltx_pipelines.ti2vid_one_stage import main as run_pipeline  # noqa: PLC0415

    run_pipeline()

    upscaler_path = _resolve_upscaler_path(cfg.anime_upscaler_choice)
    if upscaler_path is not None and "-h" not in raw_args and "--help" not in raw_args:
        output_path = _get_flag_value(sys.argv[1:], "--output-path", "output.mp4")
        assert output_path is not None
        target_w: int | None = None
        target_h: int | None = None
        if cfg.upscale_res is not None:
            target_w, target_h = cfg.upscale_res
        elif cfg.use_hq:
            target_w, target_h = (1920, 1080)

        from ltx_pipelines.anime_upscaler import upscale_video_anime  # noqa: PLC0415

        logging.getLogger(__name__).info(
            "Running post-generation anime upscaler (%s) on RX 9070 XT...", upscaler_path.name
        )
        upscale_video_anime(
            input_video_path=output_path,
            output_video_path=output_path,
            model_path=upscaler_path,
            target_width=target_w,
            target_height=target_h,
        )


if __name__ == "__main__":
    main()
