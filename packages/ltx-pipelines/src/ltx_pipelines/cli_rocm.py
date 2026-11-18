"""Simplified CLI entrypoint for LTX-2.5 on ROCm with pre-configured model paths, --hq, --animate, and anime upscalers."""

from __future__ import annotations

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


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _get_flag_value(argv: list[str], flag: str, default: str | None = None) -> str | None:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return default


def _preprocess_rocm_args(argv: list[str]) -> tuple[list[str], bool, str | None, tuple[int, int] | None]:
    """Extract custom convenience flags (--hq, --animate, --anime-upscale, --upscale-res) and expand shortcuts."""
    cleaned: list[str] = []
    use_hq = False
    anime_upscaler_choice: str | None = None
    upscale_res: tuple[int, int] | None = None

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--hq":
            use_hq = True
            i += 1
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
            cleaned.extend(["--image", img_path, "0", strength])
            continue

        if arg == "--image":
            # Collect positional arguments following --image
            pos_args: list[str] = []
            j = i + 1
            while j < len(argv) and not argv[j].startswith("-"):
                pos_args.append(argv[j])
                j += 1
            if len(pos_args) == 1:
                cleaned.extend(["--image", pos_args[0], "0", "1.0"])
                i = j
                continue
            if len(pos_args) == 2:
                # Support --image <path> <strength> (float) or --image <path> <frame_idx> (int)
                if "." in pos_args[1]:
                    cleaned.extend(["--image", pos_args[0], "0", pos_args[1]])
                else:
                    cleaned.extend(["--image", pos_args[0], pos_args[1], "1.0"])
                i = j
                continue

        if arg == "--anime-upscale" or arg.startswith("--anime-upscale="):
            if arg.startswith("--anime-upscale="):
                anime_upscaler_choice = arg.split("=", 1)[1].lower()
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                anime_upscaler_choice = argv[i + 1].lower()
                i += 2
            else:
                anime_upscaler_choice = "animesharp"
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
                upscale_res = (1920, 1080)
            elif raw_res == "720p":
                upscale_res = (1280, 720)
            elif "x" in raw_res:
                w_str, h_str = raw_res.split("x", 1)
                upscale_res = (int(w_str), int(h_str))
            else:
                raise SystemExit(f"Invalid --upscale-res {raw_res!r}; use 1080p, 720p, or WIDTHxHEIGHT.")
            continue

        cleaned.append(arg)
        i += 1

    return cleaned, use_hq, anime_upscaler_choice, upscale_res


def main() -> None:
    raw_args = sys.argv[1:]
    if "-h" in raw_args or "--help" in raw_args:
        print(
            "LTX-2.5 ROCm Shortcut Flags (ltx-run):\n"
            "  --hq                              Run 2-stage HQ pipeline (Stage 1 + 2x Latent Upsampler + Stage 2 Distilled LoRA)\n"
            "                                    Defaults: 1280x704 (720p), 97 frames (~4s), 25 steps, auto-chunking for >97 frames.\n"
            "  --animate PATH [STRENGTH]         Animate an initial frame (shortcut for --image PATH 0 STRENGTH, default 1.0).\n"
            "  --anime-upscale [MODEL]           Post-upscale output MP4 with anime upscaler on RX 9070 XT (cuda:0).\n"
            "                                    MODEL choices: animesharp (default, 2x-AnimeSharpV4 Fast), animesharp-hq, realesrgan.\n"
            "  --upscale-res {1080p,720p,WxH}    Target resolution for --anime-upscale (default: 1080p when --hq is used).\n"
            "--------------------------------------------------------------------------------"
        )

    user_args, use_hq, anime_upscaler_choice, upscale_res = _preprocess_rocm_args(raw_args)
    injected: list[str] = []

    for flag, path in _DEFAULT_FLAGS:
        if not _has_flag(user_args, flag) and path.exists():
            injected.extend([flag, str(path)])

    default_params = _HQ_DEFAULT_PARAMS if use_hq else _FAST_DEFAULT_PARAMS
    for flag, default_val in default_params:
        if not _has_flag(user_args, flag):
            injected.extend([flag, default_val])

    if use_hq:
        if not _has_flag(user_args, "--spatial-upsampler-path"):
            injected.extend(["--spatial-upsampler-path", str(_HQ_SPATIAL_UPSAMPLER)])
        if not _has_flag(user_args, "--distilled-lora"):
            injected.extend(["--distilled-lora", str(_HQ_DISTILLED_LORA), "0.8"])

    # Auto-enable sliding-window chunking for long clips (>97 frames, e.g. 10s=241 or 30s=721)
    # so VRAM usage stays constant on the two 16GB RDNA4 GPUs.
    combined_preview = [*injected, *user_args]
    num_frames_str = _get_flag_value(combined_preview, "--num-frames")
    if (
        num_frames_str is not None
        and num_frames_str.isdigit()
        and int(num_frames_str) > 97
        and not _has_flag(user_args, "--chunked")
        and not _has_flag(user_args, "--chunk-pixel-frames")
    ):
        injected.extend(["--chunk-pixel-frames", "97", "--chunk-carry-frames", "25"])

    sys.argv = [sys.argv[0], *injected, *user_args]

    if use_hq:
        from ltx_pipelines.ti2vid_two_stages import main as run_pipeline  # noqa: PLC0415
    else:
        from ltx_pipelines.ti2vid_one_stage import main as run_pipeline  # noqa: PLC0415

    run_pipeline()

    if anime_upscaler_choice is not None and "-h" not in raw_args and "--help" not in raw_args:
        if anime_upscaler_choice in _ANIME_UPSCALERS:
            upscaler_path = _ANIME_UPSCALERS[anime_upscaler_choice]
        else:
            upscaler_path = Path(anime_upscaler_choice).expanduser().resolve()
        if not upscaler_path.exists():
            raise SystemExit(
                f"Anime upscaler not found: {upscaler_path}. "
                f"Valid presets: {', '.join(_ANIME_UPSCALERS.keys())}"
            )

        output_path = _get_flag_value(sys.argv[1:], "--output-path", "output.mp4")
        assert output_path is not None
        target_w: int | None = None
        target_h: int | None = None
        if upscale_res is not None:
            target_w, target_h = upscale_res
        elif use_hq:
            # Default HQ (1280x704) + --anime-upscale targets clean 1920x1080 (1080p YouTube)
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
