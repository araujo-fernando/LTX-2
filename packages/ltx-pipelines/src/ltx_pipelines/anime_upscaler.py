"""Post-generation anime video upscaler (RealESRGAN_x4plus_anime_6B / 2x-AnimeSharpV4) on RX 9070 XT."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import tempfile

import av
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from ltx_core.devices import cleanup_accelerator_memory, get_preferred_device
from ltx_core.types import Audio
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video

logger = logging.getLogger(__name__)


class _ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNetAnime6B(nn.Module):
    """Native PyTorch implementation of RealESRGAN_x4plus_anime_6B (6 RRDB blocks, 4x scale)."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 6,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_anime_upscaler(model_path: str | Path, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """Load an anime upscaler checkpoint (RealESRGAN_x4plus_anime_6B.pth or 2x-AnimeSharpV4 via spandrel)."""
    path = Path(model_path)
    try:
        import spandrel  # noqa: PLC0415

        descriptor = spandrel.ModelLoader(device=device).load_from_file(str(path))
        return descriptor.model.eval().to(device=device, dtype=dtype)
    except ImportError:
        pass

    if path.suffix == ".safetensors":
        raise RuntimeError(f"Loading {path.name} requires spandrel; use RealESRGAN_x4plus_anime_6B.pth or install spandrel.")

    raw = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = raw.get("params_ema") or raw.get("params") or raw
    model = RRDBNetAnime6B()
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device=device, dtype=dtype)


@torch.inference_mode()
def upscale_video_anime(
    input_video_path: str | Path,
    output_video_path: str | Path,
    model_path: str | Path,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    batch_size: int = 4,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Upscale an MP4 video on the primary GPU (RX 9070 XT) and preserve its audio stream."""
    in_path = Path(input_video_path)
    out_path = Path(output_video_path)
    target_dev = device or get_preferred_device()

    cleanup_accelerator_memory()
    logger.info("Loading anime upscaler %s on %s", Path(model_path).name, target_dev)
    upscaler = load_anime_upscaler(model_path, device=target_dev, dtype=dtype)

    in_place = in_path.resolve() == out_path.resolve()
    work_out = Path(tempfile.mktemp(suffix=".mp4", dir=str(out_path.parent))) if in_place else out_path

    try:
        with av.open(str(in_path)) as in_container:
            v_in = in_container.streams.video[0]
            fps = float(v_in.average_rate or v_in.base_rate or 24.0)
            frames_rgb: list[np.ndarray] = [frame.to_ndarray(format="rgb24") for frame in in_container.decode(v_in)]

        if not frames_rgb:
            raise RuntimeError(f"No video frames decoded from {in_path}")

        audio = decode_audio_from_file(str(in_path), device=torch.device("cpu"))
        if audio is not None and audio.waveform.ndim == 3:
            audio = Audio(waveform=audio.waveform.squeeze(0), sampling_rate=audio.sampling_rate)

        first_tensor = (
            torch.from_numpy(frames_rgb[0]).to(device=target_dev, dtype=dtype).permute(2, 0, 1).unsqueeze(0) / 255.0
        )
        first_up = upscaler(first_tensor)
        _, _, up_h, up_w = first_up.shape
        final_w = target_width or up_w
        final_h = target_height or up_h
        del first_tensor, first_up

        logger.info(
            "Upscaling %d frames (%dx%d -> %dx%d) on %s",
            len(frames_rgb),
            frames_rgb[0].shape[1],
            frames_rgb[0].shape[0],
            final_w,
            final_h,
            target_dev,
        )

        upscaled_batches: list[torch.Tensor] = []
        for start in tqdm(range(0, len(frames_rgb), batch_size), desc="Anime Upscale"):
            batch_np = np.stack(frames_rgb[start : start + batch_size], axis=0)
            batch_t = (
                torch.from_numpy(batch_np)
                .to(device=target_dev, dtype=dtype)
                .permute(0, 3, 1, 2)
                .contiguous()
                / 255.0
            )
            up_t = upscaler(batch_t)
            if up_t.shape[-2] != final_h or up_t.shape[-1] != final_w:
                up_t = F.interpolate(up_t.float(), size=(final_h, final_w), mode="bicubic", align_corners=False)
            upscaled_batches.append(up_t.clamp(0.0, 1.0).permute(0, 2, 3, 1).to(dtype=torch.float32, device="cpu"))
            del batch_t, up_t

        full_video = torch.cat(upscaled_batches, dim=0)
        del upscaled_batches

        encode_video(
            video=full_video,
            fps=fps,
            audio=audio,
            output_path=work_out,
            video_chunks_number=1,
            crf=16,
            preset="fast",
        )

        if in_place:
            shutil.move(str(work_out), str(out_path))
        logger.info("Anime upscaled video saved to %s (%dx%d)", out_path, final_w, final_h)
    finally:
        del upscaler
        cleanup_accelerator_memory()
