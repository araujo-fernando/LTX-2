from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import logging
from pathlib import Path

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.loader.registry import ModelRegistry
from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    VocoderConfigurator,
)
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.types import Audio
from ltx_pipelines.utils.gpu_model import gpu_model

logger = logging.getLogger(__name__)

__all__ = ["AudioDecoder"]


class AudioDecoder:
    """Owns audio decoder + vocoder lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: ModelRegistry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        unit_registry = registry or ModelRegistry()
        self._decoder_builder = SimpleBuilder(
            AudioDecoderConfigurator,
            (Path(checkpoint_path),),
            (AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,),
            registry=unit_registry,
        )
        self._vocoder_builder = SimpleBuilder(
            VocoderConfigurator,
            (Path(checkpoint_path),),
            (VOCODER_COMFY_KEYS_FILTER,),
            registry=unit_registry,
        )
        self._alloc_trim_strategy = alloc_trim_strategy
        self._resident_pair: tuple[torch.nn.Module, torch.nn.Module] | None = None

    @contextmanager
    def resident(self) -> Iterator["AudioDecoder"]:
        """Keep audio decoder + vocoder resident across multiple calls."""
        if self._resident_pair is not None:
            yield self
            return
        logger.info("Building resident audio decoder + vocoder from %s", self._checkpoint_path)
        vocoder_dtype = torch.float32 if self._device.type == "mps" else self._dtype
        with (
            gpu_model(
                self._decoder_builder.build(device=self._device, dtype=self._dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as decoder,
            gpu_model(
                self._vocoder_builder.build(device=self._device, dtype=vocoder_dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as vocoder,
        ):
            self._resident_pair = (decoder, vocoder)
            try:
                yield self
            finally:
                self._resident_pair = None

    def __call__(self, latent: torch.Tensor) -> Audio:
        """Decode audio *latent* through VAE decoder + vocoder, then free both."""
        if self._resident_pair is not None:
            decoder, vocoder = self._resident_pair
            return vae_decode_audio(latent, decoder, vocoder)
        with self.resident():
            decoder, vocoder = self._resident_pair  # type: ignore[misc]
            return vae_decode_audio(latent, decoder, vocoder)
