"""INT8 tensor-wise / per-channel ConvRot quantization policy for LTX models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from hashlib import sha256
import math
from pathlib import Path

import torch
from torch import nn

from ltx_core.loader.checkpoint import CheckpointReader
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.units import FuseRule
from ltx_core.quantization.policy import QuantizationPolicy

try:
    import comfy_kitchen as _ck

    if torch.version.cuda is None:
        _ck.registry.disable("cuda")
    _CK_INT8_AVAILABLE = hasattr(torch.ops, "comfy_kitchen") and hasattr(torch.ops.comfy_kitchen, "int8_linear")
except ImportError:
    _CK_INT8_AVAILABLE = False

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _build_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
        device=device,
    )
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h_norm = (h / (size**0.5)).to(dtype=dtype)
    _HADAMARD_CACHE[key] = h_norm
    return h_norm


def _rotate_activation(x: torch.Tensor, group_size: int) -> torch.Tensor:
    orig_shape = x.shape
    features = orig_shape[-1]
    h = _build_hadamard(group_size, x.device, x.dtype)
    x_grouped = x.reshape(-1, features // group_size, group_size)
    return torch.matmul(x_grouped, h).reshape(orig_shape)


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    raise ValueError(f"Unsupported INT8 output dtype: {dtype}")


class Int8ConvRotLinear(nn.Module):
    """Linear layer storing pre-rotated INT8 weights and per-channel FP32 scales."""

    in_features: int
    out_features: int
    convrot: bool
    convrot_groupsize: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        scale_shape: tuple[int, ...] | None = None,
        convrot: bool = True,
        convrot_groupsize: int = 256,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convrot = convrot
        self.convrot_groupsize = convrot_groupsize
        resolved_scale_shape = (out_features, 1) if scale_shape is None else scale_shape

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.int8, device=device),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(resolved_scale_shape, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _CK_INT8_AVAILABLE and x.is_cuda:
            bias = self.bias.to(dtype=x.dtype) if self.bias is not None and self.bias.dtype != x.dtype else self.bias
            with torch.cuda.device(x.device):
                return torch.ops.comfy_kitchen.int8_linear(
                    x.contiguous(),
                    self.weight,
                    self.weight_scale,
                    bias,
                    _dtype_code(x.dtype),
                    self.convrot,
                    self.convrot_groupsize,
                )

        x_rot = _rotate_activation(x, self.convrot_groupsize) if self.convrot else x
        w_dequant = self.weight.to(dtype=x.dtype) * self.weight_scale.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x_rot, w_dequant, bias)


def _swap_linears_to_int8_convrot(
    model: nn.Module,
    scale_shapes: Mapping[str, tuple[int, ...]],
) -> nn.Module:
    replacements: list[tuple[nn.Module, str, nn.Linear, tuple[int, ...]]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, Int8ConvRotLinear):
            continue
        matched_shape = scale_shapes.get(name)
        if matched_shape is None:
            for key, shape in scale_shapes.items():
                if key.endswith(f".{name}"):
                    matched_shape = shape
                    break
        if matched_shape is None:
            continue
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name
        replacements.append((parent, attr_name, module, matched_shape))

    for parent, attr_name, linear, scale_shape in replacements:
        setattr(
            parent,
            attr_name,
            Int8ConvRotLinear(
                in_features=linear.in_features,
                out_features=linear.out_features,
                bias=linear.bias is not None,
                device=linear.weight.device,
                scale_shape=scale_shape,
            ),
        )
    return model


def selected_int8_layers(paths: Sequence[Path], sd_ops: Sequence[SDOps] = ()) -> dict[str, tuple[int, ...]]:
    """Discover INT8 layers and their weight_scale shapes from safetensors headers."""
    with CheckpointReader.open(tuple(Path(p) for p in paths)) as reader:
        _, metadata = reader.scan()
    mapped: dict[str, tuple[torch.Size, torch.dtype]] = {}
    for raw_key, meta in metadata.items():
        key: str | None = raw_key
        for op in sd_ops:
            if key is None:
                break
            key = op.apply_to_key(key)
        if key is not None:
            mapped[key] = meta

    selected: dict[str, tuple[int, ...]] = {}
    for key, (shape, _dtype) in mapped.items():
        if not key.endswith(".weight_scale"):
            continue
        prefix = key.removesuffix(".weight_scale")
        weight_meta = mapped.get(f"{prefix}.weight")
        if weight_meta is not None and weight_meta[1] == torch.int8:
            selected[prefix] = tuple(shape)
    return selected


def has_int8_convrot_layers(paths: Sequence[str | Path], sd_ops: Sequence[SDOps] = ()) -> bool:
    """Return True if the checkpoint contains INT8 weight + weight_scale pairs."""
    return bool(selected_int8_layers(tuple(Path(p) for p in paths), sd_ops=sd_ops))


def swap_int8_convrot_module_op(paths: Sequence[str | Path], sd_ops: Sequence[SDOps] = ()) -> ModuleOps:
    """Create a ModuleOps that replaces INT8 Linear modules with Int8ConvRotLinear."""
    resolved_paths = tuple(Path(p) for p in paths)
    selected = selected_int8_layers(resolved_paths, sd_ops=sd_ops)
    if not selected:
        raise ValueError("INT8 ConvRot quantization requires at least one INT8 weight with a weight_scale companion")
    fingerprint = sha256(
        "\0".join(f"{k}:{v}" for k, v in sorted(selected.items())).encode()
    ).hexdigest()[:12]
    return ModuleOps(
        f"int8-convrot-swap:{fingerprint}",
        matcher=lambda model: isinstance(model, nn.Module),
        mutator=lambda model: _swap_linears_to_int8_convrot(model, selected),
    )


def _int8_convrot_unit_patch(
    key: str, weight: torch.Tensor, delta: torch.Tensor, clean: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    if weight.dtype != torch.int8:
        if not weight.is_floating_point():
            raise ValueError(f"{key!r} is not a floating fallback weight")
        return {key: (weight.to(delta.dtype) + delta).to(weight.dtype)}
    with torch.cuda.device(weight.device) if weight.is_cuda else nullcontext():
        scale_key = f"{key.removesuffix('.weight')}.weight_scale"
        scale = clean[scale_key].to(device=weight.device, dtype=torch.float32)
        h = _build_hadamard(256, weight.device, torch.float32)
        out_f, in_f = weight.shape
        delta_rot = torch.matmul(delta.float().reshape(out_f, in_f // 256, 256), h.T).reshape(out_f, in_f)
        merged = weight.float() * scale + delta_rot
        new_scale = (merged.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
        new_weight = (merged / new_scale).round().clamp(-128.0, 127.0).to(torch.int8)
        return {key: new_weight, scale_key: new_scale.reshape(scale.shape)}


INT8_CONVROT_UNIT_FUSE_RULE = FuseRule(
    "int8-convrot",
    torch.bfloat16,
    _int8_convrot_unit_patch,
    (".weight_scale",),
)


def build_policy(checkpoint_path: str | Path) -> QuantizationPolicy:
    """Build QuantizationPolicy for pre-quantized INT8 ConvRot checkpoints."""
    return QuantizationPolicy(
        sd_ops=(),
        module_ops=(swap_int8_convrot_module_op((Path(checkpoint_path),)),),
        fuse_rule=INT8_CONVROT_UNIT_FUSE_RULE,
    )
