from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
import transformers.models
from transformers.utils.versions import require_version

from ...extras.constants import SUPPORTED_CLASS_FOR_BLOCK_DIAG_ATTN
from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = get_logger(__name__)


def get_seqlens_in_batch(attention_mask: "torch.Tensor") -> "torch.Tensor":
    r"""
    Gets the sequnce lengths in the current batch.

    e.g.
    ```python
    # input
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    # output
    [2, 3, 1, 2, 3]
    ```
    """
    bsz = attention_mask.size(0)
    dtype, device = attention_mask.dtype, attention_mask.device
    max_num = torch.max(attention_mask).item()
    counts: "torch.Tensor" = torch.zeros((bsz, max_num), dtype=dtype, device=device)
    for i in range(max_num):
        counts[:, i] = torch.sum(attention_mask == (i + 1), dim=-1)

    counts = counts.flatten()
    seqlens = counts[counts.nonzero().squeeze(dim=-1)]
    return seqlens


def get_unpad_data(attention_mask: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", int]:
    r"""
    Prepares the indices and seqlens for flash attn varlen function.

    Returns:
        indices: indices of non-masked tokens from the flattened sequence.
        cu_seqlens: the cumulative sequence lengths in the current batch, always starts from 0.
        max_seqlen_in_batch: the largest seqlen in the current batch.

    e.g.
    ```python
    # input
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    # output
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    [0, 2, 5, 6, 8, 11]
    3
    ```
    """
    seqlens_in_batch = get_seqlens_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def _patch_for_block_diag_attn(model_type: str) -> None:
    require_version("transformers>=4.41.2,<=4.42.4", "To fix: pip install transformers>=4.41.2,<=4.42.4")
    if model_type == "cohere":
        transformers.models.cohere.modeling_cohere._get_unpad_data = get_unpad_data
    elif model_type == "falcon":
        transformers.models.falcon.modeling_falcon._get_unpad_data = get_unpad_data
    elif model_type == "gemma":
        transformers.models.gemma.modeling_gemma._get_unpad_data = get_unpad_data
    elif model_type == "gemma2":
        transformers.models.gemma2.modeling_gemma2._get_unpad_data = get_unpad_data
    elif model_type == "llama":
        transformers.models.llama.modeling_llama._get_unpad_data = get_unpad_data
    elif model_type == "mistral":
        transformers.models.mistral.modeling_mistral._get_unpad_data = get_unpad_data
    elif model_type == "phi":
        transformers.models.phi.modeling_phi._get_unpad_data = get_unpad_data
    elif model_type == "phi3":
        transformers.models.phi3.modeling_phi3._get_unpad_data = get_unpad_data
    elif model_type == "qwen2":
        transformers.models.qwen2.modeling_qwen2._get_unpad_data = get_unpad_data
    elif model_type == "starcoder2":
        transformers.models.starcoder2.modeling_starcoder2._get_unpad_data = get_unpad_data


def configure_packing(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable or not model_args.block_diag_attn:
        return

    model_type = getattr(config, "model_type", None)
    if model_type in SUPPORTED_CLASS_FOR_BLOCK_DIAG_ATTN:
        _patch_for_block_diag_attn(model_type)
        logger.info("Using block diagonal attention for sequence packing without cross-attention.")
    else:
        raise ValueError("Current model does not support block diagonal attention.")
