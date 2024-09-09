from multiprocessing.sharedctypes import Value
from typing import Optional, List, Dict, Any, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from overrides import overrides
import numpy as np
import copy
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, masked_softmax

from allennlp.modules.attention.attention import Attention


def pooling_with_mask(input: torch.Tensor, mask: torch.BoolTensor):
    shape = input.shape
    # if mask.dtype != torch.bool:
    #     raise ValueError(f"mask must be torch.BoolTensor")
    if len(shape) != 3:
        raise ValueError(f"input dimension expected 3, got {len(shape)}")
    if mask.shape != shape[0:2]:
        raise ValueError(f"incompatible sizes of input {shape} and mask {mask.shape}")

    valid_input_mask = mask.unsqueeze(-1).expand(-1, -1, shape[-1]).float()

    input_sum = torch.sum(input * valid_input_mask, dim=1)

    sum_mask = valid_input_mask.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)

    return input_sum / sum_mask


