from dataclasses import dataclass
import torch


@dataclass
class DetrOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor
