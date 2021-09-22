import torch

from torch import Tensor, nn
from typing import List, Tuple


class Text2Characters(nn.Module):

    def __init__(self, symbols: str) -> None:
        super().__init__()
        self._symbol_to_id = {s: i for i, s in enumerate(symbols)}

    def _text_to_sequence(self, text: str) -> List[int]:
        return [self._symbol_to_id[s] for s in text.lower() if s in self._symbol_to_id]

    def forward(self, text: str) -> Tuple[Tensor, Tensor]:
        input = self._text_to_sequence(text)
        sequences, lengths = torch.tensor([input], dtype=torch.long), torch.tensor([len(input)], dtype=torch.long)
        return sequences, lengths
