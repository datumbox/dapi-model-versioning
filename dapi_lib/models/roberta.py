from transformers import RobertaTokenizer, RobertaModel

from functools import partial
from torch import nn

from ._api import register, Weights


class RobertaWeights(Weights):
    BASE = (
        'roberta-base',
        lambda: partial(RobertaTokenizer.from_pretrained('roberta-base'), return_tensors='pt'),
        {'params': 125_000_000},
        True
    )
    LARGE = (
        'roberta-large',
        lambda: partial(RobertaTokenizer.from_pretrained('roberta-large'), return_tensors='pt'),
        {'params': 355_000_000},
        True
    )


@register
def roberta(weights: RobertaWeights) -> nn.Module:
    model = RobertaModel.from_pretrained(weights.url)
    return model
