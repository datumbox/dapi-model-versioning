from transformers import RobertaTokenizer, RobertaModel

from functools import partial
from torch import nn

from ._api import register, Weights, WeightEntry


class RobertaWeights(Weights):
    BASE = WeightEntry(
        'roberta-base',
        lambda: partial(RobertaTokenizer.from_pretrained('roberta-base'), return_tensors='pt'),
        {'params': 125_000_000},
        True
    )
    LARGE = WeightEntry(
        'roberta-large',
        lambda: partial(RobertaTokenizer.from_pretrained('roberta-large'), return_tensors='pt'),
        {'params': 355_000_000},
        True
    )


@register
def roberta(weights: RobertaWeights) -> nn.Module:
    # Confirm we got the right weights
    RobertaWeights.check_type(weights)

    model = RobertaModel.from_pretrained(weights.url)
    return model
