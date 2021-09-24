import os
import warnings

from functools import partial
from typing import Any, Optional

from ._api import register, Weights
from ..transforms.audio_presets import Text2Characters

# Import a few stuff that we plan to keep as-is to avoid copy-pasting
from torchaudio.models.tacotron2 import Tacotron2


__all__ = ["Tacotron2"]


_DEFAULT_PARAMETERS = {
    'mask_padding': False,
    'n_mels': 80,
    'n_frames_per_step': 1,
    'symbol_embedding_dim': 512,
    'encoder_embedding_dim': 512,
    'encoder_n_convolution': 3,
    'encoder_kernel_size': 5,
    'decoder_rnn_dim': 1024,
    'decoder_max_step': 2000,
    'decoder_dropout': 0.1,
    'decoder_early_stopping': True,
    'attention_rnn_dim': 1024,
    'attention_hidden_dim': 128,
    'attention_location_n_filter': 32,
    'attention_location_kernel_size': 31,
    'attention_dropout': 0.1,
    'prenet_dim': 256,
    'postnet_n_convolution': 5,
    'postnet_kernel_size': 5,
    'postnet_embedding_dim': 512,
    'gate_threshold': 0.5,
}


class Tacotron2Weights(Weights):
    Characters_LJSpeech = (
        'https://download.pytorch.org/models/audio/tacotron2_english_characters_1500_epochs_ljspeech.pth',
        partial(Text2Characters, symbols="_-!'(),.:;? abcdefghijklmnopqrstuvwxyz"),
        {'lang': 'en', 'epochs': 1500, 'n_symbol': 38},
        True
    )
    Characters_WaveRNN_LJSpeech = (
        'https://download.pytorch.org/models/audio/tacotron2_english_characters_1500_epochs_wavernn_ljspeech.pth',
        partial(Text2Characters, symbols="_-!'(),.:;? abcdefghijklmnopqrstuvwxyz"),
        {'lang': 'en', 'epochs': 1500, 'n_symbol': 38},
        True
    )
    Phonemes_LJSpeech = (
        'https://download.pytorch.org/models/audio/tacotron2_english_phonemes_1500_epochs_ljspeech.pth',
        None,  # Phonemes preprocessing goes here
        {'lang': 'en', 'epochs': 1500, 'n_symbol': 96},
        True
    )
    Phonemes_WaveRNN_LJSpeech = (
        'https://download.pytorch.org/models/audio/tacotron2_english_phonemes_1500_epochs_wavernn_ljspeech.pth',
        None,  # Phonemes preprocessing goes here
        {'lang': 'en', 'epochs': 1500, 'n_symbol': 96},
        True
    )


@register
def tacotron2(n_symbol: Optional[int] = None, weights: Optional[Tacotron2Weights] = None, progress: bool = False,
              **kwargs: Any) -> Tacotron2:
    # Backward compatibility for checkpoint_name
    if "checkpoint_name" in kwargs:
        warnings.warn("The argument checkpoint_name is deprecated, please use weights instead.")
        checkpoint_name = kwargs.pop("checkpoint_name")
        weights = next((x for x in Tacotron2Weights if os.path.basename(x.url)[:-4] == checkpoint_name), None)
        if weights is None:
            raise ValueError(f"Unexpected checkpoint_name: '{checkpoint_name}'. ")

    # Confirm we got the right weights
    Tacotron2Weights.check_type(weights)

    if n_symbol is None and weights is None:
        raise ValueError("Both n_symbol and weights can't be None.")

    # Build parameters by overwriding the defaults
    config = {
        **_DEFAULT_PARAMETERS,
        **kwargs,
    }

    # Adjust number of symbols if necessary
    if weights is not None:
        config['n_symbol'] = weights.meta['n_symbol']

    # Initialize model
    model = Tacotron2(**config)

    # Optionally load weights
    if weights is not None:
        model.load_state_dict(weights.state_dict(progress=progress))

    return model
