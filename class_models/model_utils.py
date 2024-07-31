import transformers
transformers.logging.set_verbosity_error()

import os
import torch
import random
import numpy as np

from torch import nn
from typing import List
from urllib.parse import urlparse
from transformers import BertTokenizer
from timm.models.hub import download_cached_file

# Initialize tokenizer
def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer

# Is URL or not
def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

# Load checkpoint
def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    if 'model' in checkpoint.keys():
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('Load checkpoint from %s' % url_or_filename)
    return model, msg

# Set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Set trainable parameters
def set_trainable(model_component, layer_names, type, include_predictions, include_embeddings):
    # Freeze all parameters first
    for param in model_component.parameters():
        param.requires_grad = False

    # Enable training only for the specified layers
    for name, param in model_component.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True

    if include_predictions and type=='decoder':
        prediction_layers = [
            'cls.predictions.bias',
            'cls.predictions.transform.dense.weight',
            'cls.predictions.transform.dense.bias',
            'cls.predictions.transform.LayerNorm.weight',
            'cls.predictions.transform.LayerNorm.bias',
        ]
        for name, param in model_component.named_parameters():
            if any(layer in name for layer in prediction_layers):
                param.requires_grad = True

    if include_embeddings and type=='decoder':
        embedding_layers = [
            'bert.embeddings.word_embeddings.weight',
            'bert.embeddings.position_embeddings.weight',
            'bert.embeddings.token_type_embeddings.weight',
            'bert.embeddings.LayerNorm.weight',
            'bert.embeddings.LayerNorm.bias'
        ]
        for name, param in model_component.named_parameters():
            if any(layer in name for layer in embedding_layers):
                param.requires_grad = True

    if include_embeddings and type=='encoder':
        embedding_layers = [
            'embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight',
            'embeddings.LayerNorm.weight',
            'embeddings.LayerNorm.bias'
        ]
        for name, param in model_component.named_parameters():
            if any(layer in name for layer in embedding_layers):
                param.requires_grad = True

# Tie encoder and decoder
def tie_encoder_decoder_weights(encoder, decoder, base_model_prefix, skip_key):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        transformers.logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            skip_key: str,
            depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name + ' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                    len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                            encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)


