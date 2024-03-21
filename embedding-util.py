from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn.functional as F
import torch as Tensor

import os
import warnings

warnings.filterwarnings(action='ignore', category='ResourceWarning')
os.environ['TOKENIZERS_PARALLELISM'] = "false"

tokenizers = AutoTokenizer.from_pretrained('thenlper/gte-base')

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None]).bool(), 0.0)
    return last_hidden.sum(dim=1)/attention_mask.sum(dim=1)[...,None]

def generate_embeddings(text):
    inputs = tokenizers(text, return_tensor='pt',
                        max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    attention_mask = inputs['attention_mask']
    embeddings = average_pool(outputs.last_hidden_state, attention_mask)
    embeddings = F.normalize(embeddings,p=2, dim=1)
    return embeddings.numpy().tolist()[0]