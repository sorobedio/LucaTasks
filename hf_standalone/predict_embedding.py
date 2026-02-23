#!/usr/bin/env python
# encoding: utf-8
"""
Minimal embedding extraction backend built on HuggingFace loaders.

The function signatures intentionally match the legacy backends so
`Encoder.__get_embedding__` can keep the same extraction flow.
"""

import os
from typing import Any, Dict, Optional

import torch
from transformers import AutoModel, AutoTokenizer

def clean_seq_luca(seq_id, seq):
    del seq_id
    seq = seq.upper()
    return "".join(ch for ch in seq if "A" <= ch <= "Z")


def clean_seq_esm(seq_id, seq, return_rm_index=False):
    seq = seq.upper()
    new_seq = []
    rm_index = set()
    invalid_chars = set()
    for idx, ch in enumerate(seq):
        if "A" <= ch <= "Z" and ch != "J":
            new_seq.append(ch)
        else:
            invalid_chars.add(ch)
            rm_index.add(idx)
    if invalid_chars:
        print("id: %s. Seq: %s" % (seq_id, seq))
        print("invalid char set:", invalid_chars)
        print("return_rm_index:", rm_index)
    cleaned = "".join(new_seq)
    if return_rm_index:
        return cleaned, rm_index
    return cleaned


_DNABERT2_MODEL_ID = "zhihan1996/DNABERT-2-117M"
_DNABERTS_MODEL_ID = "zhihan1996/DNABERT-S"
_ESM2_MODEL_IDS = {
    "15B": "facebook/esm2_t48_15B_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
}

_MODEL_CACHE: Dict[str, torch.nn.Module] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}


def _normalize_sample(sample):
    if len(sample) > 2:
        return sample[0], sample[1], sample[2]
    return sample[0], "prot", sample[1]


def _resolve_device(model: torch.nn.Module, device: Optional[torch.device]) -> torch.device:
    if device is None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    if model_device != device:
        model.to(device)
    return device


def _tokenize(tokenizer, seq: str, max_length: int) -> Dict[str, torch.Tensor]:
    kwargs = {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": max_length,
    }
    try:
        batch = tokenizer(seq, **kwargs)
    except TypeError:
        batch = tokenizer(seq, add_special_tokens=True, **kwargs)
    if "input_ids" not in batch:
        token_ids = tokenizer.encode(seq)[:max_length]
        batch = {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
    if "attention_mask" not in batch and torch.is_tensor(batch["input_ids"]):
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    return batch


def _select_hidden(output: Any, seq_type: str) -> Optional[torch.Tensor]:
    # LucaOne protein branch may expose *_b fields.
    if seq_type in {"prot", "multi_prot"} and hasattr(output, "hidden_states_b"):
        hs = output.hidden_states_b
        if torch.is_tensor(hs):
            return hs
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state
    if hasattr(output, "hidden_states"):
        hs = output.hidden_states
        if torch.is_tensor(hs):
            return hs
        if isinstance(hs, (tuple, list)) and len(hs) > 0 and torch.is_tensor(hs[-1]):
            return hs[-1]
    if seq_type in {"prot", "multi_prot"} and hasattr(output, "hidden_states"):
        hs = output.hidden_states
        if torch.is_tensor(hs):
            return hs
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    if torch.is_tensor(output):
        return output
    return None


def _select_contacts(output: Any, seq_type: str) -> Optional[torch.Tensor]:
    if seq_type in {"prot", "multi_prot"} and hasattr(output, "contacts_b"):
        contacts = output.contacts_b
        if torch.is_tensor(contacts):
            return contacts
    if hasattr(output, "contacts") and torch.is_tensor(output.contacts):
        return output.contacts
    return None


def _extract_embeddings(
    output: Any,
    input_ids: Optional[torch.Tensor],
    seq_type: str,
    embedding_type: str,
    matrix_add_special_token: bool,
    truncation_seq_length: int,
):
    hidden = _select_hidden(output, seq_type)
    if hidden is None:
        return None, None
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)

    token_count = int(input_ids.shape[1]) if input_ids is not None else int(hidden.shape[1])
    valid_len = min(token_count, int(hidden.shape[1]))
    truncate_len = max(0, min(truncation_seq_length, valid_len - 2))
    processed_seq_len = truncate_len + 2 if valid_len >= 2 else valid_len

    embeddings = {}
    if "representations" in embedding_type or "matrix" in embedding_type:
        if matrix_add_special_token:
            matrix = hidden[0, 0:processed_seq_len, :]
        else:
            start = 1 if hidden.shape[1] > 1 else 0
            end = min(start + truncate_len, hidden.shape[1])
            matrix = hidden[0, start:end, :]
        embeddings["representations"] = matrix.to(device="cpu").clone().numpy()

    if "bos" in embedding_type or "vector" in embedding_type:
        vector = hidden[0, 0, :]
        embeddings["bos_representations"] = vector.to(device="cpu").clone().numpy()

    if "contacts" in embedding_type:
        contacts = _select_contacts(output, seq_type)
        if contacts is None:
            embeddings["contacts"] = None
        else:
            embeddings["contacts"] = contacts.to(device="cpu")[0, :, :].clone().numpy()

    if len(embeddings) > 1:
        return embeddings, processed_seq_len
    if len(embeddings) == 1:
        return list(embeddings.items())[0][1], processed_seq_len
    return None, None


def _load_hf_tokenizer(model_id: str, trust_remote_code: bool = True):
    if model_id not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_id] = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
    return _TOKENIZER_CACHE[model_id]


def _load_hf_model(
    model_id: str,
    trust_remote_code: bool = True,
    task_level: Optional[str] = None,
    task_type: Optional[str] = None,
):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    kwargs = {"trust_remote_code": trust_remote_code}
    if task_level is not None:
        kwargs["task_level"] = task_level
    if task_type is not None:
        kwargs["task_type"] = task_type

    try:
        model = AutoModel.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("task_level", None)
        kwargs.pop("task_type", None)
        model = AutoModel.from_pretrained(model_id, **kwargs)
    _MODEL_CACHE[model_id] = model
    return model


def _predict_embedding_hf(
    model_id: str,
    sample,
    trunc_type: str,
    embedding_type: str,
    truncation_seq_length: int,
    device: Optional[torch.device],
    matrix_add_special_token: bool,
    clean_func,
    task_level: Optional[str] = None,
    task_type: Optional[str] = None,
):
    seq_id, seq_type, seq = _normalize_sample(sample)
    processed_seq = clean_func(seq_id, seq)
    if len(processed_seq) > truncation_seq_length:
        if trunc_type == "left":
            processed_seq = processed_seq[-truncation_seq_length:]
        else:
            processed_seq = processed_seq[:truncation_seq_length]

    tokenizer = _load_hf_tokenizer(model_id=model_id, trust_remote_code=True)
    model = _load_hf_model(
        model_id=model_id,
        trust_remote_code=True,
        task_level=task_level,
        task_type=task_type,
    )
    device = _resolve_device(model, device)
    model.eval()

    batch = _tokenize(tokenizer, processed_seq, truncation_seq_length + 2)
    batch = {
        k: v.to(device=device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

    with torch.no_grad():
        try:
            output = model(**batch)
        except RuntimeError as e:
            if str(e).startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {seq_id} of length {len(seq)}.")
                print("Please reduce the 'truncation_seq_length'")
            return None, None

    return _extract_embeddings(
        output=output,
        input_ids=batch.get("input_ids"),
        seq_type=seq_type,
        embedding_type=embedding_type,
        matrix_add_special_token=matrix_add_special_token,
        truncation_seq_length=truncation_seq_length,
    )


def _resolve_lucaone_model_id(llm_dirpath, seq_type: str) -> Optional[str]:
    # Allow explicit environment override without changing existing args plumbing.
    env_global = os.environ.get("LUCAONE_HF_MODEL_ID")
    env_gene = os.environ.get("LUCAONE_GENE_HF_MODEL_ID")
    env_prot = os.environ.get("LUCAONE_PROT_HF_MODEL_ID")
    if seq_type in {"gene", "dna", "rna"} and env_gene:
        return env_gene
    if seq_type in {"prot", "multi_prot"} and env_prot:
        return env_prot
    if env_global:
        return env_global

    if isinstance(llm_dirpath, dict):
        if seq_type in {"gene", "dna", "rna"}:
            if "gene" in llm_dirpath:
                return llm_dirpath["gene"]
        if seq_type in {"prot", "multi_prot"}:
            if "protein" in llm_dirpath:
                return llm_dirpath["protein"]
            if "prot" in llm_dirpath:
                return llm_dirpath["prot"]
        # fallback: first model in dict
        for v in llm_dirpath.values():
            return v
        return None

    if isinstance(llm_dirpath, str) and llm_dirpath.strip():
        return llm_dirpath

    return None


def predict_embedding_luca(
    llm_dirpath,
    sample,
    trunc_type,
    embedding_type,
    repr_layers=[-1],
    truncation_seq_length=4094,
    device=None,
    matrix_add_special_token=False,
    use_bf16=False,
):
    assert "bos" in embedding_type or "representations" in embedding_type \
           or "matrix" in embedding_type or "vector" in embedding_type or "contacts" in embedding_type
    _, seq_type, _ = _normalize_sample(sample)
    model_id = _resolve_lucaone_model_id(llm_dirpath, seq_type=seq_type)
    if not model_id:
        raise ValueError("LucaOne model id is empty. Set llm_dirpath or LUCAONE_*_HF_MODEL_ID.")
    return _predict_embedding_hf(
        model_id=model_id,
        sample=sample,
        trunc_type=trunc_type,
        embedding_type=embedding_type,
        truncation_seq_length=truncation_seq_length,
        device=device,
        matrix_add_special_token=matrix_add_special_token,
        clean_func=clean_seq_luca,
        task_level="token_level",
        task_type="embedding",
    )


def predict_embedding_dnabert2(
    sample,
    trunc_type,
    embedding_type,
    repr_layers=[-1],
    truncation_seq_length=4094,
    device=None,
    version="dnabert2",
    matrix_add_special_token=False,
):
    assert "bos" in embedding_type or "representations" in embedding_type \
           or "matrix" in embedding_type or "vector" in embedding_type or "contacts" in embedding_type
    model_id = version if "/" in str(version) else _DNABERT2_MODEL_ID
    return _predict_embedding_hf(
        model_id=model_id,
        sample=sample,
        trunc_type=trunc_type,
        embedding_type=embedding_type,
        truncation_seq_length=truncation_seq_length,
        device=device,
        matrix_add_special_token=matrix_add_special_token,
        clean_func=clean_seq_luca,
    )


def predict_embedding_dnaberts(
    sample,
    trunc_type,
    embedding_type,
    repr_layers=[-1],
    truncation_seq_length=4094,
    device=None,
    version="dnaberts",
    matrix_add_special_token=False,
):
    assert "bos" in embedding_type or "representations" in embedding_type \
           or "matrix" in embedding_type or "vector" in embedding_type or "contacts" in embedding_type
    model_id = version if "/" in str(version) else _DNABERTS_MODEL_ID
    return _predict_embedding_hf(
        model_id=model_id,
        sample=sample,
        trunc_type=trunc_type,
        embedding_type=embedding_type,
        truncation_seq_length=truncation_seq_length,
        device=device,
        matrix_add_special_token=matrix_add_special_token,
        clean_func=clean_seq_luca,
    )


def predict_embedding_esm(
    sample,
    trunc_type,
    embedding_type,
    repr_layers=[-1],
    truncation_seq_length=4094,
    device=None,
    version="3B",
    matrix_add_special_token=False,
):
    assert "bos" in embedding_type or "representations" in embedding_type \
           or "matrix" in embedding_type or "vector" in embedding_type or "contacts" in embedding_type
    model_id = _ESM2_MODEL_IDS.get(version, version if "/" in str(version) else _ESM2_MODEL_IDS["3B"])
    return _predict_embedding_hf(
        model_id=model_id,
        sample=sample,
        trunc_type=trunc_type,
        embedding_type=embedding_type,
        truncation_seq_length=truncation_seq_length,
        device=device,
        matrix_add_special_token=matrix_add_special_token,
        clean_func=clean_seq_esm,
    )
