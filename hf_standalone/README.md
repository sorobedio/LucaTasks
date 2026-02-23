# HF Standalone Adapter

Standalone HuggingFace embedding adapter folder.

## Files
- `hf_standalone/predict_embedding.py`: minimal predictors with the same call signatures expected by `Encoder`.
- `hf_standalone/__init__.py`: exports predictor functions.
- `hf_standalone/create_embeddings.py`: standalone CLI to create embeddings from `--model_name` + `--task_name`/`--dataset_name`.

## Supported backends
- LucaOne via `AutoModel.from_pretrained(..., trust_remote_code=True)`
- ESM2 via HuggingFace model ids (`facebook/esm2_*`)
- DNABERT2 (`zhihan1996/DNABERT-2-117M`)
- DNABERT-S (`zhihan1996/DNABERT-S`)

## ESM2 case
- Explicit function: `predict_embedding_esm2(...)`
- Alias: `predict_embedding_esm(...)`
- Accepted version aliases: `esm2`, `3B`, `650M`, `150M`, `15B` (and common `esm2-*`/`esm2_*` variants).

## Notes
- This folder is self-contained and does not import project `src.utils`.
- Extraction output contract stays the same (`matrix`/`vector` modes).
- For LucaOne model id resolution:
  - first checks `LUCAONE_GENE_HF_MODEL_ID`, `LUCAONE_PROT_HF_MODEL_ID`, `LUCAONE_HF_MODEL_ID`
  - then uses `llm_dirpath` (string or dict values).

## CLI examples
- LucaOne with task name:
  - `python hf_standalone/create_embeddings.py --model_name LucaGroup/LucaOne-default-step36M --task_name CentralDogma`
- LucaOne with dataset name:
  - `python hf_standalone/create_embeddings.py --model_name LucaGroup/LucaOne-default-step36M --dataset_name ProtStab`
- ESM2 with protein task:
  - `python hf_standalone/create_embeddings.py --model_name facebook/esm2_t33_650M_UR50D --task_name PPI`
