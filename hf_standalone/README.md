# HF Standalone Adapter

Standalone HuggingFace embedding adapter folder.

## Files
- `hf_standalone/predict_embedding.py`: minimal predictors with the same call signatures expected by `Encoder`.
- `hf_standalone/__init__.py`: exports predictor functions.

## Supported backends
- LucaOne via `AutoModel.from_pretrained(..., trust_remote_code=True)`
- ESM2 via HuggingFace model ids (`facebook/esm2_*`)
- DNABERT2 (`zhihan1996/DNABERT-2-117M`)
- DNABERT-S (`zhihan1996/DNABERT-S`)

## Notes
- This folder is self-contained and does not import project `src.utils`.
- Extraction output contract stays the same (`matrix`/`vector` modes).
- For LucaOne model id resolution:
  - first checks `LUCAONE_GENE_HF_MODEL_ID`, `LUCAONE_PROT_HF_MODEL_ID`, `LUCAONE_HF_MODEL_ID`
  - then uses `llm_dirpath` (string or dict values).

