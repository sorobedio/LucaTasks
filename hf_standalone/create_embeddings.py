#!/usr/bin/env python
# encoding: utf-8
"""
Create embeddings using the standalone HuggingFace adapter and original Encoder flow.

Examples:
1) Task name only (uses repo test_data defaults):
   python hf_standalone/create_embeddings.py \
     --model_name LucaGroup/LucaOne-default-step36M \
     --task_name CentralDogma

2) Explicit dataset name:
   python hf_standalone/create_embeddings.py \
     --model_name LucaGroup/LucaOne-default-step36M \
     --dataset_name ProtStab

3) Custom input file:
   python hf_standalone/create_embeddings.py \
     --model_name LucaGroup/LucaOne-default-step36M \
     --input_file data/test_data/prot/test_prot.fasta \
     --seq_type prot \
     --output_dir embedding/custom/prot
"""

import argparse
import csv
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


TASK_TO_DATASET_TYPE = {
    "centraldogma": "gene_protein",
    "genustax": "gene",
    "infa": "gene_gene",
    "ppi": "protein",
    "protloc": "protein",
    "protstab": "protein",
    "speciestax": "gene",
    "supktax": "gene",
    "test": "gene",
    "ncrnafam": "gene",
    "ncrpi": "gene_protein",
    "translate_v3_1": "gene_protein",
}

DEFAULT_TEST_INPUT = {
    "gene": os.path.join(ROOT_DIR, "data", "test_data", "gene", "test_gene.fasta"),
    "prot": os.path.join(ROOT_DIR, "data", "test_data", "prot", "test_prot.fasta"),
}

DATASET_TYPE_TO_SEQ_TYPES = {
    "gene": ["gene"],
    "protein": ["prot"],
    "gene_gene": ["gene"],
    "gene_protein": ["gene", "prot"],
}

VALID_SEQ_TYPES = {"gene", "dna", "rna", "prot"}
VALID_LLM_TYPES = {"auto", "lucaone", "esm", "dnabert2", "dnaberts"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create embeddings from HuggingFace model and task/dataset name.")
    parser.add_argument("--model_name", required=True, type=str, help="HuggingFace model id (e.g. LucaGroup/LucaOne-default-step36M)")
    parser.add_argument(
        "--llm_type",
        default="auto",
        type=str,
        choices=sorted(list(VALID_LLM_TYPES)),
        help="Backend type. Default auto-detects from model_name.",
    )
    parser.add_argument("--dataset_name", default=None, type=str, help="Dataset name used in this repo (e.g. ProtStab)")
    parser.add_argument("--task_name", default=None, type=str, help="Task name alias used in this repo")
    parser.add_argument(
        "--dataset_type",
        default=None,
        type=str,
        choices=["gene", "protein", "gene_gene", "gene_protein"],
        help="Override resolved dataset type",
    )

    parser.add_argument("--input_file", default=None, type=str, help="Custom input file (.fasta/.csv/.tsv)")
    parser.add_argument("--input_file_gene", default=None, type=str, help="Input file for gene side (gene_protein tasks)")
    parser.add_argument("--input_file_prot", default=None, type=str, help="Input file for prot side (gene_protein tasks)")
    parser.add_argument("--seq_type", default=None, type=str, choices=sorted(list(VALID_SEQ_TYPES)), help="Required with --input_file")
    parser.add_argument("--id_idx", default=None, type=int, help="ID column index for csv/tsv")
    parser.add_argument("--seq_idx", default=None, type=int, help="Sequence column index for csv/tsv")
    parser.add_argument("--has_header", action="store_true", help="Whether csv/tsv includes header row")

    parser.add_argument("--output_dir", default=None, type=str, help="Base output dir")
    parser.add_argument("--embedding_type", default="matrix", choices=["matrix", "vector"], help="Embedding type")
    parser.add_argument("--trunc_type", default="right", choices=["left", "right"], help="Truncation side")
    parser.add_argument("--truncation_seq_length", default=4096, type=int, help="LLM truncation seq length (without BOS/EOS)")

    parser.add_argument("--matrix_add_special_token", action="store_true", help="Keep BOS/EOS vectors in matrix embedding")
    parser.add_argument("--disable_embedding_complete", action="store_true", help="Disable long sequence completion")
    parser.add_argument("--disable_embedding_complete_seg_overlap", action="store_true", help="Disable overlap completion")
    parser.add_argument("--embedding_fixed_len_a_time", default=None, type=int, help="Fixed infer length per forward for long seq")
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU id (default: 0). Set -1 for CPU")

    return parser.parse_args()


def normalize_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.strip().lower()


def sanitize_for_path(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def infer_dataset_type(args) -> str:
    if args.dataset_type:
        return args.dataset_type

    ref = normalize_name(args.dataset_name) or normalize_name(args.task_name)
    if ref and ref in TASK_TO_DATASET_TYPE:
        return TASK_TO_DATASET_TYPE[ref]

    if args.input_file and args.seq_type:
        return "protein" if args.seq_type == "prot" else "gene"

    raise ValueError("Cannot infer dataset_type. Provide --dataset_name/--task_name known in repo, or set --dataset_type.")


def infer_llm_type(model_name: str, llm_type: str) -> str:
    if llm_type and llm_type != "auto":
        return llm_type

    value = model_name.strip().lower()
    if "esm2" in value or value.startswith("facebook/esm2"):
        return "esm"
    if "dnabert-2" in value or "dnabert2" in value:
        return "dnabert2"
    if "dnabert-s" in value or "dnaberts" in value or "dnabert_s" in value:
        return "dnaberts"
    return "lucaone"


def resolve_seq_jobs(args, dataset_type: str) -> List[Tuple[str, str]]:
    """
    Return list of (seq_type, input_file).
    """
    if args.input_file:
        if not args.seq_type:
            raise ValueError("--seq_type is required with --input_file")
        return [(args.seq_type, args.input_file)]

    seq_types = DATASET_TYPE_TO_SEQ_TYPES.get(dataset_type)
    if not seq_types:
        raise ValueError(f"Unsupported dataset_type={dataset_type}")

    jobs = []
    for seq_type in seq_types:
        if seq_type == "gene":
            input_file = args.input_file_gene or DEFAULT_TEST_INPUT["gene"]
        else:
            input_file = args.input_file_prot or DEFAULT_TEST_INPUT["prot"]
        jobs.append((seq_type, input_file))
    return jobs


def validate_jobs_for_llm(llm_type: str, jobs: Sequence[Tuple[str, str]]) -> None:
    seq_types = {seq_type for seq_type, _ in jobs}
    if llm_type == "esm":
        unsupported = [v for v in seq_types if v not in {"prot"}]
        if unsupported:
            raise ValueError(
                f"llm_type=esm only supports protein sequences. Unsupported seq_type(s): {unsupported}. "
                "Use a protein task/dataset, or set --llm_type lucaone for mixed tasks."
            )
    if llm_type in {"dnabert2", "dnaberts"}:
        unsupported = [v for v in seq_types if v not in {"gene", "dna", "rna"}]
        if unsupported:
            raise ValueError(
                f"llm_type={llm_type} only supports gene/dna/rna sequences. Unsupported seq_type(s): {unsupported}. "
                "Use a gene task/dataset, or set --llm_type lucaone for mixed tasks."
            )


def configure_model_env(llm_type: str, model_name: str) -> None:
    if llm_type == "lucaone":
        os.environ["LUCAONE_HF_MODEL_ID"] = model_name
        os.environ["LUCAONE_GENE_HF_MODEL_ID"] = model_name
        os.environ["LUCAONE_PROT_HF_MODEL_ID"] = model_name
    elif llm_type == "esm":
        os.environ["ESM_HF_MODEL_ID"] = model_name
    elif llm_type == "dnabert2":
        os.environ["DNABERT2_HF_MODEL_ID"] = model_name
    elif llm_type == "dnaberts":
        os.environ["DNABERTS_HF_MODEL_ID"] = model_name


def is_seq_candidate(text: str) -> bool:
    if not text:
        return False
    upper = text.strip().upper()
    if len(upper) < 8:
        return False
    valid = sum(1 for ch in upper if "A" <= ch <= "Z")
    ratio = valid / max(len(upper), 1)
    return ratio > 0.9


def guess_id_seq_indices(header: Optional[Sequence[str]], row: Sequence[str]) -> Tuple[int, int]:
    if header:
        norm = [h.strip().lower() for h in header]
        id_candidates = ["seq_id", "protein_id", "gene_id", "id", "name"]
        seq_candidates = ["seq", "sequence", "protein_seq", "gene_seq"]
        id_idx = next((i for i, h in enumerate(norm) if h in id_candidates), None)
        seq_idx = next((i for i, h in enumerate(norm) if h in seq_candidates), None)
        if id_idx is not None and seq_idx is not None:
            return id_idx, seq_idx

    # Known common shapes:
    # 4 cols like prot test csv: clan,pfam,protein_id,protein_seq
    if len(row) >= 4 and is_seq_candidate(row[3]):
        return 2, 3
    # 2+ cols where row[1] is sequence
    if len(row) >= 2 and is_seq_candidate(row[1]):
        return 0, 1
    # Last sequence-like col fallback.
    for idx in range(len(row) - 1, -1, -1):
        if is_seq_candidate(row[idx]):
            id_idx = 0 if idx != 0 else min(1, len(row) - 1)
            return id_idx, idx
    # Final fallback.
    return 0, min(1, len(row) - 1)


def iter_sequences(
    input_file: str,
    seq_type: str,
    id_idx: Optional[int],
    seq_idx: Optional[int],
    has_header: bool,
) -> Iterable[Tuple[str, str, str]]:
    suffix = os.path.basename(input_file).lower().split(".")[-1]
    if suffix in {"fa", "fas", "fasta", "faa"}:
        from src.file_operator import fasta_reader

        for header, seq in fasta_reader(input_file):
            seq_id = header[1:] if header and header.startswith(">") else header
            yield seq_id.strip(), seq_type, seq.strip().upper()
        return

    delimiter = "," if suffix == "csv" else "\t"
    if suffix not in {"csv", "tsv"}:
        raise ValueError(f"Unsupported input file format: {input_file}")

    with open(input_file, "r", encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp, delimiter=delimiter)
        header = None
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if row_idx == 0 and has_header:
                header = row
                continue

            cur_id_idx = id_idx
            cur_seq_idx = seq_idx
            if cur_id_idx is None or cur_seq_idx is None:
                cur_id_idx, cur_seq_idx = guess_id_seq_indices(header=header, row=row)

            if cur_id_idx >= len(row) or cur_seq_idx >= len(row):
                continue
            seq_id = row[cur_id_idx].strip()
            seq = row[cur_seq_idx].strip().upper()
            if not seq_id:
                seq_id = f"{os.path.basename(input_file)}_{row_idx}"
            if seq:
                yield seq_id, seq_type, seq


def build_output_dir(args, dataset_key: str) -> str:
    if args.output_dir:
        return args.output_dir
    model_part = sanitize_for_path(args.model_name)
    return os.path.join(ROOT_DIR, "embedding", "hf_standalone", model_part, dataset_key, args.embedding_type)


def build_encoder(args, model_name: str, output_subdir: str, llm_type: str):
    from src.encoder import Encoder

    use_cpu = args.gpu_id < 0
    fixed_len = args.embedding_fixed_len_a_time if args.embedding_fixed_len_a_time else args.truncation_seq_length
    llm_version = {
        "lucaone": "lucaone",
        "esm": "esm2",
        "dnabert2": "dnabert2",
        "dnaberts": "dnaberts",
    }[llm_type]
    return Encoder(
        llm_dirpath=model_name,
        llm_type=llm_type,
        llm_version=llm_version,
        input_type="matrix" if args.embedding_type == "matrix" else "vector",
        trunc_type=args.trunc_type,
        seq_max_length=args.truncation_seq_length + 2,
        vector_dirpath=output_subdir,
        matrix_dirpath=output_subdir,
        local_rank=-1,
        use_cpu=use_cpu,
        embedding_fixed_len_a_time=fixed_len,
        matrix_add_special_token=args.matrix_add_special_token,
        embedding_complete=not args.disable_embedding_complete,
        embedding_complete_seg_overlap=not args.disable_embedding_complete_seg_overlap,
        matrix_embedding_exists=False,
    )


def run():
    args = parse_args()
    dataset_type = infer_dataset_type(args)
    jobs = resolve_seq_jobs(args, dataset_type=dataset_type)
    llm_type = infer_llm_type(args.model_name, args.llm_type)
    validate_jobs_for_llm(llm_type=llm_type, jobs=jobs)
    configure_model_env(llm_type=llm_type, model_name=args.model_name)

    dataset_key = args.dataset_name or args.task_name or "custom"
    output_base = build_output_dir(args, dataset_key=sanitize_for_path(dataset_key))
    os.makedirs(output_base, exist_ok=True)

    print("model_name:", args.model_name)
    print("llm_type:", llm_type)
    print("dataset_type:", dataset_type)
    print("jobs:", jobs)
    print("embedding_type:", args.embedding_type)
    print("output_base:", output_base)
    print("-" * 60)

    total_saved = 0
    for seq_type, input_file in jobs:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        out_dir = os.path.join(output_base, seq_type)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[{seq_type}] input={input_file}")
        print(f"[{seq_type}] out_dir={out_dir}")

        encoder = build_encoder(args=args, model_name=args.model_name, output_subdir=out_dir, llm_type=llm_type)
        done = 0
        for seq_id, cur_seq_type, seq in iter_sequences(
            input_file=input_file,
            seq_type=seq_type,
            id_idx=args.id_idx,
            seq_idx=args.seq_idx,
            has_header=args.has_header,
        ):
            emb = encoder.__get_embedding__(seq_id, cur_seq_type, seq, args.embedding_type)
            if emb is not None:
                done += 1
                total_saved += 1
                if done % 100 == 0:
                    print(f"[{seq_type}] done: {done}")

        # Print expected file pattern once for convenience.
        from src.utils import calc_emb_filename_by_seq_id
        sample_name = calc_emb_filename_by_seq_id("example_seq_id", args.embedding_type)
        print(f"[{seq_type}] completed: {done}, sample_file={sample_name}")
        print("-" * 60)

    print(f"All done. total_saved={total_saved}")


if __name__ == "__main__":
    run()
