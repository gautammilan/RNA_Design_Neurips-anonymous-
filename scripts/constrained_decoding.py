#!/usr/bin/env python3
"""
Batched inference for RNA LM with optional fast constrained decoding.

- Standard mode uses model.generate(...)
- Constrained mode uses generate() with prefix_allowed_tokens_fn for C++-accelerated, cached decoding
- Resume-from-output support to avoid re-sampling completed IDs
- NEW:
  * model_flavor choice: SL vs SL+RL
  * flavor-specific default model paths
  * test_path instead of input_path
  * if output_path is empty, derive it from test_path (e.g., ./decoding_results/foo_slrl.jsonl)
  * support Hugging Face repo subfolders, e.g. Milanmg/LLM-RNA-Design-2025/model/SL+RL
"""

import argparse
import json
import time
import logging
import os
from collections import Counter, defaultdict  # defaultdict kept in case you extend

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_pair_map(struct: str):
    """
    Map each ')' index to its matching '(' index in a dot-bracket string.
    """
    stack = []
    pair_map = {}
    for i, c in enumerate(struct):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            open_i = stack.pop()
            pair_map[i] = open_i
    return pair_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batched inference for RNA LM on a JSONL of target structures (test data)."
    )

    # --- Test data path ---
    parser.add_argument(
        "--test_path",
        type=str,
        default="./test/eterna100.jsonl",
        help="Path to test data JSONL file.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help=(
            "Where to write the generated designs (JSONL). "
            "If empty, a default will be derived as "
            "'./decoding_results/{test_file_stem}_{model_flavor}.jsonl'."
        ),
    )

    # --- Model selection: SL vs SL+RL ---
    parser.add_argument(
        "--model_flavor",
        type=str,
        choices=["sl", "slrl"],
        default="slrl",
        help="Which model flavor to use: 'sl' (supervised only) or 'slrl' (SL+RL).",
    )

    # Flavor-specific default paths
    parser.add_argument(
        "--sl_model_path",
        type=str,
        default="Milanmg/LLM-RNA-Design-2026/model/SL",
        help="Default model path for SL-only model.",
    )
    parser.add_argument(
        "--slrl_model_path",
        type=str,
        default="Milanmg/LLM-RNA-Design-2026/model/SL+RL",
        help="Default model path for SL+RL model.",
    )

    parser.add_argument("--n_repeats", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_decode_tokens", type=int, default=512)
    parser.add_argument(
        "--constrained_decode",
        action="store_true",
        help="Enforce dot-bracket pairing constraints during decode",
    )
    parser.add_argument(
        "--constrained_id",
        action="extend",
        nargs="+",
        type=int,
        default=[],
        help="Optional subset of ids to run constrained decoding on.",
    )

    # Resume behavior (defaults to True). Use --no-resume_remaining to disable.
    try:
        parser.add_argument(
            "--resume_remaining",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="If True (default), read output_path and only sample remaining repeats per id.",
        )
    except Exception:
        parser.add_argument(
            "--resume_remaining",
            type=lambda s: str(s).lower() not in {"0", "false", "no"},
            default=True,
            help="If True (default), read output_path and only sample remaining repeats per id. "
                 "Pass 0/false to disable.",
        )

    return parser.parse_args()


def load_existing_counts(output_path: str) -> Counter:
    """
    Load existing JSONL outputs and count number of samples per id.
    Ignores malformed lines with a warning.
    """
    counts = Counter()
    bad_lines = 0
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return counts

    with open(output_path, "r") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("id", None)
                if rid is not None:
                    counts[rid] += 1
            except Exception:
                bad_lines += 1
                # continue on parse error

    if bad_lines:
        logging.warning(
            f"Skipped {bad_lines} malformed lines while reading existing output from {output_path}"
        )
    return counts


def build_task_list(records, n_repeats: int, existing: Counter, constrained_ids):
    """
    From input records and existing counts, decide how many repeats remain per id.
    Returns (tasks, stats_dict)
    """
    # If constrained_ids provided, filter records accordingly
    if constrained_ids:
        id_set = set(constrained_ids)
        records = [r for r in records if r.get("id") in id_set]

    # Map id -> target_structure (assume unique per id)
    id_to_struct = {}
    for r in records:
        rid = r["id"]
        id_to_struct[rid] = r["target_structure"]

    # Compute remaining per id
    tasks = []
    fully_done = []
    partially_done = []
    not_started = []

    for rid, struct in id_to_struct.items():
        have = existing.get(rid, 0)
        need = max(0, n_repeats - have)

        if have >= n_repeats:
            fully_done.append(rid)
            continue
        elif have > 0:
            partially_done.append((rid, have, need))
        else:
            not_started.append(rid)

        for _ in range(need):
            tasks.append({"id": rid, "target_structure": struct})

    stats = {
        "total_ids_considered": len(id_to_struct),
        "already_completed_ids": len(fully_done),
        "partially_completed_ids": len(partially_done),
        "not_started_ids": len(not_started),
        "remaining_tasks": len(tasks),
        "partially_detail": partially_done,   # list of tuples (id, have, need)
        "not_started_detail": not_started[:20],  # cap preview
        "completed_detail": fully_done[:20],     # cap preview
    }
    return tasks, stats


def split_repo_and_subfolder(path_or_repo: str):
    """
    Take something like:
      - 'Milanmg/LLM-RNA-Design-2025'            -> ('Milanmg/LLM-RNA-Design-2025', None)
      - 'Milanmg/LLM-RNA-Design-2025/model/SL'  -> ('Milanmg/LLM-RNA-Design-2025', 'model/SL')
    i.e., first two components are the HF repo id, the rest (if any) is subfolder.
    """
    parts = path_or_repo.split("/")
    if len(parts) <= 2:
        return path_or_repo, None
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:])
    return repo_id, subfolder


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Resolve which "raw" model path string to use
    if args.model_flavor == "sl":
        raw_model_path = args.sl_model_path
    else:  # "slrl"
        raw_model_path = args.slrl_model_path
    logger.info(f"Using model_flavor={args.model_flavor}, model_path={raw_model_path}")

    # Split into repo_id and optional subfolder for Hugging Face Hub
    repo_id, subfolder = split_repo_and_subfolder(raw_model_path)
    if subfolder:
        logger.info(f"Resolved repo_id='{repo_id}', subfolder='{subfolder}'")
    else:
        logger.info(f"Resolved repo_id='{repo_id}', subfolder=(root)")

    # Resolve output path: if empty, derive from test_path and model_flavor
    if not args.output_path:
        test_base = os.path.basename(args.test_path)
        stem, _ext = os.path.splitext(test_base)
        # always use .jsonl for decoding outputs
        default_dir = "./decoding_results"
        os.makedirs(default_dir, exist_ok=True)
        default_name = f"{stem}_{args.model_flavor}.jsonl"
        output_path = os.path.join(default_dir, default_name)
        logger.info(
            f"No --output_path provided. Using derived path: {output_path}"
        )
    else:
        output_path = args.output_path
        logger.info(f"Using explicit output_path: {output_path}")

    # Load tokenizer & model
    logger.info(f"Loading model & tokenizer from repo_id='{repo_id}', subfolder='{subfolder or ''}'")
    tok_kwargs = {"trust_remote_code": True}
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if subfolder is not None:
        tok_kwargs["subfolder"] = subfolder
        model_kwargs["subfolder"] = subfolder

    tokenizer = AutoTokenizer.from_pretrained(repo_id, **tok_kwargs)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
    model.config.pad_token_id = model.config.eos_token_id
    device = next(model.parameters()).device

    # Read test JSONL
    with open(args.test_path, "r") as fin:
        records = [json.loads(l) for l in fin if l.strip()]
    logger.info(f"Loaded {len(records)} test records from {args.test_path}")

    # Resume logic
    existing_counts = Counter()
    if args.resume_remaining:
        existing_counts = load_existing_counts(output_path)
        logger.info(
            f"Resume enabled: found {sum(existing_counts.values())} existing samples "
            f"across {len(existing_counts)} unique ids in {output_path}"
        )
    else:
        logger.info(
            "Resume disabled (--no-resume_remaining): starting fresh without reading existing output"
        )

    # Build tasks based on remaining repeats
    tasks, stats = build_task_list(
        records, args.n_repeats, existing_counts, args.constrained_id
    )

    logger.info(
        "ID sampling status: "
        f"considered={stats['total_ids_considered']}, "
        f"completed={stats['already_completed_ids']}, "
        f"partial={stats['partially_completed_ids']}, "
        f"not_started={stats['not_started_ids']}, "
        f"remaining_tasks={stats['remaining_tasks']}"
    )
    if stats["partially_completed_ids"] > 0:
        sample = ", ".join(
            [
                f"{rid}(have={have},need={need})"
                for rid, have, need in stats["partially_detail"][:10]
            ]
        )
        logger.info(f"Partially-completed examples: {sample}")
    if stats["completed_detail"]:
        logger.info(
            f"Completed examples (preview): "
            f"{', '.join(map(str, stats['completed_detail']))}"
        )
    if stats["not_started_detail"]:
        logger.info(
            f"Not-started examples (preview): "
            f"{', '.join(map(str, stats['not_started_detail']))}"
        )

    if len(tasks) == 0:
        logger.info("Nothing to do — all requested repeats are already satisfied.")
        return

    total_batches = (len(tasks) + args.batch_size - 1) // args.batch_size
    logger.info(f"{len(tasks)} tasks in {total_batches} batches")

    # Open output file in append mode if resuming; otherwise, overwrite
    write_mode = "a" if args.resume_remaining else "w"
    with open(output_path, write_mode) as fout:
        for bidx in range(total_batches):
            batch = tasks[bidx * args.batch_size : (bidx + 1) * args.batch_size]
            structures = [t["target_structure"] for t in batch]

            # Build prompts
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "structure", "content": s}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                + (tokenizer.bos_token or "")
                for s in structures
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            prefix_len = input_ids.size(1)

            logger.info(
                f"Batch {bidx+1}/{total_batches}: size={len(batch)}, prefix_len={prefix_len}"
            )

            if args.constrained_decode:
                # Precompute pairing maps & nucleotide IDs
                pair_maps = [compute_pair_map(s) for s in structures]
                pair_rules = {
                    "A": ["U"],
                    "C": ["G"],
                    "G": ["C", "U"],
                    "U": ["A", "G"],
                }
                nucs = ["A", "C", "G", "U"]
                nuc_ids = {
                    n: tokenizer.convert_tokens_to_ids(n)
                    for n in nucs
                }
                eos_id = tokenizer.eos_token_id

                def prefix_allowed_tokens_fn(batch_id, cur_ids):
                    """
                    Called at each generation step.
                    step = total tokens so far - prefix_len
                    """
                    seq_len = cur_ids.shape[-1]
                    step = seq_len - prefix_len
                    struct = structures[batch_id]

                    # once we've generated beyond the structure, only allow EOS
                    if step >= len(struct):
                        return [eos_id]

                    c = struct[step]
                    if c in ".(":
                        return list(nuc_ids.values())

                    # c == ")': enforce pairing
                    open_i = pair_maps[batch_id].get(step, None)
                    if open_i is None:
                        return list(nuc_ids.values())
                    prev_id = cur_ids[prefix_len + open_i].item()
                    prev_tok = tokenizer.convert_ids_to_tokens(prev_id)
                    allowed_nts = pair_rules.get(prev_tok, nucs)
                    return [
                        tokenizer.convert_tokens_to_ids(n)
                        for n in allowed_nts
                    ]

                t0 = time.time()
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_decode_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temp,
                    top_p=args.top_p,
                    use_cache=True,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                )
                elapsed = time.time() - t0
                logger.info(f"  Constrained generate() took {elapsed:.2f}s")
            else:
                t0 = time.time()
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_decode_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temp,
                    top_p=args.top_p,
                )
                elapsed = time.time() - t0
                logger.info(f"  Standard generate() took {elapsed:.2f}s")

            # Decode & write out
            per_time = elapsed / len(batch)
            for task, out_ids in zip(batch, outputs):
                gen_ids = out_ids[prefix_len:]
                seq = tokenizer.decode(gen_ids, skip_special_tokens=True)
                designed_sequence = "".join(seq.split())
                fout.write(
                    json.dumps(
                        {
                            "id": task["id"],
                            "target_structure": task["target_structure"],
                            "designed_sequence": designed_sequence,
                            "time": per_time,
                        }
                    )
                    + "\n"
                )

    logger.info(f"All done — results saved to {output_path}")


if __name__ == "__main__":
    main()
