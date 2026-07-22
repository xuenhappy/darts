#!/usr/bin/env python3
"""Generate flat LTP corrections with a LoRA adapter and evaluate against gold."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pseudo_corpus import SHORT_TO_POS, validate_annotation


def parse_flat(text, source, max_atoms):
    annotation = []
    for field in text.strip().split():
        word, separator, pos = field.rpartition("/")
        if not separator or not word or pos not in SHORT_TO_POS:
            return None, "format"
        annotation.append((word, SHORT_TO_POS[pos]))
    reason = validate_annotation(source, annotation, 1, max_atoms)
    return (annotation, None) if reason is None else (None, reason)


def spans(annotation):
    offset = 0
    result = {}
    for word, pos in annotation:
        end = offset + len(word)
        result[(offset, end)] = pos
        offset = end
    return result


def add_metrics(totals, gold, predicted):
    target, output = spans(gold), spans(predicted)
    shared = target.keys() & output.keys()
    totals["tp"] += len(shared)
    totals["gold"] += len(target)
    totals["predicted"] += len(output)
    totals["typed"] += sum(target[key] == output[key] for key in shared)


def finalize(totals):
    precision = totals["tp"] / max(1, totals["predicted"])
    recall = totals["tp"] / max(1, totals["gold"])
    totals.update({
        "boundary_precision": precision, "boundary_recall": recall,
        "boundary_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "pos_accuracy_on_shared": totals["typed"] / max(1, totals["tp"]),
    })
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/darts-corpus/models/Qwen3-0.6B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", help="optional JSON file for aggregate metrics")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-atoms", type=int, default=108)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float32",
        help="Model dtype. FP32 is the stable default for the current CUDA host.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Match the stable attention path used by training on this CUDA host.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model_dtype = getattr(torch, args.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=model_dtype, local_files_only=True
    ).to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    items = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()]
    totals = {"tp": 0, "gold": 0, "predicted": 0, "typed": 0, "valid": 0, "invalid": 0}
    baseline = {"tp": 0, "gold": 0, "predicted": 0, "typed": 0}
    with Path(args.output).open("w", encoding="utf-8") as output:
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            prompts = [tokenizer.apply_chat_template(
                [item["messages"][0]], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            ) for item in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                               max_length=args.max_input_length).to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                )
            for item, output_ids in zip(batch, generated):
                add_metrics(baseline, item["gold"], item["ltp"])
                answer = tokenizer.decode(output_ids[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                annotation, reason = parse_flat(answer, item["text"], args.max_atoms)
                if annotation is None:
                    totals["invalid"] += 1
                    totals["gold"] += len(item["gold"])
                else:
                    totals["valid"] += 1
                    add_metrics(totals, item["gold"], annotation)
                output.write(json.dumps({
                    "digest": item["digest"], "text": item["text"], "annotation": annotation,
                    "reason": reason, "raw": answer,
                }, ensure_ascii=False) + "\n")
    metrics = {"corrected": finalize(totals), "ltp_baseline": finalize(baseline)}
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.metrics_output:
        target = Path(args.metrics_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
