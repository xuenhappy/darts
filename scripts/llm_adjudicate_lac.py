#!/usr/bin/env python3
"""Use a small local instruction model to adjudicate difficult teacher samples."""

import argparse
import json
from pathlib import Path
import re

from pseudo_corpus import validate_annotation


LABELS = (
    "POS_NOUN POS_VERB POS_ADJ POS_ADV POS_PRON POS_ADP POS_CCONJ POS_SCONJ "
    "POS_PART POS_AUX POS_NUM POS_PUNCT POS_X POS_nr POS_ns POS_nt POS_nz POS_t POS_f"
)
JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def prompt(item):
    return f"""/no_think
你是中文分词和词性标注仲裁器。比较两个候选，选择整体更准确的一个，不得自行重写。
文本：{json.dumps(item['text'], ensure_ascii=False)}
LTP候选：{json.dumps(item.get('ltp'), ensure_ascii=False)}
百度LAC候选：{json.dumps(item.get('paddle'), ensure_ascii=False)}
若两者都有明显错误则拒绝。只输出{{"choice":"ltp"}}、{{"choice":"paddle"}}或
{{"choice":"reject"}}，不要解释。"""


def parse_answer(answer, text, max_atoms):
    match = JSON_OBJECT.search(answer)
    if not match:
        return None, "json"
    try:
        choice = json.loads(match.group())["choice"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, "json"
    if choice not in ("ltp", "paddle", "reject"):
        return None, "choice"
    return choice, None


def selected_annotation(item, answer, max_atoms):
    choice, reason = parse_answer(answer, item["text"], max_atoms)
    if choice is None:
        return None, reason, None
    if choice == "reject":
        return None, "llm_reject", choice
    annotation = item.get(choice)
    reason = validate_annotation(item["text"], annotation, 1, max_atoms)
    return (annotation, None, choice) if reason is None else (None, reason, choice)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-AWQ")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-atoms", type=int, default=108)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("LLM adjudication requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", torch_dtype=torch.float16, low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    accepted = rejected = 0
    with Path(args.input).open(encoding="utf-8") as source, Path(args.output).open(
        "w", encoding="utf-8"
    ) as output, Path(args.rejected).open("w", encoding="utf-8") as bad:
        while batch := [json.loads(line) for line in [source.readline() for _ in range(args.batch_size)] if line]:
            messages = [[{"role": "user", "content": prompt(item)}] for item in batch]
            texts = [tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
                     for msg in messages]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=16, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                )
            for item, input_ids, output_ids in zip(batch, inputs.input_ids, generated):
                answer = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True)
                annotation, reason, choice = selected_annotation(item, answer, args.max_atoms)
                if annotation is None:
                    bad.write(json.dumps({**item, "reason": reason, "answer": answer}, ensure_ascii=False) + "\n")
                    rejected += 1
                else:
                    output.write(json.dumps(
                        {**item, "choice": choice, "annotation": annotation}, ensure_ascii=False
                    ) + "\n")
                    accepted += 1
    print(json.dumps({"accepted": accepted, "rejected": rejected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
