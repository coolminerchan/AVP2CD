import argparse
import copy
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions and calculate AVP2CD evaluation metrics.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mode", choices=["full", "partial"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--task-filter", default=None, help="Defaults to report for full mode and all tasks for partial mode.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)?|[^\s]", text.lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def bleu4(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        return float(
            sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=SmoothingFunction().method1,
            )
        )
    except Exception:
        if not pred_tokens or not ref_tokens:
            return 0.0
        precisions = []
        for n in range(1, 5):
            pred_ngrams = Counter(ngrams(pred_tokens, n))
            ref_ngrams = Counter(ngrams(ref_tokens, n))
            total = sum(pred_ngrams.values())
            matched = sum(min(count, ref_ngrams[gram]) for gram, count in pred_ngrams.items())
            precisions.append((matched + 1.0) / (total + 1.0))
        brevity_penalty = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(1, len(pred_tokens)))
        return float(brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / 4.0))


def meteor(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    try:
        from nltk.translate import meteor_score

        return float(meteor_score.meteor_score([ref_tokens], pred_tokens))
    except Exception:
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        if not pred_set or not ref_set:
            return 0.0
        precision = len(pred_set & ref_set) / len(pred_set)
        recall = len(pred_set & ref_set) / len(ref_set)
        if precision + recall == 0:
            return 0.0
        return float((10 * precision * recall) / (recall + 9 * precision))


def cider(pred_tokens: list[str], ref_tokens: list[str], max_n: int = 4) -> float:
    score = 0.0
    for n in range(1, max_n + 1):
        pred_counts = Counter(ngrams(pred_tokens, n))
        ref_counts = Counter(ngrams(ref_tokens, n))
        total = sum(pred_counts.values())
        if total == 0:
            continue
        matched = sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
        score += (matched / total) / max_n
    return float(score)


def exact_match(prediction: str, reference: str) -> float:
    normalize = lambda text: re.sub(r"\s+", " ", text.strip().lower())
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def calculate_metrics(records: dict[str, dict[str, Any]]) -> dict[str, float]:
    metrics = {"rougeL": [], "meteor": [], "bleu": [], "cider": [], "exact_match": []}
    for value in records.values():
        pred = value["prediction"]
        ref = value["reference"]
        pred_tokens = simple_tokenize(pred)
        ref_tokens = simple_tokenize(ref)
        value["rougeL"] = round(rouge_l_f1(pred_tokens, ref_tokens), 4)
        value["meteor"] = round(meteor(pred_tokens, ref_tokens), 4)
        value["bleu"] = round(bleu4(pred_tokens, ref_tokens), 4)
        value["cider"] = round(cider(pred_tokens, ref_tokens), 4)
        value["exact_match"] = round(exact_match(pred, ref), 4)

        metrics["rougeL"].append(value["rougeL"])
        metrics["meteor"].append(value["meteor"])
        metrics["bleu"].append(value["bleu"])
        metrics["cider"].append(value["cider"])
        metrics["exact_match"].append(value["exact_match"])

    summary = {key: (sum(values) / len(values)) if values else 0.0 for key, values in metrics.items()}
    summary["avg"] = sum([summary["rougeL"], summary["meteor"], summary["bleu"], summary["cider"]]) / 4.0
    return {key: round(value, 4) for key, value in summary.items()}


def batch_process_func(processor, batch, device) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    messages = copy.deepcopy(batch["Message"])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversations=messages, return_video_kwargs=False)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    for key, value in inputs.items():
        if hasattr(value, "to"):
            inputs[key] = value.to(device)
    return inputs


def filter_batch_by_task(batch: dict[str, list[Any]], task_filter: str | None) -> dict[str, list[Any]] | None:
    if not task_filter:
        return batch
    task_list = batch.get("Task")
    if not task_list:
        return batch
    selected_indices = [index for index, task in enumerate(task_list) if task == task_filter]
    if not selected_indices:
        return None
    return {key: [values[index] for index in selected_indices] for key, values in batch.items()}


def unique_record_key(records: dict[str, dict[str, Any]], patient_id: str, task: str) -> str:
    if patient_id not in records:
        return patient_id
    suffix = 2
    while f"{patient_id}#{task}_{suffix}" in records:
        suffix += 1
    return f"{patient_id}#{task}_{suffix}"


def load_model(model_dir: str, attn_implementation: str | None, device):
    import torch
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    config_kwargs = {}
    if attn_implementation:
        config_kwargs["attn_implementation"] = attn_implementation
    config = Qwen2_5_VLConfig.from_pretrained(model_dir, **config_kwargs)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    model_dir: str,
    data_root: str,
    mode: str,
    split: str,
    output_json: str,
    batch_size: int = 1,
    max_new_tokens: int = 1024,
    attn_implementation: str | None = None,
    device: str | None = None,
    task_filter: str | None = None,
) -> dict[str, float]:
    import torch
    from tqdm import tqdm
    from transformers import AutoProcessor

    from SFT_dataloader import build_dataloader

    random.seed(1234)
    torch.manual_seed(1234)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataloaders = build_dataloader(data_root, batch_size=batch_size, mode=mode)
    data_loader = dataloaders[split]
    if task_filter is None and mode == "full":
        task_filter = "report"
    processor = AutoProcessor.from_pretrained(model_dir, use_fast=True)
    model = load_model(model_dir, attn_implementation, selected_device)

    records: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {split}"):
            batch = filter_batch_by_task(batch, task_filter)
            if batch is None:
                continue
            inputs = batch_process_func(processor, batch, selected_device)
            prompt_length = inputs["input_ids"].shape[1]
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
            for optional_key in ("pixel_values", "image_grid_thw", "video_grid_thw"):
                if optional_key in inputs:
                    generation_kwargs[optional_key] = inputs[optional_key]
            generated_ids = model.generate(**generation_kwargs)
            predictions = processor.batch_decode(
                generated_ids[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for patient_id, task, prediction, reference in zip(
                batch["Patient_ID"], batch.get("Task", ["unknown"] * len(batch["Patient_ID"])), predictions, batch["Text"]
            ):
                record_key = unique_record_key(records, patient_id, task)
                records[record_key] = {
                    "patient_id": patient_id,
                    "task": task,
                    "prediction": prediction.strip(),
                    "reference": reference,
                }

    summary = calculate_metrics(records)
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=4)
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    evaluate_model(
        model_dir=args.model_dir,
        data_root=args.data_root,
        mode=args.mode,
        split=args.split,
        output_json=args.output_json,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
        device=args.device,
        task_filter=args.task_filter,
    )


if __name__ == "__main__":
    main()
