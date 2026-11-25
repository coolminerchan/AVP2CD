import csv
from tqdm import tqdm
import gc
import numpy as np
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from SFT_dataloader import build_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="SFT train lumbar MRI report generation after stage-one VQA.")
    parser.add_argument("--model-dir", default="VQA_stage_one_SFT/best")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--mode", choices=["full", "partial"], default="full")
    parser.add_argument("--save-dir", default="SFT_after_VQA")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--attn-implementation", default=None, help="Optional, e.g. flash_attention_2.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--skip-generation-eval", action="store_true")
    parser.add_argument("--eval-after-train", action="store_true")
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--eval-output-dir", default="eval")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-max-new-tokens", type=int, default=None)
    parser.add_argument("--pipeline", choices=["none", "stage2_stage3"], default="none")
    parser.add_argument("--stage2-data-root", default="stage2_dataset")
    parser.add_argument("--stage2-mode", choices=["partial"], default="partial")
    parser.add_argument("--stage2-save-dir", default="SFT_stage2_vqa")
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--stage2-lr", type=float, default=None)
    parser.add_argument("--stage3-data-root", default="stage3_dataset")
    parser.add_argument("--stage3-mode", choices=["full"], default="full")
    parser.add_argument("--stage3-model-dir", default=None, help="Defaults to --stage2-save-dir in pipeline mode.")
    parser.add_argument("--stage3-save-dir", default="SFT_stage3_report")
    parser.add_argument("--stage3-epochs", type=int, default=None)
    parser.add_argument("--stage3-lr", type=float, default=None)
    return parser.parse_args()

def load_model(model_dir, attn_implementation=None):
    config_kwargs = {}
    if attn_implementation:
        config_kwargs["attn_implementation"] = attn_implementation
    config = Qwen2_5_VLConfig.from_pretrained(model_dir, **config_kwargs)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16
    )
    print(f"[Info] Loading model from {model_dir}")
    return model

def find_subsequence(sequence, pattern):
    if not pattern or len(pattern) > len(sequence):
        return None
    for index in range(len(sequence) - len(pattern) + 1):
        if sequence[index:index + len(pattern)] == pattern:
            return index
    return None

def batch_process_func(processor, batch, device, is_training=True) -> dict:
    def extract_num_images(message):
        num_images = 0
        for dict in message:
            if dict['role'] == 'user':
                for e in dict['content']:
                    if e['type'] == 'image':
                        num_images += 1
        return num_images

    B = len(batch['Message'])
    tokenizer = processor.tokenizer
    num_images_list = [extract_num_images(messages) for messages in batch['Message']]
    if len(list(set(num_images_list))) != 1:
        raise ValueError(f"Different number of images in different conversations. {num_images_list}")

    messages = copy.deepcopy(batch['Message'])
    if is_training:
        for i in range(B):
            target_text = batch['Text'][i]
            messages[i].append({"role": "assistant", "content": target_text})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=not is_training)

    image_inputs, video_inputs = process_vision_info(conversations=messages, return_video_kwargs=False)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side='left',
        return_tensors="pt",
    )

    for key in inputs:
        if hasattr(inputs[key], 'to'):
            inputs[key] = inputs[key].to(device)

    if is_training:
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        pad_token_id = tokenizer.pad_token_id
        target_token_ids = [
            tokenizer(target_text, add_special_tokens=False)["input_ids"]
            for target_text in batch["Text"]
        ]
        for i in range(B):
            sample_ids = input_ids[i].detach().cpu().tolist()
            target_start = find_subsequence(sample_ids, target_token_ids[i])
            if target_start is None:
                assistant_start_token = tokenizer("<|im_start|>assistant", add_special_tokens=False)["input_ids"]
                target_start = find_subsequence(sample_ids, assistant_start_token)
            if target_start is None:
                raise ValueError("Assistant response not found in the input text.")
            labels[i, :target_start] = -100
            if pad_token_id is not None:
                labels[i, input_ids[i] == pad_token_id] = -100

        inputs['labels'] = labels

    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            if hasattr(inputs[key], 'to'):
                inputs[key] = inputs[key].to(device)
    return inputs

def compute_loss(model, inputs):
    """Compute training loss"""
    outputs = model(**inputs)
    return outputs.loss

import re

try:
    from nltk.translate import meteor_score
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    meteor_score = None
    sentence_bleu = None
    SmoothingFunction = None

def simple_tokenize(text):
    return re.findall(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)?|[^\s]", text.lower())

def safe_bleu(pred_tokens, ref_tokens):
    if sentence_bleu is None:
        return 0.0
    return sentence_bleu(
        [ref_tokens],
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1
    )

def safe_meteor(pred_tokens, ref_tokens):
    if meteor_score is None:
        return 0.0
    try:
        return meteor_score.meteor_score([ref_tokens], pred_tokens)
    except LookupError:
        return 0.0

def write_metric_csv(path, rows):
    if isinstance(rows, dict):
        keys = list(rows.keys())
        row_count = max((len(value) for value in rows.values()), default=0)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for index in range(row_count):
                writer.writerow({key: rows[key][index] if index < len(rows[key]) else "" for key in keys})
    else:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_loss"])
            writer.writeheader()
            for value in rows:
                writer.writerow({"train_loss": value})

def validate_model(model, processor, val_loader, accelerator, max_new_tokens=1024, skip_generation_eval=False):
    """Validation step with comprehensive metrics"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_losses = []
    all_predictions = []
    all_references = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", disable=not accelerator.is_main_process):
            inputs = batch_process_func(processor, batch, accelerator.device, True)
            loss = compute_loss(model, inputs)

            # Gather losses from all GPUs
            gathered_loss = accelerator.gather(loss)
            batch_loss = gathered_loss.mean().item()

            total_loss += batch_loss
            all_losses.append(batch_loss)
            num_batches += 1

            if skip_generation_eval:
                continue

            processed_batch = batch_process_func(processor, batch, accelerator.device, False)
            prompt_length = processed_batch["input_ids"].shape[1]

            generated_ids = accelerator.unwrap_model(model).generate(
                    input_ids=processed_batch["input_ids"],
                    attention_mask=processed_batch["attention_mask"],
                    pixel_values=processed_batch["pixel_values"],
                    image_grid_thw=processed_batch["image_grid_thw"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )

            generated_texts = processor.batch_decode(
                generated_ids[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            assistant_predictions = [text.strip() for text in generated_texts]
            true_reports = batch['Text']

            all_predictions.extend(assistant_predictions)
            all_references.extend(true_reports)

    torch.cuda.empty_cache()
    gc.collect()
    avg_loss = total_loss / num_batches

    if skip_generation_eval:
        return avg_loss, 0.0, 0.0

    assert len(all_predictions) == len(all_references)
    bleu_scores = []
    meteor_scores = []
    for pred, ref in zip(all_predictions, all_references):
        pred_tokens = simple_tokenize(pred)
        ref_tokens = simple_tokenize(ref)
        bleu_scores.append(safe_bleu(pred_tokens, ref_tokens))
        meteor_scores.append(safe_meteor(pred_tokens, ref_tokens))
    return avg_loss, np.mean(bleu_scores), np.mean(meteor_scores)

class Trainer:
    def __init__(self, args):
        self.model_dir = args.model_dir
        self.data_root = args.data_root
        self.mode = args.mode
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.num_epochs = args.epochs
        self.save_dir = args.save_dir
        self.attn_implementation = args.attn_implementation
        self.max_new_tokens = args.max_new_tokens
        self.skip_generation_eval = args.skip_generation_eval
        self.eval_after_train = args.eval_after_train
        self.eval_split = args.eval_split
        self.eval_output_dir = args.eval_output_dir
        self.eval_batch_size = args.eval_batch_size
        self.eval_max_new_tokens = args.eval_max_new_tokens or args.max_new_tokens

def run_training(config):
    model = load_model(config.model_dir, config.attn_implementation)
    accelerator = Accelerator()
    set_seed(1234)

    dataloaders = build_dataloader(config.data_root, config.batch_size, mode=config.mode)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    processor = AutoProcessor.from_pretrained(config.model_dir, use_fast = True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    global_step = 0
    train = []
    val = {
        "loss": [],
        "bleu4": [],
        "meteor": []
    }

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config.num_epochs}",
            disable=not accelerator.is_main_process
        )

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Generate inputs
                inputs = batch_process_func(processor, batch, accelerator.device, True)

                # Forward pass
                loss = compute_loss(model, inputs)

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            current_loss = loss.item()
            total_loss += current_loss
            num_batches += 1

            if accelerator.sync_gradients:
                global_step += 1
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'step': global_step
            })
            train.append(avg_loss)
        val_loss, bleu, meteor = validate_model(
            model,
            processor,
            val_loader,
            accelerator,
            max_new_tokens=config.max_new_tokens,
            skip_generation_eval=config.skip_generation_eval,
        )

        if accelerator.is_main_process:
            accelerator.print(f"Train Loss: {total_loss / num_batches:.4f}")
            accelerator.print(f"Validation at step {global_step}:")
            accelerator.print(f"Val Loss: {val_loss:.4f}")
            val['loss'].append(val_loss)
            val['bleu4'].append(bleu)
            val['meteor'].append(meteor)
        gc.collect()
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)

        # Save model using Hugging Face's save_pretrained
        unwrapped_model.save_pretrained(
            config.save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

        # Save processor
        processor.save_pretrained(config.save_dir)

        accelerator.print(f"Checkpoint saved to {config.save_dir}!")
        accelerator.print("Training finished!")

        write_metric_csv(f"{config.save_dir}_train_loss.csv", train)
        write_metric_csv(f"{config.save_dir}_val_loss.csv", val)

    should_eval = config.eval_after_train and accelerator.is_main_process
    eval_kwargs = {
        "model_dir": config.save_dir,
        "data_root": config.data_root,
        "mode": config.mode,
        "split": config.eval_split,
        "output_json": str(Path(config.eval_output_dir) / f"{Path(config.save_dir).name}_{config.eval_split}.json"),
        "batch_size": config.eval_batch_size,
        "max_new_tokens": config.eval_max_new_tokens,
        "attn_implementation": config.attn_implementation,
    }
    accelerator.wait_for_everyone()
    del model, optimizer, train_loader, val_loader, processor
    gc.collect()
    torch.cuda.empty_cache()
    if should_eval:
        from model_evaluation import evaluate_model

        evaluate_model(**eval_kwargs)

def pipeline_configs(args):
    stage2_epochs = args.stage2_epochs if args.stage2_epochs is not None else args.epochs
    stage3_epochs = args.stage3_epochs if args.stage3_epochs is not None else args.epochs
    stage2_lr = args.stage2_lr if args.stage2_lr is not None else args.lr
    stage3_lr = args.stage3_lr if args.stage3_lr is not None else args.lr
    stage3_model_dir = args.stage3_model_dir or args.stage2_save_dir

    common = {
        "batch_size": args.batch_size,
        "attn_implementation": args.attn_implementation,
        "max_new_tokens": args.max_new_tokens,
        "skip_generation_eval": args.skip_generation_eval,
        "eval_after_train": args.eval_after_train,
        "eval_split": args.eval_split,
        "eval_output_dir": args.eval_output_dir,
        "eval_batch_size": args.eval_batch_size,
        "eval_max_new_tokens": args.eval_max_new_tokens,
    }
    return [
        Trainer(
            SimpleNamespace(
                model_dir=args.model_dir,
                data_root=args.stage2_data_root,
                mode=args.stage2_mode,
                save_dir=args.stage2_save_dir,
                lr=stage2_lr,
                epochs=stage2_epochs,
                **common,
            )
        ),
        Trainer(
            SimpleNamespace(
                model_dir=stage3_model_dir,
                data_root=args.stage3_data_root,
                mode=args.stage3_mode,
                save_dir=args.stage3_save_dir,
                lr=stage3_lr,
                epochs=stage3_epochs,
                **common,
            )
        ),
    ]

def train():
    args = parse_args()
    if args.pipeline == "stage2_stage3":
        for stage_config in pipeline_configs(args):
            run_training(stage_config)
    else:
        run_training(Trainer(args))

if __name__ == "__main__":
    train()
