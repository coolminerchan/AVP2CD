import argparse
import gc
import json
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SYSTEM_PROMPT = (
    "You are a radiologist specializing in lumbar spine diseases and are good at "
    "writing structured and standardized radiology diagnostic reports based on MRI images."
)
REPORT_FORMAT_INSTRUCTION = (
    "\nThe output format must strictly adhere to the following:\n"
    "Description: [Detailed description of scan findings and imaging features here]\n"
    "Impression: [Summary of diagnostic impressions, conclusions, and key findings here]"
)


def build_parser(default_loss_type: str, default_output_dir: str, default_epochs: int, default_beta: float) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{default_loss_type.upper()} train for Stage 2 and Stage 3.")
    parser.add_argument("--model-folder", default=os.environ.get("MODEL_FOLDER", "SFT_after_VQA"))
    parser.add_argument("--data-root", default=os.environ.get("RL_DATASET", "rl_dataset"))
    parser.add_argument("--mode", choices=["full", "partial"], default="full")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output-dir", default=os.environ.get("RL_OUTPUT_DIR", default_output_dir))
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--beta", type=float, default=default_beta)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--report-to", default="none", choices=["wandb", "none"])
    parser.add_argument("--wandb-project", default=f"{default_loss_type}-lumbar-mri")
    parser.add_argument("--wandb-run-name", default=f"sft-then-{default_loss_type}")
    parser.add_argument("--eval-after-train", action="store_true")
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--eval-output-dir", default="eval")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-max-new-tokens", type=int, default=None)
    parser.add_argument("--pipeline", choices=["none", "stage2_stage3"], default="none")
    parser.add_argument("--stage2-model-folder", default=None, help="Defaults to --model-folder in pipeline mode.")
    parser.add_argument("--stage2-data-root", default="stage2_dataset")
    parser.add_argument("--stage2-mode", choices=["partial"], default="partial")
    parser.add_argument("--stage2-output-dir", default=f"{default_output_dir}_stage2")
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument(
        "--stage3-model-folder",
        default=None,
        help="Defaults to SFT_stage3_report if present, otherwise the Stage 2 RL output.",
    )
    parser.add_argument("--stage3-data-root", default="stage3_dataset")
    parser.add_argument("--stage3-mode", choices=["full"], default="full")
    parser.add_argument("--stage3-output-dir", default=f"{default_output_dir}_stage3")
    parser.add_argument("--stage3-epochs", type=int, default=None)
    parser.set_defaults(loss_type=default_loss_type)
    return parser


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        content = completion[0].get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def answer_text(answer: Any) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list) and answer:
        content = answer[0].get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def load_image(path: str) -> Any:
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        return image.convert("RGB").resize((448, 448))


def split_root(data_root: str, split: str) -> str:
    root = Path(data_root)
    split_path = root / split
    if split_path.is_dir():
        return str(split_path)
    return str(root)


def append_format_instruction(text: str, mode: str) -> str:
    if mode != "full" or "Description:" in text and "Impression:" in text:
        return text
    return text + REPORT_FORMAT_INSTRUCTION


def create_dataset_from_messages(data_root: str, mode: str, split: str) -> Any:
    from datasets import Dataset
    from SFT_dataloader import make_dataset

    random.seed(1234)
    source_dataset = make_dataset(split_root(data_root, split), mode)
    processed_data = {"prompt": [], "images": [], "answers": []}

    for item in source_dataset:
        prompt = []
        images = []
        for message in item["Message"]:
            if message["role"] == "system":
                content = message["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                prompt.append({"role": "system", "content": content})
                continue

            if message["role"] != "user":
                continue

            user_content = []
            for part in message["content"]:
                if part["type"] == "image":
                    images.append(load_image(part["image"]))
                    user_content.append({"type": "image"})
                elif part["type"] == "text":
                    user_content.append({"type": "text", "text": append_format_instruction(part["text"], mode)})
            prompt.append({"role": "user", "content": user_content})

        processed_data["prompt"].append(prompt)
        processed_data["images"].append(images)
        processed_data["answers"].append(
            [{"role": "assistant", "content": [{"type": "text", "text": item["Text"]}]}]
        )

    print(f"dataset length: {len(processed_data['prompt'])}")
    return Dataset.from_dict(processed_data)


def create_dataset_from_json(dataset_path: str, mode: str) -> Any:
    from datasets import Dataset

    json_path = os.path.join(dataset_path, "dataset.json")
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    processed_data = {"prompt": [], "images": [], "answers": []}
    for index, item in enumerate(data):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("problem"), str)
            or not isinstance(item.get("answer"), str)
            or not isinstance(item.get("images"), list)
            or not all(isinstance(path, str) for path in item["images"])
        ):
            raise ValueError(f"Dataset item {index} must have a problem, an answer, and a list of image paths.")

        user_content = [{"type": "image"} for _ in item["images"]]
        user_content.append({"type": "text", "text": append_format_instruction(item["problem"], mode)})
        prompt = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]

        image_paths = []
        for image_index, image_path in enumerate(item["images"]):
            full_image_path = os.path.join(dataset_path, image_path)
            if not os.path.isfile(full_image_path):
                raise FileNotFoundError(f"Dataset item {index}, image {image_index} is missing.")
            image_paths.append(load_image(full_image_path))

        processed_data["prompt"].append(prompt)
        processed_data["images"].append(image_paths)
        processed_data["answers"].append(
            [{"role": "assistant", "content": [{"type": "text", "text": item["answer"]}]}]
        )

    print(f"dataset length: {len(processed_data['prompt'])}")
    return Dataset.from_dict(processed_data)


def create_dataset(data_root: str, mode: str, split: str) -> Any:
    selected_root = split_root(data_root, split)
    if os.path.isfile(os.path.join(selected_root, "dataset.json")):
        return create_dataset_from_json(selected_root, mode)
    return create_dataset_from_messages(data_root, mode, split)


def init_wandb(config: SimpleNamespace, algorithm: str) -> None:
    if config.report_to == "none":
        return
    import wandb

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            "algorithm": algorithm,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "num_generations": config.num_generations,
            "mode": config.mode,
            "beta": config.beta,
        },
    )


def build_train_config(config: SimpleNamespace) -> Any:
    from trl import GRPOConfig

    kwargs = {
        "beta": config.beta,
        "bf16": True,
        "loss_type": config.loss_type,
        "output_dir": config.output_dir,
        "learning_rate": config.learning_rate,
        "num_generations": config.num_generations,
        "num_train_epochs": config.epochs,
        "temperature": config.temperature,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
        "report_to": config.report_to,
        "remove_unused_columns": False,
        "logging_steps": 5,
        "save_only_model": True,
        "save_safetensors": True,
        "save_strategy": "no",
        "reward_weights": [0.405, 0.395, 0.2],
    }
    if config.loss_type == "dapo":
        kwargs["epsilon_high"] = 0.28
        kwargs["mask_truncated_completions"] = True
    return GRPOConfig(**kwargs)


def run_rl_stage(config: SimpleNamespace, algorithm: str) -> None:
    import torch
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from trl import GRPOTrainer

    from reward_fn import BertScorer, ClinicalReward

    init_wandb(config, algorithm)
    dataset = create_dataset(config.data_root, config.mode, config.split)
    print(f"successfully created dataset from {config.data_root} ({config.mode}, {config.split})")

    clinical_reward = ClinicalReward()
    bert_scorer = BertScorer(batch_size=1)

    def format_reward_func(completions, answers, **kwargs):
        rewards = []
        pattern = r"^Description:\s*(.*?)\s*Impression:\s*(.*?)$"
        for completion in completions:
            pred = completion_text(completion).strip()
            if config.mode == "full":
                rewards.append(1.0 if re.match(pattern, pred, flags=re.S) else 0.0)
            else:
                rewards.append(1.0 if pred else 0.0)
        return rewards

    def clinical_reward_func(completions, answers, **kwargs):
        rewards = []
        for completion, answer in zip(completions, answers):
            pred = completion_text(completion)
            ref = answer_text(answer)
            reward = clinical_reward.clinical_reward(ref, pred)
            rewards.append(reward if reward > 0.5 else 0.0)
        return rewards

    def bert_reward_func(completions, answers, **kwargs):
        rewards = []
        for completion, answer in zip(completions, answers):
            pred = completion_text(completion)
            ref = answer_text(answer)
            rewards.extend(bert_scorer([ref], [pred]))
        return rewards

    model_config = Qwen2_5_VLConfig.from_pretrained(config.model_folder)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_folder, config=model_config, torch_dtype=torch.bfloat16
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[clinical_reward_func, bert_reward_func, format_reward_func],
        args=build_train_config(config),
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    should_eval = trainer.accelerator.is_main_process and config.eval_after_train
    eval_kwargs = {
        "model_dir": config.output_dir,
        "data_root": config.data_root,
        "mode": config.mode,
        "split": config.eval_split,
        "output_json": str(Path(config.eval_output_dir) / f"{Path(config.output_dir).name}_{config.eval_split}.json"),
        "batch_size": config.eval_batch_size,
        "max_new_tokens": config.eval_max_new_tokens or config.max_completion_length,
    }
    trainer.accelerator.wait_for_everyone()
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    if should_eval:
        from model_evaluation import evaluate_model

        evaluate_model(**eval_kwargs)
    if config.report_to != "none":
        import wandb

        wandb.finish()


def pipeline_configs(args: argparse.Namespace) -> list[SimpleNamespace]:
    stage2_epochs = args.stage2_epochs if args.stage2_epochs is not None else args.epochs
    stage3_epochs = args.stage3_epochs if args.stage3_epochs is not None else args.epochs
    stage2_model_folder = args.stage2_model_folder or args.model_folder
    if args.stage3_model_folder:
        stage3_model_folder = args.stage3_model_folder
    elif Path("SFT_stage3_report").exists():
        stage3_model_folder = "SFT_stage3_report"
    else:
        stage3_model_folder = args.stage2_output_dir
    common = {
        "loss_type": args.loss_type,
        "learning_rate": args.learning_rate,
        "num_generations": args.num_generations,
        "beta": args.beta,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "report_to": args.report_to,
        "split": args.split,
        "wandb_project": args.wandb_project,
        "eval_after_train": args.eval_after_train,
        "eval_split": args.eval_split,
        "eval_output_dir": args.eval_output_dir,
        "eval_batch_size": args.eval_batch_size,
        "eval_max_new_tokens": args.eval_max_new_tokens,
    }
    return [
        SimpleNamespace(
            model_folder=stage2_model_folder,
            data_root=args.stage2_data_root,
            mode=args.stage2_mode,
            output_dir=args.stage2_output_dir,
            epochs=stage2_epochs,
            wandb_run_name=f"{args.wandb_run_name}-stage2",
            **common,
        ),
        SimpleNamespace(
            model_folder=stage3_model_folder,
            data_root=args.stage3_data_root,
            mode=args.stage3_mode,
            output_dir=args.stage3_output_dir,
            epochs=stage3_epochs,
            wandb_run_name=f"{args.wandb_run_name}-stage3",
            **common,
        ),
    ]


def single_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_folder=args.model_folder,
        data_root=args.data_root,
        mode=args.mode,
        split=args.split,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        num_generations=args.num_generations,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        report_to=args.report_to,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        loss_type=args.loss_type,
        eval_after_train=args.eval_after_train,
        eval_split=args.eval_split,
        eval_output_dir=args.eval_output_dir,
        eval_batch_size=args.eval_batch_size,
        eval_max_new_tokens=args.eval_max_new_tokens,
    )


def main(default_loss_type: str, default_output_dir: str, default_epochs: int, default_beta: float) -> None:
    parser = build_parser(default_loss_type, default_output_dir, default_epochs, default_beta)
    args = parser.parse_args()
    if args.pipeline == "stage2_stage3":
        for config in pipeline_configs(args):
            run_rl_stage(config, default_loss_type)
    else:
        run_rl_stage(single_config(args), default_loss_type)
