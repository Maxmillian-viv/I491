import os
import ast
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List
from collections import Counter

import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

DATA_ROOT = "/home/chenchuang_5090/chenchuang/I491/custom_dataset/custom_dataset"
TRAIN_CSV = "/home/chenchuang_5090/chenchuang/I491/custom_dataset/custom_dataset/processed/train_proc.csv"
VAL_CSV = "/home/chenchuang_5090/chenchuang/I491/custom_dataset/custom_dataset/processed/val_proc.csv"
TRAIN_IMG_ROOT = os.path.join(DATA_ROOT, "train")
TEST_CSV = os.path.join(DATA_ROOT, "test_non_labels.csv")
TEST_IMG_ROOT = os.path.join(DATA_ROOT, "test")
OUTPUT_DIR = "./qwen2vl_clevr_lora"
SUBMISSION_PATH = "./submission.csv"


@dataclass
class Config:
    num_epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    val_max_samples: int = 800
    ensemble_size: int = 5
    gen_max_new_tokens: int = 128
    gen_temperature: float = 0.4
    gen_top_p: float = 0.9
    seed: int = 42


CFG = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CLEVRXTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, processor, max_explanation_len: int = 256):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.processor = processor
        self.max_explanation_len = max_explanation_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["file"])
        image = Image.open(img_path).convert("RGB")
        question = row["question"]
        answer = str(row["answer"]).strip()
        exp_list = ast.literal_eval(row["explanation"])
        explanation = random.choice(exp_list).strip()
        if len(explanation) > self.max_explanation_len:
            explanation = explanation[: self.max_explanation_len]
        user_text = (
            "You are a visual reasoning assistant for synthetic CLEVR-style scenes.\n"
            "Carefully look at the image and answer the question.\n\n"
            f"Question: {question}\n\n"
            "First give a short answer, then a detailed reasoning explanation.\n"
            "Answer MUST be a single word or integer token from the dataset vocabulary.\n"
            "Do NOT add units or extra words in the Answer line.\n"
            "Format:\n"
            "Answer: <short answer>\n"
            "Explanation: <step-by-step reasoning>"
        )
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]
        assistant_text = f"Answer: {answer}\nExplanation: {explanation}"
        prompt_messages = [
            {"role": "user", "content": user_content},
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_messages = [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {
            "image": image,
            "full_text": full_text,
            "prompt_text": prompt_text,
        }


@dataclass
class QwenVLDataCollator:
    processor: Any

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        full_texts = [item["full_text"] for item in batch]
        prompt_texts = [item["prompt_text"] for item in batch]
        enc_full = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            enc_prompt = self.processor(
                text=prompt_texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
        input_ids = enc_full["input_ids"]
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels = input_ids.clone()
        labels[labels == pad_token_id] = -100
        prompt_ids = enc_prompt["input_ids"]
        for i in range(input_ids.size(0)):
            prompt_len = (prompt_ids[i] != pad_token_id).sum()
            if prompt_len > 0:
                labels[i, :prompt_len] = -100
        enc_full["labels"] = labels
        return enc_full


def parse_answer_explanation(text: str):
    import re

    def normalize_answer(ans: str) -> str:
        ans = ans.strip().lower()
        ans = ans.splitlines()[0].strip()
        ans = re.split(r"\bexplanation\s*:", ans, flags=re.IGNORECASE)[0]
        ans = re.sub(r"[.,!?]", " ", ans)
        ans = ans.replace("(", " ").replace(")", " ")
        ans = re.sub(r"\s+", " ", ans).strip()
        if not ans:
            return ""
        tokens = ans.split()
        head = tokens[0]
        word2num = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        if head in word2num:
            head = word2num[head]
        color_map = {
            "grey": "gray",
            "light grey": "gray",
            "dark grey": "gray",
        }
        if ans in color_map:
            head = color_map[ans]
        elif head in color_map:
            head = color_map[head]
        return head

    ans = ""
    exp = ""
    m_ans = re.search(r"answer\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    m_exp = re.search(r"explanation\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if m_ans:
        ans_raw = m_ans.group(1)
        ans_raw = re.split(r"\n\s*explanation\s*:", ans_raw, flags=re.IGNORECASE)[0]
        ans = normalize_answer(ans_raw)
    if not ans:
        m_alt = re.search(
            r"(?:the\s+answer\s+is|answer)\s+([a-z0-9\-]+)",
            text,
            re.IGNORECASE,
        )
        if m_alt:
            ans = normalize_answer(m_alt.group(1))
    if m_exp:
        exp = m_exp.group(1).strip()
    else:
        exp = text.strip()
    return ans, exp


def load_base_processor_and_model(device):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=compute_dtype,
    ).to(device)
    return processor, model


def apply_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def generate_answer_once(
    image_path: str,
    question: str,
    processor,
    model,
    device,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.4,
    top_p: float = 0.9,
    seed: int = None,
):
    image = Image.open(image_path).convert("RGB")
    user_text = (
        "You are a visual reasoning assistant for synthetic CLEVR-style scenes.\n"
        "Carefully look at the image and answer the question.\n\n"
        f"Question: {question}\n\n"
        "First give a short answer, then a detailed reasoning explanation.\n"
        "Answer MUST be a single word or integer token from the dataset vocabulary.\n"
        "Do NOT add units or extra words in the Answer line.\n"
        "Format:\n"
        "Answer: <short answer>\n"
        "Explanation: <step-by-step reasoning>"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    ).to(device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs.update(
            {
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
        )
    if seed is not None:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
    generated_ids = model.generate(
        **inputs,
        **gen_kwargs,
    )
    gen_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
    output_text = processor.batch_decode(
        gen_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    ans, exp = parse_answer_explanation(output_text)
    return ans, exp, output_text


def evaluate_on_val(cfg: Config, model, processor, val_df: pd.DataFrame, img_dir: str, device) -> float:
    model.eval()
    if len(val_df) == 0:
        return 0.0
    if cfg.val_max_samples is not None and len(val_df) > cfg.val_max_samples:
        eval_df = val_df.sample(
            n=cfg.val_max_samples,
            random_state=cfg.seed,
        ).reset_index(drop=True)
    else:
        eval_df = val_df.reset_index(drop=True)
    correct = 0
    total = len(eval_df)
    with torch.no_grad():
        for _, row in tqdm(
            eval_df.iterrows(),
            total=len(eval_df),
            desc="Val evaluating",
        ):
            img_path = os.path.join(img_dir, row["file"])
            question = row["question"]
            gt_answer = str(row["answer"]).strip().lower()
            pred_ans, _, _ = generate_answer_once(
                image_path=img_path,
                question=question,
                processor=processor,
                model=model,
                device=device,
                max_new_tokens=cfg.gen_max_new_tokens,
                do_sample=False,
            )
            if isinstance(pred_ans, str) and pred_ans.strip() != "":
                if pred_ans.strip().lower() == gt_answer:
                    correct += 1
    acc = correct / max(1, total)
    print(f"[Val] accuracy: {acc:.4f} ({correct}/{total})")
    model.train()
    return acc


def train_lora_model(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    processor, base_model = load_base_processor_and_model(device)
    model = apply_lora(base_model)
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    train_dataset = CLEVRXTrainDataset(train_df, TRAIN_IMG_ROOT, processor)
    collator = QwenVLDataCollator(processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.grad_accum_steps)
    max_train_steps = cfg.num_epochs * num_update_steps_per_epoch
    print("Steps per epoch:", num_update_steps_per_epoch)
    print("Total train steps:", max_train_steps)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=max_train_steps,
    )
    best_val_acc = -1.0
    model.train()
    global_step = 0
    for epoch in range(cfg.num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
        )
        optimizer.zero_grad()
        for step, batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            running_loss += loss.item() * cfg.grad_accum_steps
            if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            progress_bar.set_description(
                f"Epoch {epoch+1}/{cfg.num_epochs} | Step {global_step} | Loss {running_loss / (step+1):.4f}"
            )
        val_acc = evaluate_on_val(
            cfg=cfg,
            model=model,
            processor=processor,
            val_df=val_df,
            img_dir=TRAIN_IMG_ROOT,
            device=device,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(
                f"[Epoch {epoch+1}] New best val acc: {val_acc:.4f}, saving LoRA to {OUTPUT_DIR}"
            )
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
    if best_val_acc < 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        print("Saved LoRA model to", OUTPUT_DIR)


def load_for_inference(output_dir: str, device):
    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=compute_dtype,
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    adapter_config_path = os.path.join(output_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print("Loading LoRA adapter from", output_dir)
        model = PeftModel.from_pretrained(base_model, output_dir)
    else:
        print("No LoRA adapter found in", output_dir, "- using base model only.")
        model = base_model
    model.eval()
    return processor, model


def run_inference(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    test_df = pd.read_csv(TEST_CSV)
    processor, model = load_for_inference(OUTPUT_DIR, device)
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        sample_id = row["id"]
        img_path = os.path.join(TEST_IMG_ROOT, row["file"])
        question = row["question"]
        answers = []
        explanations = []
        for k in range(max(1, cfg.ensemble_size)):
            do_sample = cfg.ensemble_size > 1
            seed = cfg.seed + k
            ans_k, exp_k, _ = generate_answer_once(
                image_path=img_path,
                question=question,
                processor=processor,
                model=model,
                device=device,
                max_new_tokens=cfg.gen_max_new_tokens,
                do_sample=do_sample,
                temperature=cfg.gen_temperature,
                top_p=cfg.gen_top_p,
                seed=seed,
            )
            answers.append(ans_k)
            explanations.append(exp_k)
        valid_answers = [a for a in answers if isinstance(a, str) and a.strip() != ""]
        if len(valid_answers) == 0:
            final_ans = answers[0] if answers else ""
            final_exp = explanations[0] if explanations else ""
        else:
            counter = Counter(valid_answers)
            final_ans = None
            best_count = -1
            best_index = len(answers)
            for a, c in counter.items():
                first_idx = answers.index(a)
                if c > best_count or (c == best_count and first_idx < best_index):
                    best_count = c
                    best_index = first_idx
                    final_ans = a
            final_exp = ""
            for a, e in zip(answers, explanations):
                if a == final_ans and isinstance(e, str) and e.strip():
                    final_exp = e
                    break
            if not final_exp:
                final_exp = explanations[0] if explanations else ""
        results.append(
            {
                "id": sample_id,
                "answer": final_ans,
                "explanation": final_exp,
            }
        )
    submission = pd.DataFrame(results)[["id", "answer", "explanation"]]
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Saved submission to", SUBMISSION_PATH)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")
    args = parser.parse_args()

    set_seed(CFG.seed)

    if args.train:
        train_lora_model(CFG)

    if args.infer:
        run_inference(CFG)

    if not args.train and not args.infer:
        train_lora_model(CFG)
        run_inference(CFG)


if __name__ == "__main__":
    main()
