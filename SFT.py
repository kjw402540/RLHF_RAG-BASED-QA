# -*- coding: utf-8 -*-

# ============================================================
# 🧠 한국어 LLaMA 기반 RLHF 전체 파이프라인 (Colab용)
# - 단계: SFT → RM → PPO
# - 기능: LoRA 학습 + 병합 + 최종 모델 저장
# - 데이터 샘플링 비율 적용 가능
# ============================================================

import os
import json
import random
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, RewardTrainer, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from transformers import pipeline

# ============================================================
# 0. 기본 설정
# ============================================================
BASE_MODEL = "beomi/llama-2-ko-7b"
device_map = "auto"
output_dir = "./weight/output_rlhf/llama2"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 1. 데이터 샘플 비율 설정
# ============================================================
DATA_SAMPLE_RATIO = 0.5  # 기본 5%, 필요 시 조절

def sample_data(data_list, ratio=0.05, seed=42):
    random.seed(seed)
    n_sample = max(1, int(len(data_list) * ratio))
    return random.sample(data_list, n_sample)

# ============================================================
# 2. JSON 파일 로드 함수
# ============================================================
def load_custom_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data["data_info"]

# ============================================================
# 3. 데이터 로드 및 샘플링
# ============================================================
sft_data = sample_data(load_custom_json("./data/train/RLHF_train/SFT.json"), DATA_SAMPLE_RATIO)
rm_data = sample_data(load_custom_json("./data/train/RLHF_train/RM.json"), DATA_SAMPLE_RATIO)
ppo_data = sample_data(load_custom_json("./data/train/RLHF_train/PPO.json"), DATA_SAMPLE_RATIO)

# ============================================================
# 4. 토크나이저 및 모델 로드
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map=device_map
)

# ============================================================
# 5. LoRA 구성
# ============================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
base_model = get_peft_model(base_model, lora_config)

# ============================================================
# 6. Step 1: SFT (Supervised Fine-Tuning)
# ============================================================
sft_dataset_formatted = Dataset.from_list([
    {"text": f"### 질문:\n{d['question']}\n\n### 답변:\n{d['answer']['contents']}"}
    for d in sft_data
])

sft_trainer = SFTTrainer(
    model=base_model,
    train_dataset=sft_dataset_formatted,
    args=TrainingArguments(
        output_dir=f"{output_dir}/sft",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        fp16=True,
        report_to="none"
    )
)
sft_trainer.train()
sft_model = sft_trainer.model
sft_model.save_pretrained(f"{output_dir}/sft_lora")

