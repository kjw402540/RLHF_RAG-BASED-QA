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
DATA_SAMPLE_RATIO = 0.1  # 기본 5%, 필요 시 조절

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

from peft import LoraConfig, get_peft_model

# ============================================================
# LoRA 설정
# ============================================================
rm_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # LLaMA에서 학습 가능한 모듈
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)


rm_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=1,
    quantization_config=bnb_config,
    device_map=device_map
)

rm_model = get_peft_model(rm_model, rm_lora_config)

rm_dataset_list = []
for d in rm_data:
    question = d["question"]
    answers = []
    for key in d:
        if key.startswith("answer") and isinstance(d[key], dict):
            ans_info = d[key]
            if "contents" in ans_info and "ranking" in ans_info:
                try:
                    ranking = int(ans_info["ranking"])
                    answers.append((ans_info["contents"], ranking))
                except (ValueError, TypeError):
                    continue
    # 모든 가능한 chosen/rejected 쌍 생성
    for i in range(len(answers)):
        for j in range(len(answers)):
            if i != j and answers[i][1] < answers[j][1]:
                rm_dataset_list.append({
                    "chosen": question + " " + answers[i][0],  # 질문 + 답변
                    "rejected": question + " " + answers[j][0]
                })

rm_dataset = Dataset.from_list(rm_dataset_list)
from trl import RewardTrainer, RewardConfig

# ============================================================
# RM 학습용 RewardConfig 정의
# ============================================================
rm_config = RewardConfig(
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    fp16=False,
    output_dir=f"{output_dir}/rm",
    save_strategy="no",
    report_to="none"
)

# ============================================================
# RewardTrainer 생성 및 학습
# ============================================================
reward_trainer = RewardTrainer(
    model=rm_model,         # 이미 로드된 RM 모델 전달
    train_dataset=rm_dataset,
    args=rm_config          # TrainingArguments 대신 RewardConfig 사용
)

reward_trainer.train()

# 학습 완료 후 RM 모델 저장
rm_model.save_pretrained(f"{output_dir}/rm_lora")