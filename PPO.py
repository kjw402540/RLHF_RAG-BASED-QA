# -*- coding: utf-8 -*-

# ============================================================
# ⚙️ Step 3: PPO (Reinforcement Learning) - 안전 버전
# ============================================================

import os
import json
import random
import torch
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, create_reference_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig

# ============================================================
# 1. 데이터 샘플 비율 설정
# ============================================================
DATA_SAMPLE_RATIO = 0.1

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
        return data.get("data_info", [])

# ============================================================
# 3. 데이터 로드 및 샘플링
# ============================================================
sft_data = sample_data(load_custom_json("./data/train/RLHF_train/SFT.json"), DATA_SAMPLE_RATIO)
rm_data = sample_data(load_custom_json("./data/train/RLHF_train/RM.json"), DATA_SAMPLE_RATIO)
ppo_data = sample_data(load_custom_json("./data/train/RLHF_train/PPO.json"), DATA_SAMPLE_RATIO)

# ============================================================
# 0️⃣ 경로 설정
# ============================================================
output_dir = "./weight/output_rlhf/llama2"
MODEL_NAME = "beomi/llama-2-ko-7b"
SFT_MODEL_PATH = f"{output_dir}/sft_lora"
RM_MODEL_PATH = f"{output_dir}/rm_lora"
PPO_SAVE_PATH = f"{output_dir}/ppo_lora"

# ============================================================
# 1️⃣ Tokenizer 로드
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
# ============================================================
# 3️⃣ SFT 모델 로드 (Policy)
# ============================================================
print("📥 SFT LoRA 모델 로드 중...")
base_policy = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
policy_model = PeftModel.from_pretrained(base_policy, SFT_MODEL_PATH)
policy_model = policy_model.merge_and_unload()
print("✅ Policy 모델 로드 완료")

# ============================================================
# 4️⃣ Reference 모델 생성
# ============================================================
ref_model = create_reference_model(policy_model)
print("✅ Reference 모델 생성 완료")

# ============================================================
# 5️⃣ RM 모델 로드 (Reward Model) 및 value 모델
# ============================================================
print("📥 RM LoRA 모델 로드 중...")
base_rm = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    quantization_config=bnb_config,
    device_map="auto"
)
reward_model = PeftModel.from_pretrained(base_rm, RM_MODEL_PATH)
print("✅ Reward 모델 로드 완료")

value_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    quantization_config=bnb_config,
    device_map="auto"
)

# ============================================================
# 6️⃣ PPO 데이터셋 구성
# ============================================================
# query 컬럼만 남기고 Dataset 생성
texts = [
     d.get("question", "").strip()
     for d in ppo_data
     if "question" in d and d["question"].strip() != ""
]

# 3️⃣ Tokenize
tokenized = tokenizer(
    texts,
    truncation=True,
    padding="max_length",
    max_length=256,  # 필요에 따라 조정 (보통 128~512)
    return_tensors="pt",  # PyTorch 텐서로 반환
)

# 4️⃣ Dataset으로 변환
ppo_dataset = Dataset.from_dict({
    "input_ids": tokenized["input_ids"],
    "attention_mask": tokenized["attention_mask"],
})

# ============================================================
# collate_fn (방어적)
# ============================================================
def collate_fn(batch):
    if len(batch) == 0:
        raise ValueError("Empty batch passed to collate_fn")

    try:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch],
            batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch],
            batch_first=True, padding_value=0
        )
    except Exception as e:
        print("❌ Collate failed on batch:")
        print(batch)
        raise e

    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ============================================================
# 7️⃣ PPO 설정
# ============================================================
ppo_config = PPOConfig(
    batch_size=2,
    mini_batch_size=1,
    learning_rate=1.4e-5,
    report_to="none",
    num_sample_generations=0
)

# ============================================================
# 8️⃣ PPO Trainer 정의
# ============================================================
ppo_trainer = PPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    value_model=value_model,
    reward_model=reward_model,
    processing_class=tokenizer,
    train_dataset=ppo_dataset,
    data_collator=collate_fn,
    args=ppo_config,
)

# ============================================================
# 9️⃣ PPO 학습 루프
# ============================================================
import time
print("🚀 PPO 학습 시작...")
start_time = time.time()

ppo_trainer.train()

end_time = time.time()
print(f"✅ PPO 학습 완료 - Total Time: {end_time - start_time:.2f}s")

# ============================================================
# 🔟 모델 저장
# ============================================================
ppo_trainer.save_model(PPO_SAVE_PATH)
tokenizer.save_pretrained(PPO_SAVE_PATH)
print(f"✅ PPO 모델 저장 완료: {PPO_SAVE_PATH}")