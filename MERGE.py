
# ============================================================
# 9. LoRA Weight 병합 및 최종 모델 저장
# ============================================================
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
import torch

output_dir = "./weight/output_rlhf/llama2"
MODEL_NAME = "beomi/llama-2-ko-7b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

merged_model = AutoModelForCausalLM.from_pretrained(
    f"{output_dir}/ppo_lora",
    torch_dtype=torch.float16,
    device_map="auto"
)

final_path = f"{output_dir}/final_model"
merged_model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

print(f"✅ 최종 모델 저장 완료: {final_path}")

# ============================================================
# 10. 최종 모델 테스트
# ============================================================
from transformers import pipeline

chat = pipeline("text-generation", model=final_path, tokenizer=tokenizer, device_map="auto")
prompt = "텐동이 뭐야?"
result = chat(prompt, max_new_tokens=150, do_sample=True, top_p=0.9)
print(f"💬 프롬프트 : {prompt}")
print("💬 모델 응답:", result[0]["generated_text"])