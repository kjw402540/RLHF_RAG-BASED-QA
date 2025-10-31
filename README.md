🧠 RLHF-RAG-BASED-QA
Reinforcement Learning from Human Feedback (RLHF) 기반의 RAG(Retrieval-Augmented Generation) QA 모델 학습 파이프라인입니다.
Supervised Fine-Tuning(SFT) → Reward Model(RM) → PPO → LoRA Merge 단계를 통해 Llama 기반 모델을 인간 피드백에 맞게 정제합니다.

📂 Project Structure
RLHF-RAG-BASED-QA/
├── SFT.py # Supervised Fine-Tuning (지도 학습)
├── RM.py # Reward Model 학습 (응답 품질 평가 모델)
├── PPO.py # Reinforcement Learning fine-tuning (PPO 알고리즘)
├── MERGE.py # LoRA 병합 및 최종 모델 저장
└── README.md

🚀 Pipeline Overview
SFT (Supervised Fine-Tuning)
Human-labeled dataset으로 기본 언어모델을 지도학습.
모델이 주어진 질문에 더 일관적이고 자연스럽게 답변하도록 학습합니다.

RM (Reward Model)
SFT 모델이 생성한 응답 중 “더 인간적인 응답”을 식별할 수 있도록 보상모델을 학습합니다.

PPO (Reinforcement Learning Fine-tuning)
PPO(Proximal Policy Optimization) 알고리즘을 통해
보상모델의 피드백을 바탕으로 언어모델을 강화학습합니다.

MERGE (LoRA Merge & Save)
LoRA 어델터를 원본 모델에 병합하여 최종 모델을 저장합니다.

🧩 Example Models
Base model: meta-llama/Llama-3.2-1B
Tokenizer: HuggingFace Transformers 기반
Quantization: BitsAndBytes (4bit, nf4)

📜 License
This project is for research and educational purposes.
All pretrained models are from publicly available sources (e.g., Hugging Face).

