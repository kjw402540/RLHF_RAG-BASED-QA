

%%writefile app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

# --- 설정 (이전 코드에서 가져옴) ---
# 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
# 예: Drive 마운트 후 'drive/MyDrive/my_qa_model' 폴더에 저장했을 경우
BASE_MODEL_NAME = "google/gemma-2-2b"
ADAPTER_PATH = "/content/drive/MyDrive/ai2/sft_lora" # Colab 경로를 가정


# GPU 사용을 위해 device_map="auto" 설정
DEVICE_MAP = "auto"
DTYPE = torch.bfloat16
OFFLOAD_DIR = "/tmp/model_offload"
# 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥

# QA 프롬프트 템플릿: 대화 히스토리를 포함할 수 있도록 수정
SYSTEM_PROMPT = "당신은 사용자 학습을 통해 특정 도메인에 대해 답변할 수 있는 유능한 QA 챗봇입니다. 사용자와의 이전 대화 내용을 참고하여 자연스럽고 정확하게 답변해 주세요."
QA_PROMPT_TEMPLATE = """
{history}
사용자의 질문: {prompt}

모델의 답변:"""

# --- 모델 및 토크나이저 로드 ---
@st.cache_resource
def load_model():
    """LoRA (PEFT) 어댑터가 적용된 4비트 양자화 모델을 로드합니다."""
    # 1. 양자화 설정 (4비트)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 2. offload 폴더 생성
    if not os.path.exists(OFFLOAD_DIR):
        os.makedirs(OFFLOAD_DIR)

    # 3. 기반 모델(Base Model) 로드
    st.info(f"기반 모델 ({BASE_MODEL_NAME}) 4비트 양자화 로드 중...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map=DEVICE_MAP,
        torch_dtype=DTYPE,
        offload_folder=OFFLOAD_DIR,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        add_eos_token=True
        )

    # 4. PEFT 어댑터 로드 및 병합
    st.info(f"LoRA 어댑터 ({ADAPTER_PATH}) 적용 중...")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )
    # 추론 모드로 전환
    model.eval()

    return tokenizer, model

# --- 대화 히스토리 포매팅 함수 ---
def format_conversation_history(messages):
    """대화 히스토리를 프롬프트에 포함하기 위한 문자열로 포맷합니다."""
    if not messages:
        return ""

    history_str = ""
    for msg in messages:
        if msg["role"] == "user":
            history_str += f"사용자의 이전 질문: {msg['content']}\n"
        elif msg["role"] == "assistant":
            # 어시스턴트 메시지는 모델의 답변으로 간주하고 답변 템플릿에 맞춤
            history_str += f"모델의 이전 답변: {msg['content']}\n"

    return history_str.strip()

# --- 챗봇 로직 함수 (멀티턴 지원) ---
def generate_response(tokenizer, model, user_prompt, history_messages):
    """Gemma 2-2B 기반 모델을 사용하여 멀티턴 응답을 생성합니다."""

    # 1. 히스토리 포맷
    history = format_conversation_history(history_messages)

    # 2. 전체 프롬프트 구성 (System Prompt + History + Current Prompt)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{QA_PROMPT_TEMPLATE.format(history=history, prompt=user_prompt)}"

    # 3. 토크나이징 및 GPU 이동
    inputs = tokenizer.encode(full_prompt, return_tensors="pt")
    inputs = inputs.to(model.device)

    # 4. 응답 생성
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id
        )

    # 5. 출력 디코딩 및 프롬프트 제거
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 입력 프롬프트 부분을 제거하고 모델이 생성한 답변만 남깁니다.
    # 복잡한 템플릿 구조 때문에 QA_PROMPT_TEMPLATE 부분을 기준으로 잘라냅니다.
    answer_prefix = QA_PROMPT_TEMPLATE.format(history=history, prompt=user_prompt).strip()

    if answer_prefix in response:
        # 템플릿 이후 부분만 추출
        answer = response.split(answer_prefix, 1)[-1].strip()
        # "모델의 답변:" 부분 제거
        answer = answer.replace("모델의 답변:", "").strip()
    else:
        # 파싱 실패 시, 전체 응답에서 시스템/히스토리만 제거 시도 (간단 처리)
        answer = response.replace(SYSTEM_PROMPT, "").strip()

    # 💡 "사용자의 질문:" 패턴도 제거될 수 있도록 최종 정리
    if "사용자의 질문:" in answer:
        answer = answer.split("사용자의 질문:", 1)[0].strip()

    return answer

# --- Streamlit UI 구성 ---
st.set_page_config(page_title="Custom Gemma 2-2B 챗봇 (Streamlit)", layout="centered")

st.title("🤖 갓기성윤의 챗봇")
st.caption("사용자 학습 기반 멀티턴 대화 지원")

# --- 모델 로드 및 세션 상태 초기화 ---
with st.spinner("사용자 학습 모델 로드 중..."):
    try:
        # 모델 로드 (st.cache_resource 사용)
        tokenizer, model = load_model()
        st.success("모델 로드 완료!")
    except Exception as e:
        st.error(f"❌ 모델 로드 중 오류 발생: 경로({BASE_MODEL_NAME})를 확인해 주세요! 상세 오류: {e}")
        st.stop()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정")
    st.caption("⚠️ **PEFT 모델 추론 환경**")

    st.divider()

    # 새 대화 시작 버튼
    if st.button("🔄 새 대화 시작", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # 현재 대화 턴 수
    st.caption(f"💬 대화 턴: {len(st.session_state.messages)//2}턴")

# --- 메인 화면: 대화 기록 표시 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 사용자 입력 및 응답 생성 ---
if prompt := st.chat_input("메시지를 입력하세요..."):

    # 1. 사용자 메시지 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 어시스턴트 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                # generate_response 함수에 현재까지의 대화 기록을 전달
                # 마지막 메시지(현재 사용자 메시지)를 제외한 기록만 전달해야 함
                history_messages = st.session_state.messages[:-1]

                answer = generate_response(tokenizer, model, prompt, history_messages)

                st.markdown(answer)

            except Exception as e:
                answer = f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
                st.error(answer)

        # 3. 어시스턴트 메시지 추가
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
with st.expander("ℹ️ 모델 정보"):
    st.markdown(f"**기반 모델:** `{BASE_MODEL_NAME}`")
    st.markdown(f"**어댑터 경로:** `{ADAPTER_PATH}`")
    st.markdown("이 봇은 PEFT 학습된 Gemma 모델을 사용하여 멀티턴 대화를 지원합니다.")

from pyngrok import ngrok
# 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
# --- ngrok 설정 ---
# ngrok 회원가입해서 토큰 받아야함
NGROK_AUTH_TOKEN = ""
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
# Gemma2-2b 사용을 위한 허깅페이스 토큰
from huggingface_hub import login
login("")
# 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
# Streamlit 앱 실행 (백그라운드에서 실행)
get_ipython().system_raw('streamlit run app.py &')

# ngrok 터널 열기
print("Streamlit 앱을 위한 ngrok 터널을 여는 중...")
try:
    # 💡 수정: port='8501'을 addr='8501'로 변경합니다.
    public_url = ngrok.connect(addr='8501')

    print(f"\n🎉 Streamlit 앱이 준비되었습니다!")
    print(f"🔗 접속 주소: {public_url}")
    print("\n⚠️ 이 주소를 클릭하여 QA 봇 UI를 확인하세요.")
except Exception as e:
    print(f"\n🚫 ngrok 터널 생성 실패: {e}")
    print("ngrok Auth Token을 올바르게 설정했는지 확인하세요.")