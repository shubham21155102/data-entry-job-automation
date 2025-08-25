import streamlit as st
from PIL import Image
import dotenv
import os
import json
import numpy as np
import cv2
from paddleocr import PaddleOCR

# --- Compatibility monkey patch ---
# Some versions of paddlepaddle used with paddleocr/paddlex call
# Config.set_optimization_level(3) which may not exist (older CPU wheels).
# Provide a harmless no-op to avoid AttributeError.
try:  # pragma: no cover
    from paddle import inference as _paddle_inference
    if not hasattr(_paddle_inference.Config, "set_optimization_level"):
        def _noop(self, *args, **kwargs):
            return None
        _paddle_inference.Config.set_optimization_level = _noop  # type: ignore[attr-defined]
except Exception:
    pass

dotenv.load_dotenv()
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Cache the OCR model so Streamlit reruns don't re-download / re-init
@st.cache_resource(show_spinner=True)
def get_ocr_models():
    """Return OCR models for Hindi + English.
    Some Paddle models are language-specific; we run both and merge.
    use_textline_orientation replaces deprecated use_angle_cls.
    """
    # Removed invalid 'show_log' argument (caused ValueError) and deprecated use_angle_cls.
    def build(lang: str):
        # Prefer new param; fallback to legacy (angle classifier) if not available
        try:
            return PaddleOCR(lang=lang, use_textline_orientation=True)
        except TypeError:
            return PaddleOCR(lang=lang, use_angle_cls=True)
    hi_model = build("hi")
    en_model = build("en")
    return hi_model, en_model

hi_ocr, en_ocr = get_ocr_models()

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Basic preprocessing to improve OCR robustness."""
    img_cv = np.array(img.convert('RGB'))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold handles uneven lighting; tweak blockSize / C if needed
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 8
    )
    denoised = cv2.medianBlur(thresh, 3)
    return denoised

def run_ocr_all_langs(img_np: np.ndarray):
    """Run OCR with both models and merge text (avoid duplicates)."""
    texts = []
    seen = set()
    for model in (hi_ocr, en_ocr):
        result = model.ocr(img_np, cls=True)
        # result is a list (one element per image); inside is list of [box, (text, conf)]
        if not result:
            continue
        for line in result[0]:
            if len(line) >= 2:
                txt = line[1][0].strip()
                if txt and txt not in seen:
                    seen.add(txt)
                    texts.append(txt)
    return "\n".join(texts)

def extract_structured_json(raw_text: str):
    if not raw_text.strip():
        return None, "No OCR text to process."
    if not GROQ_API_KEY:
        return None, "GROQ_API_KEY missing in environment." 
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are an AI data extractor.\n"
                    "I will provide raw text from documents (like mark sheets, ID cards, forms etc).\n"
                    "Your task is to extract the following fields and return them in valid JSON format:\n"
                    "- name\n- father_name\n- mother_name\n- dob\n- marks { hindi, sanskrit, maths, science, social_science }\n"
                    "- gender\n- religion\n- aadhar_number\n- aadhar_mobile_number\n- whatsapp_number\n- email_id\n- post\n- police_station\n- district\n- pincode\n- block\n- caste_category\n- state\n"
                    "Rules:\n"
                    "1. ALL values must be in CAPITAL LETTERS, EXCEPT email_id (keep original case).\n"
                    "2. If a field is missing, return it as an empty string \"\".\n"
                    "3. Output only JSON.\n"
                    "4. DOB in DD-MM-YYYY if possible.\n\n"
                    "Text:\n" + raw_text
                ),
            }
        ],
        temperature=0,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )
    content = completion.choices[0].message.content
    try:
        return json.loads(content), None
    except json.JSONDecodeError:
        return None, content  # return raw so user can inspect

# Streamlit App Title
st.title("📄 Hindi + English OCR Extractor")

# Upload multiple images
uploaded_files = st.file_uploader(
    "Upload images (Hindi/English)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

aggregate_text = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        processed = preprocess_image(img)
        st.image(processed, caption=f"Processed: {uploaded_file.name}", use_container_width=True)
        extracted_text = run_ocr_all_langs(processed)
        st.subheader(f"Extracted Text from {uploaded_file.name}")
        st.text_area(
            f"Raw OCR ({uploaded_file.name})",
            extracted_text,
            height=180,
        )
        if extracted_text:
            aggregate_text.append(extracted_text)

if aggregate_text:
    combined = "\n".join(aggregate_text)
    st.subheader("Structured Extraction")
    with st.spinner("Calling LLM to structure data..."):
        parsed, raw = extract_structured_json(combined)
    if parsed:
        st.json(parsed)
    else:
        st.text_area("LLM Response (not valid JSON)", raw or "", height=400)
elif uploaded_files:
    st.warning("No text extracted from the uploaded images.")
else:
    st.info("Upload one or more document images to begin.")
