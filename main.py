import streamlit as st
# import pytesseract  # We'll replace this with PaddleOCR
from PIL import Image
import dotenv
import os
import json
import numpy as np
import cv2
from paddleocr import PaddleOCR

dotenv.load_dotenv()
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize PaddleOCR once (with both Hindi and English support)
ocr = PaddleOCR(use_angle_cls=True, lang="hi", show_log=False)

# Function for image preprocessing
def preprocess_image(img):
    # Convert PIL Image to OpenCV format
    img_cv = np.array(img.convert('RGB'))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Denoise
    denoised = cv2.medianBlur(thresh, 3)
    
    # Deskew if needed (simplified version)
    # More advanced deskewing could be implemented if needed
    
    return denoised

# Streamlit App Title
st.title("📄 Hindi + English OCR Extractor")

# Upload multiple images
uploaded_files = st.file_uploader(
    "Upload images (Hindi/English)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

text = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Open image
        img = Image.open(uploaded_file)

        st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        # Preprocess the image
        processed_img = preprocess_image(img)
        
        # Show processed image (optional)
        st.image(processed_img, caption=f"Processed: {uploaded_file.name}", use_container_width=True)

        # OCR with PaddleOCR
        result = ocr.ocr(processed_img, cls=True)
        
        # Extract text from PaddleOCR result
        extracted_text = ""
        for line in result[0]:
            if line:
                for entry in line:
                    extracted_text += entry[1][0] + " "
                extracted_text += "\n"

        # Show extracted text
        st.subheader(f"Extracted Text from {uploaded_file.name}")
        st.text_area("Text", extracted_text, height=200)
        text += extracted_text

    if text:
        st.subheader("Processing Extracted Data...")

client = Groq(api_key=GROQ_API_KEY)
completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
              {
                "role": "user",
                "content": """
You are an AI data extractor.
I will provide raw text from documents (like mark sheets, ID cards, forms etc).
Your task is to extract the following fields and return them in **valid JSON format**:
- name
- father_name
- mother_name
- dob
- marks { hindi, sanskrit, maths, science, social_science }
- gender
- religion
- aadhar_number
- aadhar_mobile_number
- whatsapp_number
- email_id
- post
- police_station
- district
- pincode
- block
- caste_category
- state
### Rules:
1. ALL values must be in CAPITAL LETTERS, EXCEPT `email_id` (keep original case). 
2. If a field is missing in the text, return it as an empty string `""`. 
3. Ensure output is **only JSON**, no explanations, no extra text. 
4. DOB should be in DD-MM-YYYY format if found. 
### Example output:
{
  "name": "MANOJ KUMAR",
  "father_name": "ARBIND KUMAR",
  "mother_name": "PARVATI DEVI",
  "dob": "02-03-2006",
  "marks": {
    "hindi": "50",
    "sanskrit": "52",
    "maths": "32",
    "science": "48",
    "social_science": "49"
  },
  "gender": "MALE",
  "religion": "HINDU",
  "aadhar_number": "7266 8018 3657",
  "aadhar_mobile_number": "9572090206",
  "whatsapp_number": "8102489542",
  "email_id": "manojkushwahaji84@gmail.com",
  "post": "KERA",
  "police_station": "SHAMSHHERNAGAR",
  "district": "AURANGABAD",
  "pincode": "824143",
  "block": "DAUDNAGAR",
  "caste_category": "SC",
  "state": "BIHAR"
}
Now extract details from this text:
""" + text
              }
            ],
            temperature=0,
            max_completion_tokens=2048,
            top_p=1,
            reasoning_effort="medium",
            stream=False,
            stop=None
        )
        
# Get the response content
result = completion.choices[0].message.content
        
try:
    # Try to parse as JSON for better display
    parsed_json = json.loads(result)
    st.json(parsed_json)
except json.JSONDecodeError:
    # If not valid JSON, display as text
    st.text_area("Extracted Information", result, height=400)
