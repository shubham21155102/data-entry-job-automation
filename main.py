import streamlit as st
import pytesseract
from PIL import Image
import dotenv
import os
import json
dotenv.load_dotenv()
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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

        # OCR - Hindi + English
        extracted_text = pytesseract.image_to_string(img, lang="hin+eng")

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
