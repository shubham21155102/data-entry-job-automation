echo "Updating package lists..."
sudo apt update

sudo apt-get install -y libgl1
echo "Installing Tesseract OCR and dependencies..."
sudo apt install -y tesseract-ocr tesseract-ocr-eng

echo "Installing Python and pip..."
sudo apt install -y python3 python3-pip

echo "Installing other useful tools..."
sudo apt install -y imagemagick poppler-utils

echo "Installing Python libraries (from requirements.txt)..."
if [ -f requirements.txt ]; then
	pip3 install -r requirements.txt
else
	pip3 install streamlit pillow python-dotenv groq paddleocr opencv-python-headless numpy
fi

echo "Installation complete!"
echo "You can now use Tesseract OCR for text recognition in your data entry automation."