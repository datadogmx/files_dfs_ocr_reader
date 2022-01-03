
# Download yolov
git clone https://github.com/ultralytics/yolov5
cd yolov5
git checkout ed2c74218d6d46605cc5fa68ce9bd6ece213abe4
pip install -r requirements.txt
cd ..

# Install tesseract and lenguage module
sudo add-apt-repository ppa:alex-p/tesseract-ocr-devel
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt install libtesseract-dev
sudo apt-get install tesseract-ocr-spa -y

