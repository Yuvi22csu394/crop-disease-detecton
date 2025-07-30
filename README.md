This project is an AI-powered web application that detects plant diseases from uploaded leaf images using a YOLOv8 deep learning model. It then fetches relevant disease information, treatment methods, and recommended pesticides/fertilizers using Wikipedia and Google Custom Search API.

🚀 Features
🔍 Real-time plant disease detection using YOLOv8

🧠 Automatically fetches disease description from Wikipedia

🌐 Provides web search results for prevention and cure (India-focused)

🛒 Suggests Amazon.in products (pesticides/fertilizers) for the detected disease

📊 Adjustable confidence threshold via Streamlit sidebar

🧰 Tech Stack
Python

Streamlit (Web UI)

Ultralytics YOLOv8 (Deep Learning model)

Google Custom Search API (for prevention/product info)

Wikipedia API (for disease description)

OpenCV, NumPy, Pillow

📂 Folder Structure
bash
Copy
Edit
crop-disease-detection-system/
├── model/
│   └── last.pt                     # Trained YOLOv8 model
├── .env                            # API keys (not shared publicly)
├── crop.py                         # Main Streamlit application
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
⚙️ Setup Instructions
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/crop-disease-detection-system.git
cd crop-disease-detection-system
Create a virtual environment and activate it:

bash
Copy
Edit
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Add your environment variables to a .env file:

ini
Copy
Edit
GOOGLE_API_KEY=your_google_api_key
SEARCH_ENGINE_ID=your_custom_search_engine_id
Make sure your trained YOLOv8 model is saved in the model/last.pt path.

Run the app:

bash
Copy
Edit
streamlit run crop.py
📸 How It Works
Upload an image of a diseased plant leaf.

The YOLOv8 model detects the disease from the image.

The app fetches:

A short description from Wikipedia.

Prevention and cure info via Google Custom Search.

Related Amazon products for the disease.

Results are displayed interactively on the Streamlit web interface.

✅ Sample Input
Upload a clear image of a diseased plant leaf in JPG, JPEG, or PNG format.

🛡️ API Notes
You must create a Google Custom Search Engine (CSE) at: https://programmablesearchengine.google.com

Add relevant domains like:

wikipedia.org

amazon.in

krishijagran.com, etc. for more precise results

🧠 Model Training (Optional)
If you want to train your own model, use Ultralytics' YOLOv8 with your labeled dataset.

bash
Copy
Edit
yolo task=detect mode=train model=yolov8n.pt data=your_dataset.yaml epochs=50 imgsz=640
🙏 Acknowledgements
Ultralytics YOLOv8

Streamlit

Wikipedia API

Google Custom Search API

# crop-disease-detecton
