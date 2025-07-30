import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
import numpy as np
import wikipedia
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read the API key and CSE ID from the environment
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

if not API_KEY or not SEARCH_ENGINE_ID:
    st.error("API Key or Search Engine ID not found. Please check your .env file.")
    st.stop()

# Load YOLOv8 model
model = YOLO("model/last.pt")

# Sidebar for confidence level adjustment
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

# Function to search the web using Google Custom Search API
def search_web(query, num_results=3):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={SEARCH_ENGINE_ID}&num={num_results}"
    response = requests.get(search_url)
    try:
         data = response.json()
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        return []
   
    results = []
    
    if 'items' in data:
        for item in data['items']:
            title = item.get('title', 'No title')
            snippet = item.get('snippet', 'No description available')
            link = item.get('link', '')
            if 'site:.in' in query and 'amazon.in' in link:
                results.append({'title': title, 'snippet': snippet, 'link': link})
            elif 'site:.in' not in query:
                results.append({'title': title, 'snippet': snippet, 'link': link})
        return results
    else:
        st.error(f"Search error or no items found: {data.get('error', 'Unknown error')}")
        return []
    
# Function to get disease information from Wikipedia
def get_disease_info_from_wikipedia(disease_name):
    try:
        summary = wikipedia.summary(disease_name, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        st.error(f"DisambiguationError: {e}")
        return "Information is ambiguous, please refine your search."
    except wikipedia.exceptions.PageError:
        st.error(f"PageError: No Wikipedia page found for {disease_name}.")
        return "No Wikipedia page found for this disease."
    except Exception as e:
        st.error(f"An error occurred while fetching Wikipedia data: {e}")
        return "An error occurred while fetching Wikipedia data."

# Function to display disease information
def display_disease_info(disease_name):
    disease_info = get_disease_info_from_wikipedia(disease_name)
    st.write(f"### Disease Information: {disease_name}")
    st.write(f"**Description**: {disease_info}")

    prevention_query = f"{disease_name} prevention and cure site:.in"
    prevention_results = search_web(prevention_query)

    product_query = f"{disease_name} pesticides fertilizers site:amazon.in"
    product_results = search_web(product_query)

    if prevention_results:
        st.write("**Prevention and Cure Information:**")
        for result in prevention_results:
            st.write(f"- **Title**: {result['title']}")
            st.write(f"  **Description**: {result['snippet']}")
            st.write(f"  **Reference**: [Read more]({result['link']})")
    else:
        st.write("**Prevention and Cure Information:** No information found.")

    if product_results:
        st.write("### Recommended Products from Amazon India:")
        for result in product_results:
            st.write(f"- **Title**: {result['title']}")
            st.write(f"  **Description**: {result['snippet']}")
            st.write(f"  **Reference**: [Product Link]({result['link']})")
    else:
        st.write("**Product Recommendations:** No products found.")

# Main Streamlit app
st.title("Plant Disease Detection")

uploaded_file = st.file_uploader("Upload an image of the plant", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model.predict(np.array(image), conf=confidence_threshold)

    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
    labels = results[0].boxes.cls.cpu().numpy() if len(results[0].boxes) > 0 else []
    scores = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else []

    if len(boxes) > 0:
        for i, label in enumerate(labels):
            disease_name = model.names[int(label)]
            st.write(f"### Detected Disease: {disease_name}")
            display_disease_info(disease_name)
            st.write("---")
    else:
        st.write("No diseases detected. Please try another image.")
else:
    st.write("Please upload an image to detect plant diseases.")
