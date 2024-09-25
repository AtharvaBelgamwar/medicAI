import os
import cv2
import streamlit as st
from google.cloud import vision
from google.cloud.vision_v1 import types
import google.generativeai as genai
from PIL import Image
import tempfile
import json
# Set up environment variables from Streamlit Secrets
# Streamlit will automatically handle these once set in the Secrets section of the cloud deployment
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Use the Google Cloud Vision API client by securely loading the service account JSON from Streamlit Secrets
def get_vision_client():
    # Convert the AttrDict to a regular dictionary before dumping to JSON
    service_account_info = dict(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"])
    service_account_info['private_key'] = service_account_info['private_key'].replace("\\n", "\n")
    
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as temp_file:
        # Convert the dictionary to a JSON string before writing it
        temp_file.write(json.dumps(service_account_info))
        temp_file_path = temp_file.name
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path
    return vision.ImageAnnotatorClient()

# Initialize the Vision API client
client = get_vision_client()

# Preprocess the image using OpenCV
def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better text visibility
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Save the preprocessed image (optional)
    preprocessed_image_path = image_path + ".jpg"  # Just overwrite it or use another name
    cv2.imwrite(preprocessed_image_path, binary_image)

    return preprocessed_image_path

# Streamlit UI
st.title("Medical Prescription Analyzer")
st.write("Upload an image of a prescription, and click **Analyze** to get the results.")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    # Display the uploaded image using Streamlit
    st.image(Image.open(uploaded_file), caption="Uploaded Prescription", use_column_width=True)

    # Preprocess the image
    preprocessed_image_path = preprocess_image(temp_file_path)

    # When user clicks on 'Analyze'
    if st.button("Analyze"):
        # Load the preprocessed image for OCR
        with open(preprocessed_image_path, 'rb') as image_file:
            content = image_file.read()

        # Construct an image object
        image = types.Image(content=content)

        # Call the Google Cloud Vision API for text detection
        response = client.text_detection(image=image)
        texts = response.text_annotations

        # Extract and clean the detected text
        def clean_extracted_text(text):
            # Replace multiple spaces/newlines with a single space, strip trailing/leading spaces
            return text.replace("\n", " ").replace("  ", " ").strip()

        if texts:
            extracted_text = texts[0].description
            cleaned_text = clean_extracted_text(extracted_text)
           
            # Prompt for Gemini API
            prompt = (
                f"Here is a medical report: {cleaned_text}. Consider this for only testing purposes only. "
                "Please tell only what has happened to the patient and the medication, don't include patient details in your response. "
                "Also, suggest some medications other than the doctor's. "
                "Include any important advice given by the doctor."
            )

            # Send the cleaned text to Gemini for analysis
            model = genai.GenerativeModel("gemini-1.5-flash")
            gemini_response = model.generate_content(prompt)

            # Print the response from the Gemini API
            st.write(gemini_response.text)

        else:
            st.error("No text detected in the uploaded image.")
