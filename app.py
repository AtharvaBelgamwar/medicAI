import os
import streamlit as st
import requests
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.oauth2 import service_account
import google.generativeai as genai
from PIL import Image
import tempfile

# Set up environment variables from Streamlit Secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
google_places_api_key = st.secrets["MAPS_API"]

# Use the Google Cloud Vision API client by securely loading the service account JSON from Streamlit Secrets
def get_vision_client():
    service_account_info = dict(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"])
    if "private_key" in service_account_info:
        service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    return vision.ImageAnnotatorClient(credentials=credentials)

# Initialize the Vision API client
client = get_vision_client()

# Fetch the current latitude and longitude using ipinfo.io
def get_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        if 'loc' in data:
            latitude, longitude = data['loc'].split(',')
            return latitude, longitude
        else:
            st.error("Could not fetch location. Try entering it manually.")
            return None, None
    except Exception as e:
        st.error(f"Error fetching location: {str(e)}")
        return None, None

# Function to find nearby doctors using Google Places API
def find_nearby_doctors(location, radius=20000):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": location,
        "radius": radius,
        "type": "doctor",
        "key": google_places_api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("results", [])
        return results
    else:
        st.error(f"Error: Unable to fetch nearby doctors (Status code: {response.status_code})")
        return None

# Preprocess the image using PIL (no need for OpenCV)
def preprocess_image(image_path):
    image = Image.open(image_path)
    gray_image = image.convert('L')  # Convert to grayscale
    gray_image.save(image_path+ ".jpg" )
    return image_path

# Function to display doctors as clickable cards
def display_doctors(doctors):
    for doctor in doctors:
        name = doctor.get("name", "Unknown")
        address = doctor.get("vicinity", "No address available")
        rating = doctor.get("rating", "No rating")
        latitude = doctor["geometry"]["location"]["lat"]
        longitude = doctor["geometry"]["location"]["lng"]

        # Google Maps link for the doctor
        maps_url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"

        # Create a clickable card-style block for each doctor
        card_html = f"""
        <div style='border: 1px solid #ccc; padding: 16px; border-radius: 10px; margin-bottom: 16px; background-color: white; color: black;'>
            <p style='margin-bottom: 5px;'>{name}</p>
            <p style='margin: 5px 0;'>Address: {address}</p>
            <p style='margin: 5px 0;'>Rating: {rating}</p>
            <a href='{maps_url}' target='_blank'>
                <button style='background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; border-radius: 5px; border: none; cursor: pointer;'>
                    View on Maps
                </button>
            </a>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

# Streamlit UI
st.title("MedicAI")

# File uploader for prescription image
uploaded_file = st.file_uploader("Upload a prescription image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    # Display the uploaded image using Streamlit
    st.image(Image.open(uploaded_file), caption="Uploaded Prescription", use_column_width=True)

    # Preprocess the image
    preprocessed_image_path = preprocess_image(temp_file_path)

    if st.button("Analyze Prescription"):
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
            return text.replace("\n", " ").replace("  ", " ").strip()

        if texts:
            extracted_text = texts[0].description
            cleaned_text = clean_extracted_text(extracted_text)
            st.write("### Extracted text from the prescription:")
            st.write(cleaned_text)

            # Send the extracted text to Gemini for analysis
            prompt = (
                f"Here is a medical report: {cleaned_text}. Consider this for testing purposes only. "
                "Please tell what has happened to the patient and suggest medications or treatments."
            )

            model = genai.GenerativeModel("gemini-1.5-flash")
            gemini_response = model.generate_content(prompt)

            st.write("### Gemini AI Analysis:")
            st.write(gemini_response.text)
        else:
            st.error("No text detected in the uploaded image.")

# Form for user input on symptoms and information
with st.form("user_info_form"):
    st.write("### Provide Information for Diagnosis")
    name = st.text_input("Your Name")
    age = st.number_input("Your Age", min_value=0, max_value=120, step=1)
    symptoms = st.text_area("Describe your symptoms")
    allergies = st.text_input("Known Allergies (if any)")
    medical_history = st.text_area("Medical History")
    
    # Ask if the user has consulted a doctor
    visited_doctor = st.radio("Have you consulted a doctor?", ("Yes", "No"))
    
    submitted = st.form_submit_button("Submit for Diagnosis")

if submitted:
    if name and age and symptoms:
        st.write(f"### Hello, {name}")
        st.write(f"Your Age: {age}")
        st.write(f"Symptoms: {symptoms}")
        if allergies:
            st.write(f"Allergies: {allergies}")
        if medical_history:
            st.write(f"Medical History: {medical_history}")

        # Generate a diagnosis based on user input
        prompt = (
            f"Patient information:\n"
            f"Name: {name}\n"
            f"Age: {age}\n"
            f"Symptoms: {symptoms}\n"
            f"Allergies: {allergies}\n"
            f"Medical History: {medical_history}\n"
            f"Please provide a diagnosis and suggest possible treatments or medications."
        )

        # Send to Gemini for analysis
        model = genai.GenerativeModel("gemini-1.5-flash")
        diagnosis_response = model.generate_content(prompt)

        # Display the diagnosis or prescription
        st.write("### Diagnosis and Suggested Prescription")
        st.write(diagnosis_response.text)

        # Fetch and display the current location
        latitude, longitude = get_location()
        st.write(f"Latitude: {latitude}, Longitude: {longitude}")
        if latitude and longitude:

            # Find and display nearby doctors
            user_location = f"{latitude},{longitude}"
            doctors = find_nearby_doctors(user_location)

            if doctors:
                st.write(f"### Nearby Doctors")
                display_doctors(doctors)
            else:
                st.error("No doctors found near your location.")
        else:
            st.error("Unable to fetch location. Please try again.")
    else:
        st.error("Please fill out the required fields: Name, Age, and Symptoms.")
