import os
import cv2
from google.cloud import vision
from google.cloud.vision_v1 import types

# Set the path to your service account key file (downloaded JSON)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Lenovo\Downloads\medicai-436704-1e3a7b8c6ab1.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

# Preprocess the image using OpenCV
def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better text visibility
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Save the preprocessed image (optional)
    preprocessed_image_path = r'C:\Users\Lenovo\Downloads\med_pres_preprocessed.jpg'
    cv2.imwrite(preprocessed_image_path, binary_image)

    return preprocessed_image_path

image_path=r'C:\Users\Lenovo\Downloads\med_pres_1.jpg'
# Preprocess the image
preprocessed_image_path = preprocess_image(image_path)

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

# Extract and print the cleaned detected text
if texts:
    extracted_text = texts[0].description
    cleaned_text = clean_extracted_text(extracted_text)
    print("Extracted text:")
    print(cleaned_text)
else:
    print("No text detected")
    
# Check for errors in the response
if response.error.message:
    raise Exception(f'Error from Google Cloud Vision API: {response.error.message}')
