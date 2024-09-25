import os
from google.cloud import vision
from google.cloud.vision_v1 import types

# Set the path to your service account key file (downloaded JSON)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'path_to_your_service_account_key.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

# Load an image for OCR
image_path = 'path_to_your_image_file.png'
with open(image_path, 'rb') as image_file:
    content = image_file.read()

# Construct an image object
image = types.Image(content=content)

# Call the Google Cloud Vision API for text detection
response = client.text_detection(image=image)
texts = response.text_annotations

# Extract and print the detected text
if texts:
    print("Extracted text:")
    print(texts[0].description)
else:
    print("No text detected")
    
# Check for errors in the response
if response.error.message:
    raise Exception(f'Error from Google Cloud Vision API: {response.error.message}')
