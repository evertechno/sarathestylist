import streamlit as st
import cv2
import numpy as np
import dlib
from PIL import Image
import google.generativeai as genai

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Load dlib's pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Streamlit App UI
st.title("AI-Powered Hairstyle & Makeup Suggestions")
st.write("Upload a selfie to get makeup and hairstyle suggestions.")

# Upload image
uploaded_image = st.file_uploader("Upload a selfie", type=["jpg", "jpeg", "png"])

def detect_faces(image):
    """Detect faces in the uploaded image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces

def apply_makeup(image, faces):
    """Apply simple makeup (e.g., lipstick) to detected faces."""
    # This could be an enhancement: overlay makeup based on landmarks (just a placeholder logic)
    for face in faces:
        # Here, we'd use facial landmarks to place makeup
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

def suggest_hairstyles(face_image):
    """Use AI model to suggest hairstyles (simplified)."""
    prompt = "Suggest a hairstyle for a face with makeup."
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# When an image is uploaded
if uploaded_image is not None:
    # Open image with PIL and convert it to a NumPy array
    pil_image = Image.open(uploaded_image)
    image = np.array(pil_image)
    
    # Detect faces in the uploaded image
    faces = detect_faces(image)
    
    if len(faces) == 0:
        st.write("No faces detected. Please upload a clearer image.")
    else:
        # Apply makeup (simple visualization)
        image_with_makeup = apply_makeup(image, faces)
        
        # Convert NumPy image array back to PIL Image for display
        result_image = Image.fromarray(image_with_makeup)
        st.image(result_image, caption="Image with Makeup Applied", use_column_width=True)
        
        # Get hairstyle suggestions from the generative model
        hairstyle_suggestion = suggest_hairstyles(image_with_makeup)
        st.write("Suggested Hairstyle:")
        st.write(hairstyle_suggestion)

