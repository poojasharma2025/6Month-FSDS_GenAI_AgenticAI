import cv2
import numpy as np
import streamlit as st
import easyocr
from PIL import Image

# Load OCR reader
reader = easyocr.Reader(['en'])

st.title("🚘 Car Number Plate Detection App")
st.write("Upload an image of a car to detect and extract the number plate.")

# Upload image
uploaded_file = st.file_uploader("Upload Car Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Load pre-trained Haar Cascade for Number Plate Detection
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect plates
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    if len(plates) == 0:
        st.warning("⚠ No Number Plate Detected!")
    else:
        for (x,y,w,h) in plates:
            # Draw rectangle around plate
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
            plate_roi = img[y:y+h, x:x+w]

            # OCR on the plate
            results = reader.readtext(plate_roi)

            st.subheader("🔍 Detected Number Plate:")
            for res in results:
                st.success(res[1])  # res[1] is the detected text

        # Show result image with bounding box
        st.image(img, caption="Detected Plate", use_container_width=True, channels="BGR")
