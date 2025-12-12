import streamlit as st
import joblib
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# 1. Page Config
st.set_page_config(page_title="AI Posture Detector", page_icon="🪑")
st.title("🪑 AI Posture Detection System")
st.write("Upload an image of sitting posture to analyze.")

# 2. Load the Trained Model
try:
    model = joblib.load('posture_model.pkl')
    st.sidebar.success("AI Model Loaded Successfully")
except:
    st.sidebar.error("Model not found! Run 'train_model.py' first.")
    model = None

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 3. Feature Extraction Helper (Must match training script)
def process_image(image):
    # Convert PIL image to OpenCV format
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # Streamlit loads as RGB, CV2 wants BGR usually, but MediaPipe wants RGB.
    
    # MediaPipe Processing
    results = pose.process(image_np) # Send RGB directly
    
    if not results.pose_landmarks:
        return None, image_np
    
    # Draw landmarks on image for visual feedback
    annotated_image = image_np.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Extract features (Flattening coordinates)
    landmarks = results.pose_landmarks.landmark
    row = []
    for lm in landmarks:
        row.append(lm.x)
        row.append(lm.y)
        row.append(lm.z)
        row.append(lm.visibility)
        
    return np.array(row).reshape(1, -1), annotated_image

# 4. Main Interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Analyze Posture'):
        if model:
            # Process
            features, annotated_img = process_image(image)
            
            if features is not None:
                # Predict
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0]
                
                # Display Result
                with col2:
                    st.image(annotated_img, caption='AI Vision Analysis', use_container_width=True)
                    
                    if prediction == 1:
                        st.success(f"✅ Posture: GOOD ({probability[1]*100:.1f}% confidence)")
                        st.write("Great job! Keep your back straight.")
                    else:
                        st.error(f"❌ Posture: BAD ({probability[0]*100:.1f}% confidence)")
                        st.write("**Recommendation:**")
                        st.write("- Straighten your back.")
                        st.write("- Align your head with your shoulders.")
                        st.write("- Ensure your feet are flat on the floor.")
            else:
                st.warning("⚠️ No human body detected in the image.")
        else:
            st.error("Please train the model first.")