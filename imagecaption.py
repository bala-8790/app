
import streamlit as st
import pytorch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import numpy as np
import cv2

# Load models once
@st.cache_resource
def load_models():
    seg_model = maskrcnn_resnet50_fpn(pretrained=True).eval()
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return seg_model, caption_model, feature_extractor, tokenizer

seg_model, caption_model, feature_extractor, tokenizer = load_models()

st.title("Image Captioning & Segmentation App")

mode = st.radio("Choose input mode:", ["Upload Image", "Live Webcam", "USB Mobile Camera (DroidCam)"])

def process_frame(pil_image):
    transform = torchvision.transforms.ToTensor()
    img_tensor = transform(pil_image)

    with torch.no_grad():
        predictions = seg_model([img_tensor])[0]

    segmented = np.array(pil_image).copy()
    for i, score in enumerate(predictions['scores']):
        if score > 0.8:
            mask = predictions['masks'][i, 0].mul(255).byte().cpu().numpy()
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(segmented, contours, -1, (0, 255, 0), 2)

    pixel_values = feature_extractor(images=pil_image, return_tensors="pt").pixel_values
    with torch.no_grad():
        output_ids = caption_model.generate(pixel_values, max_length=16)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return segmented, caption

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)
        segmented, caption = process_frame(image)
        st.image(segmented, caption="Segmented Image", use_column_width=True)
        st.markdown(f"**Caption:** {caption}")

elif mode in ["Live Webcam", "USB Mobile Camera (DroidCam)"]:
    st.subheader("Camera Capture")
    default_index = 1 if mode == "USB Mobile Camera (DroidCam)" else 0
    camera_index = st.selectbox("Select Camera Index", [0, 1, 2, 3], index=default_index, help="Try different indexes if camera doesn't work.")
    capture_button = st.button("Capture Frame")

    if capture_button:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("Unable to access the camera. Try another index.")
        else:
            ret, frame = cap.read()
            cap.release()

            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                st.image(pil_image, caption="Captured Frame", use_column_width=True)
                segmented, caption = process_frame(pil_image)
                st.image(segmented, caption="Segmented Frame", use_column_width=True)
                st.markdown(f"**Caption:** {caption}")
            else:
                st.error("Failed to capture from camera.")
