
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Underwater Image Dehazing",
    page_icon="🌊",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #071B33;
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

.stButton button, .stDownloadButton button {
    background-color: #087EAF;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton button:hover, .stDownloadButton button:hover {
    background-color: #05A6D5;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Application title
st.title("🌊 Underwater Image Dehazing")
st.write(
    "Enhance underwater images using a trained "
    "U-Net Convolutional Neural Network."
)

st.divider()

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "best_uieb_unet.keras",
        compile=False
    )

try:
    model = load_model()
    st.success("U-Net model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader(
    "Upload an underwater image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read original image
    original_image = Image.open(uploaded_file).convert("RGB")
    original_size = original_image.size

    # Preprocessing
    input_image = original_image.resize((256, 256))

    image_array = np.array(input_image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Image enhancement
    with st.spinner("Enhancing your underwater image..."):
        prediction = model.predict(image_array, verbose=0)

    # Postprocessing
    enhanced_array = np.clip(prediction[0], 0, 1)
    enhanced_array = (enhanced_array * 255).astype(np.uint8)

    enhanced_image = Image.fromarray(enhanced_array)

    # Resize to original dimensions for display
    enhanced_image = enhanced_image.resize(original_size)

    st.success("Image enhancement completed!")

    # Display comparison
    st.subheader("Original vs Enhanced Image")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            original_image,
            caption="Original Underwater Image",
            use_container_width=True
        )

    with col2:
        st.image(
            enhanced_image,
            caption="U-Net Enhanced Image",
            use_container_width=True
        )

    st.divider()

    # Download enhanced image
    buffer = BytesIO()
    enhanced_image.save(buffer, format="PNG")

    st.download_button(
        label="⬇️ Download Enhanced Image",
        data=buffer.getvalue(),
        file_name="enhanced_underwater_image.png",
        mime="image/png",
        use_container_width=True
    )
