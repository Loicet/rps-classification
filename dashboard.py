
import os
import streamlit as st
import requests
from PIL import Image

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RPS Image Classifier",
    page_icon=":material/sports_mma:",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

CLASS_ICONS = {
    "rock": ":material/sports_mma:",
    "paper": ":material/pan_tool:",
    "scissors": ":material/content_cut:"
}

# --------------------------------------------------
# 2. HEADER
# --------------------------------------------------

st.title(":material/sports_mma: Rock Paper Scissors Classifier")

st.markdown(
    "An image classification application powered by "
    "TensorFlow, FastAPI, and Streamlit."
)

st.divider()

# --------------------------------------------------
# 3. SYSTEM STATUS
# --------------------------------------------------

st.subheader(":material/monitor_heart: System Status")

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    health_response.raise_for_status()
    health_data = health_response.json()

    col1, col2 = st.columns(2)

    with col1:
        if health_data["api_status"] == "running":
            st.success("API is running")
        else:
            st.error("API is unavailable")

    with col2:
        if health_data["model_loaded"]:
            st.success("Classification model loaded")
        else:
            st.error("Classification model unavailable")

except requests.exceptions.RequestException:
    st.error(
        "Cannot connect to the API. "
        "Please start FastAPI before using the dashboard."
    )

st.divider()

# --------------------------------------------------
# 4. IMAGE UPLOAD
# --------------------------------------------------

st.subheader(":material/image: Upload an Image")

st.write(
    "Upload an image containing a rock, paper, "
    "or scissors hand gesture."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    image_col, result_col = st.columns(2)

    with image_col:
        st.markdown("#### Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with result_col:
        st.markdown("#### Classification")

        st.write(
            "Click the button below to let the model "
            "classify your image."
        )

        if st.button(
            "Classify Image",
            type="primary",
            use_container_width=True
        ):

            try:
                with st.spinner("Analyzing image..."):

                    uploaded_file.seek(0)

                    response = requests.post(
                        f"{API_URL}/predict",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type
                            )
                        },
                        timeout=60
                    )

                    response.raise_for_status()
                    result = response.json()

                # --------------------------------------------------
                # 5. PREDICTION RESULT
                # --------------------------------------------------

                predicted_class = result["predicted_class"]
                confidence = result["confidence"]

                st.success("Classification completed!")

                st.markdown("### Prediction Result")

                st.markdown(
                    f"## {CLASS_ICONS.get(predicted_class, '')} "
                    f"{predicted_class.capitalize()}"
                )

                st.metric(
                    label="Prediction Confidence",
                    value=f"{confidence * 100:.2f}%"
                )

                # --------------------------------------------------
                # 6. CLASS PROBABILITIES
                # --------------------------------------------------

                st.divider()

                st.subheader(
                    ":material/analytics: Class Probabilities"
                )

                probabilities = result["probabilities"]

                for class_name, probability in probabilities.items():

                    st.write(
                        f"**{CLASS_ICONS.get(class_name, '')} "
                        f"{class_name.capitalize()}**"
                    )

                    st.progress(probability)

                    st.caption(
                        f"{probability * 100:.2f}%"
                    )

            except requests.exceptions.RequestException as error:
                st.error(
                    f"Could not connect to the prediction API: {error}"
                )

            except Exception as error:
                st.error(
                    f"An error occurred while processing the image: {error}"
                )

else:
    st.info(
        "Upload an image to get started with classification."
    )

# --------------------------------------------------
# 7. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Rock Paper Scissors Image Classification | "
    "Machine Learning Project"
)
