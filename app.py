import streamlit as st
import dotenv
import os
from PIL import Image
import base64
from io import BytesIO
import google.generativeai as genai
import random
from gtts import gTTS  # Import gTTS for text-to-speech
import tempfile  # To handle temporary files for audio playback

dotenv.load_dotenv()

google_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Function to convert the messages format to Gemini
def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] == "video_file":
                gemini_message["parts"].append(genai.upload_file(content["video_file"]))
            # Removed audio handling

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]

    return gemini_messages

# Function to query and stream the response from Google Gemini
def stream_llm_response(model_params, api_key=None):
    response_message = ""

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_params["model"],
        generation_config={
            "temperature": model_params["temperature"] if "temperature" in model_params else 0.3,
        }
    )
    gemini_messages = messages_to_gemini(st.session_state.messages)

    for chunk in model.generate_content(
            contents=gemini_messages,
            stream=True,
    ):
        chunk_text = chunk.text or ""
        response_message += chunk_text
        yield chunk_text

    st.session_state.messages.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})

    # Convert the final response text to speech and save as a separate audio file
    save_audio_response(response_message)

# Function to convert text to speech, save as a separate file, and display it
def save_audio_response(text):
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        st.session_state.messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "audio_file",
                    "audio_file": temp_audio.name
                }
            ]
        })

# Function to convert image file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="Adapt LearnHub",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>The Adapt LearnHub</i> 💬</h1>""")

    # Set the static Google API key
    google_api_key = "AIzaSyDon-eOs1Hh-Rl88_0u9bviCySlP_D9PVw"

    # --- Main Content ---
    if google_api_key == "" or google_api_key is None:
        st.write("#")
        st.warning("⬅️ Google API Key is required to continue...")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Displaying previous messages if any, except hidden prompts
        for message in st.session_state.messages:
            # Check for 'text' type and specific hidden prompt content before displaying
            if (message["content"][0]["type"] == "text" and
                    message["content"][0].get("text") == "Please explain the object detected in this image to a 4-year-old with a simple use case."):
                continue  # Skip rendering this specific prompt

            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":
                        st.image(content["image_url"]["url"])
                    elif content["type"] == "video_file":
                        st.video(content["video_file"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])

        # Define model parameters
        model = google_models[0]  # Default model
        model_temp = 0.3  # Default temperature

        model_params = {
            "model": model,
            "temperature": model_temp,
        }

        # Define function to reset the conversation
        def reset_conversation():
            if "messages" in st.session_state and len(st.session_state.messages) > 0:
                st.session_state.pop("messages", None)

        # Image Upload and Camera Input
        st.write(f"### **🖼️ Upload an image or capture a photo:**")

        def add_image_to_messages():
            if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                img = get_image_base64(raw_img)
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                        }]
                    }
                )
                # Automatically prompt to explain the object in the image for a 4-year-old child
                explain_image_prompt = "Please explain the object detected in this image to a 4-year-old with a simple use case."
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": explain_image_prompt
                        }]
                    }
                )

                # Generate and stream the response without displaying the prompt
                with st.chat_message("assistant"):
                    st.write_stream(
                        stream_llm_response(
                            model_params=model_params,
                            api_key=google_api_key
                        )
                    )

        cols_img = st.columns(2)
        with cols_img[0]:
            with st.popover("📁 Upload"):
                st.file_uploader(
                    f"Upload an image or a video:",
                    type=["png", "jpg", "jpeg", "mp4"],
                    accept_multiple_files=False,
                    key="uploaded_img",
                    on_change=add_image_to_messages,
                )

        with cols_img[1]:
            with st.popover("📸 Camera"):
                activate_camera = st.checkbox("Activate camera")
                if activate_camera:
                    st.camera_input(
                        "Take a picture",
                        key="camera_img",
                        on_change=add_image_to_messages,
                    )

        # Removed audio upload section
        # st.write("#")
        # st.write(f"### **🎤 Add an audio:**")
        # audio_prompt = None
        # speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395")
        # if speech_input:
        #     # Save the audio file
        #     audio_id = random.randint(100000, 999999)
        #     with open(f"audio_{audio_id}.wav", "wb") as f:
        #         f.write(speech_input)

        #     st.session_state.messages.append(
        #         {
        #             "role": "user",
        #             "content": [{
        #                 "type": "audio_file",
        #                 "audio_file": f"audio_{audio_id}.wav",
        #             }]
        #         }
        #     )

        #     # Display the audio file
        #     with st.chat_message("user"):
        #         st.audio(f"audio_{audio_id}.wav")

        # Chat input
        if prompt := st.chat_input("Hi! Ask me anything..."):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt,
                    }]
                }
            )

            # Display the new messages
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and stream the response
            with st.chat_message("assistant"):
                st.write_stream(
                    stream_llm_response(
                        model_params=model_params,
                        api_key=google_api_key
                    )
                )

if __name__ == "__main__":
    main()
