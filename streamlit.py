import streamlit as st
import os
import time
import google.generativeai as genai

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="Video Analyzer", # Replace with your desired title
    page_icon="🎬",  # You can also set an icon
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            width: 500px !important;  /* Set your desired width here */
        }
    </style>
    """, unsafe_allow_html=True,
)

@st.cache_resource
def load_gemini_model(model_name, generation_config, system_instruction):
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        system_instruction=system_instruction,
    )

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini."""
  genai.configure(api_key=GEMINI_API_KEY)
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  """Waits for the given files to be active."""
  genai.configure(api_key=GEMINI_API_KEY)
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()


def analyze_video_from_url(url, selected_model_name, generation_config, system_instruction, input_prompt):
    """Analyzes video from URL using Gemini."""
    model = load_gemini_model(selected_model_name, generation_config, system_instruction)
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [url],
            },
        ]
    )
    response = chat_session.send_message(input_prompt)
    return response.text


def analyze_video_from_files(video_files, selected_model_name, generation_config, system_instruction, input_prompt):
    """Analyzes video from uploaded files using Gemini."""
    uploaded_files = []
    temp_video_paths = []

    for video_file in video_files:
        video_path = "temp_video_" + video_file.name
        temp_video_paths.append(video_path)
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        uploaded_files.append(upload_to_gemini(video_path, mime_type=f"video/{video_path.split('.')[-1]}"))

    wait_for_files_active(uploaded_files)

    model = load_gemini_model(selected_model_name, generation_config, system_instruction)

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": uploaded_files,
            },
        ]
    )
    response = chat_session.send_message(input_prompt)

    for video_path in temp_video_paths:
        if os.path.exists(video_path):
            os.remove(video_path)
    return response.text


# st.title("Gemini Video Analyzer")

# Sidebar for Input Options and Instructions
with st.sidebar:
    st.subheader("Input Options")
    input_option = st.radio("Select Video Input Source:", ("Upload Video File(s)", "Enter Video URL"))

    video_files = None
    video_url = None

    if input_option == "Upload Video File(s)":
        video_files = st.file_uploader("Upload one or more video files", type=["mp4", "avi", "mov", "mpeg", "wmv"], accept_multiple_files=True)
    elif input_option == "Enter Video URL":
        video_url = st.text_input("Enter video URL (YouTube Shorts, etc.)")

    st.subheader("Model and Instructions")
    model_options = [
        "gemini-2.0-flash-thinking-exp-01-21",
        "gemini-2.0-flash",
        "gemini-2.0-pro-exp-02-05",
        "gemini-1.5-pro",
        "gemini-1.5-flash"
    ]
    selected_model_name = st.selectbox("Select Gemini Model", model_options)

    system_instruction_text = st.text_area("System Instructions",
        value="Analyze the video in detail, focusing on its main political topic, content, and key highlights. Pay close attention to the spoken language, identifying any specific dialects or accents. Note any background music, specifying the song if recognizable, and describe its impact on the video's tone. Identify and highlight the appearance of any notable public figures, politicians, or celebrities. Consider the video's visual elements, including body language, crowd reactions, and setting, to provide context and insight into the social and political implications. If multiple people are speaking, use the captions to accurately attribute statements to the correct individuals. Incorporate time frames to structure the analysis effectively when required. If multiple videos are given, provide timestamps for within each video that have crucial information. Ensure the analysis is comprehensive and written in English, directly summarizing the content without introductory phrases. \n\n",
        height=450
    )

    input_prompt_text = st.text_area("Input Prompt",
        value="Analyze each video separately, generate a comprehensive summary, and include relevant subheadings for better organization. Do not miss any video",
        height=150
    )


generation_config = {
    "temperature": 0.3,
    "top_p": 0.3,
    "top_k": 4,
    "max_output_tokens": 65536,
    "response_mime_type": "text/plain",
}


# Main area for output
st.subheader("Video Analysis Output")
if st.button("Analyze Video(s)", use_container_width=True):
    if input_option == "Upload Video File(s)":
        if video_files:
            with st.spinner("Analyzing video(s) from files..."):
                try:
                    summary_text = analyze_video_from_files(video_files, selected_model_name, generation_config, system_instruction_text, input_prompt_text)
                    
                    st.markdown(summary_text)
                    with st.expander("View raw markdown with Copy Button"):
                        st.code(summary_text, language='markdown')
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload video file(s).")

    elif input_option == "Enter Video URL":
        if video_url:
            with st.spinner("Analyzing video from URL..."):
                try:
                    summary_text = analyze_video_from_url(video_url, selected_model_name, generation_config, system_instruction_text, input_prompt_text)
                    
                    st.markdown(summary_text) # Use st.markdown to render markdown
                    with st.expander("View raw markdown with Copy Button"):
                        st.code(summary_text, language='markdown')
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a video URL.")
else:
    st.info("Upload video file(s) or enter a video URL in the sidebar to analyze.")