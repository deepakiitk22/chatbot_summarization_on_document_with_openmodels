import os
from groq import Groq
import streamlit as st
import PyPDF2
from io import StringIO

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Function to handle Summarization
def handle_summarization(text=None):
    st.subheader("Summarization")

    # Text input for the document to summarize, shown only if text is not passed
    if not text:
        text = st.text_area("Enter the text to summarize:")

    # Input for the number of words for the summary
    num_words = st.slider("Number of words in the summary:", min_value=50, max_value=500, value=150)

    # Model selection dropdown
    model = st.selectbox("Select summarization model:", 
                         ["llama-3.1-70b-versatile", "llama3-8b-8192", 
                          "mixtral-8x7b-32768", "gemma2-9b-it", "llava-v1.5-7b-4096-preview"])

    if st.button("Generate Summary"):
        st.write(f"Generating summary with {model} for {num_words} words...")

        # Create a prompt for summarization
        prompt = f"Please summarize the following text in about {num_words} words:\n\n{text}"

        # Make API call to OpenAI
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract and display the summary
            summary = response.choices[0].message.content
            st.write(summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Function to handle Generic Question-Answer
def handle_question_answer(text=None):
    st.subheader("Generic Question-Answer")

    # Text input for the question
    query = st.text_input("Enter your question:")

    # Model selection dropdown
    model = st.selectbox("Select question-answer model:", 
                         ["llama-3.1-70b-versatile", "llama3-8b-8192", 
                          "mixtral-8x7b-32768", "gemma2-9b-it", "llava-v1.5-7b-4096-preview"])

    if st.button("Get Answer"):
        st.write(f"Getting answer with {model}...")

        # Create a prompt for question-answering
        prompt = f"Answer the following question:\n\n{query} on {text}"

        # Make API call to OpenAI
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract and display the answer
            answer = response.choices[0].message.content
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    text = ""
    pdf = PyPDF2.PdfReader(file)
    for page in pdf.pages:
        text += page.extract_text()
    return text


def save_uploaded_file(uploaded_file):
    try:
        # Get the current working directory
        current_directory = os.getcwd()
        
        # Create a file path in the current directory
        file_path = os.path.join(current_directory, uploaded_file.name)
        
        # Save the uploaded file to the specified path
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        return file_path
    except Exception as e:
        st.error(f"Failed to save file: {e}")
        return None

# Function to convert MP3 file to text
def mp3_text(filepath):
    audio_file = open(filepath, "rb")
    try:
        translation = client.audio.translations.create(
            file=audio_file,
            model="whisper-large-v3",
        )
        return translation.text
    except Exception as e:
        st.error(f"Failed to process audio: {e}")
        return ""

# Main function to handle app layout and options
def main():
    st.title("Text Processing Options")

    # Session state to keep track of the selected option
    if 'selected_mode' not in st.session_state:
        st.session_state.selected_mode = None

    # Layout for clickable boxes
    col1, col2 = st.columns(2)

    # Style for clickable boxes
    box_style = """
    <style>
    .box {
        padding: 20px;
        margin: 10px;
        border: 2px solid #ddd;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .box:hover {
        background-color: #f0f0f0;
    }
    .box.selected {
        border-color: #007bff;
        background-color: #e7f0ff;
    }
    </style>
    """

    st.markdown(box_style, unsafe_allow_html=True)

    with col1:
        if st.button("Summarization", key="summarization"):
            st.session_state.selected_mode = "Summarization"

    with col2:
        if st.button("Generic Question-Answer", key="qa"):
            st.session_state.selected_mode = "Generic Question-Answer"

    # File uploader
    uploaded_file = st.file_uploader("Choose a file (text, PDF, MP3)", type=["txt", "pdf", "mp3"])

    if uploaded_file:
        # Debug: Print the MIME type of the uploaded file
        st.write(f"Detected file type: {uploaded_file.type}")
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = StringIO(uploaded_file.read().decode("utf-8")).read()
        elif uploaded_file.type == "audio/mpeg":
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                print(f"File path: {file_path}")
                text = mp3_text(file_path)  
            else:
                print("there is not file path")
        else:
            st.error("Unsupported file type.")
            text = None

        if text:
            if st.session_state.selected_mode == "Summarization":
                handle_summarization(text)
            elif st.session_state.selected_mode == "Generic Question-Answer":
                handle_question_answer(text)
        else:
            st.error("No text extracted from the file. Please upload a text, PDF, or MP3 file.")

if __name__ == "__main__":
    main()
