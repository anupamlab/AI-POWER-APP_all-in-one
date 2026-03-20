# ═══════════════════════════════════════════════════════════════════════════
# IMPORT LIBRARIES
# ═══════════════════════════════════════════════════════════════════════════

import streamlit as st
import PyPDF2
import pandas as pd
import requests
from bs4 import BeautifulSoup
from groq import Groq
import json
from datetime import datetime
from PIL import Image
import io
import os

# Optional libraries (will be used if available)
try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import easyocr
except ImportError:
    easyocr = None

# ═══════════════════════════════════════════════════════════════════════════
# API SETUP
# ═══════════════════════════════════════════════════════════════════════════

# Initialize API client
# Try local environment variable first
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# If not found, try Streamlit secrets (for cloud)
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except:
        GROQ_API_KEY = None

# Final check
if not GROQ_API_KEY:
    st.error("API key not found. Please set API_KEY")
    st.stop()

# Initialize Groq client maildarkkreator console groq
groq_client = Groq(api_key=GROQ_API_KEY)

AI_MODEL = "llama-3.1-8b-instant"

MAX_TEXT_LENGTH = 3000
MAX_RESEARCH_LENGTH = 5000

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - These are reusable functions for common tasks
# ═══════════════════════════════════════════════════════════════════════════

def call_groq_api(prompt_text, max_tokens=500):
    """
    Send a prompt to the Groq AI API and get a response.
    
    This is the core function that communicates with the AI.
    
    Parameters:
    - prompt_text (str): The prompt to send to AI
    - max_tokens (int): Maximum length of response
    
    Returns:
    - str: The AI's response
    """
    try:
        response = groq_client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=max_tokens,
            temperature=0.7  # Controls creativity (0=precise, 1=creative)
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"API Error: {str(e)}")


def ask_question_with_context(context_text, question):
    """
    Ask a question about a document or text.
    
    This function combines the document and question into a single prompt.
    
    Parameters:
    - context_text (str): The document/context to analyze
    - question (str): The question to ask
    
    Returns:
    - str: The AI's answer
    """
    # Truncate text if too long
    context_text = context_text[:MAX_TEXT_LENGTH]
    
    # Create a clear prompt
    prompt = f"""Context:
{context_text}

Question: {question}

Please answer the question based on the context above."""
    
    return call_groq_api(prompt)


def summarize_text(text, length="medium"):
    """
    Create a summary of the provided text.
    
    Parameters:
    - text (str): Text to summarize
    - length (str): "short" = 2-3 sentences, "medium" = 4-5 sentences, "long" = 1 paragraph
    
    Returns:
    - str: The summary
    """
    # Truncate if too long
    text = text[:MAX_TEXT_LENGTH]
    
    # Create instruction based on desired length
    length_instructions = {
        "short": "Summarize in 2-3 sentences",
        "medium": "Summarize in 4-5 sentences",
        "long": "Summarize in 1 detailed paragraph"
    }
    
    prompt = f"""{length_instructions.get(length, length_instructions['medium'])}:

{text}"""
    
    return call_groq_api(prompt, max_tokens=300)


def extract_text_from_pdf(uploaded_pdf):
    """
    Extract text from a PDF file.
    
    Parameters:
    - uploaded_pdf: Streamlit uploaded PDF file object
    
    Returns:
    - tuple: (extracted_text, num_pages, success_message)
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
        extracted_text = ""
        
        # Go through each page and extract text
        for page_num, page in enumerate(pdf_reader.pages):
            extracted_text += f"\n--- Page {page_num + 1} ---\n"
            extracted_text += page.extract_text()
        
        return extracted_text, len(pdf_reader.pages), "✅ PDF successfully loaded"
    
    except Exception as e:
        return "", 0, f"❌ Error reading PDF: {str(e)}"


def extract_text_from_txt(uploaded_txt):
    """
    Extract text from a TXT file.
    
    Parameters:
    - uploaded_txt: Streamlit uploaded TXT file object
    
    Returns:
    - tuple: (extracted_text, success_message)
    """
    try:
        text = uploaded_txt.read().decode("utf-8")
        return text, "✅ Text file successfully loaded"
    except Exception as e:
        return "", f"❌ Error reading text file: {str(e)}"


def fetch_url_content(url):
    """
    Download and extract readable text from a URL.
    
    Parameters:
    - url (str): The website URL
    
    Returns:
    - tuple: (extracted_text, success_message)
    """
    try:
        # Add browser headers so websites don't block us
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Download the webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        
        # Parse HTML to extract text
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "meta", "link", "noscript"]):
            element.decompose()
        
        # Extract readable text
        text = soup.get_text(separator="\n", strip=True)
        
        if len(text) < 20:
            return "", "❌ Could not extract content from this website"
        
        return text, f"✅ Content loaded ({len(text)} characters)"
    
    except requests.exceptions.Timeout:
        return "", "❌ Website took too long to respond (timeout)"
    except requests.exceptions.ConnectionError:
        return "", "❌ Could not connect - check your internet"
    except requests.exceptions.HTTPError as e:
        return "", f"❌ Website error: {e.response.status_code}"
    except Exception as e:
        return "", f"❌ Error: {str(e)}"


def transcribe_audio_to_text(audio_data):
    """
    Convert audio to text using speech recognition with aggressive retry logic.
    
    Parameters:
    - audio_data: Audio bytes or UploadedFile from Streamlit audio input
    
    Returns:
    - tuple: (transcribed_text, success_message)
    """
    if sr is None:
        return "", "❌ Speech Recognition not available. Install: pip install SpeechRecognition"
    
    if audio_data is None:
        return "", "❌ No audio recorded. Please record something and try again."
    
    try:
        import tempfile
        import os
        import time
        
        # Convert UploadedFile to bytes if needed
        if hasattr(audio_data, 'getvalue'):  # It's an UploadedFile
            audio_bytes = audio_data.getvalue()
        elif isinstance(audio_data, bytes):
            audio_bytes = audio_data
        else:
            return "", f"❌ Invalid audio format. Got {type(audio_data).__name__}"
        
        if not audio_bytes or len(audio_bytes) == 0:
            return "", "❌ Audio data is empty. Please record something and try again."
        
        # Initialize speech recognizer
        recognizer = sr.Recognizer()
        
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        try:
            # Load audio from the temporary file
            with sr.AudioFile(temp_audio_path) as source:
                audio = recognizer.record(source)
            
            # Try to recognize speech with aggressive retry logic for connection errors
            max_retries = 5  # Increased from 3 to 5
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    text = recognizer.recognize_google(audio, language='en-US')
                    return text, f"✅ Audio transcribed successfully"
                
                except sr.UnknownValueError as e:
                    # Audio couldn't be understood - no point retrying
                    return "", "❌ Could not understand audio. Please speak clearly and try again."
                
                except sr.RequestError as e:
                    last_error = e
                    error_str = str(e).lower()
                    error_full = str(e)
                    
                    # Check if it's a connection error (worth retrying)
                    is_connection_error = (
                        "connection" in error_str or 
                        "closed" in error_str or 
                        "10054" in error_full or
                        "timeout" in error_str or
                        "refused" in error_str or
                        "unreachable" in error_str
                    )
                    
                    if is_connection_error and attempt < max_retries - 1:
                        # This is a connection error - RETRY with increasing delays
                        wait_time = 3 + (attempt * 2)  # 3, 5, 7, 9, 11 seconds
                        import sys
                        sys.stderr.write(f"[Attempt {attempt + 1}/{max_retries}] Connection error. Retrying in {wait_time}s...\n")
                        time.sleep(wait_time)
                        continue
                    elif not is_connection_error and attempt < max_retries - 1:
                        # Not a connection error but try again anyway
                        time.sleep(2)
                        continue
                    else:
                        # All retries exhausted or non-retriable error
                        break
            
            # All retries failed - return helpful guidance
            error_msg = """❌ **Audio Transcription Failed**

This appears to be a network connectivity issue with Google's Speech API.

**✨ RECOMMENDED: Use Text Input Instead**
• Go to **Method 2: ⌨️ Type Your Message**
• Type your question directly  
• Click "Get AI Response"

Text input works perfectly and is often faster! 🎯"""
            
            return "", error_msg
        
        finally:
            # Clean up the temporary file
            try:
                os.remove(temp_audio_path)
            except:
                pass
    
    except ValueError as e:
        return "", f"❌ Audio format error: {str(e)}"
    except ImportError as e:
        return "", f"❌ Missing library: {str(e)}"
    except Exception as e:
        return "", f"❌ Error transcribing audio: {str(e)}"


def extract_text_from_image(uploaded_image):
    """
    Extract text from an image (JPG, PNG, etc.) using EasyOCR.
    
    Parameters:
    - uploaded_image: Streamlit uploaded image file object
    
    Returns:
    - tuple: (extracted_text, success_message)
    """
    if easyocr is None:
        return "", "❌ EasyOCR not available. Install: pip install easyocr pillow"
    
    if uploaded_image is None:
        return "", "❌ No image provided. Please upload an image."
    
    try:
        # Open the image
        image = Image.open(uploaded_image)
        
        # Display image info
        img_info = f"Image size: {image.size} | Format: {image.format}"
        
        st.write(f"📸 {img_info}")
        
        # Initialize OCR reader (English language by default, can support multiple languages)
        # Note: First time will download the model (~100MB)
        with st.spinner("🔄 Loading OCR model (this may take a moment on first use)..."):
            reader = easyocr.Reader(['en'], gpu=False)
        
        # Convert PIL image to array for EasyOCR
        import numpy as np
        image_array = np.array(image)
        
        # Extract text from image
        with st.spinner("📖 Extracting text from image..."):
            results = reader.readtext(image_array)
        
        # Combine all detected text
        extracted_text = "\n".join([text[1] for text in results])
        
        if not extracted_text.strip():
            return "", "⚠️ No text detected in the image. Try a clearer image."
        
        # Calculate confidence score average
        confidence_scores = [text[2] for text in results]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        success_msg = f"✅ Text extracted successfully | Confidence: {avg_confidence*100:.1f}% | Words detected: {len(extracted_text.split())}"
        return extracted_text, success_msg
    
    except Exception as e:
        return "", f"❌ Error extracting text from image: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🤖 anupamLab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
        .main {
            padding-top: 0px;
        }
        .header {
            text-align: center;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("#  All-in-One AI Assistant_By anupamLab")
st.markdown("*Your personal AI companion for 13+ different tasks* 👤Follow: https://www.linkedin.com/in/palanupam")

# ═══════════════════════════════════════════════════════════════════════════
# CREATE TABS FOR DIFFERENT TOOLS
# ═══════════════════════════════════════════════════════════════════════════

# Create tabs for different tools
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
    "📄 Document Analyzer",
    "🌐 URL Analyzer",
    "📄 Resume Analyzer",
    "💬 Chat with Doc",
    "🔍 Multi-Tool Research",
    "📚 Notes Generator",
    "🎙️ Voice Assistant",
    "📊 Data Analyzer",
    "💰 Finance Analyzer",
    "💻 Code Explainer",
    "✉️ Content Generator",
    "🎓 Learn Anything",
    "🖼️ Image to Text",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: DOCUMENT ANALYZER  by anupamLab
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("📄 Upload & Analyze Your Documents")
    st.write("Upload a PDF or TXT file to extract text, get a summary, and ask questions.")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a file (PDF or TXT)",
        type=["pdf", "txt"],
        key="doc_analyzer"
    )
    
    if uploaded_file:
        # Determine file type
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        # Extract text based on file type
        if file_type == "pdf":
            extracted_text, num_pages, message = extract_text_from_pdf(uploaded_file)
            st.info(f"{message} ({num_pages} pages)")
        
        elif file_type == "txt":
            extracted_text, message = extract_text_from_txt(uploaded_file)
            st.info(message)
        
        else:
            st.error("❌ Unsupported file type")
            extracted_text = ""
        
        # If text was successfully extracted
        if extracted_text:
            # Show file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", len(extracted_text))
            with col2:
                st.metric("Words", len(extracted_text.split()))
            with col3:
                st.metric("Paragraphs", extracted_text.count('\n'))
            
            st.divider()
            
            # Summary section
            st.subheader("📝 Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                summary_type = st.radio("Summary length:", ["Short", "Medium", "Long"], horizontal=True)
            
            if st.button("Generate Summary", key="doc_summary_btn"):
                with st.spinner("Generating summary..."):
                    summary = summarize_text(extracted_text, summary_type.lower())
                    st.success("Summary generated!")
                    st.info(summary)
            
            st.divider()
            
            # Q&A section
            st.subheader("❓ Ask Questions About Your Document")
            user_question = st.text_input(
                "What would you like to know about this document?",
                placeholder="e.g., What is the main topic?",
                key="doc_question"
            )
            
            if user_question:
                if st.button("Get Answer", key="doc_answer_btn"):
                    with st.spinner("AI is thinking..."):
                        answer = ask_question_with_context(extracted_text, user_question)
                        st.success("Answer ready!")
                        st.write(answer)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: URL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("🌐 Analyze Website Content")
    st.write("Enter a URL to extract content, generate a summary, and ask questions.")
    
    url_input = st.text_input(
        "Enter website URL:",
        placeholder="https://example.com",
        key="url_analyzer"
    )
    
    if url_input:
        # Validate URL
        if not url_input.startswith(("http://", "https://")):
            st.warning("⚠️ URL should start with http:// or https://")
        else:
            # Fetch content
            with st.spinner("Fetching website content..."):
                url_text, message = fetch_url_content(url_input)
            
            st.info(message)
            
            if url_text:
                # Show website info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Characters", len(url_text))
                with col2:
                    st.metric("Words", len(url_text.split()))
                
                st.divider()
                
                # Website summary
                st.subheader("📝 Website Summary")
                if st.button("Generate Summary", key="url_summary_btn"):
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(url_text, "medium")
                        st.info(summary)
                
                st.divider()
                
                # Q&A about website
                st.subheader("❓ Ask About This Website")
                url_question = st.text_input(
                    "Your question:",
                    placeholder="What is this website about?",
                    key="url_question"
                )
                
                if url_question:
                    if st.button("Get Answer", key="url_answer_btn"):
                        with st.spinner("AI is thinking..."):
                            answer = ask_question_with_context(url_text, url_question)
                            st.write(answer)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: RESUME ANALYZER  by anupamLab
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("📄 Resume Analyzer")
    st.write("Upload your resume to get ATS feedback, skill analysis, and improvement suggestions.")
    
    resume_file = st.file_uploader(
        "Upload your resume (PDF or TXT)",
        type=["pdf", "txt"],
        key="resume_analyzer"
    )
    
    if resume_file:
        # Extract resume text
        if resume_file.name.endswith(".pdf"):
            resume_text, num_pages, message = extract_text_from_pdf(resume_file)
            st.info(message)
        else:
            resume_text, message = extract_text_from_txt(resume_file)
            st.info(message)
        
        if resume_text:
            st.success(f"✅ Resume loaded ({len(resume_text)} characters)")
            
            # Analysis tabs
            analysis_choice = st.radio(
                "What analysis would you like?",
                ["ATS Feedback", "Skill Gap Analysis", "Improvement Suggestions", "Job Role Match"],
                horizontal=True
            )
            
            if st.button("Analyze Resume", key="resume_btn"):
                with st.spinner("Analyzing your resume..."):
                    if analysis_choice == "ATS Feedback":
                        prompt = f"""Analyze this resume from an Applicant Tracking System (ATS) perspective:

{resume_text[:MAX_TEXT_LENGTH]}

Provide:
1. ATS Score (1-10)
2. Issues that might cause rejection
3. Missing keywords
4. Formatting issues
5. Recommendations

Be specific and constructive."""

                    elif analysis_choice == "Skill Gap Analysis":
                        prompt = f"""Based on this resume:

{resume_text[:MAX_TEXT_LENGTH]}

Provide:
1. Current skills identified
2. In-demand skills that are missing
3. Recommendations for skill development
4. Learning resources suggestions

Be specific."""

                    elif analysis_choice == "Improvement Suggestions":
                        prompt = f"""Review this resume and suggest improvements:

{resume_text[:MAX_TEXT_LENGTH]}

Provide:
1. Top 5 improvements
2. Bullet point enhancements
3. Achievement quantification suggestions
4. Section reorganization advice
5. Keywords to add

Be actionable and specific."""

                    else:  # Job Role Match
                        prompt = f"""Based on this resume, analyze potential job roles:

{resume_text[:MAX_TEXT_LENGTH]}

Provide:
1. Top 5 suitable job roles
2. Why each role fits
3. Additional experience needed for each
4. Career progression path
5. Remote opportunities

Be realistic and encouraging."""

                    result = call_groq_api(prompt, max_tokens=800)
                    st.success("Analysis complete!")
                    st.write(result)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: CHAT WITH DOCUMENT (RAG LITE)  by anupamLab
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("💬 Chat with Your Document")
    st.write("Upload a document and have an ongoing conversation about it.")
    
    # Initialize session state for conversation history
    if "doc_chat_history" not in st.session_state:
        st.session_state.doc_chat_history = []
    
    if "doc_chat_content" not in st.session_state:
        st.session_state.doc_chat_content = ""
    
    # File upload
    chat_upload = st.file_uploader(
        "Upload document (PDF or TXT)",
        type=["pdf", "txt"],
        key="chat_uploader"
    )
    
    if chat_upload:
        # Extract text
        if chat_upload.name.endswith(".pdf"):
            chat_content, _, msg = extract_text_from_pdf(chat_upload)
        else:
            chat_content, msg = extract_text_from_txt(chat_upload)
        
        st.session_state.doc_chat_content = chat_content
        st.success(msg)
        
        if chat_content:
            st.divider()
            
            # Display conversation history
            st.subheader("💬 Conversation")
            
            # Show all previous messages
            for message in st.session_state.doc_chat_history:
                if message["role"] == "user":
                    st.write(f"**You:** {message['content']}")
                else:
                    st.info(f"**AI:** {message['content']}")
            
            # Input new message
            user_message = st.text_input(
                "Ask something about the document:",
                placeholder="What is the main topic?",
                key="chat_input"
            )
            
            if user_message:
                if st.button("Send", key="chat_send"):
                    # Add user message to history
                    st.session_state.doc_chat_history.append({
                        "role": "user",
                        "content": user_message
                    })
                    
                    # Get AI response
                    with st.spinner("Thinking..."):
                        ai_response = ask_question_with_context(
                            st.session_state.doc_chat_content,
                            user_message
                        )
                    
                    # Add AI response to history
                    st.session_state.doc_chat_history.append({
                        "role": "assistant",
                        "content": ai_response
                    })
                    
                    # Rerun to display new messages
                    st.rerun()
            
            # Clear history button
            if st.button("Clear Conversation", key="clear_chat"):
                st.session_state.doc_chat_history = []
                st.session_state.doc_chat_content = ""
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: MULTI RESEARCH TOOL  by anupamLab
# ═══════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader(" Multi Research Tool")
    st.write("Professional research platform: Upload PDFs, text files, and URLs to compare sources and conduct deep research analysis with citations.")
    
    # Initialize session state
    if "research_sources" not in st.session_state:
        st.session_state.research_sources = {}  # {source_id: {name, type, content, source_ref}}
    if "source_counter" not in st.session_state:
        st.session_state.source_counter = 0
    
    st.divider()
    
    # INPUT SECTION - THREE INDEPENDENT UPLOAD OPTIONS
    st.subheader("📥 Add Research Sources")
    
    col1, col2, col3 = st.columns(3)
    
    # COLUMN 1: PDF UPLOAD
    with col1:
        st.write("**📄 PDF Documents**")
        pdf_file = st.file_uploader("Upload PDF:", type=["pdf"], key="pdf_upload")
        if pdf_file and st.button("➕ Add PDF", key="add_pdf_btn"):
            try:
                pdf_text, _, msg = extract_text_from_pdf(pdf_file)
                if pdf_text:
                    source_id = f"pdf_{st.session_state.source_counter}"
                    st.session_state.source_counter += 1
                    st.session_state.research_sources[source_id] = {
                        "name": pdf_file.name,
                        "type": "PDF",
                        "content": pdf_text[:MAX_RESEARCH_LENGTH],
                        "source_ref": f"PDF: {pdf_file.name}",
                        "char_count": len(pdf_text)
                    }
                    st.success(f"✅ Added: {pdf_file.name}")
                else:
                    st.error("Could not extract text from PDF")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    # COLUMN 2: TEXT FILE UPLOAD
    with col2:
        st.write("**📝 Text Documents**")
        text_file = st.file_uploader("Upload TXT:", type=["txt"], key="text_upload")
        if text_file and st.button("➕ Add Text", key="add_text_btn"):
            try:
                text_content = text_file.read().decode("utf-8")
                if text_content:
                    source_id = f"text_{st.session_state.source_counter}"
                    st.session_state.source_counter += 1
                    st.session_state.research_sources[source_id] = {
                        "name": text_file.name,
                        "type": "Text",
                        "content": text_content[:MAX_RESEARCH_LENGTH],
                        "source_ref": f"Text: {text_file.name}",
                        "char_count": len(text_content)
                    }
                    st.success(f"✅ Added: {text_file.name}")
            except Exception as e:
                st.error(f"Error processing text file: {str(e)}")
    
    # COLUMN 3: URL INPUT
    with col3:
        st.write("**🌐 Web Sources**")
        url_input = st.text_input("Enter URL:", placeholder="https://example.com", key="url_input_research")
        if url_input and st.button("➕ Add URL", key="add_url_btn_research"):
            if url_input.startswith(("http://", "https://")):
                content, message = fetch_url_content(url_input)
                if content:
                    source_id = f"url_{st.session_state.source_counter}"
                    st.session_state.source_counter += 1
                    st.session_state.research_sources[source_id] = {
                        "name": url_input.split("/")[2],  # domain name
                        "type": "URL",
                        "content": content[:MAX_RESEARCH_LENGTH],
                        "source_ref": url_input,
                        "char_count": len(content)
                    }
                    st.success(f"✅ Added: {url_input}")
                else:
                    st.error(f"Could not fetch URL: {message}")
            else:
                st.error("❌ Invalid URL. Must start with http:// or https://")
    
    st.divider()
    
    # SOURCES MANAGER
    if st.session_state.research_sources:
        st.subheader(f"📚 Research Sources ({len(st.session_state.research_sources)})")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sources", len(st.session_state.research_sources))
        with col2:
            total_chars = sum(s["char_count"] for s in st.session_state.research_sources.values())
            st.metric("Total Content", f"{total_chars:,} chars")
        with col3:
            pdf_count = sum(1 for s in st.session_state.research_sources.values() if s["type"] == "PDF")
            st.metric("PDFs", pdf_count)
        with col4:
            url_count = sum(1 for s in st.session_state.research_sources.values() if s["type"] == "URL")
            st.metric("URLs", url_count)
        
        st.divider()
        
        # Display all sources
        st.subheader("📖 Source Details")
        
        source_list = list(st.session_state.research_sources.items())
        for idx, (source_id, source_data) in enumerate(source_list):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                type_emoji = "📄" if source_data["type"] == "PDF" else "📝" if source_data["type"] == "Text" else "🌐"
                with st.expander(f"{type_emoji} [{source_data['type']}] {source_data['name']} - {source_data['char_count']:,} chars"):
                    st.caption(f"**Source:** {source_data['source_ref']}")
                    st.write(source_data["content"][:300] + "..." if len(source_data["content"]) > 300 else source_data["content"])
            
            with col2:
                if st.button("🔄", key=f"view_src_{source_id}", help="View full"):
                    st.info(source_data["content"][:1000] + "..." if len(source_data["content"]) > 1000 else source_data["content"])
            
            with col3:
                if st.button("❌", key=f"remove_src_{source_id}", help="Remove"):
                    del st.session_state.research_sources[source_id]
                    st.rerun()
        
        st.divider()
        
        # DEEP RESEARCH ANALYSIS SECTION
        st.subheader("🔬 Deep Research Analysis")
        
        research_focus = st.radio(
            "Select research focus:",
            [
                "Source Evaluation",
                "Key Findings & Patterns",
                "Evidence-Based Analysis",
                "Research Recommendations"
            ],
            horizontal=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Conduct Deep Research", key="deep_research_btn", use_container_width=True):
                with st.spinner("Analyzing sources with research methodology..."):
                    # Combine all sources with citations
                    combined_analysis = ""
                    sources_summary = []
                    
                    for source_id, source_data in st.session_state.research_sources.items():
                        combined_analysis += f"\n\n[SOURCE: {source_data['source_ref']}]\n{source_data['content']}"
                        sources_summary.append(f"[{source_data['type']}] {source_data['name']}")
                    
                    sources_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sources_summary)])
                    
                    if research_focus == "Source Evaluation":
                        prompt = f"""You are a research methodology expert. Evaluate and critically assess these {len(st.session_state.research_sources)} research sources.

SOURCES:
{sources_text}

CONTENT:
{combined_analysis[:MAX_RESEARCH_LENGTH]}

Provide comprehensive source evaluation:
1. **Source Credibility Assessment** - Reliability score for each source [0-10]
2. **Source Type Analysis** - Strengths/weaknesses of PDF vs Text vs URL sources
3. **Information Contradictions** - Where sources disagree and why [cite both]
   - Source A states: [quote]
   - Source B states: [quote]
4. **Information Overlap** - Which sources cover same topics (redundancy analysis)
5. **Complementary Data** - How sources fill each other's gaps
6. **Bias Detection** - Potential biases in [Source_Name]
7. **Consensus Points** - Where all sources agree [cite sources]
8. **Overall Source Quality Score** - Rate 0-10: credibility, reliability, bias level, usefulness
9. **Recommendation** - Which source(s) most reliable and why

Use citations: [Source_1_name], [Source_2_name], etc."""

                    elif research_focus == "Key Findings & Patterns":
                        prompt = f"""You are a research analyst. Extract key findings and identify patterns across these {len(st.session_state.research_sources)} sources.

SOURCES ({sources_text}):
{combined_analysis[:MAX_RESEARCH_LENGTH]}

Identify and report:
1. **Major Findings** - Top 5 most important discoveries [source citations]
2. **Recurring Patterns** - What repeats across sources [cite each occurrence]
3. **Emerging Trends** - Changes, developments, or progressions identified
4. **Statistical Insights** - All numbers/data points with source attribution
5. **Cause & Effect Relationships** - Connections and causality [with evidence]
6. **Unique Discoveries** - Critical insights found in only one source
7. **Pattern Validation Score** - How strongly patterns are supported [0-10]
8. **Research Implication** - What these patterns mean for the research area

Citation format: [Source_1] and [Source_2] both mention | Only [Source_Name] reports"""

                    elif research_focus == "Evidence-Based Analysis":
                        prompt = f"""You are a research writer. Create a comprehensive evidence-based analysis of these {len(st.session_state.research_sources)} sources.

SOURCES ANALYZED:
{sources_text}

SOURCE MATERIALS:
{combined_analysis[:MAX_RESEARCH_LENGTH]}

Write professional evidence-based analysis:
1. **Research Topic Summary** - What these sources collectively cover
2. **Primary Findings** - Major evidence-backed discoveries [cite each]
3. **Supporting Evidence** - Key claims with proof from sources
   - Claim: [statement] - Evidence: [Source_Name]
4. **Data & Statistics** - All quantified information with source attribution
5. **Knowledge Synthesis** - How pieces connect across sources
6. **Information Gaps** - Missing data needed for complete picture
7. **Expert Perspectives** - Key viewpoints and schools of thought mentioned
8. **Final Conclusion** - Evidence-based summary and takeaway
9. **Analysis Confidence Score** - Rate 0-10: evidence strength, source consensus, clarity

Every claim MUST have citation. Format: "Text [Source_Name]" """

                    else:  # Research Recommendations
                        prompt = f"""You are a senior research consultant. Based on these {len(st.session_state.research_sources)} sources, provide actionable research recommendations.

SOURCES STUDIED:
{sources_text}

RESEARCH MATERIALS:
{combined_analysis[:MAX_RESEARCH_LENGTH]}

Provide professional recommendations:
1. **Current Research Status** - What we know and don't know from these sources
2. **Critical Gaps** - Most important missing information to address
3. **Next Research Phase** - Specific investigations recommended
4. **Data Collection Needs** - What and where to source additional data
5. **Methodology** - How to verify/expand on current findings
6. **Timeline & Milestones** - Phases and realistic schedule
7. **Resource Allocation** - Where to invest research effort
8. **Risk Factors** - Limitations and constraints to manage
9. **Priority Actions** - Top 3 recommendations ranked by importance

Cite reasoning: "Recommended because [Source_Name] found..." """

                    result = call_groq_api(prompt, max_tokens=1500)
                    st.success("✅ Deep Research Analysis Complete!")
                    st.write(result)
        
        with col2:
            if st.button("🗑️ Clear All Sources", key="clear_all_research", use_container_width=True):
                st.session_state.research_sources = {}
                st.session_state.source_counter = 0
                st.rerun()
    
    else:
        st.info("👆 **Get Started:** Upload a PDF, add text, or enter a URL to begin your research analysis")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: AI NOTES GENERATOR  by anupamLab
# ═══════════════════════════════════════════════════════════════════════════

with tab6:
    st.subheader("📚 AI Notes Generator")
    st.write("Enter a topic to generate comprehensive study notes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        topic = st.text_input(
            "Enter a topic:",
            placeholder="e.g., Photosynthesis, Artificial Intelligence, Renaissance Art",
            key="notes_topic"
        )
    
    with col2:
        note_type = st.selectbox(
            "Note type:",
            ["Short Notes", "Detailed Notes", "Bullet Summary", "Study Guide"],
            key="note_type"
        )
    
    if topic and st.button("Generate Notes", key="notes_btn"):
        with st.spinner("Generating notes..."):
            if note_type == "Short Notes":
                prompt = f"""Create concise study notes on '{topic}'. 
                
Include:
1. Definition/Overview (2-3 sentences)
2. Key concepts (5-7 points)
3. Examples
4. Importance
5. Quick summary

Keep it beginner-friendly and under 300 words."""

            elif note_type == "Detailed Notes":
                prompt = f"""Create comprehensive study notes on '{topic}'.

Include:
1. Introduction (what is it?)
2. Historical background
3. Key concepts and theories
4. Important figures/discoveries
5. Real-world applications
6. Current status/trends
7. Further learning resources

Make it detailed but easy to understand."""

            elif note_type == "Bullet Summary":
                prompt = f"""Create bullet-point notes on '{topic}'.

Provide:
• Definition
• Key points (at least 8)
• Sub-points under each
• Examples
• Similar topics
• Quick facts

Format as clear bullet points."""

            else:  # Study Guide
                prompt = f"""Create a study guide for '{topic}'.

Include:
1. Learning objectives
2. Core concepts explained
3. Common misconceptions
4. Practice questions (3-4)
5. Key terms and definitions
6. Study tips
7. Real-world connections

Make it suitable for students."""

            notes = call_groq_api(prompt, max_tokens=1000)
            st.success("Notes generated!")
            st.write(notes)
            
            # Copy button (for better UX)
            st.download_button(
                label="📥 Download Notes",
                data=notes,
                file_name=f"{topic}_notes.txt",
                mime="text/plain"
            )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 7: VOICE AI ASSISTANT by anupamLab
# ═══════════════════════════════════════════════════════════════════════════

with tab7:
    st.subheader("🎙️ Voice AI Assistant")
    st.write("Record your voice and get AI responses back in real-time.")
    
    # Initialize session state for voice interaction
    if "voice_transcribed" not in st.session_state:
        st.session_state.voice_transcribed = ""
    if "voice_response" not in st.session_state:
        st.session_state.voice_response = ""
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    # ===== METHOD 1: RECORD AUDIO =====
    with col1:
        st.write("**Method 1: 🎤 Record Your Voice**")
        
        if sr is None:
            st.error("❌ Speech recognition library not installed. Run: pip install SpeechRecognition")
        else:
            st.info("🎙️ Click the microphone button below to start recording")
            
            # Streamlit audio input for recording
            audio_data = st.audio_input(
                "Record your message:",
                label_visibility="collapsed"
            )
            
            if audio_data is not None:
                st.success("✅ Audio recorded!")
                
                # Transcribe audio to text
                with st.spinner("🔄 Converting speech to text..."):
                    transcribed_text, transcribe_msg = transcribe_audio_to_text(audio_data)
                
                st.write(transcribe_msg)
                
                if transcribed_text:
                    st.session_state.voice_transcribed = transcribed_text
                    st.write(f"**You said:** *{transcribed_text}*")
                else:
                    st.warning("⚠️ Could not transcribe audio. Try speaking again.")
                    st.session_state.voice_transcribed = ""
    
    # ===== METHOD 2: TYPE MESSAGE =====
    with col2:
        st.write("**Method 2: ⌨️ Type Your Message**")
        
        typed_message = st.text_area(
            "Or type your question/message:",
            placeholder="What would you like to know?",
            key="voice_text",
            height=120
        )
        
        if typed_message:
            st.session_state.voice_transcribed = typed_message
    
    st.divider()
    
    # ===== GET AI RESPONSE =====
    st.subheader("🤖 AI Response:")
    
    user_input = st.session_state.voice_transcribed
    
    if user_input:
        st.write(f"**Your message:** {user_input}")
        st.divider()
        
        if st.button("🚀 Get AI Response", key="voice_ask_btn", use_container_width=True):
            with st.spinner("⏳ AI is processing your message..."):
                try:
                    # Send to AI for response
                    response = call_groq_api(
                        f"Please answer this question in a helpful and conversational way:\n\n{user_input}",
                        max_tokens=500
                    )
                    st.session_state.voice_response = response
                except Exception as e:
                    st.error(f"❌ Error getting response: {str(e)}")
                    st.session_state.voice_response = ""
        
        # Display response if available
        if st.session_state.voice_response:
            st.success("✅ Response ready!")
            st.info(st.session_state.voice_response)
            
            # Show response options
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.caption("💾 Copy the response above")
            with col2:
                st.caption("🔊 Use browser text-to-speech")
            with col3:
                st.caption("📝 Ask a follow-up")
            
            # Clear for next input
            if st.button("🔄 Clear & Start New", key="voice_clear"):
                st.session_state.voice_transcribed = ""
                st.session_state.voice_response = ""
                st.rerun()
    else:
        st.info("👆 Record audio or type a message above, then click 'Get AI Response'")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 8: DATA ANALYZER by anupamLab
# ═══════════════════════════════════════════════════════════════════════════

with tab8:
    st.subheader("📊 Data Analyzer")
    st.write("Upload a CSV file to analyze data and get insights.")
    
    data_file = st.file_uploader(
        "Upload CSV file:",
        type=["csv"],
        key="data_analyzer"
    )
    
    if data_file:
        try:
            # Read CSV
            df = pd.read_csv(data_file)
            st.success("✅ CSV file loaded successfully")
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Data Types", len(df.dtypes.unique()))
            
            st.divider()
            
            # Show preview with option to view all
            st.subheader("📋 Data Preview")
            rows_to_show = st.slider("Number of rows to display:", 5, len(df), 10)
            st.dataframe(df.head(rows_to_show), use_container_width=True)
            
            st.divider()
            
            # Show statistics in table format
            st.subheader("📊 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.divider()
            
            # Show Data Quality info
            st.subheader("✅ Data Quality Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Complete Rows", len(df[df.isnull().sum(axis=1) == 0]))
            with col2:
                st.metric("Rows with Missing", len(df[df.isnull().sum(axis=1) > 0]))
            with col3:
                completeness = round((100 - df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100), 1)
                st.metric("Completeness %", f"{completeness}%")
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Show missing values by column
            missing_by_col = df.isnull().sum().to_frame("Missing Count")
            missing_by_col["Missing %"] = (missing_by_col["Missing Count"] / len(df) * 100).round(2)
            
            if missing_by_col["Missing Count"].sum() > 0:
                st.subheader("❌ Missing Values by Column")
                st.dataframe(missing_by_col[missing_by_col["Missing Count"] > 0], use_container_width=True)
            
            st.divider()
            
            # Analysis options
            analysis_type = st.radio(
                "Choose analysis:",
                ["Summary", "Statistical Analysis", "Data Quality Report", "Insights"],
                horizontal=True
            )
            
            if st.button("Analyze Data", key="data_btn"):
                with st.spinner("Analyzing..."):
                    # Create a cleaner summary for AI
                    summary_stats = df.describe().to_csv(index=True)
                    
                    if analysis_type == "Summary":
                        prompt = f"""Provide a simple summary of this dataset with {len(df)} rows:

Columns: {', '.join(df.columns)}

Sample Statistics:
{summary_stats[:MAX_TEXT_LENGTH]}

Include:
1. What the data represents
2. Key fields
3. Number of records
4. Data quality
5. Potential uses"""

                    elif analysis_type == "Statistical Analysis":
                        prompt = f"""Analyze the statistics of this dataset:

Columns: {', '.join(df.columns)}
Total Records: {len(df)}

Statistics:
{summary_stats[:MAX_TEXT_LENGTH]}

Provide:
1. Average values
2. Distribution insights
3. Outliers
4. Correlations (if applicable)
5. Statistical significance"""

                    elif analysis_type == "Data Quality Report":
                        # Create clean missing values summary (not as code)
                        missing = df.isnull().sum()
                        missing_pct = (missing / len(df) * 100).round(2)
                        quality_data = f"""MISSING VALUES BY COLUMN:
{chr(10).join([f'{col}: {missing[col]} ({missing_pct[col]}%)' for col in df.columns])}

COMPLETENESS SUMMARY:
Total Records: {len(df)}
Complete Rows (no missing values): {len(df[df.isnull().sum(axis=1) == 0])}
Rows with Missing Values: {len(df[df.isnull().sum(axis=1) > 0])}
Overall Data Completeness: {(100 - df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%"""
                        
                        prompt = f"""Evaluate data quality based on this report:

{quality_data[:MAX_TEXT_LENGTH]}

Column Details:
{summary_stats[:MAX_TEXT_LENGTH]}

Provide (numbered list):
1. Data completeness percentage
2. Issues found
3. Data consistency concerns
4. Recommendations
5. Priority fixes needed"""

                    else:  # Insights
                        prompt = f"""Extract key insights from this dataset:

Columns: {', '.join(df.columns)}
Total Records: {len(df)}

Statistics:
{summary_stats[:MAX_TEXT_LENGTH]}

Provide:
1. Top 5 insights
2. Surprising findings
3. Patterns or trends
4. Business implications
5. Recommendations"""

                    result = call_groq_api(prompt, max_tokens=800)
                    st.success("✅ Analysis complete!")
                    st.write(result)
        
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 9: FINANCIAL REPORT ANALYZER by anupamLab
# ═══════════════════════════════════════════════════════════════════════════

with tab9:
    st.subheader("💰 Financial Report Analyzer")
    st.write("Upload financial data (CSV/TXT) for analysis and insights.")
    
    fin_file = st.file_uploader(
        "Upload financial data (CSV or TXT):",
        type=["csv", "txt"],
        key="finance_analyzer"
    )
    
    if fin_file:
        try:
            # Read file
            if fin_file.name.endswith(".csv"):
                fin_df = pd.read_csv(fin_file)
                st.success("✅ Financial data loaded")
                
                # Display data as clean table
                st.subheader("📊 Financial Data")
                st.dataframe(fin_df, use_container_width=True)
                
                # Display statistics
                st.subheader("📈 Financial Statistics")
                st.dataframe(fin_df.describe(), use_container_width=True)
                
                # Prepare cleaner data for AI
                fin_content = fin_df.to_csv(index=False)
            else:
                fin_content = fin_file.read().decode("utf-8")
                st.success("✅ Financial data loaded")
                st.subheader("📄 Data Preview")
                st.write(fin_content[:500])  # Show first 500 chars
            
            st.divider()
            
            # Analysis options
            fin_analysis = st.radio(
                "Analysis type:",
                ["Simple Explanation", "Key Metrics", "Business Health", "Investment Potential"],
                horizontal=True
            )
            
            if st.button("Analyze", key="finance_btn"):
                with st.spinner("Analyzing financial data..."):
                    if fin_analysis == "Simple Explanation":
                        prompt = f"""In simple terms, explain this financial data with clean numbers only:

{fin_content[:MAX_TEXT_LENGTH]}

Provide (numbered list):
1. What the numbers represent
2. Overall financial health
3. Strong areas
4. Areas of concern
5. Simple summary for non-experts"""

                    elif fin_analysis == "Key Metrics":
                        prompt = f"""Identify and explain key financial metrics from this data:

{fin_content[:MAX_TEXT_LENGTH]}

Provide (numbered list):
1. Important ratios/metrics found
2. What each metric means
3. Good vs concerning values
4. Comparisons if available
5. Implications"""

                    elif fin_analysis == "Business Health":
                        prompt = f"""Assess the overall business health from this financial data:

{fin_content[:MAX_TEXT_LENGTH]}

Provide (numbered list):
1. Overall financial health (score 1-10)
2. Strengths
3. Weaknesses
4. Risks
5. Recommendations"""

                    else:  # Investment Potential
                        prompt = f"""Evaluate investment potential from this financial data:

{fin_content[:MAX_TEXT_LENGTH]}

Provide (numbered list):
1. Investment attractiveness (1-10)
2. Growth potential
3. Risk level (low/medium/high)
4. Comparison to industry standards
5. Investment recommendation (buy/hold/avoid)"""

                    result = call_groq_api(prompt, max_tokens=800)
                    st.success("✅ Analysis complete!")
                    st.write(result)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 10: CODE EXPLAINER by anupamLab
# ═══════════════════════════════════════════════════════════════════════════

with tab10:
    st.subheader("💻 Code Explainer")
    st.write("Upload a code file or paste code to get explanation, logic breakdown, and improvements.")
    
    # Initialize session state for code
    if "uploaded_code" not in st.session_state:
        st.session_state.uploaded_code = ""
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = ""
    
    st.divider()
    
    # Two methods: Upload or Paste
    method_col1, method_col2 = st.columns(2)
    
    # METHOD 1: UPLOAD CODE FILE
    with method_col1:
        st.write("**Method 1: 📁 Upload Code File**")
        code_file = st.file_uploader(
            "Upload code file (.py, .js, .java, .cpp, .txt, etc):",
            type=["py", "js", "java", "cpp", "c", "rb", "go", "rs", "ts", "jsx", "tsx", "txt"],
            key="code_file_uploader"
        )
        
        if code_file is not None:
            try:
                code_content = code_file.read().decode("utf-8")
                st.session_state.uploaded_code = code_content
                st.session_state.uploaded_filename = code_file.name
                st.success(f"✅ File loaded: {code_file.name}")
                
                # Show file size
                file_size = len(code_content)
                st.caption(f"📊 {file_size:,} characters uploaded")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # METHOD 2: PASTE CODE
    with method_col2:
        st.write("**Method 2: ⌨️ Paste Code**")
        manual_code = st.text_area(
            "Or paste your code here:",
            placeholder="# Paste Python, JavaScript, or any code\n\ndef hello():\n    print('Hello')",
            key="code_input",
            height=150
        )
        
        if manual_code:
            st.session_state.uploaded_code = manual_code
            st.session_state.uploaded_filename = "pasted_code"
    
    st.divider()
    
    # Get the code to analyze
    code_input = st.session_state.uploaded_code
    
    if code_input:
        st.success(f"✅ Code ready for analysis ({len(code_input):,} characters)")
        
        # Auto-detect language or let user choose
        col1, col2 = st.columns(2)
        
        with col1:
            lang = st.selectbox(
                "Programming language:",
                ["Python", "JavaScript", "Java", "C++", "C", "Ruby", "Go", "Rust", "TypeScript", "Other"],
                key="code_lang"
            )
        
        with col2:
            detail_level = st.selectbox(
                "Explanation level:",
                ["Simple (Beginner)", "Moderate (Intermediate)", "Detailed (Advanced)"],
                key="code_detail"
            )
        
        # Show code preview
        st.subheader("📝 Code Preview (First 20 Lines)")
        lines = code_input.split('\n')
        preview_lines = min(20, len(lines))
        st.code(code_input[:1000] if len(code_input) > 1000 else code_input, language=lang.lower())
        
        st.info(f"⚡ **Entire Code Will Be Analyzed**: {len(code_input):,} characters across {len(lines)} lines will be sent to AI for complete analysis")
        
        if st.button("🚀 Explain Code", key="code_btn", use_container_width=True):
            with st.spinner("Analyzing entire code..."):
                # Use higher limit for code analysis (not the normal MAX_TEXT_LENGTH)
                CODE_ANALYSIS_LIMIT = 15000
                code_to_analyze = code_input if len(code_input) <= CODE_ANALYSIS_LIMIT else code_input[:CODE_ANALYSIS_LIMIT] + "\n... (truncated)"
                
                if detail_level == "Simple (Beginner)":
                    prompt = f"""Explain this {lang} code in very simple terms (as if teaching a beginner):

```{lang}
{code_to_analyze}
```

Provide (numbered list):
1. What does this code do?
2. What are the main parts?
3. How does it work step-by-step?
4. Example input/output
5. Common mistakes beginners make"""

                elif detail_level == "Moderate (Intermediate)":
                    prompt = f"""Explain this {lang} code at an intermediate level:

```{lang}
{code_to_analyze}
```

Provide (numbered list):
1. Purpose and overview
2. Algorithm explanation
3. Key functions/methods
4. Data structures used
5. Time/space complexity (if applicable)
6. Potential issues"""

                else:  # Detailed
                    prompt = f"""Provide a detailed analysis of this code:

```{lang}
{code_to_analyze}
```

Provide (numbered list):
1. Code design patterns
2. Algorithmic analysis
3. Edge cases
4. Performance considerations
5. Security implications
6. Refactoring suggestions
7. Best practices"""

                explanation = call_groq_api(prompt, max_tokens=1500)
                st.success("✅ Complete analysis ready!")
                st.write(explanation)
        
        # Clear button
        if st.button("🔄 Clear Code", key="code_clear"):
            st.session_state.uploaded_code = ""
            st.session_state.uploaded_filename = ""
            st.rerun()
    else:
        st.info("👆 Upload a code file or paste code above to analyze")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 11: EMAIL/CONTENT GENERATOR by anupamLab
# ═══════════════════════════════════════════════════════════════════════════

with tab11:
    st.subheader("✉️ Email & Content Generator")
    st.write("Generate professional emails, social media posts, and marketing content.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        content_type = st.selectbox(
            "Content type:",
            ["Professional Email", "Follow-up Email", "LinkedIn Post", 
             "Marketing Post", "Product Description", "Social Media Caption"],
            key="content_type"
        )
    
    with col2:
        tone = st.selectbox(
            "Tone:",
            ["Professional", "Casual", "Friendly", "Formal", "Creative"],
            key="content_tone"
        )
    
    # Get content requirements
    st.write("**What's it about?**")
    content_topic = st.text_area(
        "Describe the purpose, product, or message:",
        placeholder="e.g., Following up on a meeting, promoting a new product, etc.",
        key="content_topic",
        height=100
    )
    
    if content_topic and st.button("Generate Content", key="content_btn"):
        with st.spinner("Generating..."):
            if content_type == "Professional Email":
                prompt = f"""Write a professional email with {tone} tone about:

{content_topic}

Include:
1. Proper greeting
2. Clear subject line
3. Main message
4. Call to action
5. Professional closing

Keep it concise and impactful."""

            elif content_type == "Follow-up Email":
                prompt = f"""Write a professional follow-up email with {tone} tone:

{content_topic}

Include:
1. Reference to previous communication
2. Recap of key points
3. Next steps
4. Soft call to action
5. Professional closing"""

            elif content_type == "LinkedIn Post":
                prompt = f"""Write an engaging LinkedIn post with {tone} tone:

{content_topic}

Include:
1. Attention-grabbing opening
2. Main message/value
3. Relevant insights
4. Call to engagement
5. Appropriate emojis (if {tone} allows)
6. Hashtags

Keep it professional yet engaging."""

            elif content_type == "Marketing Post":
                prompt = f"""Write a marketing post with {tone} tone:

{content_topic}

Include:
1. Attention hook
2. Problem statement
3. Solution highlight
4. Benefits (3-4)
5. Call to action
6. Emojis and formatting

Make it persuasive and action-oriented."""

            elif content_type == "Product Description":
                prompt = f"""Write a product description with {tone} tone:

{content_topic}

Include:
1. Product overview
2. Key features
3. Unique benefits
4. Target audience
5. Value proposition
6. Call to action

Be compelling and clear."""

            else:  # Social Media Caption
                prompt = f"""Write a social media caption with {tone} tone:

{content_topic}

Include:
1. Hook/opening
2. Main message
3. Engagement element
4. Emojis (moderate use)
5. Call to action
6. Relevant hashtags

Keep it short, punchy, and engaging."""

            generated = call_groq_api(prompt, max_tokens=500)
            st.success("Content generated!")
            st.write(generated)
            
            # Copy button
            st.download_button(
                label="📥 Download Content",
                data=generated,
                file_name=f"{content_type}.txt",
                mime="text/plain"
            )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 12: LEARNING ASSISTANT by anupamLab
# ═══════════════════════════════════════════════════════════════════════════

with tab12:
    st.subheader("🎓 Learning Assistant")
    st.write("Ask anything and get step-by-step explanations designed for beginners.")
    
    st.write("**Ask your question:**")
    learn_question = st.text_area(
        "What would you like to learn about?",
        placeholder="e.g., How does photosynthesis work? How to code in Python? What is cryptocurrency?",
        key="learn_question",
        height=100
    )
    
    if learn_question:
        col1, col2 = st.columns(2)
        
        with col1:
            learning_style = st.selectbox(
                "Learning style:",
                ["Step-by-Step", "Visual Analogy", "Question & Answer", "Real Examples"],
                key="learn_style"
            )
        
        with col2:
            detail = st.selectbox(
                "Detail level:",
                ["Very Simple", "Beginner-Friendly", "Intermediate", "Comprehensive"],
                key="learn_detail"
            )
        
        if st.button("Explain", key="learn_btn"):
            with st.spinner("Preparing explanation..."):
                if learning_style == "Step-by-Step":
                    prompt = f"""Explain this concept step-by-step for a beginner:
{learn_question}

Format:
1. Simple definition
2. Step-by-step breakdown (numbered)
3. Key terms explained
4. Real-world example
5. Common misconceptions
6. Further learning tips

Use {detail} language."""

                elif learning_style == "Visual Analogy":
                    prompt = f"""Explain this using analogies and comparisons:
{learn_question}

Include:
1. Simple analogy/comparison
2. How it's similar
3. How it's different
4. Visual description
5. Real-world connection
6. Example scenario

Make it easy to visualize."""

                elif learning_style == "Question & Answer":
                    prompt = f"""Explain in Q&A format:
{learn_question}

Include:
1. What is it?
2. Why is it important?
3. How does it work?
4. When is it used?
5. What are examples?
6. Common questions answered

Use {detail} language."""

                else:  # Real Examples
                    prompt = f"""Explain using real-world examples:
{learn_question}

Include:
1. Simple definition
2. Real examples (3-4)
3. How it works in practice
4. Benefits of understanding
5. Common applications
6. Resources to learn more

Make examples relatable."""

                explanation = call_groq_api(prompt, max_tokens=1000)
                st.success("Explanation ready!")
                st.write(explanation)
                
                # Encourage further learning
                st.divider()
                if st.button("Ask a Follow-up Question", key="followup_enable"):
                    followup = st.text_input("Your follow-up question:", key="followup_input")
                    if followup and st.button("Answer Follow-up", key="followup_btn"):
                        with st.spinner("Thinking..."):
                            followup_answer = call_groq_api(
                                f"Based on the topic '{learn_question}', answer this follow-up: {followup}",
                                max_tokens=500
                            )
                            st.write(followup_answer)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 13: IMAGE TO TEXT (OCR)  by anupamLab
# ═══════════════════════════════════════════════════════════════════════════

with tab13:
    st.subheader("🖼️ Image to Text (OCR)")
    st.write("Upload an image (JPG, PNG, etc.) to extract text from it using AI-powered OCR.")
    
    # Initialize session state for image operations
    if "image_extracted_text" not in st.session_state:
        st.session_state.image_extracted_text = ""
    if "image_uploaded_filename" not in st.session_state:
        st.session_state.image_uploaded_filename = ""
    
    # Check if EasyOCR is available
    if easyocr is None:
        st.error("❌ EasyOCR is not installed")
        st.info("To enable this feature, install EasyOCR:")
        st.code("pip install easyocr pillow", language="bash")
    else:
        # File upload
        uploaded_image = st.file_uploader(
            "Upload an image file (JPG, PNG, BMP, TIFF, etc.)",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            key="image_uploader"
        )
        
        if uploaded_image:
            # Show the uploaded image
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📸 Uploaded Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("🔧 Extract Text")
                
                # Extract text button
                if st.button("📖 Extract Text from Image", key="extract_image_btn", use_container_width=True):
                    with st.spinner("🔄 Loading OCR model..."):
                        extracted_text, message = extract_text_from_image(uploaded_image)
                    
                    st.write(message)
                    
                    if extracted_text:
                        # Store in session state
                        st.session_state.image_extracted_text = extracted_text
                        st.session_state.image_uploaded_filename = uploaded_image.name
                        st.success("✅ Text extraction complete! Scroll down to analyze.")
                    else:
                        st.error("❌ Failed to extract text from image")
            
            # Display extracted text section (if text was extracted)
            if st.session_state.image_extracted_text:
                st.divider()
                st.subheader("📝 Extracted Text")
                
                extracted_text = st.session_state.image_extracted_text
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Characters", len(extracted_text))
                with col2:
                    st.metric("Words", len(extracted_text.split()))
                with col3:
                    st.metric("Lines", len(extracted_text.split('\n')))
                
                st.divider()
                
                # Show the extracted text in a text area
                st.text_area(
                    "Extracted Text:",
                    value=extracted_text,
                    height=250,
                    disabled=True,
                    key="extracted_text_display"
                )
                
                st.divider()
                
                # Download button
                st.download_button(
                    label="📥 Download Extracted Text",
                    data=extracted_text,
                    file_name=f"extracted_text_{st.session_state.image_uploaded_filename}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                st.divider()
                
                # AI Analysis of extracted text
                st.subheader("🤖 AI Analysis Options")
                
                # Initialize analysis session state
                if "image_analysis_type" not in st.session_state:
                    st.session_state.image_analysis_type = "Summarize Text"
                
                # Radio button to select analysis type
                st.session_state.image_analysis_type = st.radio(
                    "What would you like to do with the extracted text?",
                    ["Summarize Text", "Ask Questions", "Translate to Simple", "Get Key Points"],
                    index=["Summarize Text", "Ask Questions", "Translate to Simple", "Get Key Points"].index(st.session_state.image_analysis_type)
                )
                
                st.divider()
                
                # ===== ANALYSIS OPTION 1: SUMMARIZE TEXT =====
                if st.session_state.image_analysis_type == "Summarize Text":
                    st.write("**📋 Summary Options:**")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        summary_length = st.selectbox(
                            "Select summary length:",
                            ["Short", "Medium", "Long"],
                            index=1,
                            key="img_summary_length_select"
                        )
                    with col2:
                        st.write("")  # Spacer
                        st.write("")  # Spacer
                        generate_summary_btn = st.button("📋 Generate", key="img_gen_summary", use_container_width=True)
                    
                    if generate_summary_btn:
                        with st.spinner("✨ Generating summary..."):
                            summary = summarize_text(st.session_state.image_extracted_text, summary_length.lower())
                            st.success("✅ Summary generated!")
                            st.info(summary)
                            st.divider()
                
                # ===== ANALYSIS OPTION 2: ASK QUESTIONS =====
                elif st.session_state.image_analysis_type == "Ask Questions":
                    st.write("**❓ Ask About the Text:**")
                    question = st.text_input(
                        "Enter your question:",
                        placeholder="What is this text about? Who is mentioned? What happened?",
                        key="img_question_input"
                    )
                    
                    if question:
                        if st.button("❓ Get Answer", key="img_get_answer", use_container_width=True):
                            with st.spinner("🤔 Thinking..."):
                                answer = ask_question_with_context(st.session_state.image_extracted_text, question)
                                st.success("✅ Answer ready!")
                                st.info(answer)
                                st.divider()
                    else:
                        st.info("💡 Ask a question about the extracted text above")
                
                # ===== ANALYSIS OPTION 3: TRANSLATE TO SIMPLE =====
                elif st.session_state.image_analysis_type == "Translate to Simple":
                    st.write("**🔤 Simplify Complex Text:**")
                    st.write("Convert the text into simple, easy-to-understand language")
                    
                    if st.button("✨ Simplify Now", key="img_simplify_text", use_container_width=True):
                        with st.spinner("🔧 Simplifying text..."):
                            prompt = f"""Rewrite this text in VERY SIMPLE language that a 10-year-old can understand:

{st.session_state.image_extracted_text[:MAX_TEXT_LENGTH]}

Requirements:
1. Use simple, common words
2. Use short sentences
3. No jargon or technical terms
4. Clear and easy to follow
5. Keep the main meaning

Output only the simplified text."""
                            
                            simplified = call_groq_api(prompt, max_tokens=1000)
                            st.success("✅ Simplified text ready!")
                            st.info(simplified)
                            st.divider()
                
                # ===== ANALYSIS OPTION 4: GET KEY POINTS =====
                elif st.session_state.image_analysis_type == "Get Key Points":
                    st.write("**📌 Extract Key Information:**")
                    st.write("Find the main ideas and important details")
                    
                    if st.button("🎯 Extract Key Points", key="img_keypoints_btn", use_container_width=True):
                        with st.spinner("🔍 Analyzing..."):
                            prompt = f"""From this text, extract ONLY the most important key points:

{st.session_state.image_extracted_text[:MAX_TEXT_LENGTH]}

Provide:
1. **Main Topic** - What is this about?
2. **Key Points** - 3-5 most important facts (numbered)
3. **Important Details** - Critical information
4. **Main Conclusion** - The takeaway

Keep each point short and clear."""
                            
                            key_points = call_groq_api(prompt, max_tokens=800)
                            st.success("✅ Key points extracted!")
                            st.info(key_points)
                            st.divider()
                
                # Clear extracted text button
                st.divider()
                if st.button("🗑️ Clear & Upload New Image", key="img_clear_text"):
                    st.session_state.image_extracted_text = ""
                    st.session_state.image_uploaded_filename = ""
                    st.session_state.image_analysis_type = "Summarize Text"
                    st.rerun()
        
        else:
            col1, col2, col3 = st.columns(3)
            with col2:
                st.info("👆 Upload an image to get started")

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("""

**mail to** [infoanupampal@gmail.com]
""")

# Display session info (optional)
with st.sidebar:
    st.divider()
    st.subheader("ℹ️ About This App")
    st.write("""
    The solution provides 13 AI-powered tools for different tasks.
    """)
    
    # Features list
    st.markdown("""
    
    1. 📄 **Document Analyzer** - PDF/TXT analysis
    2. 🌐 **URL Analyzer** - Website content extraction
    3. 📄 **Resume Analyzer** - ATS feedback & skills
    4. 💬 **Chat with Doc** - Continuous conversations
    5. 🔍 **Multi-URL Research** - Compare websites
    6. 📚 **Notes Generator** - Study notes on any topic
    7. 🎙️ **Voice Assistant** - Conversational AI
    8. 📊 **Data Analyzer** - CSV analysis & insights
    9. 💰 **Finance Analyzer** - Financial reports
    10. 💻 **Code Explainer** - Code explanation
    11. ✉️ **Email Generator** - Professional content
    12. 🎓 **Learning Assistant** - Learn anything
    13. 🖼️ **Image to Text** - OCR image extraction
    """)
    
    st.divider()
    
    st.write("**API Model:** llama-3.1-8b-instant")
    st.write("**Developed By:** Anupam Pal")
