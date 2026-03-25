## 🚀 AI POWER APP – All-in-One AI Toolkit

A powerful AI-driven multi-utility web application built with Streamlit + Groq (LLaMA 3.1) that combines multiple real-world AI tools into a single platform for productivity, learning, and analysis.


**Click here 👇**
  ## 🔗 **Live App:** <https://aianupamlab.streamlit.app>

---

## 🔥 Overview

AI POWER APP is designed as a centralized AI workspace where users can:

- Analyze documents
- Generate content
- Interact with AI via voice
- Perform research
- Work with data
- Extract insights from multiple sources

All features are accessible through a clean and interactive UI.

---

## 🧠 Core 13 AI Features

**📄 Document Intelligence**

- Upload PDFs or text files
- Get summaries instantly
- Ask questions based on document content

**🔗 URL Analyzer**

- Extract and analyze webpage content
- Summarize articles or blogs
- Generate insights from live URLs

**📚 AI Research Tool**

- Combine multiple inputs (text, PDF, URL)
- Perform structured research
- Get summarized insights with reasoning

**📝 Notes Generator**

- Convert raw content into structured notes
- Helpful for students and learners

**🎓 Learn Anything**

- Input any topic
- Get simplified explanations and breakdowns

**💼 Resume Analyzer**

- ATS-style analysis
- Identify missing skills
- Get improvement suggestions

**💻 Code Explainer**

- Paste code in any language
- Get detailed explanation
- Understand logic, optimization & improvements

**📊 Data Analyzer**

- Upload CSV files
- Get:
  - Summary statistics
  - Data insights
  - Trend understanding

**💰 Financial Analyzer**

- Analyze financial data or input
- Get:
  - Business insights
  - Financial breakdown
  - Suggestions

**🖼️ Image to Text (OCR)**

- Upload image
- Extract text using OCR (Optical Character Recognition)

**🎙️ Voice Assistant**

- Speak your query
- Convert speech → text → AI response

**✍️ Content Generator**

Generate:
- Emails
- LinkedIn posts
- Marketing content
- Social media captions

**⚙️ Tech Stack**

- Frontend: Streamlit
- Backend: Python
- LLM: Groq API (LLaMA 3.1)

## 📦 Libraries & Their Purpose

- **streamlit**  
  Used to build the **interactive web UI** of the application.  
  Handles layouts, user inputs, and displays outputs in real time.

- **groq**  
  Used to connect with **Groq LLM (LLaMA 3.1)**.  
  Powers AI features like summarization, Q&A, and content generation.

- **pandas**  
  Used for **data manipulation and analysis**.  
  Processes CSV files, generates statistics, and extracts insights.

- **PyPDF2**  
  Used to **extract text from PDF files**.  
  Enables document reading and analysis.

- **requests**  
  Used to **make HTTP requests**.  
  Fetches data from websites and APIs.

- **beautifulsoup4**  
  Used for **web scraping and HTML parsing**.  
  Extracts readable text from web pages.

- **SpeechRecognition**  
  Used for **speech-to-text conversion**.  
  Enables voice-based interaction.

- **EasyOCR**  
  Used for **image-to-text extraction (OCR)**.  
  Reads text from uploaded images.

- **Pillow (PIL)**  
  Used for **image processing**.  
  Handles image loading and preprocessing.

---

## 🛠️ Installation

Install Dependencies

```
pip install -r requirements.txt
```

Add API Key **(.env) file**

```
GROQ_API_KEY=apiValue
```

Run the App

```
python -m streamlit run app.py
```

## 🔑 Configuration

The application requires a Groq API Key.

You can set it via:

- Environment variable ("GROQ_API_KEY")
- Or Streamlit secrets ("st.secrets")

---

## 🎯 Use Cases

- Students → Notes, learning, summaries
- Developers → Code understanding
- Job Seekers → Resume analysis
- Analysts → Data insights
- Researchers → Multi-source research
- Content Creators → Automated writing

---

## ⚠️ Limitations

- Large files may be truncated
- OCR requires initial model download
- Voice feature depends on microphone + internet

---

## 🔮 Future Scope

- Vector database (RAG implementation)
- Chat history & user sessions
- Multi-language support
- Advanced visual analytics
- Deployment scaling

---
## This project was developed as part of my learning journey in Python. As a learning assistant to understand concepts and structure the code took help of GPT Models. The final implementation, testing, and project setup were completed by "infoanupampal@gmail.com"
