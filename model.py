# --- VERY FIRST LINES: Prevent Streamlit inspection issues ---
# (Keep these lines if they are necessary for your environment, but the torch.classes patch is removed)
# import torch
# torch.classes = type("classes", (), {"__path__": None}) # Removed this problematic patch
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false" # Keep attempting to disable watcher

# --- Configuration ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Keep if needed for your environment
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true" # Keep if needed
import warnings # Import the warnings library
warnings.filterwarnings("ignore") # Suppress warnings

# --- Now do imports ---

import streamlit as st
import sqlite3
import pandas as pd
import fitz # PyMuPDF
import requests
from sentence_transformers import SentenceTransformer, util
from typing import Union, List # Added for type hinting
import traceback # For detailed error logging

# --- Constants ---
DB_PATH = "candidates.db"
OLLAMA_MODEL = "llama3" # Or your preferred Ollama model (e.g., "mistral", "phi3")
OLLAMA_TIMEOUT = 180 # Increased timeout to 3 minutes for potentially longer generations
SIMILARITY_THRESHOLD = 75.0 # Score percentage to shortlist

# --- Performance Optimizations & Model Loading ---
@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """Loads the Sentence Transformer model using Streamlit's caching."""
    st.write("Attempting to load embedding model...") # More visible feedback
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.write("Embedding model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"CRITICAL ERROR: Failed to load embedding model: {e}")
        st.error("The application cannot function without the embedding model. Please check your internet connection and library installation.")
        st.stop() # Stop execution if model fails to load

# --- Database Functions ---
def init_db():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shortlisted (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_name TEXT NOT NULL,
                    job_title TEXT NOT NULL,
                    score REAL NOT NULL,
                    email_draft TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        print("Database initialized successfully.")
    except sqlite3.Error as e:
        st.error(f"Database Error: Failed to initialize database - {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during DB init: {e}")

def save_to_db(candidate_name: str, score: float, email_draft: str, job_title: str):
    """Saves shortlisted candidate details to the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO shortlisted (candidate_name, job_title, score, email_draft) VALUES (?, ?, ?, ?)",
                (candidate_name, job_title, score, email_draft)
            )
            conn.commit()
        print(f"Saved {candidate_name} for {job_title} to DB.")
    except sqlite3.Error as e:
        st.error(f"Database Error: Failed to save candidate {candidate_name} - {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred saving to DB: {e}")


# --- Text Processing & AI Functions ---
def extract_text_from_pdf(uploaded_file) -> str:
    """Extracts text from an uploaded PDF file with progress tracking."""
    text = ""
    progress_bar = None # Initialize progress_bar
    try:
        # Read the file content into memory
        file_bytes = uploaded_file.read()
        if not file_bytes:
             st.warning(f"Uploaded file '{uploaded_file.name}' is empty.")
             return "" # Return empty string for empty files

        # Create a progress bar context outside the fitz block
        progress_bar = st.progress(0.0, text=f"Opening PDF: {uploaded_file.name}...")
        total_pages = 0

        # Open the PDF from bytes
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            total_pages = len(doc)
            if total_pages == 0:
                st.warning(f"PDF '{uploaded_file.name}' has 0 pages.")
                progress_bar.progress(1.0, text=f"Extraction complete (0 pages) for {uploaded_file.name}.")
                # Reset file pointer (important if reading again)
                uploaded_file.seek(0)
                return ""

            for i, page in enumerate(doc):
                text += page.get_text()
                # Update progress bar based on pages processed
                progress = float((i + 1) / total_pages)
                progress_bar.progress(progress, text=f"Extracting text from '{uploaded_file.name}'... Page {i+1}/{total_pages}")

        progress_bar.progress(1.0, text=f"Text extraction complete for {uploaded_file.name}.")
        # Reset file pointer (important if reading again)
        uploaded_file.seek(0)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF '{uploaded_file.name}': {e}")
        # Ensure progress bar completes even on error, maybe show error status
        if progress_bar:
             progress_bar.progress(1.0, text=f"Error during extraction for {uploaded_file.name}")
        # Reset file pointer if possible
        try:
            uploaded_file.seek(0)
        except Exception:
            pass # Ignore errors if seek fails after an error
        return f"Error extracting text: {e}" # Return error message

def ollama_generate(model_name: str, prompt: str) -> str:
    """Generates text using Ollama with timeout and error handling."""
    api_url = "http://localhost:11434/api/generate" # Ensure this endpoint is correct
    try:
        response = requests.post(
            api_url,
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT # Use the increased timeout
        )
        response.raise_for_status() # Raises HTTPError for bad responses (4XX, 5XX)
        data = response.json()
        generated_text = data.get("response", "").strip()
        if not generated_text:
            # Check for context window issues or other common problems if response is empty
             error_context = data.get("error", "Received empty response without specific error.")
             return f"❌ Ollama Error: Model '{model_name}' returned an empty response. Possible reason: {error_context}"
        return generated_text
    except requests.exceptions.ConnectionError:
        return f"❌ Ollama Error: Connection refused at {api_url}. Is Ollama running?"
    except requests.exceptions.Timeout:
        return f"❌ Ollama Error: Request timed out after {OLLAMA_TIMEOUT} seconds. Ollama might be too slow, the model too large for your hardware, or the timeout needs increasing further."
    except requests.exceptions.RequestException as e:
        # Catch other request errors (like 404 if URL is wrong, or server errors)
        return f"❌ Ollama Error: Request failed - {str(e)}"
    except Exception as e:
        # Catch unexpected errors (e.g., JSON decoding errors)
        return f"❌ Ollama Error: An unexpected error occurred - {str(e)}"

def summarize_jd(jd_text: str) -> str:
    """Summarizes the job description using Ollama."""
    if not jd_text or not jd_text.strip():
        return "Error: Job Description text is empty."

    prompt = f"""
    Please carefully analyze the following job description and provide a concise summary highlighting ONLY the essential information needed to match candidate skills and experience. Focus strictly on:
    1. Core Responsibilities (main duties).
    2. Required Technical Skills (programming languages, frameworks, tools, software).
    3. Required Experience (years, specific domains).
    4. Key Qualifications (education level, certifications if mandatory).

    Exclude information about company culture, benefits, mission statements, diversity statements, application instructions, and equal opportunity clauses. Keep the summary factual and keyword-rich for semantic matching.

    Job Description:
    ---
    {jd_text}
    ---

    Concise Summary for Matching:
    """
    return ollama_generate(OLLAMA_MODEL, prompt)

def calculate_similarity(text1: str, text2: str) -> Union[float, str]:
    """Calculates cosine similarity between two texts using the sentence transformer model."""
    # Input validation
    if not isinstance(text1, str) or not text1.strip():
        return "Error: Cannot calculate similarity, input text 1 is invalid or empty."
    if not isinstance(text2, str) or not text2.strip():
        return "Error: Cannot calculate similarity, input text 2 is invalid or empty."
    if text1.startswith("Error:") or text2.startswith("Error:") or text2.startswith("❌ Ollama Error:"):
         return "Error: Cannot calculate similarity due to preceding errors in text generation or extraction."

    try:
        model = load_embedding_model() # Get the cached model
        # Check if model loading failed earlier (st.stop() should prevent reaching here, but double-check)
        if model is None:
            return "Error: Embedding model is not available. Application cannot proceed."

        # It's possible the model exists but encode fails if text is too large or problematic
        embedding1 = model.encode(text1, convert_to_tensor=True, show_progress_bar=False)
        embedding2 = model.encode(text2, convert_to_tensor=True, show_progress_bar=False)

        # Compute cosine similarity
        cosine_scores = util.cos_sim(embedding1, embedding2)
        similarity_score = cosine_scores.item() * 100 # Get single score and convert to percentage

        # Clamp score between 0 and 100
        similarity_score = max(0.0, min(similarity_score, 100.0))

        return similarity_score
    except RuntimeError as e:
         # Catch potential OOM errors from sentence-transformers
         if "out of memory" in str(e).lower():
             return f"Error calculating similarity: Ran out of memory. Input text might be too long. Details: {str(e)}"
         else:
             return f"Error calculating similarity: A runtime error occurred. Details: {str(e)}"
    except Exception as e:
        st.error(f"Unexpected error calculating similarity: {e}")
        return f"Error calculating similarity: {str(e)}"


def generate_interview_email(candidate_name: str, job_title: str) -> str:
    """Generates a draft email for inviting a candidate to an interview using Ollama."""
    prompt = f"""
    Please write a polite and professional email draft to invite a candidate named '{candidate_name}'
    for an initial interview for the '{job_title}' position.

    Key points to include:
    - Acknowledge their application for the specific role.
    - State that their profile appears to be a good match based on initial review.
    - Express interest in learning more about their experience.
    - Suggest proceeding to the next step, an interview.
    - Request their availability for a brief call/video conference in the coming days.
    - Include placeholders like "[Interviewer Name]", "[Your Company Name]", "[Your Name/HR Department]", and "[Contact Information/Scheduling Link]".
    - Keep the tone welcoming and professional.

    Subject: Interview Invitation: {job_title} Position - {candidate_name}

    Email Draft:
    """
    return ollama_generate(OLLAMA_MODEL, prompt)


# --- Main App Logic ---
st.set_page_config(layout="wide") # Use wider layout
st.title("🚀 AI-Powered Job Screening Tool")

# Initialize DB and load model once using session state
# Model loading is triggered by the first call to load_embedding_model() via cache
if "initialized" not in st.session_state:
    with st.spinner("Initializing database..."):
        init_db()
        # Trigger model loading here explicitly within the init block if needed,
        # otherwise it loads on first calculate_similarity call.
        # load_embedding_model() # Uncomment if you want to load upfront
        st.session_state.initialized = True
        st.success("Initialization complete. Ready to upload files.")


# --- File Upload and Processing Form ---
with st.form("upload_form"):
    st.header("1. Upload Files")
    col1, col2 = st.columns(2)
    with col1:
        jd_file = st.file_uploader("📄 Upload Job Descriptions CSV", type=['csv'], help="CSV must contain 'Job Title' and 'Job Description' columns.")
    with col2:
        cv_files = st.file_uploader("📎 Upload Candidate CVs (PDF)", type=['pdf'], accept_multiple_files=True, help="Upload one or more PDF resumes.")

    submitted = st.form_submit_button("🚀 Start Screening", type="primary")

# --- Processing Logic ---
if submitted and jd_file and cv_files:
    st.header("2. Screening Results")
    try:
        # Use specified encoding, consider adding an option or detecting encoding
        # Wrap CSV reading in try-except for specific file reading errors
        try:
             jd_df = pd.read_csv(jd_file, encoding="ISO-8859-1")
        except FileNotFoundError:
            st.error("Job Description CSV file not found.")
            st.stop()
        except pd.errors.EmptyDataError:
            st.error("Job Description CSV file is empty.")
            st.stop()
        except Exception as e:
             st.error(f"Error reading Job Description CSV: {e}")
             st.stop()


        # Validate CSV columns
        required_cols = {'Job Title', 'Job Description'}
        if not required_cols.issubset(jd_df.columns):
            st.error(f"CSV file is missing required columns. It must contain: {', '.join(required_cols)}")
        else:
            # Process each job description
            for jd_index, jd_row in jd_df.iterrows():
                job_title = jd_row['Job Title']
                jd_text = jd_row['Job Description']

                # Basic validation of row data
                if pd.isna(job_title) or not str(job_title).strip():
                     st.warning(f"Skipping Job Description at index {jd_index} due to missing or empty 'Job Title'.")
                     continue
                if pd.isna(jd_text) or not str(jd_text).strip():
                     st.warning(f"Skipping Job Description '{job_title}' (index {jd_index}) due to missing or empty 'Job Description'.")
                     continue

                st.subheader(f"Processing Job: {job_title}")

                # Summarize JD
                with st.spinner(f"Analyzing job description for '{job_title}' (this may take time)..."):
                    jd_summary = summarize_jd(jd_text)

                # Display JD Summary (handle Ollama errors)
                with st.expander("View Job Summary Analysis", expanded=False):
                    if jd_summary.startswith("❌ Ollama Error:") or jd_summary.startswith("Error:"):
                        st.error(f"Could not summarize Job Description for '{job_title}': {jd_summary}")
                    else:
                        st.success("Job Description Summarized:")
                        st.markdown(jd_summary) # Use markdown for better formatting potential

                # If JD summary failed, skip CV matching for this job
                if jd_summary.startswith("❌ Ollama Error:") or jd_summary.startswith("Error:"):
                     st.warning(f"Skipping CV matching for '{job_title}' due to error in processing the job description.")
                     st.markdown("---") # Add separator even if skipped
                     continue

                st.markdown("---") # Separator before CVs for this job

                # Process each CV against the current JD
                st.write(f"**Matching Candidates for {job_title}:**")
                num_cvs_processed = 0
                for cv_file in cv_files:
                    cv_name = os.path.splitext(cv_file.name)[0] # Get name without extension
                    st.markdown(f"*Candidate: {cv_name}*")

                    # Use columns for better layout of CV processing
                    col_cv_extract, col_cv_score, col_cv_status = st.columns([2,1,2])

                    with col_cv_extract:
                         with st.spinner(f"Extracting text from {cv_name}.pdf..."):
                            cv_text = extract_text_from_pdf(cv_file)

                    # Handle PDF extraction errors
                    if isinstance(cv_text, str) and cv_text.startswith("Error extracting text:"):
                         with col_cv_status:
                             st.error(f"PDF Error: {cv_text}")
                         st.markdown("---") # Separator between candidates
                         continue # Skip to next CV

                    if not cv_text.strip():
                         with col_cv_status:
                             st.warning("CV appears empty after text extraction.")
                         st.markdown("---") # Separator between candidates
                         continue # Skip to next CV


                    # Calculate Similarity
                    with col_cv_score:
                        with st.spinner(f"Matching {cv_name}..."):
                            score = calculate_similarity(cv_text, jd_summary)

                        # Display Score or Error
                        if isinstance(score, str): # Check if calculate_similarity returned an error string
                            with col_cv_status: # Display error in status column
                                st.error(f"Similarity Error: {score}")
                        else:
                             # Display score in metric column
                            st.metric(label="Match Score", value=f"{score:.1f}%")

                            # Determine status based on score
                            with col_cv_status:
                                if score >= SIMILARITY_THRESHOLD:
                                    st.success("✅ Potential Match - Shortlisted!")
                                    with st.spinner(f"Generating interview email draft for {cv_name}..."):
                                        email_draft = generate_interview_email(cv_name, job_title)

                                    # Display Email Draft (handle Ollama errors)
                                    email_expander = st.expander("View Interview Email Draft", expanded=False)
                                    with email_expander:
                                        if email_draft.startswith("❌ Ollama Error:") or email_draft.startswith("Error:"):
                                            st.error(f"Could not generate email: {email_draft}")
                                            # Save anyway, but indicate email generation failed
                                            save_to_db(cv_name, score, f"EMAIL GENERATION FAILED: {email_draft}", job_title)
                                        else:
                                            st.code(email_draft, language=None) # Use st.code for preformatted text
                                            save_to_db(cv_name, score, email_draft, job_title)
                                else:
                                    st.info("ℹ️ Below Threshold - Not Shortlisted")

                    st.markdown("<hr>", unsafe_allow_html=True) # Thinner separator between candidates
                    num_cvs_processed += 1

                if num_cvs_processed == 0:
                    st.info("No CVs were processed for this job description.")
                st.markdown("---") # Separator after processing all CVs for a job

    except Exception as e:
        st.error(f"An critical error occurred during the main processing loop: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}") # More detailed error for debugging

elif submitted:
    st.warning("⚠️ Please upload both a Job Description CSV and at least one CV PDF file.")

# --- Results Sidebar ---
st.sidebar.header("📊 Shortlisted Candidates")
if st.sidebar.button("🔄 Refresh Results"):
    try:
        # Use 'with' statement for automatic connection closing
        with sqlite3.connect(DB_PATH) as conn:
            # Check if table exists before querying
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='shortlisted';")
            if cursor.fetchone():
                 # Select specific columns and provide better names for display
                 df = pd.read_sql("""
                     SELECT
                         candidate_name AS Candidate,
                         job_title AS "Job Title",
                         printf('%.1f%%', score) AS Score, -- Format score
                         CASE
                             WHEN email_draft LIKE 'EMAIL GENERATION FAILED:%' THEN 'Failed'
                             WHEN email_draft IS NULL OR email_draft = '' THEN 'N/A' -- Should not happen if saved correctly
                             ELSE 'Generated'
                         END AS "Email Status",
                         strftime('%Y-%m-%d %H:%M', timestamp) AS Timestamp -- Format timestamp
                     FROM shortlisted
                     ORDER BY job_title, score DESC
                 """, conn)
                 st.sidebar.dataframe(df, use_container_width=True)
                 st.sidebar.success(f"Loaded {len(df)} shortlisted record(s).")
                 if len(df) == 0:
                     st.sidebar.info("No candidates have been shortlisted yet.")
            else:
                st.sidebar.info("No results found. Process files first.")

    except sqlite3.Error as e:
        st.sidebar.error(f"Database Error: Failed to load results - {e}")
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred loading results: {e}")
else:
     st.sidebar.info("Click 'Refresh Results' to view shortlisted candidates after screening.")