# Ai-resume-assistant
# 📄 ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) friendliness and suggests concrete improvements, powered by Google Gemini Flash.

## Features
- Upload a resume as **PDF** or **DOCX**
- Optional **job description** for a targeted score and keyword gap analysis
- Overall ATS score (0-100) with a category breakdown
- Strengths, prioritised improvements, missing keywords and rewrite examples
- Instant rule-based checks (email, phone, links, sections, length, metrics)
- Download the report as JSON

## Run locally
1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free API key from https://aistudio.google.com/apikey
4. Set the key (pick one):
   - Create `.streamlit/secrets.toml` containing:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```
   - or set an environment variable: `export GEMINI_API_KEY="your-key-here"`
   - or paste it into the app sidebar at runtime
5. Start the app:
   ```bash
   streamlit run app.py
   ```

## Configuration
| Name | Purpose | Default |
|------|---------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | none (required) |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash` |

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (never commit your API key).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Notes
- Scanned or image-only resumes cannot be read; use a text-based PDF or DOCX.
- The ATS score is an AI estimate, not the output of any real employer's ATS.
- Resume text is sent to Google's Gemini API for analysis and is not stored by this app.

## Project structure
```
app.py            # Streamlit app
requirements.txt  # Python dependencies
README.md         # This file
```
