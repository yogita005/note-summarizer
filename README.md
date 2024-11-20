# note-summarizer

Key Features:
- OCR with image preprocessing
- Multiple summarization techniques
- Named entity extraction
- Question-answering capability
- Streamlit interactive interface

Dependencies:
- `streamlit`
- `pytesseract`
- `opencv-python`
- `spacy`
- `nltk`
- `transformers`
- `torch`
- `scikit-learn`

Additional Setup:
1. Install Tesseract OCR
2. Download spaCy model: `python -m spacy download en_core_web_sm`
3. Install dependencies

Summarization Techniques:
1. Extractive: Selects key sentences
2. Abstractive: Generates concise summary
3. Hybrid: Combines both approaches
