import os
import zipfile
import pdfplumber
import re
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Step 1: Unzip the model archive
model_zip_path = "Enviro_bart_model.zip"
model_dir = "Enviro_bart_model"

if not os.path.exists(model_dir):
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)
    print(f"Model unzipped to {model_dir}")

# Step 2: Load the model and tokenizer
tokenizer = BartTokenizer.from_pretrained(model_dir)
model = BartForConditionalGeneration.from_pretrained(model_dir)

# Step 3: Function to identify environmental issues
def identify_environmental_issues(text):
    environmental_keywords = [
        'climate change', 'global warming', 'pollution', 'deforestation',
        'biodiversity', 'renewable energy', 'carbon emissions', 'sustainability',
        'conservation', 'ecosystem', 'endangered species', 'greenhouse gas',
        'waste management', 'environmental protection', 'air quality',
        'water quality', 'habitat loss', 'natural resources'
    ]

    issues = []
    text_lower = text.lower()

    for keyword in environmental_keywords:
        if keyword in text_lower:
            issues.append(keyword)

    return list(set(issues))

# Step 4: Function to preprocess text
def preprocess_pdf_text(text):
    """
    Comprehensive preprocessing of PDF text to remove unnecessary elements and retain important content.
    """
    lines = text.split('\n')
    cleaned_lines = []

    patterns_to_remove = {
        'email': r'\S+@\S+\.\S+',
        'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        'date': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}',
        'time': r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?',
        'figure_references': r'(?i)figure\s+\d+|fig\.\s*\d+|table\s+\d+|appendix\s+[a-z]',
        'citations': r'\[\d+\]|\(\w+\s+et\s+al\.,\s+\d{4}\)|\(\w+,\s+\d{4}\)',
        'page_numbers': r'Page\s+\d+|\d+\s+of\s+\d+',
        'author_affiliations': r'(?i)^[^a-z]*department\s+of|^[^a-z]*university\s+of|^[^a-z]*institute\s+of',
    }

    def clean_line(line):
        for pattern in patterns_to_remove.values():
            line = re.sub(pattern, '', line)
        return line.strip()

    for line in lines:
        if not line.strip():
            continue
        if len(line.strip()) < 30 and (re.search(r'\d', line) or re.search(r'[^\w\s]', line)):
            continue
        if re.match(r'^([A-Z][a-z]+\s+){2,}$', line.strip()):
            continue
        cleaned_line = clean_line(line)
        if len(cleaned_line) < 30:
            continue
        cleaned_lines.append(cleaned_line)

    text = ' '.join(cleaned_lines)
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)  # Remove special characters except basic punctuation

    sentences = re.split(r'(?<=[.!?]) +', text)  # Simple sentence splitting
    important_sentences = [sent for sent in sentences if len(sent.split()) > 5]
    cleaned_text = ' '.join(important_sentences)

    return cleaned_text.strip()

def analyze_text_importance(text):
    important_phrases = [
        'in conclusion', 'we found', 'results show', 'key findings',
        'significant', 'important', 'furthermore', 'however',
        'therefore', 'thus', 'consequently', 'in summary',
        'notably', 'primary', 'essential', 'critical'
    ]

    sentences = re.split(r'(?<=[.!?]) +', text)  # Simple sentence splitting
    important_sentences = []

    for i, sentence in enumerate(sentences):
        is_important = (
            i < 3 or
            i > len(sentences) - 3 or
            any(phrase in sentence.lower() for phrase in important_phrases) or
            len(sentence.split()) > 20
        )
        if is_important:
            important_sentences.append(sentence)

    return ' '.join(important_sentences)

def process_pdf_text(raw_text):
    cleaned_text = preprocess_pdf_text(raw_text)
    important_text = analyze_text_importance(cleaned_text)
    return important_text

# Step 5: Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                pdf_text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    return pdf_text.strip()

# Step 6: Function to generate summary
def generate_summary(text, max_input_length=1024, max_output_length=250):
    """
    Generate a summary for the given text using the loaded model.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        max_length=max_input_length
    )
    
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=max_output_length,
        min_length=20,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Step 7: Chunking function to break the text into smaller chunks for processing
def chunk_text(text, max_chunk_length=1024):
    """
    Chunk the text into smaller parts so that each part is below the model's input length.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)  # Simple sentence splitting
    chunks = []
    chunk = ""

    for sentence in sentences:
        # Check if adding this sentence would exceed the chunk limit
        if len(tokenizer.encode(chunk + sentence)) <= max_chunk_length:
            chunk += " " + sentence
        else:
            # Add the current chunk and start a new chunk
            chunks.append(chunk.strip())
            chunk = sentence

    # Add the last chunk if any
    if chunk:
        chunks.append(chunk.strip())

    return chunks

# Step 8: Streamlit App Interface
st.title("Environmental Article Summarizer")

# PDF file uploader
uploaded_file = st.file_uploader("Upload a PDF file to generate its summary and identify key environmental issues", type=["pdf"])

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    print("PDF Text:\n\n", pdf_text)

    if pdf_text:
        # Process the extracted text
        clean_text = process_pdf_text(pdf_text)
        print("Cleaned Text:\n\n", clean_text)

        # Identify environmental issues
        st.subheader("Identified Environmental Issues:")
        environmental_issues = identify_environmental_issues(clean_text)

        # Capitalize each word in the identified issues
        capitalized_issues = [issue.title() for issue in environmental_issues]

        if capitalized_issues:
            st.markdown("- " + "\n- ".join(capitalized_issues))
        else:
            st.write("No environmental issues identified.")

        # Chunk the cleaned text into smaller parts
        text_chunks = chunk_text(clean_text)
        print("Text Chunks:\n\n", text_chunks)

        # Generate summary for each chunk and combine them
        full_summary = ""
        for chunk in text_chunks:
            summary = generate_summary(chunk)
            full_summary += summary + "\n"

        st.subheader("Generated Full Summary:")
        st.write(full_summary)

    else:
        st.write("No text found in the PDF.")
