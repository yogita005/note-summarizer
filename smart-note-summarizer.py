import os
import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import cv2
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import torch
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class NoteSummarizer:
    def __init__(self):
        """
        Initialize the Note Summarizer with NLP models and configurations
        """
        # Load spaCy English model for advanced NLP
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load summarization and question-answering models
        self.summarizer = pipeline("summarization")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
        self.qa_tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
    
    def perform_ocr(self, image_path):
        """
        Perform Optical Character Recognition on an image
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            str: Extracted text from the image
        """
        # Image preprocessing
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # OCR extraction
        text = pytesseract.image_to_string(threshold)
        return text
    
    def extract_key_entities(self, text):
        """
        Extract key named entities from text
        
        Args:
            text (str): Input text
        
        Returns:
            dict: Named entities by type
        """
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities
    
    def generate_extractive_summary(self, text, max_sentences=5):
        """
        Generate an extractive summary using TextRank algorithm
        
        Args:
            text (str): Input text
            max_sentences (int): Maximum number of summary sentences
        
        Returns:
            list: Key sentences forming the summary
        """
        # Sentence tokenization
        sentences = sent_tokenize(text)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        sentence_vectors = vectorizer.fit_transform(sentences)
        
        # Create similarity graph
        similarity_graph = nx.from_numpy_array(
            cosine_similarity(sentence_vectors.toarray())
        )
        
        # Apply PageRank
        scores = nx.pagerank(similarity_graph)
        
        # Rank sentences and select top sentences
        ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), 
            reverse=True
        )
        
        return [sent for _, sent in ranked_sentences[:max_sentences]]
    
    def abstractive_summary(self, text, max_length=150, min_length=50):
        """
        Generate abstractive summary using transformer model
        
        Args:
            text (str): Input text
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
        
        Returns:
            str: Condensed summary
        """
        summary = self.summarizer(
            text, 
            max_length=max_length, 
            min_length=min_length, 
            do_sample=False
        )[0]['summary_text']
        
        return summary
    
    def answer_questions(self, context, questions):
        """
        Answer specific questions about the text
        
        Args:
            context (str): Input text
            questions (list): Questions to be answered
        
        Returns:
            dict: Answers to questions
        """
        answers = {}
        for question in questions:
            inputs = self.qa_tokenizer.encode_plus(
                question, context, 
                return_tensors="pt", 
                max_length=512
            )
            
            outputs = self.qa_model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores)
            
            answer = self.qa_tokenizer.decode(
                inputs["input_ids"][0][start_index:end_index+1]
            )
            
            answers[question] = answer
        
        return answers

def main():
    st.title("🧠 Smart Note Summarizer")
    
    # Sidebar for configuration
    st.sidebar.header("Summarization Settings")
    summary_type = st.sidebar.selectbox(
        "Summary Type", 
        ["Extractive", "Abstractive", "Hybrid"]
    )
    max_sentences = st.sidebar.slider(
        "Max Summary Sentences", 
        min_value=1, max_value=10, value=5
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Image/Document", 
        type=["png", "jpg", "jpeg", "pdf", "txt"]
    )
    
    # Initialize summarizer
    summarizer = NoteSummarizer()
    
    if uploaded_file is not None:
        # Process based on file type
        if uploaded_file.type.startswith('image'):
            # Save uploaded image temporarily
            with open("temp_image.png", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Perform OCR
            extracted_text = summarizer.perform_ocr("temp_image.png")
            os.remove("temp_image.png")
        else:
            # Read text directly
            extracted_text = uploaded_file.read().decode('utf-8')
        
        # Display extracted text
        st.subheader("Extracted Text")
        st.write(extracted_text)
        
        # Summarization
        st.subheader("Summary")
        if summary_type == "Extractive":
            summary = summarizer.generate_extractive_summary(
                extracted_text, 
                max_sentences=max_sentences
            )
            st.write("\n".join(summary))
        
        elif summary_type == "Abstractive":
            summary = summarizer.abstractive_summary(extracted_text)
            st.write(summary)
        
        else:  # Hybrid
            extractive = summarizer.generate_extractive_summary(
                extracted_text, 
                max_sentences=max_sentences
            )
            abstractive = summarizer.abstractive_summary(extracted_text)
            
            st.write("Extractive Summary:")
            st.write("\n".join(extractive))
            st.write("\nAbstractive Summary:")
            st.write(abstractive)
        
        # Named Entities
        st.subheader("Key Entities")
        entities = summarizer.extract_key_entities(extracted_text)
        st.json(entities)
        
        # Q&A Section
        st.subheader("Ask Questions")
        question_input = st.text_input("Enter a question about the text")
        if question_input:
            answers = summarizer.answer_questions(
                extracted_text, 
                [question_input]
            )
            st.write(answers[question_input])

if __name__ == "__main__":
    main()
