import os
import re
import zipfile
import glob

import numpy as np
import pandas as pd
import PyPDF2
from PIL import Image
from gtts import gTTS
from IPython.display import Audio
import pytesseract
from pdf2image import convert_from_path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel

# Libraries for visualization
import matplotlib.pyplot as plt
import graphviz


def extract_pdf_text(pdf_file_path):
    """Extracts text from a PDF file."""
    with open(pdf_file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        all_text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            all_text += page.extract_text()
    return all_text


def preprocess_text(text):
    """Preprocesses text by converting to lowercase and removing punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text


def generate_embeddings(sentences):
    """Generates embeddings for sentences using Legal-BERT."""
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return embeddings


def perform_clustering(embeddings, sentences):
    """Performs KMeans clustering on sentence embeddings."""
    num_clusters = min(20, len(sentences))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42) #added random state for consistency
    kmeans.fit(np.vstack(embeddings))
    cluster_labels = kmeans.labels_
    return cluster_labels


def extract_strong_points(sentences, cluster_labels):
    """Extracts strong points from clusters."""
    strong_points = []
    num_strong_points_per_cluster = 5  # Select top 5 sentences per cluster

    num_clusters = len(set(cluster_labels)) #get the correct number of clusters.

    for cluster_id in range(num_clusters):
        cluster_sentences = [sentences[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        strong_points.extend(cluster_sentences[:num_strong_points_per_cluster])

    if len(strong_points) < 15:
        cluster_sizes = {cluster_id: len([label for label in cluster_labels if label == cluster_id])
                         for cluster_id in range(num_clusters)}
        sorted_clusters = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)
        for cluster_id in sorted_clusters:
            cluster_sentences = [sentences[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            strong_points.extend(cluster_sentences[num_strong_points_per_cluster:15 - len(strong_points)])
            if len(strong_points) >= 15:
                break
    return strong_points


def generate_pdf_report(strong_points):
    """Generates a PDF report of strong points."""
    doc = SimpleDocTemplate("strong_points.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Strong Points:", styles['h1'])]
    for i, point in enumerate(strong_points):
        story.append(Paragraph(f"{i + 1}. {point}", styles['Normal']))
        story.append(Spacer(1, 12))
    doc.build(story)


def generate_training_diagram():
    """Generates a diagram of Legal-BERT training."""
    dot = graphviz.Digraph(comment='Legal-BERT Training')
    dot.node('A', 'Legal Documents')
    dot.node('B', 'Preprocessing')
    dot.node('C', 'Legal-BERT Model')
    dot.node('D', 'Fine-tuning')
    dot.node('E', 'Trained Legal-BERT')
    dot.edges(['AB', 'BC', 'CD', 'DE'])
    dot.render('legal_bert_training', view=True)


def extract_text_from_pdf_images(pdf_file_path):
    """Extracts text from PDF images using OCR."""
    images = convert_from_path(pdf_file_path)
    extracted_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        extracted_text += f"--- Page {i + 1} ---\n{text}\n\n"
    return extracted_text


def text_to_speech(text):
    """Converts text to speech."""
    tts = gTTS(text=text, lang='en')
    tts.save("extracted_text.mp3")
    os.system("extracted_text.mp3") #open the file with default player.

def main():
    """Main function to orchestrate the process."""
    pdf_file_path = input("Enter the path to the PDF file: ")

    # PDF to Text, and strong points
    all_text = extract_pdf_text(pdf_file_path)
    all_text = preprocess_text(all_text)
    sentences = re.split(r'\n|\. ', all_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    embeddings = generate_embeddings(sentences)
    cluster_labels = perform_clustering(embeddings, sentences)
    strong_points = extract_strong_points(sentences, cluster_labels)

    print("Strong Points:")
    for i, point in enumerate(strong_points):
        print(f"{i + 1}. {point}")

    generate_pdf_report(strong_points)
    generate_training_diagram()

    #PDF to images to text, and speech.
    extracted_text = extract_text_from_pdf_images(pdf_file_path)
    print(extracted_text)
    text_to_speech(extracted_text)

if __name__ == "__main__":
    main()