# Legal Text Summarizer

This program helps extract key insights from legal documents using AI-powered text analysis. It utilizes advanced natural language processing (NLP) techniques to summarize legal text and highlight the most critical sentences.

[Visit our project on google colab]([https://www.example.com](https://colab.research.google.com/drive/1HaKfQf_nTephfoSubHuj71uhgGgoxbON?usp=sharing))

## Requirements

### 1. Install Dependencies
Before running the program, install the necessary Python libraries by executing the following command:

```sh
pip install transformers[torch] sentencepiece PyPDF2 scikit-learn
```

If additional functionalities like data processing or visualization are needed, install these packages as well:

```sh
pip install kaggle
pip install kagglehub[pandas-datasets]
pip install dask matplotlib graphviz reportlab
apt-get install poppler-utils tesseract-ocr libtesseract-dev
pip install pytesseract pdf2image Pillow gTTS
```

### 2. Supported Data Formats

The program can process legal documents from various sources:

- **PDF Files** – Upload your legal documents in PDF format for summarization.
- **Kaggle Dataset** – To analyze Indian Supreme Court judgments, download the dataset from Kaggle: [Indian Supreme Court Judgments](https://www.kaggle.com/datasets/vangap/indian-supreme-court-judgments). A Kaggle API key is required.
- **ZIP Files of CSVs** – Upload multiple CSV files in a ZIP archive for bulk processing.

### 3. Setting Up Kaggle API Key (If Using Kaggle Dataset)

To access the Kaggle dataset, authenticate using a Kaggle API key:

1. Go to [Kaggle](https://www.kaggle.com/), sign in, and navigate to 'Account'.
2. Scroll down to 'API' and click 'Create New API Token'.
3. A `kaggle.json` file will be downloaded.
4. Upload this file to your working directory.

## Usage Instructions

1. **Install Dependencies** – Ensure all required libraries are installed.
2. **Upload Document/Data**
   - PDFs – Upload the document for text extraction.
   - Kaggle dataset – The program will fetch it if the API key is set up.
   - CSV files – Upload as a ZIP file for bulk processing.
3. **Run the Summarizer**
   - The program will extract important sentences and highlight key legal insights.
   - If applicable, it may suggest relevant Indian laws based on the document’s content.
4. **View Results**
   - The extracted summaries will be displayed on the screen.
   - The results can be exported as a PDF or an image.

## How It Works

The program utilizes **Legal-BERT**, an advanced NLP model trained for legal texts, to analyze documents effectively. Here’s the process:

1. **Text Extraction** – The program reads and breaks down legal documents into individual sentences.
2. **Sentence Embedding** – Using Legal-BERT, it converts sentences into numerical representations for better analysis.
3. **Clustering** – A machine learning technique (K-means clustering) groups similar sentences together.
4. **Key Sentence Selection** – The most representative sentences from each cluster are selected as key highlights.
5. **Output Presentation** – The results are displayed in an easy-to-read format.

This tool provides an efficient way to analyze large legal texts quickly, making it useful for law professionals, researchers, and students.

For further questions or improvements, feel free to contribute!



# Legal Text Summarizer

## Overview
This project is a **Legal Text Summarizer** that extracts key points from legal documents. It utilizes **Legal-BERT** for understanding legal text, applies **K-Means clustering** to find important sentences, and provides features like **PDF text extraction, OCR (image-to-text), and text-to-speech conversion**.

## Features
- Extracts text from **PDF documents** (both normal and scanned PDFs)
- Uses **Legal-BERT** to generate embeddings for sentences
- Applies **K-Means clustering** to group similar sentences and extract key points
- Generates a **PDF report** of the key points
- Converts **scanned PDF images into text using OCR** (Tesseract)
- Supports **text-to-speech conversion** for extracted text
- Creates a **Legal-BERT training process visualization**

## Installation

Before using this project, install the required dependencies using:

```bash
pip install transformers[torch] sentencepiece PyPDF2 scikit-learn
pip install pytesseract pdf2image Pillow gTTS
pip install reportlab graphviz matplotlib numpy pandas
```

Additionally, install **Poppler** and **Tesseract OCR** (for image-based PDFs):

```bash
apt-get install poppler-utils tesseract-ocr libtesseract-dev
```

## Usage

### 1. Running the Program
To use the program, run the following command:

```bash
python main.py
```

### 2. Entering the PDF File Path
After running the script, it will prompt you to enter the path to the PDF file:

```bash
Enter the path to the PDF file: your_document.pdf
```

### 3. Output
The program will:
- Extract and display the **most important sentences** from the document
- Save a **PDF report** containing the extracted key points
- Convert scanned images into text (if needed)
- Convert extracted text into speech and play the audio
- Generate a **visual representation** of the Legal-BERT model training

## How It Works

1. **Extracting Text:** Reads text from PDFs (both standard and scanned documents using OCR).
2. **Preprocessing:** Converts text to lowercase and removes special characters.
3. **Generating Embeddings:** Uses **Legal-BERT** to understand sentence meanings.
4. **Clustering Sentences:** Groups similar sentences using **K-Means clustering**.
5. **Extracting Key Sentences:** Selects the most representative sentences from each cluster.
6. **Generating Output:** Displays results, creates a PDF report, and plays text-to-speech audio.

## Example Output
```
Strong Points:
1. The contract must be executed in accordance with the laws of the state.
2. Any disputes shall be resolved through arbitration.
3. The parties agree to the terms stated in this agreement.
...
```

## Dependencies
- **Python 3.8+**
- **PyPDF2** (for PDF text extraction)
- **Transformers (Hugging Face)** (for Legal-BERT embeddings)
- **Scikit-Learn** (for K-Means clustering)
- **Tesseract OCR** (for image-based PDFs)
- **gTTS** (for text-to-speech conversion)
- **ReportLab** (for PDF report generation)
- **Graphviz** (for training process visualization)

## Notes
- Ensure **Tesseract OCR** is installed properly for text extraction from scanned PDFs.
- You can modify the **number of extracted key points** in the `extract_strong_points` function.
- If you encounter an error with **Graphviz**, try reinstalling it using:
  ```bash
  pip install graphviz
  ```

## Author
Developed by **[Your Name]** 🚀

