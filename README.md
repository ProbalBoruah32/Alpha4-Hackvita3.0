# Legal Text Summarizer

This program helps extract key insights from legal documents using AI-powered text analysis. It utilizes advanced natural language processing (NLP) techniques to summarize legal text and highlight the most critical sentences.

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

