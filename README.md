# Legal Document Analyzer

A machine learning project for analyzing, summarizing, and extracting key clauses from legal documents.

## Features

- **PDF Text Extraction**: Extract text from PDF legal documents
- **Key Clause Identification**: Identify important legal clauses (termination, confidentiality, liability, etc.)
- **Document Summarization**: Generate concise summaries of lengthy legal documents
- **Sentence Segmentation**: Properly segment legal text into sentences for analysis
- **Evaluation**: Benchmark clause extraction against the CUAD dataset
- **HTML Report Generation**: Create visual HTML reports of document analysis

## Technical Highlights

- **Legal-BERT**: Uses domain-specific BERT models fine-tuned for legal text
- **Semantic Similarity**: Identifies clauses using cosine similarity with legal templates
- **Extractive Summarization**: Implements extractive summarization techniques
- **Quantitative Evaluation**: Provides precision, recall, and F1 metrics for clause extraction

## Using the Program

### Prerequisites

- Python 3.8+
- Pip package manager

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/legal-document-analyzer.git
   cd legal-document-analyzer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

### Usage

Process a single document:
```
python main.py --pdf path/to/your/document.pdf --output html
```

Run evaluation on CUAD dataset:
```
python evaluate.py
```

## Project Structure

- **main.py**: Core document processing functionality
- **evaluate.py**: Evaluation script for benchmarking against CUAD dataset
- **requirements.txt**: Project dependencies

## Model Architecture

The system uses a combination of NLP techniques:

1. **Document Parsing**: Extracts text from PDFs and segments into sentences
2. **Embedding Generation**: Creates embeddings using Legal-BERT
3. **Clause Identification**: Identifies clauses through semantic similarity
4. **Summarization**: Generates summaries

## Performance

When evaluated on the CUAD dataset, the system outputs:
- **Precision**: Measures the accuracy of identified clauses
- **Recall**: Measures the system's ability to find all relevant clauses
- **F1 Score**: Harmonic mean of precision and recall

## Future Improvements

- Fine-tune Legal-BERT for clause detection
- Implement more sophisticated summarization techniques
- Add support for more document formats
- Create a web interface for document uploading and analysis

## Acknowledgments

- CUAD dataset for legal document annotations
- MIT License
- Legal-BERT from NLPAUEB for domain-specific embeddings
