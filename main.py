import argparse
import os
from pdfminer.high_level import extract_text
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from summa import summarizer

class LegalDocumentProcessor:
    def __init__(self):
        print("Initializing Legal Document Processor...")
        # Load NLP models
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load Legal-BERT for embeddings
        print("Loading Legal-BERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        
        # Define common legal clause templates
        self.clause_templates = {
            "termination": [
                "This agreement shall terminate",
                "This contract terminates",
                "Termination of this agreement",
                "Either party may terminate"
            ],
            "confidentiality": [
                "shall keep confidential",
                "shall maintain the confidentiality",
                "confidential information",
                "non-disclosure"
            ],
            "liability": [
                "limitation of liability",
                "shall not be liable",
                "liability is limited to",
                "no liability for"
            ],
            "payment": [
                "payment terms",
                "shall pay",
                "fees and payment",
                "compensation"
            ],
            "governing_law": [
                "governed by the laws",
                "jurisdiction",
                "governing law",
                "applicable law"
            ]
        }
        print("Initialization complete!")

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        try:
            print(f"Extracting text from {pdf_path}...")
            return extract_text(pdf_path)
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def segment_sentences(self, text):
        """Segment text into sentences."""
        print("Segmenting text into sentences...")
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def get_embedding(self, text):
        """Get embedding for a piece of text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def identify_clause_type(self, sentence):
        """Identify the type of legal clause in a sentence."""
        sentence_embedding = self.get_embedding(sentence)
        
        max_similarity = -1
        best_clause_type = "other"
        
        for clause_type, templates in self.clause_templates.items():
            for template in templates:
                template_embedding = self.get_embedding(template)
                similarity = cosine_similarity(sentence_embedding, template_embedding)[0][0]
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_clause_type = clause_type
        
        # Only classify if similarity is above threshold
        if max_similarity > 0.7:
            return best_clause_type, max_similarity
        return "other", max_similarity

    def extract_key_clauses(self, sentences):
        """Extract key legal clauses from sentences."""
        print("Extracting key clauses...")
        key_clauses = {}
        
        for sentence in sentences:
            if len(sentence.split()) > 5:  # Skip very short sentences
                clause_type, confidence = self.identify_clause_type(sentence)
                if clause_type != "other":
                    if clause_type not in key_clauses:
                        key_clauses[clause_type] = []
                    key_clauses[clause_type].append((sentence, confidence))
        
        # Sort clauses by confidence and take the top ones
        for clause_type in key_clauses:
            key_clauses[clause_type].sort(key=lambda x: x[1], reverse=True)
            if len(key_clauses[clause_type]) > 3:  # Limit to top 3 clauses per type
                key_clauses[clause_type] = key_clauses[clause_type][:3]
        
        return key_clauses

    def generate_summary(self, text, ratio=0.2):
        """Generate a summary of the text."""
        print("Generating summary...")
        return summarizer.summarize(text, ratio=ratio)

    def process_document(self, pdf_path, output_format="text"):
        """Process a legal document and extract information."""
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "Failed to extract text from document"}
        
        # Segment into sentences
        sentences = self.segment_sentences(text)
        
        # Extract key clauses
        key_clauses = self.extract_key_clauses(sentences)
        
        # Generate summary
        summary = self.generate_summary(text)
        
        result = {
            "summary": summary,
            "key_clauses": key_clauses,
            "total_sentences": len(sentences)
        }
        
        if output_format == "html":
            self.generate_html_output(result, os.path.splitext(pdf_path)[0] + "_analysis.html")
        
        return result

    def generate_html_output(self, result, output_path):
        """Generate HTML output for visualization."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Legal Document Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; }
                .clause { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }
                .confidence { color: #7f8c8d; font-size: 0.8em; }
                .summary { background-color: #e8f4f8; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Legal Document Analysis</h1>
            
            <h2>Document Summary</h2>
            <div class="summary">
                {summary}
            </div>
            
            <h2>Key Clauses Extracted</h2>
            {clauses}
            
            <p>Total sentences analyzed: {total_sentences}</p>
        </body>
        </html>
        """
        
        clauses_html = ""
        for clause_type, clauses in result["key_clauses"].items():
            clauses_html += f"<h3>{clause_type.title()} Clauses</h3>"
            for clause, confidence in clauses:
                clauses_html += f'<div class="clause">{clause} <span class="confidence">(Confidence: {confidence:.2f})</span></div>'
        
        html_content = html_content.format(
            summary=result["summary"].replace("\n", "<br>"),
            clauses=clauses_html,
            total_sentences=result["total_sentences"]
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"HTML output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Legal Document Processor")
    parser.add_argument("--pdf", required=True, help="Path to the PDF document")
    parser.add_argument("--output", choices=["text", "html"], default="text", help="Output format")
    args = parser.parse_args()
    
    processor = LegalDocumentProcessor()
    result = processor.process_document(args.pdf, args.output)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\nSummary:")
        print(result["summary"])
        
        print("\nKey Clauses:")
        for clause_type, clauses in result["key_clauses"].items():
            print(f"\n{clause_type.upper()}:")
            for clause, confidence in clauses:
                print(f"- {clause} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()