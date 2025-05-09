"""
Simplified demo for Legal Document Analyzer
This version uses minimal dependencies to demonstrate core functionality
"""

import os
import re
import argparse
import math

class SimpleLegalDocumentProcessor:
    """
    A simplified version of the legal document processor that works without
    heavy dependencies like transformers, spacy, etc.
    """
    
    def __init__(self):
        print("Initializing Simplified Legal Document Processor...")
        
        # Define common legal clause patterns
        self.clause_patterns = {
            "termination": [
                r"(?i)this\s+agreement\s+shall\s+terminate",
                r"(?i)termination\s+of\s+this\s+agreement",
                r"(?i)either\s+party\s+may\s+terminate",
                r"(?i)right\s+to\s+terminate"
            ],
            "confidentiality": [
                r"(?i)confidential\s+information",
                r"(?i)shall\s+keep\s+confidential",
                r"(?i)non-disclosure",
                r"(?i)shall\s+not\s+disclose"
            ],
            "liability": [
                r"(?i)limitation\s+of\s+liability",
                r"(?i)shall\s+not\s+be\s+liable",
                r"(?i)liability\s+is\s+limited",
                r"(?i)no\s+liability\s+for"
            ],
            "payment": [
                r"(?i)payment\s+terms",
                r"(?i)shall\s+pay",
                r"(?i)fees\s+and\s+payment",
                r"(?i)compensation"
            ],
            "governing_law": [
                r"(?i)governed\s+by\s+the\s+laws",
                r"(?i)jurisdiction",
                r"(?i)governing\s+law",
                r"(?i)applicable\s+law"
            ]
        }
        
        print("Initialization complete!")
    
    def extract_text_from_file(self, file_path):
        """Extract text from a file (PDF or TXT)."""
        try:
            print(f"Reading text from {file_path}...")
            if file_path.lower().endswith('.pdf'):
                # For PDFs, try to use pdfminer if available
                try:
                    from pdfminer.high_level import extract_text
                    return extract_text(file_path)
                except ImportError:
                    print("Warning: pdfminer not available. Cannot extract PDF text.")
                    print("Please install pdfminer.six or provide a text file instead.")
                    return ""
            else:
                # For text files, just read the content
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Error extracting text from file: {e}")
            return ""
    
    def segment_sentences(self, text):
        """Segment text into sentences using simple rules."""
        # Simple sentence segmentation based on common punctuation
        text = re.sub(r'([.!?])\s+', r'\1\n', text)
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return sentences
    
    def identify_clause_type(self, sentence):
        """Identify the type of legal clause in a sentence using regex patterns."""
        best_clause_type = "other"
        highest_confidence = 0.0
        
        for clause_type, patterns in self.clause_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sentence):
                    # Simple confidence calculation - longer matches get higher confidence
                    match = re.search(pattern, sentence)
                    match_length = match.end() - match.start()
                    confidence = min(0.9, match_length / len(sentence) * 2)
                    
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_clause_type = clause_type
        
        return best_clause_type, highest_confidence
    
    def extract_key_clauses(self, sentences):
        """Extract key legal clauses from sentences."""
        print("Extracting key clauses...")
        key_clauses = {}
        
        for sentence in sentences:
            if len(sentence.split()) > 5:  # Skip very short sentences
                clause_type, confidence = self.identify_clause_type(sentence)
                if clause_type != "other" and confidence > 0.3:
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
        """Generate a simple summary by selecting important sentences."""
        sentences = self.segment_sentences(text)
        if not sentences:
            return ""
            
        # Calculate sentence scores based on word importance
        # (simple version: longer sentences with legal terms get higher scores)
        legal_terms = ['agreement', 'contract', 'party', 'parties', 'shall', 'liability',
                      'term', 'terminate', 'confidential', 'payment', 'law', 'provision']
        
        scores = []
        for sentence in sentences:
            # Basic score based on sentence length (favor medium-length sentences)
            words = sentence.split()
            length_score = min(1.0, len(words) / 20)
            
            # Increase score for sentences with legal terms
            legal_term_count = sum(1 for word in words if word.lower() in legal_terms)
            legal_score = min(1.0, legal_term_count / 3)
            
            # Final score is a combination of both factors
            final_score = (length_score + legal_score) / 2
            scores.append(final_score)
        
        # Select top sentences for the summary
        num_summary_sentences = max(1, int(len(sentences) * ratio))
        
        # Use a simple approach instead of numpy
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in indexed_scores[:num_summary_sentences]]
        top_indices.sort()  # Sort by position in document
        
        summary_sentences = [sentences[i] for i in top_indices]
        return " ".join(summary_sentences)
    
    def process_document(self, file_path, output_format="text"):
        """Process a legal document and extract information."""
        # Extract text from file
        text = self.extract_text_from_file(file_path)
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
            self.generate_html_output(result, os.path.splitext(file_path)[0] + "_analysis.html")
        
        return result
    
    def generate_html_output(self, result, output_path):
        """Generate HTML output for visualization."""
        # Using triple braces to escape curly braces in CSS
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Legal Document Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; }}
                .clause {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }}
                .confidence {{ color: #7f8c8d; font-size: 0.8em; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
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
        
        html_content = html_template.format(
            summary=result["summary"].replace("\n", "<br>"),
            clauses=clauses_html,
            total_sentences=result["total_sentences"]
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"HTML output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Simple Legal Document Processor")
    parser.add_argument("--file", required=True, help="Path to the legal document file (PDF or TXT)")
    parser.add_argument("--output", choices=["text", "html"], default="html", help="Output format")
    args = parser.parse_args()
    
    processor = SimpleLegalDocumentProcessor()
    result = processor.process_document(args.file, args.output)
    
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