import os
import argparse
from main import LegalDocumentProcessor

def run_demo(pdf_path):
    """
    Run a demonstration of the legal document processor
    """
    print("=" * 80)
    print("LEGAL DOCUMENT ANALYZER DEMO")
    print("=" * 80)
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found")
        return
    
    print(f"\nProcessing document: {pdf_path}")
    print("-" * 80)
    
    # Initialize processor
    processor = LegalDocumentProcessor()
    
    # Process the document and generate HTML report
    result = processor.process_document(pdf_path, output_format="html")
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Print summary
    print("\nDOCUMENT SUMMARY")
    print("-" * 80)
    print(result["summary"])
    
    # Print key clauses
    print("\nKEY CLAUSES EXTRACTED")
    print("-" * 80)
    for clause_type, clauses in result["key_clauses"].items():
        print(f"\n{clause_type.upper()} CLAUSES:")
        for i, (clause, confidence) in enumerate(clauses, 1):
            print(f"{i}. {clause}")
            print(f"   Confidence: {confidence:.2f}")
    
    # Print statistics
    print("\nSTATISTICS")
    print("-" * 80)
    print(f"Total sentences analyzed: {result['total_sentences']}")
    print(f"Total clauses extracted: {sum(len(clauses) for clauses in result['key_clauses'].values())}")
    print(f"Clause types found: {', '.join(result['key_clauses'].keys())}")
    
    # Print HTML report path
    html_path = os.path.splitext(pdf_path)[0] + "_analysis.html"
    print(f"\nHTML report generated: {html_path}")
    print("\nDemo completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Legal Document Analyzer Demo")
    parser.add_argument("--pdf", required=True, help="Path to the PDF document")
    args = parser.parse_args()
    
    run_demo(args.pdf)

if __name__ == "__main__":
    main()