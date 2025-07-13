#!/usr/bin/env python3
"""
LLM Model Specification Generator
Main entry point for the pipeline
"""

import os
import argparse
import json
import yaml
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from rag_pipeline import RAGModelExtractor
from utils.preprocess import process_survey_data

# Import TopicAnalyzer only when needed
def get_topic_analyzer():
    """Import TopicAnalyzer only when needed to avoid dependency issues."""
    try:
        from topic_analysis import TopicAnalyzer
        return TopicAnalyzer
    except ImportError as e:
        print(f"Warning: Topic analysis not available due to dependency issue: {e}")
        return None

def run_complete_pipeline(
    input_file: str,
    groq_api_key: str = None,
    use_rag: bool = True,
    perform_topic_analysis: bool = True,
    output_dir: str = "outputs",
    llm_model: str = "llama3-70b-8192"
) -> Dict[str, Any]:
    """
    Run the complete pipeline from data processing to model extraction.
    
    Args:
        input_file: Path to input data file
        openai_api_key: OpenAI API key
        use_rag: Whether to use RAG enhancement
        perform_topic_analysis: Whether to perform topic analysis
        output_dir: Output directory for results
    
    Returns:
        Dictionary containing all results
    """
    
    print("Starting LLM Model Specification Pipeline")
    print("=" * 50)
    
    # Initialize extractor
    extractor = RAGModelExtractor(
        groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
        llm_model=llm_model,
        temperature=0.1
    )
    
    # Step 1: Process and store data
    print("\nStep 1: Processing and storing data...")
    processed_chunks = extractor.process_and_store_data(
        input_file,
        max_tokens=500,
        save_processed=True
    )
    print(f"Processed {len(processed_chunks)} text chunks")
    
    # Step 2: Extract models
    print("\nStep 2: Extracting models...")
    extraction_results = extractor.extract_models_from_all_chunks(
        use_rag=use_rag,
        save_results=True
    )
    
    successful_extractions = [r for r in extraction_results if r['success']]
    print(f"Extracted models from {len(successful_extractions)}/{len(extraction_results)} chunks")
    
    # Step 3: Topic analysis (optional)
    topic_results = None
    if perform_topic_analysis:
        print("\nStep 3: Performing topic analysis...")
        TopicAnalyzer = get_topic_analyzer()
        if TopicAnalyzer:
            topic_analyzer = TopicAnalyzer(
                nr_topics=10,
                min_topic_size=5
            )
            
            texts = [chunk['text'] for chunk in processed_chunks]
            topic_results = topic_analyzer.analyze_topics(texts, save_results=True)
            
            # Generate visualizations
            plot_paths = topic_analyzer.create_topic_visualizations(topic_results)
            print(f"Created {len(plot_paths)} visualizations")
            
            # Generate summary
            summary = topic_analyzer.generate_topic_summary(topic_results)
            print("Generated topic summary")
        else:
            print("Topic analysis skipped due to dependency issues")
    
    # Step 4: Generate comprehensive report
    print("\nStep 4: Generating comprehensive report...")
    
    report = {
        'pipeline_info': {
            'input_file': input_file,
            'use_rag': use_rag,
            'perform_topic_analysis': perform_topic_analysis,
            'total_chunks': len(processed_chunks)
        },
        'extraction_results': {
            'total_extractions': len(extraction_results),
            'successful_extractions': len(successful_extractions),
            'success_rate': len(successful_extractions) / len(extraction_results) if extraction_results else 0,
            'models': successful_extractions
        },
        'topic_analysis': topic_results if topic_results else None
    }
    
    # Save comprehensive report
    report_path = f"{output_dir}/comprehensive_report.json"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Comprehensive report saved to {report_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Input file: {input_file}")
    print(f"Total chunks processed: {len(processed_chunks)}")
    print(f"Models extracted: {len(successful_extractions)}/{len(extraction_results)}")
    print(f"Success rate: {report['extraction_results']['success_rate']*100:.1f}%")
    
    if topic_results:
        topic_info = topic_results['model_info']
        print(f"Topics identified: {len([t for t in topic_results['topic_info'] if t['Topic'] != -1])}")
    
    print(f"Results saved to: {output_dir}")
    print("=" * 50)
    
    return report

def run_interactive_mode():
    """Run the pipeline in interactive mode."""
    print("Interactive Mode")
    print("=" * 30)
    
    # Get input file
    input_file = input("Enter path to input file (CSV or TXT): ").strip()
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    # Get API key
    api_key = input("Enter Groq API key (or press Enter to use environment variable): ").strip()
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("No API key provided")
            return
    
    # Get options
    use_rag = input("Use RAG enhancement? (y/n, default: y): ").strip().lower() != 'n'
    perform_topic_analysis = input("Perform topic analysis? (y/n, default: y): ").strip().lower() != 'n'
    
    # Run pipeline
    try:
        results = run_complete_pipeline(
            input_file=input_file,
            openai_api_key=api_key,
            use_rag=use_rag,
            perform_topic_analysis=perform_topic_analysis
        )
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")

def create_sample_data():
    """Create sample data for testing."""
    sample_data = [
        {
            "speaker_id": "respondent_1",
            "text": "I feel overwhelmed when I have too many deadlines at work. My manager doesn't provide clear guidance, which makes it worse.",
            "timestamp": "2024-01-15"
        },
        {
            "speaker_id": "respondent_2", 
            "text": "Team support really helps when I'm stressed about deadlines. Having colleagues to bounce ideas off of reduces my anxiety.",
            "timestamp": "2024-01-15"
        },
        {
            "speaker_id": "respondent_3",
            "text": "When I have a clear project timeline and regular check-ins with my supervisor, I feel much more confident and productive.",
            "timestamp": "2024-01-16"
        },
        {
            "speaker_id": "respondent_4",
            "text": "The workload is manageable when I can prioritize tasks effectively. However, unexpected urgent requests throw everything off.",
            "timestamp": "2024-01-16"
        },
        {
            "speaker_id": "respondent_5",
            "text": "I perform best when I have autonomy in my work but also know I can ask for help when needed.",
            "timestamp": "2024-01-17"
        }
    ]
    
    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(sample_data)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/sample_survey_data.csv", index=False)
    print("Sample data created: data/raw/sample_survey_data.csv")

def main():
    parser = argparse.ArgumentParser(description="LLM Model Specification Generator")
    parser.add_argument("--input", "-i", help="Input file path")
    parser.add_argument("--api-key", "-k", help="Groq API key")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG enhancement")
    parser.add_argument("--no-topic-analysis", action="store_true", help="Disable topic analysis")
    parser.add_argument("--interactive", "-I", action="store_true", help="Run in interactive mode")
    parser.add_argument("--create-sample", action="store_true", help="Create sample data")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Output directory")
    parser.add_argument("--llm-model", default="llama3-70b-8192", help="Groq model name (default: llama3-70b-8192)")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
        return
    
    if args.interactive:
        run_interactive_mode()
        return
    
    if not args.input:
        print("Please provide an input file or use --interactive mode")
        parser.print_help()
        return
    
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    
    try:
        results = run_complete_pipeline(
            input_file=args.input,
            groq_api_key=args.api_key,
            use_rag=not args.no_rag,
            perform_topic_analysis=not args.no_topic_analysis,
            output_dir=args.output_dir,
            llm_model=args.llm_model
        )
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")

if __name__ == "__main__":
    main() 