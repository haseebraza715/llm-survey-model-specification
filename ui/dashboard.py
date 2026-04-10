import streamlit as st
import pandas as pd
import json
import yaml
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import sys
import os

# Add src directory to import package modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from llm_survey.rag_pipeline import RAGModelExtractor
from llm_survey.topic_analysis import TopicAnalyzer

# Page configuration
st.set_page_config(
    page_title="LLM Model Specification Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🧠 LLM Model Specification Generator</h1>', unsafe_allow_html=True)
    st.markdown("Generate structured scientific models from qualitative survey data using LLMs and RAG")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key")
        
        # Model settings
        st.subheader("Model Settings")
        llm_model = st.selectbox(
            "LLM Model",
            ["google/gemma-4-31b-it", "openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct"],
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        base_url = st.text_input("Base URL", value="https://openrouter.ai/api/v1", help="Base URL for OpenRouter API")
        referer = st.text_input("HTTP Referer (optional)", value="", help="Site URL for OpenRouter rankings")
        x_title = st.text_input("X-Title (optional)", value="", help="Site title for OpenRouter rankings")
        extra_headers = {}
        if referer:
            extra_headers["HTTP-Referer"] = referer
        if x_title:
            extra_headers["X-Title"] = x_title
        
        # RAG settings
        st.subheader("RAG Settings")
        use_rag = st.checkbox("Use RAG Enhancement", value=True)
        use_literature = st.checkbox("Use Literature Retrieval", value=True)
        num_context_docs = st.slider("Number of Context Documents", 1, 10, 3)
        
        # Topic analysis settings
        st.subheader("Topic Analysis Settings")
        nr_topics = st.slider("Number of Topics", 5, 20, 10)
        min_topic_size = st.slider("Minimum Topic Size", 2, 10, 5)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Upload", "🔍 Model Extraction", "📊 Topic Analysis", "📈 Results"])
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'extractor' not in st.session_state:
        st.session_state.extractor = None
    if 'topic_analyzer' not in st.session_state:
        st.session_state.topic_analyzer = None
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    if 'topic_results' not in st.session_state:
        st.session_state.topic_results = None
    if 'gap_report' not in st.session_state:
        st.session_state.gap_report = None
    if 'clarification_plan' not in st.session_state:
        st.session_state.clarification_plan = None
    
    # Tab 1: Data Upload
    with tab1:
        st.markdown('<h2 class="section-header">📁 Data Upload</h2>', unsafe_allow_html=True)
        
        upload_option = st.radio(
            "Choose upload method:",
            ["Upload CSV/Text File", "Paste Text", "Use Sample Data"]
        )
        
        if upload_option == "Upload CSV/Text File":
            uploaded_file = st.file_uploader(
                "Upload your survey data",
                type=['csv', 'txt', 'pdf', 'docx'],
                help="Upload CSV/TXT/PDF/DOCX input"
            )
            
            if uploaded_file is not None:
                # Save uploaded file
                file_path = f"data/raw/{uploaded_file.name}"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"File uploaded: {uploaded_file.name}")
                
                # Process data
                if st.button("Process Data"):
                    with st.spinner("Processing data..."):
                        try:
                            # Initialize extractor
                            if api_key:
                                st.session_state.extractor = RAGModelExtractor(
                                    openai_api_key=api_key,
                                    llm_model=llm_model,
                                    temperature=temperature,
                                    base_url=base_url,
                                    extra_headers=extra_headers,
                                    enable_literature_retrieval=use_literature,
                                )
                                
                                # Process and store data
                                st.session_state.processed_data = st.session_state.extractor.process_and_store_data(
                                    file_path,
                                    max_tokens=500,
                                    save_processed=True
                                )
                                
                                st.success(f"Processed {len(st.session_state.processed_data)} text chunks")
                                
                                # Display sample
                                st.subheader("Sample Processed Data")
                                sample_df = pd.DataFrame(st.session_state.processed_data[:5])
                                st.dataframe(sample_df[['id', 'text', 'metadata']])
                                
                            else:
                                st.error("Please enter your OpenRouter API key in the sidebar")
                                
                        except Exception as e:
                            st.error(f"Error processing data: {str(e)}")
        
        elif upload_option == "Paste Text":
            text_input = st.text_area(
                "Paste your qualitative data here",
                height=200,
                placeholder="Enter your survey responses, interview transcripts, or other qualitative data..."
            )
            
            if text_input and st.button("Process Text"):
                with st.spinner("Processing text..."):
                    try:
                        # Save text to file
                        file_path = "data/raw/pasted_text.txt"
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(text_input)
                        
                        # Initialize extractor
                        if api_key:
                            st.session_state.extractor = RAGModelExtractor(
                                openai_api_key=api_key,
                                llm_model=llm_model,
                                temperature=temperature,
                                base_url=base_url,
                                extra_headers=extra_headers,
                                enable_literature_retrieval=use_literature,
                            )
                            
                            # Process data
                            st.session_state.processed_data = st.session_state.extractor.process_and_store_data(
                                file_path,
                                max_tokens=500,
                                save_processed=True
                            )
                            
                            st.success(f"Processed {len(st.session_state.processed_data)} text chunks")
                            
                        else:
                            st.error("Please enter your OpenRouter API key in the sidebar")
                            
                    except Exception as e:
                        st.error(f"Error processing text: {str(e)}")
        
        else:  # Sample data
            st.info("Using sample data for demonstration")
            
            sample_data = [
                "I feel overwhelmed when I have too many deadlines at work. My manager doesn't provide clear guidance, which makes it worse.",
                "Team support really helps when I'm stressed about deadlines. Having colleagues to bounce ideas off of reduces my anxiety.",
                "When I have a clear project timeline and regular check-ins with my supervisor, I feel much more confident and productive.",
                "The workload is manageable when I can prioritize tasks effectively. However, unexpected urgent requests throw everything off.",
                "I perform best when I have autonomy in my work but also know I can ask for help when needed."
            ]
            
            if st.button("Load Sample Data"):
                with st.spinner("Processing sample data..."):
                    try:
                        # Save sample data
                        file_path = "data/raw/sample_data.txt"
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write("\n\n".join(sample_data))
                        
                        # Initialize extractor
                        if api_key:
                            st.session_state.extractor = RAGModelExtractor(
                                openai_api_key=api_key,
                                llm_model=llm_model,
                                temperature=temperature,
                                base_url=base_url,
                                extra_headers=extra_headers,
                                enable_literature_retrieval=use_literature,
                            )
                            
                            # Process data
                            st.session_state.processed_data = st.session_state.extractor.process_and_store_data(
                                file_path,
                                max_tokens=500,
                                save_processed=True
                            )
                            
                            st.success(f"Processed {len(st.session_state.processed_data)} text chunks")
                            
                        else:
                            st.error("Please enter your OpenRouter API key in the sidebar")
                            
                    except Exception as e:
                        st.error(f"Error processing sample data: {str(e)}")
    
    # Tab 2: Model Extraction
    with tab2:
        st.markdown('<h2 class="section-header">🔍 Model Extraction</h2>', unsafe_allow_html=True)
        
        if st.session_state.processed_data is None:
            st.warning("Please upload and process data first")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Processed Chunks", len(st.session_state.processed_data))
                st.metric("Use RAG", "Yes" if use_rag else "No")
            
            with col2:
                st.metric("LLM Model", llm_model)
                st.metric("Context Docs", num_context_docs if use_rag else 0)
            
            if st.button("Extract Models", type="primary"):
                with st.spinner("Extracting models from all chunks..."):
                    try:
                        st.session_state.extraction_results = st.session_state.extractor.extract_models_from_all_chunks(
                            use_rag=use_rag,
                            save_results=True
                        )
                        st.session_state.gap_report = st.session_state.extractor.detect_cross_chunk_gaps(
                            st.session_state.extraction_results,
                            save_results=True
                        )
                        st.session_state.clarification_plan = st.session_state.extractor.generate_clarification_plan(
                            st.session_state.gap_report,
                            save_results=True
                        )
                        
                        st.success(f"Extracted models from {len(st.session_state.extraction_results)} chunks")
                        
                        # Display results summary
                        successful_extractions = [r for r in st.session_state.extraction_results if r['success']]
                        failed_extractions = [r for r in st.session_state.extraction_results if not r['success']]
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Successful", len(successful_extractions))
                        col2.metric("Failed", len(failed_extractions))
                        col3.metric("Success Rate", f"{len(successful_extractions)/len(st.session_state.extraction_results)*100:.1f}%")
                        if st.session_state.gap_report:
                            st.caption(
                                "Cross-chunk completeness: "
                                f"{st.session_state.gap_report.get('overall_model_completeness', 0)*100:.1f}% | "
                                "testability: "
                                f"{st.session_state.gap_report.get('model_testability_score', 0)*100:.1f}%"
                            )
                        if st.session_state.clarification_plan:
                            st.caption(
                                "Clarification: "
                                f"{len(st.session_state.clarification_plan.get('questions', []))} questions | "
                                f"{len(st.session_state.clarification_plan.get('auto_answers', []))} literature auto-answers"
                            )
                        
                    except Exception as e:
                        st.error(f"Error extracting models: {str(e)}")
    
    # Tab 3: Topic Analysis
    with tab3:
        st.markdown('<h2 class="section-header">📊 Topic Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.processed_data is None:
            st.warning("Please upload and process data first")
        else:
            # Initialize topic analyzer
            st.session_state.topic_analyzer = TopicAnalyzer(
                nr_topics=nr_topics,
                min_topic_size=min_topic_size
            )
            
            if st.button("Perform Topic Analysis", type="primary"):
                with st.spinner("Performing topic analysis..."):
                    try:
                        # Extract texts
                        texts = [chunk['text'] for chunk in st.session_state.processed_data]
                        
                        # Perform analysis
                        st.session_state.topic_results = st.session_state.topic_analyzer.analyze_topics(
                            texts,
                            save_results=True
                        )
                        
                        st.success("Topic analysis completed!")
                        
                        # Display results
                        topic_info = pd.DataFrame(st.session_state.topic_results['topic_info'])
                        valid_topics = topic_info[topic_info['Topic'] != -1]
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Topics Found", len(valid_topics))
                        col2.metric("Total Documents", st.session_state.topic_results['model_info']['total_documents'])
                        
                        # Topic distribution chart
                        fig = px.bar(
                            valid_topics,
                            x='Topic',
                            y='Count',
                            title='Topic Distribution',
                            labels={'Count': 'Number of Documents', 'Topic': 'Topic ID'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error performing topic analysis: {str(e)}")
    
    # Tab 4: Results
    with tab4:
        st.markdown('<h2 class="section-header">📈 Results</h2>', unsafe_allow_html=True)
        
        # Model extraction results
        if st.session_state.extraction_results:
            st.subheader("Model Extraction Results")
            
            # Show successful extractions
            successful_results = [r for r in st.session_state.extraction_results if r['success']]
            
            if successful_results:
                # Display first few models
                for i, result in enumerate(successful_results[:3]):
                    with st.expander(f"Model {i+1} (Chunk: {result['chunk_id']})"):
                        if result['model']:
                            st.json(result['model'])
                        else:
                            st.text(result['raw_response'])
                
                # Download results
                if st.button("Download All Models"):
                    # Create downloadable file
                    models_data = {
                        'models': successful_results,
                        'metadata': {
                            'total_chunks': len(st.session_state.extraction_results),
                            'successful_extractions': len(successful_results),
                            'success_rate': len(successful_results)/len(st.session_state.extraction_results)
                        }
                    }
                    
                    json_str = json.dumps(models_data, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name="extracted_models.json",
                        mime="application/json"
                    )
        
        # Topic analysis results
        if st.session_state.topic_results:
            st.subheader("Topic Analysis Results")
            
            topic_info = pd.DataFrame(st.session_state.topic_results['topic_info'])
            valid_topics = topic_info[topic_info['Topic'] != -1]
            
            # Display topics
            for _, topic in valid_topics.iterrows():
                with st.expander(f"Topic {topic['Topic']}: {topic['Name']}"):
                    col1, col2 = st.columns(2)
                    col1.metric("Documents", topic['Count'])
                    col1.metric("Percentage", f"{topic['Count']/st.session_state.topic_results['model_info']['total_documents']*100:.1f}%")
                    
                    # Show keywords
                    keywords = st.session_state.topic_results['topic_keywords'].get(topic['Topic'], [])
                    if keywords:
                        col2.write("**Top Keywords:**")
                        for kw, score in keywords[:5]:
                            col2.write(f"- {kw} ({score:.3f})")
            
            # Download topic analysis
            if st.button("Download Topic Analysis"):
                yaml_str = st.session_state.topic_analyzer.export_topic_data(
                    st.session_state.topic_results,
                    output_format="yaml"
                )
                st.success(f"Topic analysis exported to {yaml_str}")

        if st.session_state.gap_report:
            st.subheader("Cross-Chunk Gap Detection")
            report = st.session_state.gap_report
            c1, c2, c3 = st.columns(3)
            c1.metric("Detected Gaps", len(report.get("gaps", [])))
            c2.metric("Completeness", f"{report.get('overall_model_completeness', 0)*100:.1f}%")
            c3.metric("Testability", f"{report.get('model_testability_score', 0)*100:.1f}%")

            priority_gaps = report.get("priority_gaps", [])
            if priority_gaps:
                st.write("**Priority Gaps**")
                for gap in priority_gaps:
                    st.write(f"- {gap}")

        if st.session_state.clarification_plan:
            st.subheader("Clarification Plan")
            plan = st.session_state.clarification_plan
            c1, c2, c3 = st.columns(3)
            c1.metric("Questions", len(plan.get("questions", [])))
            c2.metric("Auto Answers", len(plan.get("auto_answers", [])))
            c3.metric("Proceed w/ Literature", "Yes" if plan.get("can_proceed_with_literature") else "No")

            questions = plan.get("questions", [])
            if questions:
                st.write("**Follow-up Questions**")
                for q in questions[:8]:
                    st.write(f"- {q.get('question_id')}: {q.get('question_text')} ({q.get('answer_source')}, {q.get('priority')})")

if __name__ == "__main__":
    main() 
