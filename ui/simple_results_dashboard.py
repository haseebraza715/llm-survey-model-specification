import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="LLM Extraction Results Viewer",
    layout="wide"
)

st.title("LLM Extraction Results Viewer")
st.write("This dashboard displays the processed data and extraction results from the latest pipeline run.")

# Paths
processed_path = "data/processed/processed_chunks.json"
results_path = "outputs/extracted_models.json"

# Load processed data
processed_chunks = []
if os.path.exists(processed_path):
    with open(processed_path, "r", encoding="utf-8") as f:
        processed_chunks = json.load(f)
else:
    st.warning("No processed data found.")

# Load extraction results
extraction_results = []
if os.path.exists(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        extraction_results = json.load(f)
else:
    st.warning("No extraction results found.")

# Show processed data with better design
st.header("📊 Data Overview")
if processed_chunks:
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chunks", len(processed_chunks))
    with col2:
        avg_words = sum(chunk.get('metadata', {}).get('word_count', 0) for chunk in processed_chunks) / len(processed_chunks)
        st.metric("Avg Words/Chunk", f"{avg_words:.1f}")
    with col3:
        success_rate = sum(1 for result in extraction_results if result.get('success', False)) / len(extraction_results) * 100 if extraction_results else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        unique_speakers = len(set(chunk.get('metadata', {}).get('speaker_id', '') for chunk in processed_chunks))
        st.metric("Unique Speakers", unique_speakers)
    
    st.header("📝 Processed Data Chunks")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Table View", "Card View"])
    
    with tab1:
        # Enhanced table view
        df = pd.DataFrame(processed_chunks)
        if 'metadata' in df.columns:
            # Expand metadata for better display
            metadata_df = pd.json_normalize(df['metadata'])
            display_df = pd.concat([
                df[['id', 'text']], 
                metadata_df[['speaker_id', 'word_count', 'sentence_count', 'sentiment']]
            ], axis=1)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    
    with tab2:
        # Card view for better readability
        for i, chunk in enumerate(processed_chunks):
            with st.expander(f"Chunk {i+1}: {chunk.get('id', '')}", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Text:** {chunk.get('text', '')}")
                with col2:
                    metadata = chunk.get('metadata', {})
                    st.markdown(f"""
                    **Speaker:** {metadata.get('speaker_id', 'N/A')}  
                    **Words:** {metadata.get('word_count', 0)}  
                    **Sentences:** {metadata.get('sentence_count', 0)}  
                    **Sentiment:** {metadata.get('sentiment', 0):.2f}
                    """)
else:
    st.info("No processed data to display.")

# Show extraction results with better design
st.header("🔍 Extraction Results")
if extraction_results:
    # Summary metrics
    successful = sum(1 for result in extraction_results if result.get('success', False))
    failed = len(extraction_results) - successful
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Successful Extractions", successful)
    with col2:
        st.metric("Failed Extractions", failed)
    with col3:
        st.metric("Total Variables Extracted", 
                 sum(len(result.get('model', {}).get('Variables', [])) 
                     for result in extraction_results if result.get('success', False)))
    
    # Results display
    for i, result in enumerate(extraction_results):
        success_status = "✅" if result.get('success', False) else "❌"
        with st.expander(f"{success_status} Chunk {i+1}: {result.get('chunk_id', '')}", expanded=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Original Text:**")
                st.text(result.get('chunk_text', ''))
            
            with col2:
                if result.get('success', False):
                    model = result.get('model', {})
                    
                    # Variables
                    if 'Variables' in model:
                        st.markdown("**Variables:**")
                        for var in model['Variables']:
                            for name, desc in var.items():
                                st.markdown(f"• **{name}**: {desc}")
                    
                    # Relationships
                    if 'Relationships' in model:
                        st.markdown("**Relationships:**")
                        for rel in model['Relationships']:
                            st.markdown(f"• {rel}")
                    
                    # Hypotheses
                    if 'Hypotheses' in model:
                        st.markdown("**Hypotheses:**")
                        for hyp in model['Hypotheses']:
                            if isinstance(hyp, dict):
                                for name, desc in hyp.items():
                                    st.markdown(f"• **{name}**: {desc}")
                            else:
                                st.markdown(f"• {hyp}")
                else:
                    st.error(f"Extraction failed: {result.get('error', 'Unknown error')}")
else:
    st.info("No extraction results to display.") 