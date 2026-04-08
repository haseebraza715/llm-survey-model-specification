import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any
import pandas as pd
from textblob import TextBlob

def ensure_nltk_resources() -> None:
    """Lazy-download tokenizer resources only when needed."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    """Break text into overlapping chunks."""
    ensure_nltk_resources()
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        
        if current_length + sentence_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep some overlap
            overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
            current_chunk = overlap_sentences
            current_length = sum(len(s.split()) for s in overlap_sentences)
        
        current_chunk.append(sentence)
        current_length += sentence_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def extract_metadata(text: str, speaker_id: str = None, timestamp: str = None) -> Dict[str, Any]:
    """Extract metadata from text."""
    ensure_nltk_resources()
    blob = TextBlob(text)
    
    # Handle NaN values from pandas
    if pd.isna(speaker_id):
        speaker_id = None
    if pd.isna(timestamp):
        timestamp = None
    
    metadata = {
        'speaker_id': speaker_id,
        'timestamp': timestamp,
        'word_count': len(text.split()),
        'sentence_count': len(sent_tokenize(text)),
        'sentiment': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }
    
    return metadata

def process_survey_data(file_path: str, max_tokens: int = 500) -> List[Dict[str, Any]]:
    """Process survey data from CSV or text file."""
    processed_chunks = []
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        
        for idx, row in df.iterrows():
            text = str(row.get('text', row.get('response', '')))
            if pd.isna(text) or text.strip() == '' or text.lower() == 'nan':
                continue
                
            cleaned_text = clean_text(text)
            if not cleaned_text.strip():
                continue
                
            chunks = chunk_text(cleaned_text, max_tokens)
            
            for chunk_idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                # Handle NaN values from pandas DataFrame
                speaker_id = row.get('speaker_id')
                if pd.isna(speaker_id):
                    speaker_id = f'respondent_{idx}'
                
                timestamp = row.get('timestamp')
                if pd.isna(timestamp):
                    timestamp = None
                
                metadata = extract_metadata(
                    chunk,
                    speaker_id=speaker_id,
                    timestamp=timestamp
                )
                
                processed_chunks.append({
                    'id': f"{idx}_{chunk_idx}",
                    'text': chunk,
                    'metadata': metadata,
                    'original_index': idx
                })
    
    else:  # Text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text, max_tokens)
        
        for chunk_idx, chunk in enumerate(chunks):
            metadata = extract_metadata(chunk)
            
            processed_chunks.append({
                'id': f"chunk_{chunk_idx}",
                'text': chunk,
                'metadata': metadata,
                'original_index': chunk_idx
            })
    
    return processed_chunks

def save_processed_data(chunks: List[Dict[str, Any]], output_path: str):
    """Save processed chunks to JSON file."""
    import json
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False) 