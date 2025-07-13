import os
import yaml
import json
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from prompts.model_extraction_prompts import get_prompt_template, format_prompt
from utils.preprocess import process_survey_data, save_processed_data

class RAGModelExtractor:
    """Main pipeline for extracting scientific models from qualitative data using RAG."""
    
    def __init__(self, 
                 groq_api_key: str = None,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model: str = "llama3-70b-8192",
                 temperature: float = 0.1):
        """
        groq_api_key: Groq API key
        llm_model: Groq model name (e.g., llama3-70b-8192, mixtral-8x7b-32768, gemma2-9b-it)
        """
        # Set up Groq API key
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        elif not os.getenv("GROQ_API_KEY"):
            raise ValueError("Groq API key must be provided or set as environment variable")
        
        # Initialize components
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.llm = ChatGroq(
            model=llm_model,
            temperature=temperature,
            api_key=groq_api_key or os.getenv("GROQ_API_KEY")
        )
        self.vector_store = None
        self.processed_chunks = []
        
    def process_and_store_data(self, 
                             file_path: str, 
                             max_tokens: int = 500,
                             save_processed: bool = True) -> List[Dict[str, Any]]:
        """Process survey data and store in vector database."""
        
        print(f"Processing data from {file_path}...")
        
        # Process the data
        self.processed_chunks = process_survey_data(file_path, max_tokens)
        
        # Save processed data if requested
        if save_processed:
            output_path = "data/processed/processed_chunks.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_processed_data(self.processed_chunks, output_path)
            print(f"Processed data saved to {output_path}")
        
        # Create documents for vector store
        documents = []
        for chunk in self.processed_chunks:
            doc = Document(
                page_content=chunk['text'],
                metadata={
                    'id': chunk['id'],
                    'speaker_id': chunk['metadata'].get('speaker_id'),
                    'timestamp': chunk['metadata'].get('timestamp'),
                    'sentiment': chunk['metadata'].get('sentiment'),
                    'original_index': chunk['original_index']
                }
            )
            documents.append(doc)
        
        # Create vector store
        print("Creating vector store...")
        self.vector_store = FAISS.from_documents(documents, self.embedding_model)
        
        # Save vector store
        self.vector_store.save_local("embeddings/faiss_index")
        print("Vector store created and saved.")
        
        return self.processed_chunks
    
    def extract_model_from_chunk(self, 
                                chunk_text: str, 
                                use_rag: bool = True,
                                num_context_docs: int = 3) -> Dict[str, Any]:
        """Extract scientific model from a single text chunk."""
        
        if use_rag and self.vector_store:
            # Get relevant context
            similar_docs = self.vector_store.similarity_search(
                chunk_text, 
                k=num_context_docs
            )
            context = "\n\n".join([doc.page_content for doc in similar_docs])
            
            # Use RAG-enhanced prompt
            prompt = format_prompt(
                get_prompt_template("rag"),
                context=context,
                input_text=chunk_text
            )
        else:
            # Use base prompt
            prompt = format_prompt(
                get_prompt_template("base"),
                context="",
                input_text=chunk_text
            )
        
        # Get LLM response
        response = self.llm.invoke(prompt)
        
        # Parse YAML response
        try:
            model_spec = yaml.safe_load(response.content)
            return {
                'model': model_spec,
                'raw_response': response.content,
                'success': True
            }
        except yaml.YAMLError as e:
            return {
                'model': None,
                'raw_response': response.content,
                'success': False,
                'error': str(e)
            }
    
    def extract_models_from_all_chunks(self, 
                                     use_rag: bool = True,
                                     save_results: bool = True) -> List[Dict[str, Any]]:
        """Extract models from all processed chunks."""
        
        if not self.processed_chunks:
            raise ValueError("No processed chunks available. Run process_and_store_data first.")
        
        results = []
        
        print(f"Extracting models from {len(self.processed_chunks)} chunks...")
        
        for i, chunk in enumerate(self.processed_chunks):
            print(f"Processing chunk {i+1}/{len(self.processed_chunks)}")
            
            result = self.extract_model_from_chunk(
                chunk['text'], 
                use_rag=use_rag
            )
            
            # Add metadata
            result['chunk_id'] = chunk['id']
            result['chunk_metadata'] = chunk['metadata']
            result['chunk_text'] = chunk['text']
            
            results.append(result)
        
        # Save results if requested
        if save_results:
            output_path = "outputs/extracted_models.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved to {output_path}")
        
        return results
    
    def perform_thematic_analysis(self, 
                                text_excerpts: List[str],
                                save_results: bool = True) -> Dict[str, Any]:
        """Perform thematic analysis across multiple text excerpts."""
        
        combined_text = "\n\n---\n\n".join(text_excerpts)
        
        prompt = format_prompt(
            get_prompt_template("thematic"),
            text_excerpts=combined_text
        )
        
        response = self.llm.invoke(prompt)
        
        try:
            thematic_analysis = yaml.safe_load(response.content)
            result = {
                'thematic_analysis': thematic_analysis,
                'raw_response': response.content,
                'success': True
            }
        except yaml.YAMLError as e:
            result = {
                'thematic_analysis': None,
                'raw_response': response.content,
                'success': False,
                'error': str(e)
            }
        
        # Save results if requested
        if save_results:
            output_path = "outputs/thematic_analysis.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Thematic analysis saved to {output_path}")
        
        return result
    
    def refine_model(self, 
                    original_model: Dict[str, Any],
                    context: str = "",
                    save_results: bool = True) -> Dict[str, Any]:
        """Refine and validate a model specification."""
        
        # Convert model to YAML string
        model_yaml = yaml.dump(original_model, default_flow_style=False)
        
        prompt = format_prompt(
            get_prompt_template("refinement"),
            original_model=model_yaml,
            context=context
        )
        
        response = self.llm.invoke(prompt)
        
        try:
            refinement = yaml.safe_load(response.content)
            result = {
                'refined_model': refinement,
                'raw_response': response.content,
                'success': True
            }
        except yaml.YAMLError as e:
            result = {
                'refined_model': None,
                'raw_response': response.content,
                'success': False,
                'error': str(e)
            }
        
        # Save results if requested
        if save_results:
            output_path = "outputs/model_refinement.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Model refinement saved to {output_path}")
        
        return result 