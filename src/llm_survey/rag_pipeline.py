import json
import os
import time
from typing import Any, Dict, List

import chromadb
import yaml
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from llm_survey.prompts.model_extraction_prompts import format_prompt, get_prompt_template
from llm_survey.utils.preprocess import process_survey_data, save_processed_data


class ScientificModel(BaseModel):
    """Strict structure expected from each model extraction."""

    Variables: List[Dict[str, str]] = Field(default_factory=list)
    Relationships: List[str] = Field(default_factory=list)
    Hypotheses: List[Any] = Field(default_factory=list)
    Moderators: List[Dict[str, str]] = Field(default_factory=list)
    Themes: List[Any] = Field(default_factory=list)


class RAGModelExtractor:
    """LlamaIndex + Chroma pipeline for structured model extraction."""

    def __init__(
        self,
        openai_api_key: str = "",
        llm_model: str = "google/gemma-4-31b-it",
        embedding_model: str = "google/embeddinggemma-300m",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.1,
        extra_headers: Dict[str, str] | None = None,
        chroma_path: str = "data/chroma",
        chroma_collection: str = "survey_chunks",
        max_retries: int = 2,
    ):
        api_key = openai_api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key is required (OPENROUTER_API_KEY).")

        self.embedding_model_name = embedding_model
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.embed_model = self.embed_model

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers or {},
        )
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.processed_chunks: List[Dict[str, Any]] = []
        self.nodes_by_chunk_id: Dict[str, BaseNode] = {}

        os.makedirs(chroma_path, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        chroma_collection_obj = chroma_client.get_or_create_collection(chroma_collection)
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection_obj)
        self.index: VectorStoreIndex | None = None

    def process_and_store_data(
        self,
        file_path: str,
        max_tokens: int = 500,
        save_processed: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process source files, semantically split chunks, and persist to Chroma."""
        print(f"Processing data from {file_path}...")
        raw_chunks = process_survey_data(file_path, max_tokens=max_tokens)
        docs: List[Document] = []
        for chunk in raw_chunks:
            metadata = {
                "chunk_id": chunk["id"],
                "speaker_id": chunk["metadata"].get("speaker_id"),
                "timestamp": str(chunk["metadata"].get("timestamp") or ""),
                "source_file": os.path.basename(file_path),
                "original_index": chunk["original_index"],
            }
            docs.append(Document(text=chunk["text"], metadata=metadata))

        parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model,
        )
        nodes = parser.get_nodes_from_documents(docs)

        self.processed_chunks = []
        self.nodes_by_chunk_id = {}
        for i, node in enumerate(nodes):
            node_id = node.metadata.get("chunk_id") or f"node_{i}"
            text = node.get_content(metadata_mode=MetadataMode.NONE).strip()
            if not text:
                continue
            record = {
                "id": f"{node_id}_s{i}",
                "text": text,
                "metadata": {
                    "speaker_id": node.metadata.get("speaker_id"),
                    "timestamp": node.metadata.get("timestamp"),
                    "source_file": node.metadata.get("source_file"),
                    "original_index": node.metadata.get("original_index"),
                },
                "original_index": node.metadata.get("original_index", i),
            }
            self.processed_chunks.append(record)
            self.nodes_by_chunk_id[record["id"]] = node

        self.index = VectorStoreIndex(nodes, vector_store=self.vector_store)
        print(f"Stored {len(self.processed_chunks)} semantic chunks in Chroma.")

        if save_processed:
            output_path = "data/processed/processed_chunks.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_processed_data(self.processed_chunks, output_path)
            print(f"Processed data saved to {output_path}")
        return self.processed_chunks

    def _extract_yaml_with_validation(self, prompt: str) -> Dict[str, Any]:
        last_error = "Unknown parsing error"
        last_response = ""
        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": "Return only valid YAML. No markdown or commentary.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                raw_response = self._safe_completion_text(completion)
            except Exception as err:
                raw_response = ""
                last_error = f"OpenRouter request failed: {err}"

            last_response = raw_response
            if not raw_response.strip():
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return {
                    "model": None,
                    "raw_response": last_response,
                    "success": False,
                    "error": last_error or "Empty response from LLM.",
                }

            try:
                parsed = yaml.safe_load(raw_response)
                scientific_model = ScientificModel.model_validate(parsed)
                return {
                    "model": scientific_model.model_dump(),
                    "raw_response": raw_response,
                    "success": True,
                }
            except (yaml.YAMLError, ValidationError, TypeError, ValueError) as err:
                last_error = str(err)
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        return {
            "model": None,
            "raw_response": last_response,
            "success": False,
            "error": last_error,
        }

    def _safe_completion_text(self, completion: Any) -> str:
        """Extract completion text defensively across provider edge cases."""
        choices = getattr(completion, "choices", None)
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return ""

    def extract_model_from_chunk(
        self,
        chunk_text: str,
        use_rag: bool = True,
        num_context_docs: int = 3,
    ) -> Dict[str, Any]:
        """Extract one structured model from a chunk."""
        context = ""
        if use_rag and self.index is not None:
            retriever = self.index.as_retriever(similarity_top_k=num_context_docs)
            matches = retriever.retrieve(chunk_text)
            context = "\n\n".join([m.node.get_content() for m in matches])

        prompt_type = "rag" if use_rag else "base"
        prompt = format_prompt(
            get_prompt_template(prompt_type),
            context=context,
            input_text=chunk_text,
        )
        return self._extract_yaml_with_validation(prompt)

    def extract_models_from_all_chunks(
        self,
        use_rag: bool = True,
        num_context_docs: int = 3,
        save_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """Extract scientific models from all available chunks."""
        if not self.processed_chunks:
            raise ValueError("No processed chunks available. Run process_and_store_data first.")

        results: List[Dict[str, Any]] = []
        print(f"Extracting models from {len(self.processed_chunks)} chunks...")
        for i, chunk in enumerate(self.processed_chunks, start=1):
            print(f"Processing chunk {i}/{len(self.processed_chunks)}")
            result = self.extract_model_from_chunk(
                chunk_text=chunk["text"],
                use_rag=use_rag,
                num_context_docs=num_context_docs,
            )
            result["chunk_id"] = chunk["id"]
            result["chunk_metadata"] = chunk["metadata"]
            result["chunk_text"] = chunk["text"]
            results.append(result)

        if save_results:
            output_path = "outputs/extracted_models.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        return results

    def _call_yaml(self, prompt: str) -> Dict[str, Any]:
        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "Return only valid YAML. No markdown."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw_response = self._safe_completion_text(completion)
        except Exception as err:
            return {"payload": None, "raw_response": "", "success": False, "error": str(err)}
        try:
            return {"payload": yaml.safe_load(raw_response), "raw_response": raw_response, "success": True}
        except yaml.YAMLError as err:
            return {"payload": None, "raw_response": raw_response, "success": False, "error": str(err)}

    def perform_thematic_analysis(
        self,
        text_excerpts: List[str],
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Perform thematic analysis across multiple text excerpts."""
        combined_text = "\n\n---\n\n".join(text_excerpts)

        prompt = format_prompt(
            get_prompt_template("thematic"),
            text_excerpts=combined_text,
        )

        response = self._call_yaml(prompt)
        result = {
            "thematic_analysis": response.get("payload"),
            "raw_response": response.get("raw_response"),
            "success": response.get("success", False),
        }
        if not response.get("success", False):
            result["error"] = response.get("error", "Failed to parse YAML.")

        if save_results:
            output_path = "outputs/thematic_analysis.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"Thematic analysis saved to {output_path}")

        return result

    def refine_model(
        self,
        original_model: Dict[str, Any],
        context: str = "",
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Refine and validate a model specification."""
        model_yaml = yaml.dump(original_model, default_flow_style=False)

        prompt = format_prompt(
            get_prompt_template("refinement"),
            original_model=model_yaml,
            context=context,
        )

        response = self._call_yaml(prompt)
        result = {
            "refined_model": response.get("payload"),
            "raw_response": response.get("raw_response"),
            "success": response.get("success", False),
        }
        if not response.get("success", False):
            result["error"] = response.get("error", "Failed to parse YAML.")

        if save_results:
            output_path = "outputs/model_refinement.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"Model refinement saved to {output_path}")

        return result

