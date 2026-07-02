"""
Core RAG pipeline for the Veterinary First-Aid Guide.

Pipeline:
    Document Loader -> Text Splitter -> HuggingFace Embeddings -> ChromaDB -> Retriever -> Groq LLM
"""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

try:
    import streamlit as st
    cache_decorator = st.cache_resource
except ImportError:
    def cache_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"vet_rag_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


DATA_DIR = Path("data")
CHROMA_ROOT_DIR = Path("chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.2
MAX_TOKENS = 1024
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_DOCS = 4
MAX_HISTORY_TURNS = 4
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

SAFETY_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Disclaimer:** This information is intended for emergency first-aid guidance "
    "only and is NOT a substitute for professional veterinary care. Always consult a "
    "licensed veterinarian as soon as possible."
)

VET_SYSTEM_PROMPT = """You are a certified veterinary first-aid assistant.
Your role is to provide clear, calm, and safe emergency first-aid guidance for pets using ONLY the retrieved context.
Use the conversation history only to understand what "it", "this", "that", or other follow-up references mean.

STRICT RULES YOU MUST FOLLOW:
1. Respond with numbered steps (maximum 5 steps) outlining the immediate first aid actions.
2. Use simple, plain language that any pet owner can understand in an emergency.
3. NEVER recommend dangerous, invasive, or surgical procedures.
4. ALWAYS include a reminder to consult a veterinarian as one of your steps.
5. If the answer is not found in the context, respond ONLY with:
   "I don't know. Please contact your veterinarian or an emergency animal clinic immediately."
6. Do NOT fabricate medical information or go beyond the provided context.
7. If the situation sounds immediately life-threatening, make urgent transport part of the response.
8. Keep your answer focused, concise, and actionable.
9. If the current question is a follow-up, keep the answer connected to the same pet/problem from the chat history when the retrieved context supports it.
10. STRICT REFERENCE RULE: You MUST cite at least one retrieved source for every single recommendation step you make. Do not state any medical advice or emergency action without putting its corresponding Source [index] citation in brackets like `[1]`, `[2]`, etc. immediately next to the sentence.
11. At the very end of your response, after a blank line, output exactly three highly relevant, context-aware follow-up questions that the user might want to ask next regarding this pet emergency. Format them exactly as:
SUGGESTED_QUESTIONS:
- [Question 1]
- [Question 2]
- [Question 3]

CONVERSATION HISTORY:
{conversation_history}

CONTEXT:
{context}

QUESTION: {question}

FIRST-AID RESPONSE (numbered steps, max 5 with citations, followed by SUGGESTED_QUESTIONS):"""

URL_PATTERN = re.compile(r"https?://[^\s)]+")
SAFETY_DISCLAIMER = SAFETY_DISCLAIMER.replace("⚠️ ", "Warning: ")


def strip_disclaimer(answer: str) -> str:
    """Remove the UI disclaimer when storing prior assistant turns as memory."""
    return (answer or "").replace(SAFETY_DISCLAIMER, "").strip()


def extract_reference_urls(text: str) -> list[str]:
    """Extract unique URLs embedded inside curated source documents."""
    found = []
    seen = set()
    for match in URL_PATTERN.findall(text or ""):
        cleaned = match.rstrip(".,;")
        if cleaned not in seen:
            found.append(cleaned)
            seen.add(cleaned)
    return found


def format_chat_history(chat_history: Optional[list[dict]], max_turns: int = MAX_HISTORY_TURNS) -> str:
    """Convert recent chat turns into compact memory for the prompt."""
    if not chat_history:
        return "No previous conversation."

    lines = []
    for turn in chat_history[-max_turns:]:
        question = (turn.get("question") or "").strip()
        answer = strip_disclaimer(turn.get("answer", ""))
        if question:
            lines.append(f"User: {question}")
        if answer:
            lines.append(f"Assistant: {answer}")

    return "\n".join(lines) if lines else "No previous conversation."


def build_retrieval_query(question: str, chat_history: Optional[list[dict]]) -> str:
    """Rewrite follow-up questions into a standalone, search-optimized query using the LLM."""
    clean_question = question.strip()
    if not chat_history:
        return clean_question

    try:
        # Load the LLM (which is cached and fast)
        llm = get_llm()
        
        # Convert chat history into the compact string
        history_str = format_chat_history(chat_history, max_turns=3)
        
        prompt_text = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone search query that contains all necessary context (like species and primary symptoms) optimized for semantic search in a veterinary database.
Do NOT answer the question. Just output the standalone search query and nothing else.

CONVERSATION HISTORY:
{history_str}

FOLLOW-UP QUESTION: {clean_question}

STANDALONE SEARCH QUERY:"""

        logger.info("Condensing follow-up question with LLM...")
        condensed = llm.invoke(prompt_text).content.strip()
        # Clean quotes
        condensed = condensed.replace('"', '').replace("'", "")
        if condensed.lower().startswith("standalone search query:"):
            condensed = condensed[len("standalone search query:"):].strip()
        
        logger.info("Condensed standalone query: %s", condensed)
        return condensed
    except Exception as e:
        logger.warning("Failed to condense query using LLM, falling back to heuristic: %s", e)
        # Fallback to simple heuristic
        recent_questions = [
            (turn.get("question") or "").strip()
            for turn in chat_history[-2:]
            if (turn.get("question") or "").strip()
        ]
        if not recent_questions:
            return clean_question
        return f"Recent pet situation: {' | '.join(recent_questions)}\nCurrent question: {clean_question}"

class BM25Retriever:
    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        """
        corpus is a list of dicts: [{'content': str, 'metadata': dict, 'id': str}]
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(self._tokenize(doc['content'])) for doc in corpus]
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len) if corpus else 1.0
        self.doc_freqs = []
        self.idf = {}
        self._initialize()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _initialize(self):
        nd = len(self.corpus)
        df = Counter()
        for doc in self.corpus:
            tokens = self._tokenize(doc['content'])
            self.doc_freqs.append(Counter(tokens))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1
        for token, freq in df.items():
            self.idf[token] = math.log(1 + (nd - freq + 0.5) / (freq + 0.5))

    def score(self, query: str, filter_dict: Optional[dict] = None) -> list[tuple[dict, float]]:
        query_tokens = self._tokenize(query)
        scores = []
        for idx, doc in enumerate(self.corpus):
            if filter_dict:
                species_allowed = filter_dict.get("species", [])
                doc_species = doc["metadata"].get("species", "general")
                if doc_species not in species_allowed:
                    continue

            score = 0.0
            doc_len = self.doc_len[idx]
            freqs = self.doc_freqs[idx]
            for token in query_tokens:
                if token in freqs:
                    tf = freqs[token]
                    idf = self.idf.get(token, 0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf * (numerator / denominator)
            scores.append((doc, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def intercept_safety_guardrails(question: str) -> Optional[str]:
    """
    Analyze the user prompt for extreme clinical hazards and return a targeted alert message.
    """
    q_lower = question.lower()
    
    # Check for Toxic NSAIDs
    ibuprofen_words = {"ibuprofen", "advil", "motrin", "nurofen"}
    acetaminophen_words = {"acetaminophen", "tylenol", "paracetamol"}
    aspirin_words = {"aspirin", "salicylate"}
    
    # Check for Dangerous Home Procedures
    surgery_words = {"surgery", "stitch", "suture", "lance", "cut open", "drain cyst", "pop cyst", "amputate"}
    
    if any(word in q_lower for word in ibuprofen_words):
        return (
            "🚨 **CRITICAL VETERINARY TOXICITY WARNING: IBUPROFEN**\n\n"
            "Ibuprofen (Advil, Motrin) is **extremely toxic** to dogs and cats. It blocks protective prostaglandins, "
            "leading rapidly to severe gastric ulceration, life-threatening stomach perforation, and acute renal (kidney) failure. "
            "Never administer human NSAIDs. **Seek immediate emergency veterinary care.**"
        )
    elif any(word in q_lower for word in acetaminophen_words):
        return (
            "🚨 **CRITICAL VETERINARY TOXICITY WARNING: ACETAMINOPHEN (TYLENOL)**\n\n"
            "Acetaminophen (Tylenol, Paracetamol) is **highly toxic and lethal**, especially to cats. Cats lack the liver "
            "enzymes to metabolize it, causing rapid methemoglobinemia (destruction of red blood cells leading to severe oxygen deprivation, "
            "chocolate-colored gums, and respiratory failure) and acute liver necrosis. **Seek immediate emergency veterinary care.**"
        )
    elif any(word in q_lower for word in aspirin_words):
        return (
            "🚨 **CRITICAL VETERINARY WARNING: ASPIRIN**\n\n"
            "Aspirin poses severe risks of gastrointestinal bleeding, ulceration, and platelet dysfunction in pets. "
            "It has a narrow safety margin and must never be administered without explicit dosing instructions from a licensed veterinarian."
        )
    elif any(word in q_lower for word in surgery_words):
        return (
            "⚠️ **CRITICAL CLINICAL ADVISORY: HOME SURGICAL RISK**\n\n"
            "Do NOT attempt home surgical procedures, lancing, suturing, popping, or draining wounds on your pet. "
            "These invasive actions carry extreme risk of inducing septic shock, causing fatal hemorrhage, introducing lethal deep-tissue infections, "
            "and inflicting severe trauma. Seek professional emergency stabilization."
        )
        
    return None


def discover_default_data_files() -> list[Path]:
    """Return all default knowledge-base files shipped with the project."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: '{DATA_DIR}'")

    files = sorted(
        path for path in DATA_DIR.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No supported dataset files found in '{DATA_DIR}'. Add .txt or .pdf knowledge files."
        )
    return files


def resolve_source_files(extra_file_paths: Optional[Iterable[str]] = None) -> list[Path]:
    """Resolve the built-in data files plus any optional user-supplied references."""
    source_files = discover_default_data_files()
    for raw_path in extra_file_paths or []:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Custom data file not found: '{path}'")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{path.suffix}'. Only .txt and .pdf are supported."
            )
        source_files.append(path)

    deduped = []
    seen = set()
    for path in source_files:
        resolved = str(path.resolve())
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def build_dataset_signature(source_files: list[Path]) -> tuple[str, list[dict]]:
    """Build a stable signature for the active dataset composition."""
    manifest = []
    for path in source_files:
        stat = path.stat()
        manifest.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    # Append a structural schema version to force rebuild when chunking/metadata rules change
    schema_version = "v3_hybrid_rrf"
    payload = json.dumps({"manifest": manifest, "schema_version": schema_version}, sort_keys=True).encode("utf-8")
    signature = hashlib.sha1(payload).hexdigest()[:12]
    return signature, manifest


def load_single_document(path: Path) -> list:
    """Load one knowledge file and annotate metadata for source tracing."""
    logger.info("Loading document source: %s", path)

    if path.suffix.lower() == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    elif path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        raise ValueError(f"Unsupported file format '{path.suffix}'")

    docs = loader.load()
    for index, doc in enumerate(docs, start=1):
        metadata = dict(doc.metadata or {})
        metadata["source_path"] = str(path)
        metadata["source_name"] = path.name
        metadata["source_type"] = "custom_upload" if "uploads" in path.parts else "built_in"
        metadata["page_label"] = metadata.get("page", index)
        doc.metadata = metadata

    logger.info("Loaded %d document page(s) from %s", len(docs), path.name)
    return docs


def load_documents(extra_file_paths: Optional[Iterable[str]] = None) -> tuple[list, list[Path], str, list[dict]]:
    """
    Load all built-in knowledge files plus optional extra reference files.

    Returns:
        tuple: (docs, source_files, dataset_signature, manifest)
    """
    source_files = resolve_source_files(extra_file_paths)
    signature, manifest = build_dataset_signature(source_files)

    docs = []
    for path in source_files:
        docs.extend(load_single_document(path))

    logger.info("Loaded %d total documents across %d source files.", len(docs), len(source_files))
    return docs, source_files, signature, manifest


def detect_species(text: str) -> str:
    """Analyze the text content of a chunk to classify its species association."""
    text_lower = text.lower()
    
    # Canine indicators
    dog_words = {"dog", "canine", "puppy", "gdv", "bloat", "macadamia"}
    # Feline indicators
    cat_words = {"cat", "feline", "kitten", "urinary blockage", "lily", "lilies", "permethrin"}
    
    has_dog = any(word in text_lower for word in dog_words)
    has_cat = any(word in text_lower for word in cat_words)
    
    if has_dog and has_cat:
        return "general"
    elif has_dog:
        return "dog"
    elif has_cat:
        return "cat"
    else:
        return "general"


def split_documents(docs: list) -> list:
    """Split documents into embedding-friendly chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    
    # Tag each chunk with its species relevance
    for chunk in chunks:
        chunk.metadata["species"] = detect_species(chunk.page_content)
        
    logger.info("Split into %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


@cache_decorator
def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialise local sentence-transformer embeddings."""
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@cache_decorator
def get_cross_encoder() -> Optional[CrossEncoder]:
    """Initialise local sentence-transformer cross-encoder re-ranker."""
    if CrossEncoder is None:
        logger.warning("sentence-transformers not installed, CrossEncoder re-ranker unavailable.")
        return None
    try:
        logger.info("Loading CrossEncoder re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    except Exception as e:
        logger.error("Failed to load CrossEncoder model: %s", e)
        return None


def get_or_create_vectorstore(
    chunks: list,
    embeddings: HuggingFaceEmbeddings,
    dataset_signature: str,
    force_rebuild: bool = False,
) -> tuple[Chroma, Path]:
    """Create or load a persisted ChromaDB scoped to the active dataset signature."""
    persist_dir = CHROMA_ROOT_DIR / dataset_signature

    if force_rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)
        logger.info("Deleted ChromaDB collection for dataset signature %s", dataset_signature)

    # Clean up any old collection folders that are not active to save disk space
    try:
        if CHROMA_ROOT_DIR.exists():
            for path in CHROMA_ROOT_DIR.iterdir():
                if path.is_dir() and path.name != dataset_signature:
                    logger.info("Pruning old unused collection directory: %s", path)
                    shutil.rmtree(path)
    except Exception as e:
        logger.warning("Could not prune old database directory: %s", e)

    if persist_dir.exists():
        logger.info("Loading existing ChromaDB from: %s", persist_dir)
        vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
        logger.info("ChromaDB loaded. Collection contains %d documents.", vectorstore._collection.count())
        return vectorstore, persist_dir

    logger.info("Creating new ChromaDB at: %s", persist_dir)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    logger.info("ChromaDB created with %d chunks.", vectorstore._collection.count())
    return vectorstore, persist_dir


@cache_decorator
def get_llm() -> ChatGroq:
    """Initialise the Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Please add it to your .env file: GROQ_API_KEY=your_key_here"
        )

    logger.info("Initialising Groq LLM: %s", GROQ_MODEL)
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=api_key,
    )


def build_rag_chain(vectorstore: Chroma, llm: ChatGroq):
    """Assemble the retrieval + prompt + generation chain."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_DOCS},
    )
    prompt = PromptTemplate(
        input_variables=["context", "question", "conversation_history"],
        template=VET_SYSTEM_PROMPT,
    )
    chain = prompt | llm | StrOutputParser()

    logger.info("RAG chain assembled successfully.")
    return chain, retriever


def initialise_rag(
    extra_file_paths: Optional[Iterable[str]] = None,
    force_rebuild: bool = False,
) -> dict:
    """
    Full pipeline initialisation: load -> split -> embed -> store -> chain.

    Returns:
        dict with chain, retriever, and dataset metadata.
    """
    load_dotenv()

    docs, source_files, dataset_signature, manifest = load_documents(extra_file_paths)
    chunks = split_documents(docs)
    embeddings = get_embeddings()
    vectorstore, persist_dir = get_or_create_vectorstore(
        chunks,
        embeddings,
        dataset_signature=dataset_signature,
        force_rebuild=force_rebuild,
    )
    llm = get_llm()
    rag_chain, retriever = build_rag_chain(vectorstore, llm)

    # Reconstruct corpus for BM25 from ChromaDB
    collection_data = vectorstore._collection.get()
    bm25_corpus = []
    if collection_data and "documents" in collection_data:
        for idx in range(len(collection_data["documents"])):
            bm25_corpus.append({
                "content": collection_data["documents"][idx],
                "metadata": collection_data["metadatas"][idx] if collection_data["metadatas"] else {},
                "id": collection_data["ids"][idx]
            })
    bm25_retriever = BM25Retriever(bm25_corpus)

    return {
        "chain": rag_chain,
        "retriever": retriever,
        "vectorstore": vectorstore,
        "bm25_retriever": bm25_retriever,
        "dataset_signature": dataset_signature,
        "dataset_manifest": manifest,
        "source_files": [str(path) for path in source_files],
        "persist_dir": str(persist_dir),
    }


def query_rag(
    rag_runtime: dict,
    question: str,
    animal_type: Optional[str] = None,
    chat_history: Optional[list[dict]] = None,
    use_reranker: bool = True,
) -> dict:
    """Run a question through the RAG chain and return answer plus source metadata."""
    if not question.strip():
        return {
            "answer": "Please enter a valid question.",
            "sources": [],
            "question": question,
            "source_files": [],
            "safety_warning": None,
        }

    clean_question = question.strip()
    safety_warning = intercept_safety_guardrails(clean_question)
    retrieval_query = build_retrieval_query(clean_question, chat_history)

    if animal_type and animal_type.lower() not in ("auto", "all pets"):
        augmented_question = f"[{animal_type}] {clean_question}"
        retrieval_query = f"[{animal_type}] {retrieval_query}"
    else:
        augmented_question = clean_question

    logger.info("Query: %s (Reranker: %s)", augmented_question, use_reranker)

    try:
        rag_chain = rag_runtime["chain"]
        vectorstore = rag_runtime["vectorstore"]
        bm25_retriever = rag_runtime.get("bm25_retriever")
        conversation_history = format_chat_history(chat_history)
        
        search_filter = None
        if animal_type and animal_type.lower() in ("dog", "cat"):
            search_filter = {"species": {"$in": [animal_type.lower(), "general"]}}
            logger.info("Applying species metadata filter: %s", search_filter)

        # 1. Dense Semantic Retrieval (retrieve top 12 if reranking, else 10)
        k_retrieve = 12 if use_reranker else 10
        vector_results = vectorstore.similarity_search_with_score(
            retrieval_query, k=k_retrieve, filter=search_filter
        )

        # 2. Sparse Keyword Retrieval (retrieve top 12 if reranking, else 10)
        bm25_filter = {"species": [animal_type.lower(), "general"]} if (animal_type and animal_type.lower() in ("dog", "cat")) else None
        bm25_results = bm25_retriever.score(retrieval_query, filter_dict=bm25_filter)[:k_retrieve] if bm25_retriever else []

        # 3. Reciprocal Rank Fusion (RRF, k=60)
        rrf_scores = {}
        doc_map = {} # Maps content hash -> (Document, similarity_score_str, path_label)
        
        # Process Vector results
        for rank, (doc, score) in enumerate(vector_results, start=1):
            doc_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
            relevance_percentage = f"{round((1.0 / (1.0 + float(score))) * 100, 1)}%"
            doc_map[doc_hash] = {
                "doc": doc,
                "relevance_score": relevance_percentage,
                "path": "Semantic (Dense)"
            }
            rrf_scores[doc_hash] = rrf_scores.get(doc_hash, 0.0) + 1.0 / (60.0 + rank)
            
        # Process BM25 results
        for rank, (bm25_doc, score) in enumerate(bm25_results, start=1):
            content = bm25_doc["content"]
            doc_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if doc_hash in doc_map:
                doc_map[doc_hash]["path"] = "Both (Hybrid Fusion)"
            else:
                from langchain_core.documents import Document
                doc = Document(page_content=content, metadata=bm25_doc["metadata"])
                doc_map[doc_hash] = {
                    "doc": doc,
                    "relevance_score": f"BM25 Score: {round(score, 2)}",
                    "path": "Keyword (Sparse)"
                }
            rrf_scores[doc_hash] = rrf_scores.get(doc_hash, 0.0) + 1.0 / (60.0 + rank)
            
        # Sort by RRF score descending and select candidates
        sorted_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)
        
        candidates_limit = 12 if use_reranker else TOP_K_DOCS
        candidate_hashes = sorted_hashes[:candidates_limit]
        
        hybrid_results = []
        for rank, h in enumerate(candidate_hashes, start=1):
            mapped = doc_map[h]
            doc = mapped["doc"]
            doc.metadata["hybrid_rank"] = rank
            doc.metadata["rrf_score"] = round(rrf_scores[h], 4)
            doc.metadata["retrieval_path"] = mapped["path"]
            doc.metadata["original_relevance"] = mapped["relevance_score"]
            hybrid_results.append(doc)
            
        # 4. Neural Cross-Encoder Re-ranking (Stage-2)
        if use_reranker and len(hybrid_results) > 1:
            try:
                cross_encoder = get_cross_encoder()
                if cross_encoder is not None:
                    logger.info("Executing Stage-2 Neural Re-ranking on %d candidates...", len(hybrid_results))
                    pairs = [(retrieval_query, doc.page_content) for doc in hybrid_results]
                    scores = cross_encoder.predict(pairs)
                    
                    for doc, score in zip(hybrid_results, scores):
                        doc.metadata["rerank_score"] = float(score)
                        
                    hybrid_results.sort(key=lambda d: d.metadata.get("rerank_score", -9999.0), reverse=True)
                    
                    for rank, doc in enumerate(hybrid_results[:TOP_K_DOCS], start=1):
                        doc.metadata["hybrid_rank"] = rank
                        doc.metadata["retrieval_path"] = f"{doc.metadata['retrieval_path']} + Neural Re-rank"
                        
                    hybrid_results = hybrid_results[:TOP_K_DOCS]
                else:
                    hybrid_results = hybrid_results[:TOP_K_DOCS]
            except Exception as e:
                logger.error("Error running cross-encoder re-ranking: %s", e)
                hybrid_results = hybrid_results[:TOP_K_DOCS]
        else:
            hybrid_results = hybrid_results[:TOP_K_DOCS]
        
        # Format context with source indexes for grounded citations
        context_parts = []
        for idx, doc in enumerate(hybrid_results, start=1):
            source_name = doc.metadata.get("source_name", "unknown")
            context_parts.append(f"Source [{idx}] (File: {source_name}):\n{doc.page_content.strip()}")
        context = "\n\n".join(context_parts)

        raw_answer = rag_chain.invoke(
            {
                "question": augmented_question,
                "conversation_history": conversation_history,
                "context": context,
            }
        )

        # Parse out dynamic suggested follow-up questions from the LLM response
        suggested_questions = []
        if "SUGGESTED_QUESTIONS:" in raw_answer:
            parts = raw_answer.split("SUGGESTED_QUESTIONS:")
            clean_answer = parts[0].strip()
            questions_text = parts[1].strip()
            
            for line in questions_text.splitlines():
                line = line.strip()
                if line.startswith(("-", "*", "1.", "2.", "3.")):
                    q = line.lstrip("-*123. ").strip()
                    if q:
                        suggested_questions.append(q)
        else:
            clean_answer = raw_answer.strip()

        sources = []
        source_names = []
        for index, doc in enumerate(hybrid_results, start=1):
            source_name = doc.metadata.get("source_name", "unknown")
            source_names.append(source_name)
            
            relevance_score = f"RRF: {doc.metadata.get('rrf_score')} | {doc.metadata.get('original_relevance')}"
            if "rerank_score" in doc.metadata:
                relevance_score = f"Neural Rerank: {round(doc.metadata['rerank_score'], 3)} | {relevance_score}"

            sources.append(
                {
                    "index": index,
                    "source_name": source_name,
                    "source_path": doc.metadata.get("source_path", ""),
                    "page_label": doc.metadata.get("page_label", "n/a"),
                    "relevance_score": relevance_score,
                    "retrieval_path": doc.metadata.get("retrieval_path", "Hybrid Fusion"),
                    "hybrid_rank": doc.metadata.get("hybrid_rank", 1),
                    "reference_urls": extract_reference_urls(doc.page_content),
                    "content": doc.page_content.strip()[:450],
                }
            )

    except Exception as exc:
        logger.error("Error during RAG query: %s", exc, exc_info=True)
        return {
            "answer": f"An error occurred while generating the response: {exc}",
            "sources": [],
            "question": clean_question,
            "source_files": [],
            "safety_warning": None,
        }

    final_answer = clean_answer + SAFETY_DISCLAIMER
    if safety_warning:
        final_answer = f"{safety_warning}\n\n---\n\n" + final_answer
        
    logger.info("Response generated. Sources used: %d", len(sources))

    return {
        "answer": final_answer,
        "sources": sources,
        "question": clean_question,
        "source_files": sorted(set(source_names)),
        "conversation_used": bool(chat_history),
        "suggested_questions": suggested_questions,
        "safety_warning": safety_warning,
    }


def run_cli() -> None:
    """Interactive command-line interface."""
    print("\n" + "═" * 60)
    print("   🐾  Veterinary First-Aid Guide  |  RAG Assistant")
    print("═" * 60)
    print("Type your pet emergency question and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    try:
        rag_runtime = initialise_rag()
    except EnvironmentError as exc:
        print(f"\nConfiguration Error: {exc}\n")
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"\nFile Error: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        logger.error("Unexpected initialisation error: %s", exc, exc_info=True)
        print(f"\nUnexpected Error: {exc}\n")
        sys.exit(1)

    print("System ready. Ask your question below:\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Keep your pets safe.")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Keep your pets safe.")
            break

        result = query_rag(rag_runtime, question)
        print("\n" + "─" * 60)
        print("VetAid Assistant:")
        print(result["answer"])
        if result["source_files"]:
            print(f"\nSources: {', '.join(result['source_files'])}")
        print("─" * 60 + "\n")


if __name__ == "__main__":
    run_cli()
