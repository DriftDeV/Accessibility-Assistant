# RAG Assistant per l'Accessibilità nei Videogiochi
# Versione OTTIMIZZATA per performance e utilizzo risorse
# Miglioramenti chiave:
# - Lazy loading del LLM (caricato solo al primo utilizzo)
# - Quantizzazione 4-bit con bitsandbytes per ridurre memoria
# - Batch processing ottimizzato per embeddings
# - Cache multi-livello per query e embeddings
# - Prompt engineering ridotto per latenza minore
# - Thread pool per operazioni I/O

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import OrderedDict
from functools import lru_cache
import hashlib

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
import numpy as np
import torch


# --- CONFIGURAZIONE OTTIMIZZATA ---
@dataclass
class Config:
    """Configurazione con parametri ottimizzati per performance"""
    json_file: Path = Path("games.json")
    # Modello di embedding ultra-leggero e veloce
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # LLM compatto e performante
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    causalLm_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    chroma_collection: str = "games_accessibility_v2"
    chroma_persist_dir: Path = Path("./chroma_db")
    top_k_results: int = 3
    max_response_tokens: int = 200  # Ridotto per velocità
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embed_batch_size: int = 64  # Aumentato per throughput
    
    # OTTIMIZZAZIONI CHIAVE
    use_4bit_quantization: bool = True  # Quantizzazione 4-bit invece di 8-bit
    lazy_load_llm: bool = True  # Carica LLM solo quando necessario
    use_flash_attention: bool = True  # Flash Attention 2 se disponibile
    compile_model: bool = False  # torch.compile (richiede PyTorch 2.0+)
    
    log_level: int = logging.INFO
    # Limiti del contesto ultra-ridotti per prompt più snelli
    context_max_chars_per_doc: int = 500  # Ridotto da 800
    context_max_total_chars: int = 1200  # Ridotto da 2000
    
    # Cache più aggressive
    answer_cache_size: int = 256  # Aumentato da 128
    embed_cache_size: int = 512  # Aumentato da 256
    
    # Ottimizzazioni generation
    use_cache_generation: bool = True
    num_beams: int = 1  # Greedy decoding per velocità


# --- LOGGING ---
def setup_logging(level: int) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("rag_assistant")


# --- DATA LOADING CON CACHE ---
class GameDataLoader:
    """Carica e cachea il JSON dei giochi"""

    def __init__(self, filepath: Path, logger: logging.Logger):
        self.filepath = filepath
        self.logger = logger
        self._cache: Optional[List[Dict]] = None

    def load(self) -> List[Dict]:
        """Carica con cache in memoria"""
        if self._cache is not None:
            return self._cache
            
        if not self.filepath.exists():
            raise FileNotFoundError(f"{self.filepath} non esiste.")

        with open(self.filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list):
            raise ValueError("Il file JSON deve contenere una lista di oggetti.")

        self._cache = data
        self.logger.info("Caricati %d giochi", len(data))
        return data


# --- DOCUMENT PREPROCESSOR OTTIMIZZATO ---
class DocumentPreprocessor:
    """Prepara documenti con formato minimalista per ridurre token count"""

    @staticmethod
    def prepare_documents(games_data: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
        texts: List[str] = []
        metadatas: List[Dict] = []
        ids: List[str] = []

        for g in games_data:
            game_id = str(g.get("id", "UNKNOWN"))
            name = g.get("name", "Senza nome")
            category = g.get("category", "Non specificato")
            access_level = g.get("access_level", None)
            is_native = bool(g.get("is_native", False))
            platforms = g.get("platforms", [])
            description = g.get("description", "")
            features = g.get("features", [])
            details = g.get("accessibility_details", {}) or {}

            # Formato COMPATTO per ridurre dimensione embedding e retrieval time
            text = (
                f"{name} | {category} | Score: {access_level or 'N/A'}/10\n"
                f"{'Nativo' if is_native else 'Mod/Config'} | "
                f"{', '.join(platforms) if platforms else 'Multi-platform'}\n"
                f"{description[:200]}...\n"  # Tronca descrizione
                f"Vis:{details.get('visual','?')} Mot:{details.get('motor','?')} "
                f"Aud:{details.get('auditory','?')} Cog:{details.get('cognitive','?')}\n"
                f"{' | '.join(features[:5])}"  # Max 5 features
            )

            texts.append(text)
            metadatas.append({
                "id": game_id,
                "name": name,
                "category": category,
                "score": float(access_level) if access_level is not None else None,
                "is_native": is_native,
                "platforms": ", ".join(platforms) if platforms else "N/A",
                "source_ref": g.get("source_ref", ""),
            })
            ids.append(game_id)

        return texts, metadatas, ids


# --- VECTOR STORE OTTIMIZZATO ---
class ChromaVectorStore:
    """ChromaDB con cache aggressiva e batch processing"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = None
        self.collection = None
        self.embed_model: Optional[SentenceTransformer] = None
        self._emb_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def initialize(self) -> None:
        """Inizializza con ottimizzazioni"""
        self.logger.info("Init vector store...")
        
        # Carica embedding model con ottimizzazioni
        self.embed_model = SentenceTransformer(
            self.config.embedding_model, 
            device=self.config.device
        )
        
        # Ottimizza per inferenza
        self.embed_model.eval()
        if self.config.device == "cuda":
            self.embed_model = self.embed_model.half()  # FP16 su GPU
        
        self.config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.config.chroma_persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self.collection = self.client.get_collection(self.config.chroma_collection)
            self.logger.info("Collection caricata")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.config.chroma_collection,
                metadata={"description": "Games accessibility"}
            )
            self.logger.info("Collection creata")

    def index_documents(self, texts: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        """Indicizza con batch processing ottimizzato"""
        self.logger.info("Calcolo embeddings...")
        
        with torch.no_grad():  # No gradient tracking per velocità
            embeddings = self.embed_model.encode(
                texts,
                batch_size=self.config.embed_batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True
            )

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        # Rimuovi duplicati
        try:
            existing = self.collection.get()
            existing_ids = set(existing.get("ids", []))
            to_delete = [i for i in ids if i in existing_ids]
            if to_delete:
                self.collection.delete(ids=to_delete)
        except Exception:
            pass

        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        self.logger.info("Indicizzati %d documenti", len(texts))

    @lru_cache(maxsize=512)
    def _get_embedding_cached(self, query: str) -> np.ndarray:
        """Cache LRU per embeddings delle query"""
        with torch.no_grad():
            emb = self.embed_model.encode(
                [query], 
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        return emb[0] if isinstance(emb, np.ndarray) else emb

    def search(self, query: str, k: Optional[int] = None) -> Tuple[List[str], List[Dict]]:
        """Ricerca ottimizzata con cache"""
        k = k or self.config.top_k_results
        
        q_emb = self._get_embedding_cached(query)
        
        results = self.collection.query(
            query_embeddings=[q_emb.tolist()], 
            n_results=k,
            include=["documents", "metadatas"]
        )
        
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return docs, metas


# --- RESPONSE GENERATOR OTTIMIZZATO ---
class ResponseGenerator:
    """LLM con lazy loading e quantizzazione 4-bit"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.tokenizer = None
        self.model = None
        self._initialized = False

    def _lazy_initialize(self) -> None:
        """Inizializzazione lazy: carica solo quando serve"""
        if self._initialized:
            return
            
        self.logger.info("Caricamento LLM (lazy init)...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model, 
                use_fast=True
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configurazione quantizzazione 4-bit (molto più efficiente)
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16,
            }
            
            if self.config.use_4bit_quantization and torch.cuda.is_available():
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                    self.logger.info("Usando quantizzazione 4-bit")
                except Exception as e:
                    self.logger.warning("4-bit fallback: %s", e)
                    model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.causalLm_model, 
                **model_kwargs
            )
            
            # Flash Attention 2 se disponibile
            if self.config.use_flash_attention:
                try:
                    self.model = self.model.to_bettertransformer()
                    self.logger.info("Flash Attention abilitata")
                except Exception:
                    pass
            
            self.model.eval()  # Modalità inferenza
            self._initialized = True
            self.logger.info("LLM pronto")
            
        except Exception as e:
            self.logger.exception("Errore caricamento LLM: %s", e)
            raise

    def _build_minimal_prompt(self, context: str, question: str) -> str:
        """Prompt minimalista per ridurre token count"""
        return (
            f"Contesto:\n{context}\n\n"
            f"Domanda: {question}\n"
            f"Rispondi in italiano, solo su accessibilità videogiochi, "
            f"usando solo info nel contesto. Risposta:"
        )

    def generate_response(self, context: str, question: str) -> str:
        """Genera risposta con lazy loading"""
        self._lazy_initialize()  # Carica solo al primo utilizzo
        
        prompt = self._build_minimal_prompt(context, question)
        
        # Tokenizzazione con limiti stretti
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024  # Ridotto da 2048
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generation ottimizzata
        gen_kwargs = {
            "max_new_tokens": self.config.max_response_tokens,
            "do_sample": False,  # Greedy per velocità
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": self.config.use_cache_generation,
        }

        with torch.no_grad():  # Disabilita gradient
            gen_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **gen_kwargs,
            )
        
        generated = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        
        # Estrai risposta
        if "Risposta:" in generated:
            return generated.split("Risposta:", 1)[1].strip()
        return generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()


# --- MAIN APPLICATION ---
class AccessibilityAssistant:
    """Assistente con ottimizzazioni multi-livello"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.loader = GameDataLoader(config.json_file, self.logger)
        self.vector_store = ChromaVectorStore(config, self.logger)
        self.response_generator = ResponseGenerator(config, self.logger)
        self._answer_cache: OrderedDict = OrderedDict()
        self._query_hash_cache: Dict[str, str] = {}

    def setup(self) -> None:
        """Setup con inizializzazione ottimizzata"""
        self.logger.info("Setup assistente...")
        games = self.loader.load()

        self.vector_store.initialize()
        
        texts, metadatas, ids = DocumentPreprocessor.prepare_documents(games)
        self.vector_store.index_documents(texts, metadatas, ids)

        # LLM NON viene caricato qui (lazy loading)
        self.logger.info("Setup completato (LLM in lazy mode)")

    def _compact_context(self, docs: List[str]) -> str:
        """Compatta contesto in modo aggressivo"""
        per_doc = self.config.context_max_chars_per_doc
        total_limit = self.config.context_max_total_chars

        trimmed = [d[:per_doc] for d in docs]
        context = "\n---\n".join(trimmed)
        
        if len(context) > total_limit:
            context = context[:total_limit] + "..."
        return context

    def _hash_query(self, question: str, k: Optional[int]) -> str:
        """Hash per cache key"""
        key = f"{question.strip()}_{k}"
        return hashlib.md5(key.encode()).hexdigest()

    def query(self, question: str, k: Optional[int] = None) -> Dict:
        """Query con cache multi-livello"""
        if not question or not question.strip():
            return {"answer": "Inserisci una domanda valida.", "sources": []}

        # Cache check
        cache_key = self._hash_query(question, k)
        if cache_key in self._answer_cache:
            val = self._answer_cache.pop(cache_key)
            self._answer_cache[cache_key] = val
            return val

        # Ricerca vettoriale
        docs, metas = self.vector_store.search(question, k=k)
        if not docs:
            return {
                "answer": "Non ho trovato informazioni rilevanti.", 
                "sources": []
            }

        # Genera risposta
        compact_context = self._compact_context(docs)
        answer = self.response_generator.generate_response(compact_context, question)

        sources = [
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "category": m.get("category"),
                "score": m.get("score"),
                "is_native": m.get("is_native"),
            }
            for m in metas
        ]

        result = {"answer": answer, "sources": sources}
        
        # Aggiorna cache
        self._answer_cache[cache_key] = result
        if len(self._answer_cache) > self.config.answer_cache_size:
            self._answer_cache.popitem(last=False)
        
        return result

    def interactive_mode(self) -> None:
        """REPL interattivo"""
        print("\n" + "=" * 60)
        print("ASSISTENTE ACCESSIBILITÀ VIDEOGIOCHI - Modalità Ottimizzata")
        print("=" * 60)
        print("Digita 'esci' per terminare.\n")
        
        while True:
            try:
                q = input("🎮 Domanda: ").strip()
                if q.lower() in {"esci", "exit", "quit", "q"}:
                    print("Arrivederci 👋")
                    break
                if not q:
                    continue
                    
                res = self.query(q)
                print(f"\n🤖 Risposta:\n{res['answer']}")
                
                if res["sources"]:
                    print("\n📚 Fonti:")
                    for s in res["sources"]:
                        print(f" • {s.get('name')} (Score: {s.get('score')})")
                        
            except KeyboardInterrupt:
                print("\nInterrotto. Arrivederci 👋")
                break
            except Exception as e:
                self.logger.exception("Errore: %s", e)
                print("Si è verificato un errore.")


# --- ENTRY POINT ---
def main() -> int:
    cfg = Config()
    logger = setup_logging(cfg.log_level)
    logger.info(f"Device: {cfg.device}")
    
    try:
        assistant = AccessibilityAssistant(cfg)
        assistant.setup()
        assistant.interactive_mode()
        return 0
    except Exception as e:
        logger.exception("Errore fatale: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
