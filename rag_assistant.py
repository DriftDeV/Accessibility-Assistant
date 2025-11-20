
# RAG Assistant per l'Accessibilità nei Videogiochi
# Versione rifattorizzata per Mistral-7B con best practice industriali
# - Leggibile e modulare
# - Usa Chromadb per il vector store e SentenceTransformer per embeddings
# - Usa Mistral-7B (instruct) come LLM principale, con fallback robusto
# - Prompt vincolato a rispondere SOLO a domande su accessibilità nei videogiochi

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import torch


# --- CONFIGURAZIONE ---
@dataclass
class Config:
    """
    Configurazione centralizzata.
    Modifica qui i percorsi o i nomi dei modelli.
    """
    json_file: Path = Path("games.json")
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Usare la versione instruct di Mistral 7B per comportamento orientato alle istruzioni
    llm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    chroma_collection: str = "games_accessibility_v2"
    chroma_persist_dir: Path = Path("./chroma_db")
    top_k_results: int = 3
    max_response_tokens: int = 256
    # se hai GPU disponibile, verrà usata; altrimenti CPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # batch size per embedding
    embed_batch_size: int = 32
    # opzioni per il caricamento del modello LLM (se bitsandbytes installato abilita 8bit)
    allow_8bit: bool = True 
    # livello di logging
    log_level: int = logging.INFO


# --- LOGGING ---
def setup_logging(level: int) -> logging.Logger:
    """Configura il logger con formato pulito e ritorna l'istanza"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("rag_assistant")


# --- DATA LOADING ---
class GameDataLoader:
    """Carica e valida il JSON dei giochi"""

    def __init__(self, filepath: Path, logger: logging.Logger):
        self.filepath = filepath
        self.logger = logger

    def load(self) -> List[Dict]:
        """Carica il file JSON e restituisce la lista di giochi validata"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"{self.filepath} non esiste. Posiziona il file nella directory corretta.")

        with open(self.filepath, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as e:
                raise ValueError(f"Errore parsing JSON: {e}")

        if not isinstance(data, list):
            raise ValueError("Il file JSON deve contenere una lista di oggetti (giochi).")

        self.logger.info("Caricati %d giochi da %s", len(data), self.filepath)
        return data


# --- DOCUMENT PREPROCESSOR ---
class DocumentPreprocessor:
    """Prepara testi e metadati per l'indicizzazione"""

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
            source_ref = g.get("source_ref", "")

            platforms_str = ", ".join(platforms) if platforms else "Non specificato"
            features_str = "; ".join(features) if features else "Nessuna feature elencata"

            visual = details.get("visual", "Non specificato")
            motor = details.get("motor", "Non specificato")
            auditory = details.get("auditory", "Non specificato")
            cognitive = details.get("cognitive", "Non specificato")

            native_status = "Supporto Nativo" if is_native else "Richiede modifiche/mod"

            # Documento strutturato pensato per la ricerca semantica e per fornire informazioni chiare al LLM.
            text = (
                f"TITOLO: {name}\n"
                f"ID: {game_id}\n"
                f"CATEGORIA: {category}\n"
                f"PUNTEGGIO_ACCESSIBILITA: {access_level if access_level is not None else 'N/A'}/10\n"
                f"TIPO_SUPPORTO: {native_status}\n"
                f"PIATTAFORME: {platforms_str}\n"
                f"SOURCE_REF: {source_ref}\n\n"
                f"DESCRIZIONE:\n{description}\n\n"
                f"FEATURES:\n{features_str}\n\n"
                f"DETTAGLI_ACCESSIBILITA:\n"
                f"- Visiva: {visual}\n"
                f"- Motoria: {motor}\n"
                f"- Uditiva: {auditory}\n"
                f"- Cognitiva: {cognitive}\n"
            )

            texts.append(text)
            # Converti platforms (lista) in stringa per compatibilità ChromaDB
            platforms_meta = ", ".join(platforms) if platforms else "N/A"
            metadatas.append({
                "id": game_id,
                "name": name,
                "category": category,
                "score": float(access_level) if access_level is not None else None,
                "is_native": is_native,
                "platforms": platforms_meta,  # Ora è una stringa, non una lista
                "source_ref": source_ref,
            })
            ids.append(game_id)

        return texts, metadatas, ids


# --- VECTOR STORE (ChromaDB) ---
class ChromaVectorStore:
    """Wrapper per ChromaDB che gestisce initialization, indexing e search"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client: Optional[chromadb.api.models.APIClient] = None
        self.collection = None
        self.embed_model: Optional[SentenceTransformer] = None

    def initialize(self) -> None:
        """Inizializza SentenceTransformer e ChromaDB con persistenza"""
        self.logger.info("Inizializzazione vector store e modello di embedding...")
        self.embed_model = SentenceTransformer(self.config.embedding_model, device=self.config.device)
        self.config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        # Uso PersistentClient per persistenza locale
        self.client = chromadb.PersistentClient(path=str(self.config.chroma_persist_dir),
                                                settings=Settings(anonymized_telemetry=False))

        # crea o recupera collection
        try:
            self.collection = self.client.get_collection(name=self.config.chroma_collection)
            self.logger.info("Collection '%s' caricata", self.config.chroma_collection)
        except Exception:
            self.collection = self.client.create_collection(
                name=self.config.chroma_collection,
                metadata={"description": "Video games accessibility data"}
            )
            self.logger.info("Collection '%s' creata", self.config.chroma_collection)

    def index_documents(self, texts: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        """Calcola embeddings e inserisce/aggiorna la collection"""
        if self.embed_model is None or self.collection is None:
            raise RuntimeError("Vector store non inizializzato. Chiamare initialize() prima.")

        self.logger.info("Calcolo embeddings (%d documenti)...", len(texts))
        embeddings = self.embed_model.encode(texts,
                                            show_progress_bar=True,
                                            batch_size=self.config.embed_batch_size,
                                            normalize_embeddings=True)

        # garantiamo che embeddings sia lista di liste
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        # rimuovere eventuali documenti con gli stessi id per evitare duplicati
        try:
            existing = self.collection.get()
            existing_ids = set(existing.get("ids", []))
            to_delete = [i for i in ids if i in existing_ids]
            if to_delete:
                self.collection.delete(ids=to_delete)
                self.logger.info("Rimossi %d documenti esistenti prima dell'aggiornamento", len(to_delete))
        except Exception:
            self.logger.debug("Nessuna pulizia della collection necessaria o errore non critico.")

        # aggiungi documenti
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        self.logger.info("Indicizzati %d documenti", len(texts))

    def search(self, query: str, k: Optional[int] = None) -> Tuple[List[str], List[Dict]]:
        """Esegue ricerca semantica e restituisce documenti + metadati"""
        if k is None:
            k = self.config.top_k_results

        if self.embed_model is None or self.collection is None:
            raise RuntimeError("Vector store non inizializzato. Chiamare initialize() prima.")

        q_emb = self.embed_model.encode([query], normalize_embeddings=True)
        if isinstance(q_emb, np.ndarray):
            q_emb = q_emb.tolist()[0]
        else:
            q_emb = q_emb[0]

        results = self.collection.query(query_embeddings=[q_emb], n_results=k,
                                        include=["documents", "metadatas", "distances"])
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        self.logger.debug("Search distances: %s", distances)
        return docs, metas


# --- RESPONSE GENERATOR (LLM usando Mistral 7B) ---
class ResponseGenerator:
    """Gestisce il caricamento del LLM e la generazione di testo"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.tokenizer = None
        self.model = None

    def initialize(self) -> None:
        """Carica il tokenizer e il modello Mistral 7B in modo robusto"""
        self.logger.info("Caricamento LLM: %s", self.config.llm_model)
        try:
            # tentativo di usare bitsandbytes (8bit) se disponibile e consentito
            load_in_8bit = False
            try:
                if self.config.allow_8bit:
                    import bitsandbytes  # type: ignore
                    load_in_8bit = True
                    self.logger.info("bitsandbytes rilevato: tenterò caricamento in 8-bit per risparmio memoria")
            except Exception:
                load_in_8bit = False

            from transformers import AutoTokenizer, AutoModelForCausalLM

            # tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model, use_fast=True)
            if self.tokenizer.pad_token_id is None:
                # assegna pad_token se assente
                if self.tokenizer.eos_token_id is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

            # modello: proviamo ad utilizzare device_map automatico per sfruttare GPU/CPU in modo ottimale
            model_kwargs = dict(low_cpu_mem_usage=True)
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
                if load_in_8bit:
                    # parametro load_in_8bit è supportato quando bitsandbytes è installato
                    model_kwargs["load_in_8bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(self.config.llm_model, **model_kwargs)

            # Se abbiamo aggiunto token al tokenizer, resize
            try:
                self.model.resize_token_embeddings(len(self.tokenizer))
            except Exception:
                pass

            self.logger.info("LLM caricato correttamente")
        except Exception as e:
            self.logger.exception("Errore caricamento LLM: %s", e)
            raise RuntimeError("Impossibile caricare il LLM. Controlla la configurazione ed i requisiti di memoria.") from e

    def _build_prompt(self, context: str, question: str) -> str:
        """
        Prompt con regole vincolanti:
        - Rispondi SOLO in italiano.
        - Usa ESCLUSIVAMENTE le informazioni presenti nel contesto.
        - Se la domanda non riguarda accessibilità o videogiochi rispondi con messaggio di limitazione.
        - Specifica sempre se una feature è nativa o richiede mod.
        - Fornisci punteggio di accessibilità se disponibile nel contesto.
        - Se non ci sono informazioni sufficienti dichiara l'incertezza.
        """
        prompt = (
            "Sei un assistente esperto in accessibilità nei videogiochi. "
            "Rispondi in italiano basandoti ESCLUSIVAMENTE sul CONTENUTO fornito nella sezione CONTESTO.\n\n"
            "CONTESTO:\n"
            f"{context}\n\n"
            "ISTRUZIONI:\n"
            "1) Rispondi SOLO in italiano.\n"
            "2) Usa SOLO le informazioni presenti nel contesto.\n"
            "3) Se la domanda non riguarda accessibilità o videogiochi, rispondi esattamente: "
            "\"Mi dispiace, posso rispondere solo a domande sull'accessibilità nei videogiochi.\"\n"
            "4) Specifica sempre se una funzionalità è nativa o richiede modifiche/mod.\n"
            "5) Fornisci il punteggio di accessibilità quando disponibile.\n"
            "6) Sii conciso e pratico; se non trovi dati adeguati, dillo chiaramente.\n\n"
            f"DOMANDA: {question}\n\nRISPOSTA:"
        )
        return prompt

    def generate_response(self, context: str, question: str) -> str:
        """Genera la risposta usando model.generate per avere maggiore controllo"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM non inizializzato. Chiamare initialize() prima.")

        prompt = self._build_prompt(context, question)
        tok = self.tokenizer
        model = self.model

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            # modello su device_map=auto gestirà i dispositivi, ma assicuriamone lo stato
            try:
                model = model.to("cuda")
            except Exception:
                pass

        # Parametri conservativi per risposte coerenti
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.config.max_response_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=0.95,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

        # Usare model.generate; gestiamo eventuali errori e fallback
        try:
            gen_ids = model.generate(input_ids=inputs["input_ids"],
                                     attention_mask=inputs.get("attention_mask"),
                                     max_new_tokens=self.config.max_response_tokens,
                                     do_sample=False,
                                     temperature=0.0,
                                     top_p=0.95,
                                     eos_token_id=tok.eos_token_id,
                                     pad_token_id=tok.pad_token_id)
            generated = tok.decode(gen_ids[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.exception("Errore durante generation: %s", e)
            raise

        # Cerchiamo la parte dopo 'RISPOSTA:' se presente
        if "RISPOSTA:" in generated:
            return generated.split("RISPOSTA:", 1)[1].strip()
        # altrimenti rimuoviamo il prompt (se il modello ha replicato il prompt)
        if generated.startswith(prompt):
            return generated[len(prompt):].strip()
        return generated.strip()


# --- MAIN APPLICATION ---
class AccessibilityAssistant:
    """Classe che aggrega loader, vector store e LLM per rispondere alle query"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.loader = GameDataLoader(config.json_file, self.logger)
        self.vector_store = ChromaVectorStore(config, self.logger)
        self.response_generator = ResponseGenerator(config, self.logger)

    def setup(self) -> None:
        """Carica dati, inizializza componenti e indicizza i documenti"""
        self.logger.info("Setup dell'assistente in corso...")
        games = self.loader.load()

        # Inizializza vector store e embeddings
        self.vector_store.initialize()

        # Prepara documenti e indicizza
        texts, metadatas, ids = DocumentPreprocessor.prepare_documents(games)
        self.vector_store.index_documents(texts, metadatas, ids)

        # Inizializza LLM (alla fine, per evitare uso memoria non necessario durante indexing)
        self.response_generator.initialize()
        self.logger.info("Setup completato. Sistema pronto per le query.")

    def query(self, question: str, k: Optional[int] = None) -> Dict:
        """Esegue ricerca e genera la risposta; ritorna anche le fonti utilizzate"""
        if not question or not question.strip():
            return {"answer": "Inserisci una domanda valida.", "sources": []}

        docs, metas = self.vector_store.search(question, k=k)
        if not docs:
            return {"answer": "Non ho trovato informazioni rilevanti per rispondere alla tua domanda.", "sources": []}

        # Creiamo un contesto compatto: concatenazione dei documenti recuperati
        context = "\n\n---\n\n".join(docs)
        answer = self.response_generator.generate_response(context, question)

        # Normalizziamo le fonti per la UI
        sources = []
        for m in metas:
            sources.append({
                "id": m.get("id"),
                "name": m.get("name"),
                "category": m.get("category"),
                "score": m.get("score"),
                "is_native": m.get("is_native"),
                "platforms": m.get("platforms"),
                "source_ref": m.get("source_ref")
            })

        return {"answer": answer, "sources": sources}

    def interactive_mode(self) -> None:
        """Semplice REPL per interrogare l'assistente da terminale"""
        print("\n" + "=" * 60)
        print(f"ASSISTENTE ACCESSIBILITÀ VIDEOGIOCHI {Config.llm_model} - Modalità interattiva")
        print("=" * 60)
        print("Digita 'esci' per terminare.")
        while True:
            try:
                q = input("\n🎮 Domanda: ").strip()
                if q.lower() in {"esci", "exit", "quit", "q"}:
                    print("Arrivederci 👋")
                    break
                if not q:
                    continue
                res = self.query(q)
                print("\n🤖 Risposta:\n" + res["answer"])
                if res["sources"]:
                    print("\n📚 Fonti:")
                    for s in res["sources"]:
                        print(f" • {s.get('name')} (categoria: {s.get('category')}, score: {s.get('score')})")
            except KeyboardInterrupt:
                print("\nInterrotto dall'utente. Arrivederci 👋")
                break
            except Exception as e:
                self.logger.exception("Errore durante l'interazione: %s", e)
                print("Si è verificato un errore. Controlla i log.")

# --- ENTRY POINT ---
def main() -> int:
    cfg = Config()
    logger = setup_logging(cfg.log_level)
    logger.info(f"Using device {cfg.device}")
    try:
        assistant = AccessibilityAssistant(cfg)
        assistant.setup()
        assistant.interactive_mode()
        return 0
    except Exception as e:
        logger.exception("Errore fatale durante l'esecuzione: %s", e)
        print("Errore fatale: ", e)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())