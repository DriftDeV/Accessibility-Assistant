"""
Assistente RAG per l'Accessibilità nei Videogiochi.

Questo modulo implementa un sistema di domande e risposte (RAG) che utilizza:
- ChromaDB per il vector store
- Ollama per i language model (embedding e generazione)
- Best practice di Python a livello industriale

Esempio di utilizzo:
    config = RAGConfig()
    assistant = AccessibilityAssistant(config)
    assistant.setup()
    response = assistant.query("Quali giochi sono accessibili per daltonici?")
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================


@dataclass
class RAGConfig:
    """Configurazione centralizzata del sistema RAG.
    
    Attributes:
        games_file: Percorso al file JSON con i dati dei giochi
        chroma_dir: Directory per la persistenza di ChromaDB
        chroma_collection: Nome della collection in ChromaDB
        ollama_embedding_model: Modello di embedding di Ollama
        ollama_llm_model: Modello LLM di Ollama per generare risposte
        ollama_base_url: URL base di Ollama
        top_k_results: Numero di risultati di ricerca da recuperare
        max_tokens: Token massimi per le risposte generate
    """

    games_file: Path = Path("games.json")
    chroma_dir: Path = Path("./chroma_db")
    chroma_collection: str = "games_accessibility"
    
    # Modelli Ollama
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_llm_model: str = "llama3:8b"
    ollama_base_url: str = "http://localhost:11434"
    
    # Parametri di ricerca e generazione
    top_k_results: int = 3
    max_tokens: int = 300
    
    # Logging
    log_level: int = logging.INFO


# ==============================================================================
# LOGGING
# ==============================================================================


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configura il logging dell'applicazione.
    
    Args:
        level: Livello di logging (default: INFO)
        
    Returns:
        Logger configurato
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# ==============================================================================
# DATA LOADING
# ==============================================================================
class GameDataLoader:
    """Carica e valida i dati dei giochi dal file JSON.
    
    Attributes:
        filepath: Percorso del file JSON
        logger: Logger dell'applicazione
        _cache: Cache in memoria dei dati caricati
    """

    def __init__(self, filepath: Path, logger: logging.Logger) -> None:
        """Inizializza il loader.
        
        Args:
            filepath: Percorso del file JSON
            logger: Logger dell'applicazione
        """
        self.filepath = filepath
        self.logger = logger
        self._cache: Optional[list[dict]] = None

    def load(self) -> list[dict]:
        """Carica i dati dal JSON con caching in memoria.
        
        Returns:
            Lista di dizionari con i dati dei giochi
            
        Raises:
            FileNotFoundError: Se il file non esiste
            ValueError: Se il JSON non contiene una lista valida
        """
        if self._cache is not None:
            return self._cache

        if not self.filepath.exists():
            raise FileNotFoundError(f"File non trovato: {self.filepath}")

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON non valido: {e}") from e

        if not isinstance(data, list):
            raise ValueError("Il JSON deve contenere una lista di oggetti")

        if not data:
            raise ValueError("Il JSON contiene una lista vuota")

        self._cache = data
        self.logger.info("Caricati %d giochi", len(data))
        return data


# ==============================================================================
# PREPROCESSING DOCUMENTI
# ==============================================================================

class DocumentPreprocessor:
    """Prepara i documenti per l'indicizzazione in ChromaDB."""

    @staticmethod
    def prepare_documents(games: list[dict]) -> tuple[list[str], list[dict], list[str]]:
        """Prepara testi, metadati e IDs per l'indicizzazione.
        
        Args:
            games: Lista di dizionari con i dati dei giochi
            
        Returns:
            Tupla di (testi, metadati, ids)
        """
        texts = []
        metadatas = []
        ids = []

        for game in games:
            game_id = str(game.get("id", "unknown"))
            name = game.get("name", "Unknown")
            category = game.get("category", "Unknown")
            access_level = game.get("access_level", 0)
            is_native = game.get("is_native", False)
            platforms = game.get("platforms", [])
            description = game.get("description", "")
            features = game.get("features", [])
            details = game.get("accessibility_details", {}) or {}

            # Formato compatto ma informativo
            text = DocumentPreprocessor._format_game_text(
                name, category, access_level, is_native, platforms,
                description, features, details
            )

            texts.append(text)
            metadatas.append({
                "id": game_id,
                "name": name,
                "category": category,
                "score": float(access_level),
                "is_native": is_native,
                "platforms": ", ".join(platforms) if platforms else "N/A",
            })
            ids.append(game_id)

        return texts, metadatas, ids

    @staticmethod
    def _format_game_text(
        name: str,
        category: str,
        access_level: float,
        is_native: bool,
        platforms: list[str],
        description: str,
        features: list[str],
        details: dict,
    ) -> str:
        """Formatta un gioco in testo per il vettore.
        
        Args:
            name: Nome del gioco
            category: Categoria del gioco
            access_level: Livello di accessibilità (0-10)
            is_native: Se è nativamente accessibile
            platforms: Piattaforme disponibili
            description: Descrizione del gioco
            features: Features di accessibilità
            details: Dettagli di accessibilità per tipologia
            
        Returns:
            Testo formattato per l'embedding
        """
        parts = [
            f"Gioco: {name}",
            f"Categoria: {category}",
            f"Accessibilità: {access_level}/10",
            f"Supporto nativo: {'Sì' if is_native else 'No'}",
            f"Piattaforme: {', '.join(platforms) if platforms else 'Multi-platform'}",
            f"Descrizione: {description[:300]}",  # Tronca per performance
        ]

        # Aggiungi features se presenti
        if features:
            parts.append(f"Features: {', '.join(features[:5])}")

        # Aggiungi dettagli di accessibilità
        accessibility_parts = []
        if details.get("visual"):
            accessibility_parts.append(f"Visiva: {details['visual'][:100]}")
        if details.get("motor"):
            accessibility_parts.append(f"Motoria: {details['motor'][:100]}")
        if details.get("auditory"):
            accessibility_parts.append(f"Uditiva: {details['auditory'][:100]}")
        if details.get("cognitive"):
            accessibility_parts.append(f"Cognitiva: {details['cognitive'][:100]}")

        if accessibility_parts:
            parts.append("Dettagli: " + " | ".join(accessibility_parts))

        return "\n".join(parts)


# ==============================================================================
# VECTOR STORE
# ==============================================================================

class VectorStore:
    """Gestisce ChromaDB per l'indicizzazione e la ricerca semantica.
    
    Attributes:
        config: Configurazione del RAG
        logger: Logger dell'applicazione
        client: Client di ChromaDB
        collection: Collection di ChromaDB
    """

    def __init__(self, config: RAGConfig, logger: logging.Logger) -> None:
        """Inizializza il vector store.
        
        Args:
            config: Configurazione del RAG
            logger: Logger dell'applicazione
        """
        self.config = config
        self.logger = logger
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None

    def initialize(self) -> None:
        """Inizializza la connessione a ChromaDB."""
        self.logger.info("Inizializzo ChromaDB...")

        # Crea directory se non esiste
        self.config.chroma_dir.mkdir(parents=True, exist_ok=True)

        # Inizializza client
        self.client = chromadb.PersistentClient(
            path=str(self.config.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Prova a caricare la collection, altrimenti crea una nuova
        try:
            self.collection = self.client.get_collection(
                name=self.config.chroma_collection
            )
            self.logger.info("Collection caricata")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.config.chroma_collection,
                metadata={"description": "Videogiochi con informazioni di accessibilità"},
            )
            self.logger.info("Collection creata")

    def index_documents(
        self, texts: list[str], metadatas: list[dict], ids: list[str]
    ) -> None:
        """Indicizza i documenti in ChromaDB.
        
        Utilizza Ollama per generare gli embeddings in modo locale.
        
        Args:
            texts: Lista di testi da indicizzare
            metadatas: Lista di metadati
            ids: Lista di IDs
        """
        if not self.collection:
            raise RuntimeError("Vector store non inizializzato")

        self.logger.info("Indicizzazione di %d documenti...", len(texts))

        try:
            # Rimuovi IDs duplicati se esistono
            existing_ids = set(self.collection.get(include=[]).get("ids", []))
            to_delete = [id_ for id_ in ids if id_ in existing_ids]
            if to_delete:
                self.collection.delete(ids=to_delete)
                self.logger.info("Rimossi %d documenti duplicati", len(to_delete))
        except Exception as e:
            self.logger.warning("Errore durante la rimozione di duplicati: %s", e)

        # Usa Ollama per gli embeddings
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        self.logger.info("Indicizzazione completata")

    def search(self, query: str, k: Optional[int] = None) -> tuple[list[str], list[dict]]:
        """Ricerca semantica nel vector store.
        
        Args:
            query: Testo della query
            k: Numero di risultati da restituire (default: config.top_k_results)
            
        Returns:
            Tupla di (documenti, metadati)
        """
        if not self.collection:
            raise RuntimeError("Vector store non inizializzato")

        k = k or self.config.top_k_results

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas"],
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            self.logger.debug("Trovati %d risultati per: %s", len(documents), query)
            return documents, metadatas
        except Exception as e:
            self.logger.error("Errore nella ricerca: %s", e)
            return [], []


# ==============================================================================
# GENERAZIONE RISPOSTE
# ==============================================================================

class ResponseGenerator:
    """Genera risposte utilizzando Ollama.
    
    Attributes:
        config: Configurazione del RAG
        logger: Logger dell'applicazione
    """

    def __init__(self, config: RAGConfig, logger: logging.Logger) -> None:
        """Inizializza il generatore di risposte.
        
        Args:
            config: Configurazione del RAG
            logger: Logger dell'applicazione
        """
        self.config = config
        self.logger = logger

    def generate(self, context: str, question: str) -> str:
        """Genera una risposta utilizzando Ollama.
        
        Args:
            context: Contesto (risultati della ricerca)
            question: Domanda dell'utente
            
        Returns:
            Risposta generata
        """
        try:
            import ollama
        except ImportError:
            self.logger.error(
                "ollama non è installato. Installalo con: pip install ollama"
            )
            return "Errore: libreria ollama non disponibile."

        prompt = self._build_prompt(context, question)

        try:
            response = ollama.generate(
                model=self.config.ollama_llm_model,
                prompt=prompt,
                stream=False,
                options={"temperature": 0.3, "top_k": 40, "top_p": 0.9},
            )

            answer = response.get("response", "").strip()
            return answer if answer else "Scusa, non ho potuto generare una risposta."
        except Exception as e:
            self.logger.error("Errore nella generazione: %s", e)
            return f"Errore durante la generazione della risposta: {e}"

    @staticmethod
    def _build_prompt(context: str, question: str) -> str:
        """Costruisce il prompt per Ollama.
        
        Args:
            context: Contesto dalla ricerca
            question: Domanda dell'utente
            
        Returns:
            Prompt formattato
        """
        return f"""Sei un esperto di accessibilità nei videogiochi. 
Rispondi SOLO alla domanda basandoti sul contesto fornito.
Se le informazioni nel contesto non permettono di rispondere, 
rispondi che non hai informazioni sufficienti.

Rispondi sempre in italiano e sii il più conciso possibile.

CONTESTO:
{context}

DOMANDA:
{question}

RISPOSTA:"""


# ==============================================================================
# ASSISTENTE PRINCIPALE
# ==============================================================================

class AccessibilityAssistant:
    """Assistente RAG per domande sull'accessibilità nei videogiochi.
    
    Attributes:
        config: Configurazione del RAG
        logger: Logger dell'applicazione
        loader: Loader per i dati dei giochi
        vector_store: Vector store
        response_generator: Generatore di risposte
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        """Inizializza l'assistente.
        
        Args:
            config: Configurazione (default: RAGConfig())
        """
        self.config = config or RAGConfig()
        self.logger = setup_logging(self.config.log_level)

        self.loader = GameDataLoader(self.config.games_file, self.logger)
        self.vector_store = VectorStore(self.config, self.logger)
        self.response_generator = ResponseGenerator(self.config, self.logger)

    def setup(self) -> None:
        """Carica i dati e indicizza i documenti."""
        self.logger.info("Setup dell'assistente...")

        # Carica i dati
        games = self.loader.load()

        # Inizializza il vector store
        self.vector_store.initialize()

        # Prepara i documenti
        texts, metadatas, ids = DocumentPreprocessor.prepare_documents(games)

        # Indicizza
        self.vector_store.index_documents(texts, metadatas, ids)

        self.logger.info("Setup completato")

    def query(self, question: str) -> dict:
        """Elabora una domanda e restituisce una risposta.
        
        Args:
            question: Domanda dell'utente
            
        Returns:
            Dizionario con 'answer' e 'sources'
        """
        # Validazione input
        if not question or not question.strip():
            return {
                "answer": "Inserisci una domanda valida.",
                "sources": [],
            }

        # Ricerca semantica
        documents, metadatas = self.vector_store.search(question)

        if not documents:
            return {
                "answer": "Non ho trovato informazioni sull'accessibilità per questa domanda.",
                "sources": [],
            }

        # Costruisci il contesto
        context = self._build_context(documents)

        # Genera la risposta
        answer = self.response_generator.generate(context, question)

        # Formatta le fonti
        sources = [
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "category": m.get("category"),
                "score": m.get("score"),
                "is_native": m.get("is_native"),
            }
            for m in metadatas
        ]

        return {"answer": answer, "sources": sources}

    @staticmethod
    def _build_context(documents: list[str]) -> str:
        """Costruisce il contesto dai documenti recuperati.
        
        Args:
            documents: Lista di documenti
            
        Returns:
            Contesto formattato
        """
        return "\n\n".join([f"[Documento {i + 1}]\n{doc}" for i, doc in enumerate(documents)])

    def interactive_mode(self) -> None:
        """Avvia la modalità interattiva."""
        print("\n" + "=" * 70)
        print("ASSISTENTE PER L'ACCESSIBILITÀ NEI VIDEOGIOCHI")
        print("=" * 70)
        print("Digita una domanda sull'accessibilità nei videogiochi.")
        print("Digita 'esci' per terminare.\n")

        while True:
            try:
                question = input("❓ Domanda: ").strip()

                if question.lower() in {"esci", "exit", "quit", "q"}:
                    print("\n👋 Arrivederci!")
                    break

                if not question:
                    continue

                print("\n⏳ Elaborazione...\n")
                result = self.query(question)

                print(f"💬 Risposta:\n{result['answer']}\n")

                if result["sources"]:
                    print("📚 Fonti:")
                    for source in result["sources"]:
                        native_label = "✓ Nativo" if source["is_native"] else "✗ Mod"
                        print(
                            f"  • {source['name']} ({source['category']}) "
                            f"- Accessibilità: {source['score']}/10 [{native_label}]"
                        )

                print()

            except KeyboardInterrupt:
                print("\n\n👋 Arrivederci!")
                break
            except Exception as e:
                self.logger.exception("Errore durante l'elaborazione")
                print(f"❌ Errore: {e}\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================


def main() -> int:
    """Punto di ingresso dell'applicazione.
    
    Returns:
        Codice di uscita (0 = successo, 1 = errore)
    """
    try:
        config = RAGConfig()
        assistant = AccessibilityAssistant(config)
        assistant.setup()
        assistant.interactive_mode()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logger = setup_logging()
        logger.exception("Errore fatale")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
