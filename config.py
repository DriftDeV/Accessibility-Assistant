"""
Gestione della configurazione da variabili d'ambiente.

Questo modulo legge i valori da .env e li rende disponibili
all'applicazione. Supporta override via variabili d'ambiente.

Esempio:
    from config import get_config
    
    config = get_config()
    print(config.ollama_base_url)
"""

import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os

# Carica variabili da .env se esiste
load_dotenv(".env", override=False)


def get_env(key: str, default: str = "") -> str:
    """Legge una variabile d'ambiente con fallback.
    
    Args:
        key: Nome della variabile
        default: Valore di default
        
    Returns:
        Valore della variabile o default
    """
    return os.getenv(key, default)


def get_int_env(key: str, default: int = 0) -> int:
    """Legge una variabile d'ambiente come intero.
    
    Args:
        key: Nome della variabile
        default: Valore di default
        
    Returns:
        Valore come intero
    """
    try:
        return int(get_env(key, str(default)))
    except ValueError:
        return default


class AppConfig:
    """Configurazione centralizzata dell'applicazione.
    
    Attributes:
        ollama_base_url: URL di base di Ollama
        ollama_embedding_model: Modello per embeddings
        ollama_llm_model: Modello LLM per generazione
        games_file: Path al file JSON dei giochi
        chroma_db_dir: Directory per ChromaDB
        top_k_results: Numero di risultati di ricerca
        max_tokens: Token massimi per risposte
        log_level: Livello di logging
    """

    def __init__(self) -> None:
        """Inizializza la configurazione da variabili d'ambiente."""
        # Ollama
        self.ollama_base_url = get_env(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )
        self.ollama_embedding_model = get_env(
            "OLLAMA_EMBEDDING_MODEL",
            "nomic-embed-text",
        )
        self.ollama_llm_model = get_env(
            "OLLAMA_LLM_MODEL",
            "mistral",
        )

        # File paths
        self.games_file = Path(
            get_env("GAMES_FILE", "games.json")
        )
        self.chroma_db_dir = Path(
            get_env("CHROMA_DB_DIR", "./chroma_db")
        )

        # RAG parameters
        self.top_k_results = get_int_env("TOP_K_RESULTS", 3)
        self.max_tokens = get_int_env("MAX_TOKENS", 300)

        # Logging
        log_level_str = get_env("LOG_LEVEL", "INFO").upper()
        self.log_level = getattr(logging, log_level_str, logging.INFO)

    def __repr__(self) -> str:
        """Rappresentazione stringa della configurazione."""
        return (
            f"AppConfig("
            f"ollama_base_url={self.ollama_base_url}, "
            f"embedding_model={self.ollama_embedding_model}, "
            f"llm_model={self.ollama_llm_model}, "
            f"games_file={self.games_file}, "
            f"top_k_results={self.top_k_results}, "
            f"log_level={logging.getLevelName(self.log_level)}"
            f")"
        )


def get_config() -> AppConfig:
    """Factory function per ottenere la configurazione.
    
    Returns:
        Istanza di AppConfig
        
    Example:
        config = get_config()
        print(config.ollama_base_url)
    """
    return AppConfig()
