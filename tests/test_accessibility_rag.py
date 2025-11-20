"""
Test unit per il modulo accessibility_rag.

Questo modulo contiene test per validare il comportamento
dei componenti principali del sistema RAG.

Run tests:
    pytest tests/test_accessibility_rag.py -v
"""

import json
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from accessibility_rag import (
    RAGConfig,
    GameDataLoader,
    DocumentPreprocessor,
    ResponseGenerator,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def temp_games_file() -> Generator[Path, None, None]:
    """Crea un file JSON temporaneo con dati di test."""
    games = [
        {
            "id": "TEST_001",
            "name": "Test Game 1",
            "category": "Action",
            "access_level": 9.0,
            "is_native": True,
            "platforms": ["PC"],
            "description": "Test description 1",
            "features": ["Feature1", "Feature2"],
            "accessibility_details": {
                "visual": "Good",
                "motor": "Fair",
            },
        },
        {
            "id": "TEST_002",
            "name": "Test Game 2",
            "category": "Adventure",
            "access_level": 7.0,
            "is_native": False,
            "platforms": ["PlayStation", "Xbox"],
            "description": "Test description 2",
            "features": ["Feature3"],
            "accessibility_details": {
                "auditory": "Excellent",
            },
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(games, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def mock_logger():
    """Crea un logger mock."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    return logger


# ==============================================================================
# TEST: GameDataLoader
# ==============================================================================


class TestGameDataLoader:
    """Test per GameDataLoader."""

    def test_load_valid_file(self, temp_games_file, mock_logger):
        """Test caricamento file valido."""
        loader = GameDataLoader(temp_games_file, mock_logger)
        games = loader.load()

        assert len(games) == 2
        assert games[0]["id"] == "TEST_001"
        assert games[1]["name"] == "Test Game 2"
        mock_logger.info.assert_called()

    def test_load_cache(self, temp_games_file, mock_logger):
        """Test che la cache funziona."""
        loader = GameDataLoader(temp_games_file, mock_logger)

        games1 = loader.load()
        games2 = loader.load()

        # Dovrebbe essere lo stesso oggetto (cache)
        assert games1 is games2
        # Logger dovrebbe essere chiamato solo una volta
        assert mock_logger.info.call_count == 1

    def test_load_nonexistent_file(self, mock_logger):
        """Test caricamento file inesistente."""
        loader = GameDataLoader(Path("nonexistent.json"), mock_logger)

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_invalid_json(self, mock_logger):
        """Test caricamento JSON non valido."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            f.write("invalid json {")
            f.flush()

            loader = GameDataLoader(Path(f.name), mock_logger)
            with pytest.raises(ValueError, match="JSON non valido"):
                loader.load()

    def test_load_not_list(self, mock_logger):
        """Test che il JSON sia una lista."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump({"key": "value"}, f)
            f.flush()

            loader = GameDataLoader(Path(f.name), mock_logger)
            with pytest.raises(ValueError, match="deve contenere una lista"):
                loader.load()


# ==============================================================================
# TEST: DocumentPreprocessor
# ==============================================================================


class TestDocumentPreprocessor:
    """Test per DocumentPreprocessor."""

    def test_prepare_documents(self, temp_games_file, mock_logger):
        """Test preparazione documenti."""
        loader = GameDataLoader(temp_games_file, mock_logger)
        games = loader.load()

        texts, metadatas, ids = DocumentPreprocessor.prepare_documents(games)

        assert len(texts) == 2
        assert len(metadatas) == 2
        assert len(ids) == 2

        # Controlla contenuto
        assert "TEST_001" in ids
        assert metadatas[0]["name"] == "Test Game 1"
        assert metadatas[0]["score"] == 9.0

    def test_format_game_text(self):
        """Test formattazione testo gioco."""
        text = DocumentPreprocessor._format_game_text(
            name="Test Game",
            category="Action",
            access_level=8.0,
            is_native=True,
            platforms=["PC", "PS5"],
            description="Test description",
            features=["F1", "F2"],
            details={"visual": "Good", "motor": "Fair"},
        )

        assert "Test Game" in text
        assert "Action" in text
        assert "8.0/10" in text
        assert "Sì" in text  # is_native=True
        assert "PC" in text


# ==============================================================================
# TEST: ResponseGenerator
# ==============================================================================


class TestResponseGenerator:
    """Test per ResponseGenerator."""

    def test_build_prompt(self):
        """Test costruzione prompt."""
        context = "Context about games"
        question = "What games are accessible?"

        prompt = ResponseGenerator._build_prompt(context, question)

        assert context in prompt
        assert question in prompt
        assert "CONTESTO:" in prompt
        assert "DOMANDA:" in prompt
        assert "RISPOSTA:" in prompt

    @patch("ollama.generate")
    def test_generate_success(self, mock_generate, mock_logger):
        """Test generazione risposta."""
        mock_generate.return_value = {"response": "Test answer"}

        config = RAGConfig()
        generator = ResponseGenerator(config, mock_logger)
        answer = generator.generate("context", "question")

        assert answer == "Test answer"
        mock_generate.assert_called_once()

    @patch("ollama.generate")
    def test_generate_error(self, mock_generate, mock_logger):
        """Test gestione errore nella generazione."""
        mock_generate.side_effect = Exception("Connection error")

        config = RAGConfig()
        generator = ResponseGenerator(config, mock_logger)
        answer = generator.generate("context", "question")

        assert "Errore" in answer or "error" in answer.lower()


# ==============================================================================
# TEST: RAGConfig
# ==============================================================================


class TestRAGConfig:
    """Test per RAGConfig."""

    def test_default_config(self):
        """Test configurazione di default."""
        config = RAGConfig()

        assert config.games_file == Path("games.json")
        assert config.top_k_results == 3
        assert config.max_tokens == 300

    def test_custom_config(self):
        """Test configurazione personalizzata."""
        config = RAGConfig(
            top_k_results=5,
            max_tokens=500,
        )

        assert config.top_k_results == 5
        assert config.max_tokens == 500


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestIntegration:
    """Test di integrazione."""

    def test_full_pipeline(self, temp_games_file, mock_logger):
        """Test pipeline completo caricamento."""
        loader = GameDataLoader(temp_games_file, mock_logger)
        games = loader.load()
        texts, metadatas, ids = DocumentPreprocessor.prepare_documents(games)

        # Validazioni
        assert len(games) > 0
        assert len(texts) == len(games)
        assert len(metadatas) == len(games)
        assert len(ids) == len(games)

        # Ogni documento ha un ID
        for i, game_id in enumerate(ids):
            assert metadatas[i]["id"] == game_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
