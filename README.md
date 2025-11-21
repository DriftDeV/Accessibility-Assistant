# Assistente RAG per l'Accessibilità nei Videogiochi

**RAG (Retrieval-Augmented Generation)** basato su Ollama per rispondere a domande sull'accessibilità nei videogiochi.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ollama](https://img.shields.io/badge/Ollama-Ready-brightgreen)](https://ollama.ai)

## 📋 Prerequisiti

- **Python** 3.10 o superiore
- **Ollama** con modelli scaricati:
  - `nomic-embed-text` (embeddings)
  - `mistral` (generazione)

## 🚀 Installation

### 1. Installa Ollama

```bash
# Visita https://ollama.ai e scarica
# Oppure con package manager:
brew install ollama          # macOS
sudo apt install ollama      # Ubuntu
# Windows: https://ollama.ai/download
```

### 2. Scarica i modelli

```bash
ollama pull nomic-embed-text  # ~274MB
ollama pull llama3:8b           # ~4.1GB
```

### 3. Clona il repository

```bash
git clone https://github.com/DriftDeV/Accessibility-Assistant.git
cd Accessibility-Assistant
```

### 4. Setup Python

```bash
# Crea virtual environment (consigliato)
python -m venv venv

# Attiva
source venv/bin/activate        # Linux/macOS
# oppure
venv\Scripts\activate           # Windows PowerShell

# Installa dipendenze
pip install -r requirements.txt

# (Opzionale) Installa dev tools
pip install -r requirements-dev.txt
```

### 5. (Opzionale) Configura `.env`

```bash
cp .env.example .env
# Modifica i valori secondo necessità
```

## 🎮 Utilizzo

### Modalità Interattiva

```bash
# Avvia Ollama (in un terminale separato)
ollama serve

# In un altro terminale:
python accessibility_rag.py
```

```
❓ Domanda: Quali giochi sono accessibili per daltonici?

💬 Risposta:
Forza Motorsport offre filtri avanzati per daltonismo (Tritanopia, Protanopia, 
Deuteranopia). Anche The Last of Us Part II ha modalità Alto Contrasto che aiuta...

📚 Fonti:
  • Forza Motorsport (Racing Simulation) - Accessibilità: 10/10 [✓ Nativo]
  • The Last of Us Part II (Action-Adventure) - Accessibilità: 10/10 [✓ Nativo]
```

### Utilizzo Programmativo

```python
from accessibility_rag import AccessibilityAssistant

# Setup
assistant = AccessibilityAssistant()
assistant.setup()

# Query
result = assistant.query("Quali giochi hanno Text-to-Speech?")
print(result["answer"])

# Con fonti
for source in result["sources"]:
    print(f"📖 {source['name']}: {source['score']}/10")
```

### Configurazione Custom

```python
from accessibility_rag import AccessibilityAssistant, RAGConfig
from pathlib import Path

config = RAGConfig(
    games_file=Path("games.json"),
    top_k_results=5,        # Più risultati
    max_tokens=500,         # Risposte più lunghe
    ollama_base_url="http://localhost:11434"
)

assistant = AccessibilityAssistant(config)
assistant.setup()
```

## 📁 Struttura

```
Accessibility-Assistant/
├── accessibility_rag.py      # 🎯 Main module
├── config.py                 # ⚙️  Configurazione
├── games.json                # 📚 Database giochi (213 entries)
├── ARCHITECTURE.md           # 📖 Documentazione tecnica
├── requirements.txt          # 📦 Dipendenze
├── .env.example              # 🔧 Variabili d'ambiente
├── tests/
│   └── test_accessibility_rag.py  # ✅ Unit tests
└── chroma_db/                # 💾 Vector store (creato al setup)
```

## 🔍 Architettura

```
User Query
    ↓
VectorStore (ChromaDB)
    ├─ Ricerca semantica
    └─ Top 3 risultati
    ↓
Context Builder
    ├─ Formatta documenti
    └─ Limita lunghezza
    ↓
ResponseGenerator (Ollama Mistral)
    └─ Genera risposta
    ↓
Output con Fonti
```

Dettagli completi in [ARCHITECTURE.md](./ARCHITECTURE.md)

## ⚙️ Configurazione

Variabili d'ambiente (`.env`):

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=mistral

# Percorsi
GAMES_FILE=games.json
CHROMA_DB_DIR=./chroma_db

# RAG
TOP_K_RESULTS=3
MAX_TOKENS=300

# Logging
LOG_LEVEL=INFO
```

## 🧪 Test

```bash
# Esegui tutti i test
pytest tests/ -v

# Con coverage
pytest tests/ --cov=. --cov-report=html

# Test specifico
pytest tests/test_accessibility_rag.py::TestGameDataLoader -v
```

## 🐛 Troubleshooting

### Ollama non disponibile

```
❌ ConnectionError: Cannot connect to localhost:11434
```

**Soluzione:**
```bash
# Verifica Ollama in esecuzione
ollama serve

# Oppure verifica URL
curl http://localhost:11434/api/tags
```

### Modelli non trovati

```
❌ Model not found: mistral
```

**Soluzione:**
```bash
ollama pull mistral
ollama pull nomic-embed-text
ollama list  # Verifica
```

### ChromaDB corrotto

```bash
rm -rf ./chroma_db
python accessibility_rag.py  # Ricrea
```

## 📊 Performance

| Operazione | Tempo |
|------------|-------|
| Setup | ~2 sec |
| Ricerca | ~100 ms |
| Generazione | ~2-3 sec |
| **Total Q&A** | ~2.5-3.5 sec |

## 📚 Documentazione

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Dettagli tecnici completi
- **[Docstrings](./accessibility_rag.py)** - Documentazione inline (Google style)

## 🛠️ Best Practice Implementate

✅ Type hints completi (`PEP 484`)  
✅ Docstring esaustivi (Google style)  
✅ Separation of Concerns  
✅ Logging strutturato  
✅ Configurazione centralizzata  
✅ Error handling robusto  
✅ Test unit comprensivi  
✅ Nessun magic number  
✅ Validazione input  
✅ Nessuna dipendenza pesante  

## 🚀 Estensioni Future

- [x] **Web UI** (Gradio/Streamlit)
- [ ] **Ui Accessibile** Accessibilità su Ui
- [ ] **Consigli per devs** Consigli per devs su come implementare l'accessibilià nei videogiochi che vorrebbero creare
- [ ] **Multi-language support** Supporto Multilingua
- [ ] **Fine-tuning su dataset custom**

## 📝 Licenza

MIT

## 👤 Autore

[Luigi Santini](https://github.com/DriftDeV)

## 🤝 Contributi

Contributi sono benvenuti! Per favore:

1. Fork il repository
2. Crea un branch (`git checkout -b feature/AmazingFeature`)
3. Commit i cambiamenti (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## ⭐ Se ti piace, metti una stella!

---

**Made with ❤️ per l'accessibilità nei videogiochi**
