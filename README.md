# 🎮 RAG Assistant per Accessibilità nei Videogiochi

Sistema RAG (Retrieval-Augmented Generation) per fornire consigli su videogiochi accessibili, con focus sul mercato italiano.

## 📦 Installazione Rapida (con GPU CUDA)

### Per GPU NVIDIA (consigliato)

```powershell
# Crea virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Installa PyTorch CUDA 12.1 e dipendenze
pip install -r requirements.txt

# Avvia
python .\rag_assistant.py
```

**Nota**: Se hai CUDA 11.8 o 12.4, modifica `requirements.txt` sostituendo `cu121` con `cu118` o `cu124`.

Verifica CUDA:
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Per CPU-only (più lento)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
python .\rag_assistant.py
```

## ✨ Caratteristiche Principali

- **Ricerca Semantica**: Trova i giochi più rilevanti usando embeddings multilingua
- **Persistenza Dati**: Database vettoriale ChromaDB con salvataggio su disco
- **Modelli Locali**: Funziona completamente offline (no API key necessarie)
- **Ottimizzato per Italiano**: Usa Mistral-7B e modelli specifici per italiano
- **Best Practices**: Codice modulare, type hints, logging strutturato
- **GPU Support**: Utilizza automaticamente CUDA se disponibile

## 📋 Requisiti

- Python 3.10 o superiore
- 8GB RAM minimo (16GB consigliato)
- GPU NVIDIA con CUDA (opzionale ma consigliato)
- ~20GB spazio disco (per modelli Mistral-7B)

## 📦 Perché pyproject.toml?

Questo progetto usa `pyproject.toml` invece di `requirements.txt` per diversi vantaggi:

### ✅ Vantaggi

1. **Standard Moderno**: È lo standard PEP 518/621 per progetti Python
2. **Tutto in un File**: Configurazione progetto, dipendenze, tool settings
3. **Dipendenze Opzionali**: Installa solo ciò che serve (`[dev]`, `[api]`, etc.)
4. **Metadati Ricchi**: Autori, licenza, keywords in formato strutturato
5. **Tool Configuration**: Black, Ruff, MyPy, Pytest in un posto solo
6. **Installabile**: `pip install -e .` rende il progetto un package Python
7. **Riproducibilità**: Lock file con Poetry/PDM per ambienti deterministici

### 📚 Gestori Compatibili

Puoi usare diversi tool per gestire `pyproject.toml`:

#### Opzione 1: pip (Standard, già installato)
```bash
pip install -e .
pip install -e ".[dev]"
```

#### Opzione 2: Poetry (Consigliato per progetti complessi)
```bash
# Installa Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Installa dipendenze
poetry install

# Con extra
poetry install --extras "dev api"

# Attiva virtual environment
poetry shell

# Esegui comandi
poetry run python main.py
```

#### Opzione 3: PDM (Moderno e veloce)
```bash
# Installa PDM
pip install pdm

# Installa dipendenze
pdm install

# Con extra
pdm install -G dev -G api

# Esegui comandi
pdm run python main.py
```

#### Opzione 4: Hatch (Nuovo standard)
```bash
# Installa Hatch
pip install hatch

# Crea environment e installa
hatch env create

# Esegui comandi
hatch run python main.py
```

### 🔄 Migrazione da requirements.txt

Se preferisci ancora `requirements.txt`, puoi generarlo da `pyproject.toml`:

```bash
# Con pip-tools
pip install pip-tools
pip-compile pyproject.toml -o requirements.txt

# Con Poetry
poetry export -f requirements.txt --output requirements.txt

# Manuale (solo dipendenze base)
pip install -e . && pip freeze > requirements.txt
```

## 🚀 Installazione

### 1. Clona o scarica il progetto

```bash
git clone <repository-url>
cd accessibility-assistant-rag
```

### 2. Crea un ambiente virtuale

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Installa il progetto con pip

#### Installazione Standard (Consigliata)

```bash
pip install --upgrade pip
pip install -e .
```

L'opzione `-e` (editable mode) permette di modificare il codice senza reinstallare.

#### Installazione con Dipendenze Opzionali

**Per sviluppo (con testing e linting):**
```bash
pip install -e ".[dev]"
```

**Per quantizzazione modelli (riduce uso RAM):**
```bash
pip install -e ".[quantization]"
```

**Per API REST:**
```bash
pip install -e ".[api]"
```

**Per interfaccia Web:**
```bash
pip install -e ".[ui]"
```

**Installazione completa (tutto insieme):**
```bash
pip install -e ".[all]"
```

### 4. Download automatico dei modelli

Al primo avvio, i modelli verranno scaricati automaticamente da Hugging Face:

- **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` (~420MB)
- **LLM**: `sapienzanlp/modello-italia-9b` (~18GB) o fallback a GPT-2

### 5. Configurazione (Opzionale)

Copia il file di esempio `.env.example` in `.env` e personalizza:

```bash
cp .env.example .env
```

Apri `.env` e modifica i parametri secondo le tue esigenze:

```bash
# Modelli da usare
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=sapienzanlp/modello-italia-9b

# Hardware
DEVICE=auto  # auto, cuda, cpu, mps
ENABLE_8BIT_QUANTIZATION=false

# Parametri ricerca
TOP_K_RESULTS=3
MAX_RESPONSE_TOKENS=512

# Logging
LOG_LEVEL=INFO
```

**Nota**: Se non crei il file `.env`, verranno usati i valori predefiniti.

### 6. Verifica installazione

```bash
# Verifica che il pacchetto sia installato
pip show accessibility-assistant-rag

# Avvia l'applicazione
python main.py
```

## 💡 Modelli Consigliati per l'Italiano

### Opzione 1: Modello Leggero (Predefinito)
```python
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
llm_model = "sapienzanlp/modello-italia-9b"  # Richiede ~18GB
```

### Opzione 2: Modelli Alternativi

**Per embedding migliori:**
```python
embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

**Per LLM più piccoli (se hai poca RAM):**
```python
llm_model = "GroNLP/gpt2-small-italian"  # ~500MB
```

**Per LLM più potenti (se hai GPU potente):**
```python
llm_model = "meta-llama/Llama-2-7b-chat-hf"  # Richiede account HF
# oppure
llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
```

### Opzione 3: Usare API Esterne (Consigliato per Produzione)

Per risultati migliori, considera di integrare:
- **OpenAI GPT-4** (migliore qualità)
- **Anthropic Claude** (ottimo per l'italiano)
- **Cohere** (buon rapporto qualità/prezzo)

## 🎯 Utilizzo

### Modalità Interattiva

```bash
python main.py
```

Esempio di interazione:
```
🎮 Domanda: Quali giochi sono accessibili per non vedenti?

🤖 Risposta: 
Ci sono diversi giochi eccellenti per non vedenti:

1. **The Last of Us Part II** (10/10) - Supporto nativo completo con:
   - Navigation Assist con scansione audio
   - Text-to-Speech per menu
   - High Contrast Mode
   
2. **A Blind Legend** (10/10) - Progettato specificamente per non vedenti:
   - Audio binaurale 3D
   - Nessuna grafica necessaria
   
3. **Forza Motorsport 2023** (10/10) - Primo simulatore accessibile:
   - Blind Driving Assist
   - Audio cues per pista e curve

📚 Fonti consultate:
  • The Last of Us Part II (Remastered) (Categoria: Action-Adventure, Score: 10.0/10)
  • A Blind Legend (Categoria: Audio Game, Score: 10.0/10)
  • Forza Motorsport (2023) (Categoria: Racing Simulation, Score: 10.0/10)
```

### Uso Programmatico

```python
from main import AccessibilityAssistant, Config, GameDataLoader

# Carica configurazione e dati
config = Config()
loader = GameDataLoader(config.json_file)
games_data = loader.load()

# Inizializza assistente
assistant = AccessibilityAssistant(config)
assistant.setup(games_data)

# Fai una domanda
result = assistant.query("Quali giochi hanno buon supporto motorio?")
print(result['answer'])
```

## ⚙️ Configurazione Avanzata

### Usando il file .env

Il progetto supporta configurazione tramite variabili d'ambiente. Crea un file `.env` nella root:

```bash
# Copia il template
cp .env.example .env

# Modifica con il tuo editor
nano .env  # oppure vim, code, etc.
```

**Variabili principali:**

| Variabile | Descrizione | Predefinito |
|-----------|-------------|-------------|
| `EMBEDDING_MODEL` | Modello per embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| `LLM_MODEL` | Modello per generazione risposte | `sapienzanlp/modello-italia-9b` |
| `DEVICE` | Hardware (auto/cuda/cpu/mps) | `auto` |
| `TOP_K_RESULTS` | Documenti da recuperare | `3` |
| `MAX_RESPONSE_TOKENS` | Token max risposta | `512` |
| `LOG_LEVEL` | Livello logging | `INFO` |
| `ENABLE_8BIT_QUANTIZATION` | Quantizzazione 8-bit | `false` |

### Modifica della classe Config

Modifica la classe `Config` in `main.py`:

```python
@dataclass
class Config:
    json_file: Path = field(
        default_factory=lambda: Path(os.getenv("GAMES_JSON_PATH", "games.json"))
    )
    
    # Modello embedding (multilingual per italiano)
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", 
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    
    # Altri parametri...
```

### Workflow con Poetry (Consigliato)

Poetry semplifica la gestione delle dipendenze e degli ambienti virtuali:

```bash
# Installa Poetry (se non l'hai già)
curl -sSL https://install.python-poetry.org | python3 -

# Installa dipendenze dal pyproject.toml
poetry install

# Installa con extra opzionali
poetry install --extras "dev api ui"

# Attiva l'ambiente virtuale
poetry shell

# Oppure esegui comandi senza attivare
poetry run python main.py

# Aggiungi nuove dipendenze
poetry add nome-pacchetto

# Aggiungi dipendenze di sviluppo
poetry add --group dev pytest black

# Aggiorna dipendenze
poetry update

# Esporta requirements.txt (per compatibilità)
poetry export -f requirements.txt --output requirements.txt

# Build del pacchetto
poetry build

# Pubblica su PyPI (se pubblico)
poetry publish
```

### Workflow con PDM (Alternativa moderna)

```bash
# Installa PDM
pip install pdm

# Installa dipendenze
pdm install

# Con gruppi opzionali
pdm install -G dev -G api

# Aggiungi pacchetti
pdm add requests
pdm add -G dev pytest

# Esegui comandi
pdm run python main.py

# Lock file
pdm lock

# Build
pdm build
```

## 🔧 Risoluzione Problemi

### Errore: "Out of Memory"

**Soluzione 1**: Usa un modello più piccolo
```python
llm_model = "gpt2"  # Molto leggero ma qualità inferiore
```

**Soluzione 2**: Quantizzazione 8-bit (richiede GPU)
```bash
pip install bitsandbytes
```

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

### Modello non si scarica

Verifica la connessione e lo spazio disco:
```bash
# Test download manuale
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

### Risposte di bassa qualità

1. Aumenta `top_k_results` a 5
2. Usa un modello LLM più grande
3. Considera l'uso di API esterne (GPT-4, Claude)

## 📊 Performance

| Componente | Tempo Primo Avvio | Tempo Query | RAM Usata |
|------------|-------------------|-------------|-----------|
| Embedding | 30-60s | 0.1-0.3s | 1-2GB |
| Indexing | 5-10s | - | 500MB |
| LLM (9B) | 60-120s | 3-10s | 18GB |
| LLM (GPT-2) | 10-20s | 1-3s | 500MB |

*Tempi su CPU Intel i7, GPU RTX 3060*

## 🎨 Estensioni Possibili

- **Web UI**: Aggiungi Streamlit o Gradio
- **API REST**: Usa FastAPI
- **Multi-tenancy**: Collezioni separate per utente
- **Feedback Loop**: Sistema di rating risposte
- **Cache**: Redis per query frequenti
- **Monitoring**: Prometheus + Grafana

## 📝 Struttura Progetto

```
accessibility-assistant-rag/
├── 📄 pyproject.toml       # ⭐ Configurazione progetto (dipendenze, metadata, tool config)
├── 🐍 main.py              # ⭐ Applicazione principale RAG
├── 🎮 games.json           # ⭐ Database giochi (dati da indicizzare)
│
├── 📋 README.md            # Documentazione completa
├── 🚀 QUICKSTART.md        # Guida rapida per iniziare
├── 🔨 Makefile             # Comandi utili semplificati
│
├── 🔐 .env                 # Configurazione runtime (da creare)
├── 📝 .env.example         # Template configurazione
├── 🚫 .gitignore           # File da ignorare in git
│
├── 📁 chroma_db/           # Vector store ChromaDB (auto-generato)
│   ├── chroma.sqlite3
│   └── ...
│
├── 📁 tests/               # Test unitari (opzionale)
│   ├── __init__.py
│   ├── test_loader.py
│   ├── test_preprocessor.py
│   └── test_rag.py
│
└── 📁 docs/                # Documentazione generata (opzionale)
    └── ...
```

### File Principali

| File | Scopo | Modificabile |
|------|-------|--------------|
| `main.py` | Codice applicazione RAG | ✅ Sì |
| `games.json` | Database giochi | ✅ Sì (aggiungi giochi) |
| `pyproject.toml` | Dipendenze e config | ✅ Sì (aggiungi deps) |
| `.env` | Config runtime | ✅ Sì (personalizza) |
| `Makefile` | Automazione comandi | ⚠️ Opzionale |
| `chroma_db/` | Cache vettoriale | ❌ No (auto-generato) |

## 🔨 Comandi Utili (Makefile)

Il progetto include un `Makefile` per semplificare i comandi comuni:

### Setup e Installazione
```bash
make quickstart       # Setup completo per nuovi utenti
make install          # Installa solo dipendenze base
make install-dev      # Installa con tool di sviluppo
make install-all      # Installa tutto
```

### Esecuzione
```bash
make run              # Avvia l'applicazione
make run-dev          # Avvia in modalità debug
```

### Sviluppo
```bash
make test             # Esegui test
make lint             # Controlla codice con ruff
make format           # Formatta con black
make check            # Type checking con mypy
make quality          # Esegui tutti i check
```

### Manutenzione
```bash
make clean            # Pulisci file temporanei
make clean-cache      # Pulisci cache modelli HuggingFace
make clean-db         # Pulisci database vettoriale
make clean-all        # Pulizia completa
```

### Utility
```bash
make check-system     # Verifica sistema e dipendenze
make check-data       # Verifica games.json
make setup-env        # Crea .env da template
make build            # Build package per distribuzione
```

**Esempio workflow completo:**
```bash
# Setup iniziale
make quickstart

# Modifica .env se necessario
nano .env

# Avvia
make run

# Durante sviluppo
make format lint test
```

## 🤝 Contribuire

Per aggiungere nuovi giochi, modifica `games.json` seguendo il formato esistente:

```json
{
  "id": "GAME_XXX",
  "name": "Nome Gioco",
  "access_level": 8.5,
  "is_native": true,
  "category": "Categoria",
  "platforms": ["PC", "Console"],
  "description": "Descrizione dettagliata...",
  "features": ["Feature 1", "Feature 2"],
  "accessibility_details": {
    "visual": "Dettagli accessibilità visiva",
    "motor": "Dettagli accessibilità motoria",
    "auditory": "Dettagli accessibilità uditiva",
    "cognitive": "Dettagli accessibilità cognitiva"
  }
}
```

## 📄 Licenza

MIT License - vedi file LICENSE per dettagli

## 🆘 Supporto

Per problemi o domande:
- Apri una issue su GitHub
- Consulta la documentazione ChromaDB: https://docs.trychroma.com/
- Documentazione Sentence Transformers: https://www.sbert.net/

---

**Sviluppato con ❤️ per rendere il gaming più accessibile a tutti**
