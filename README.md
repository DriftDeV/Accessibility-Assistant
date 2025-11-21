#  Assistente RAG per l'Accessibilità nei Videogiochi

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ollama](https://img.shields.io/badge/Ollama-Ready-brightgreen)](https://ollama.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Introduzione

**Accessibility Assistant** è un sistema di **Retrieval-Augmented Generation (RAG)** progettato per rispondere a domande sull'accessibilità nei videogiochi. 

Utilizza:
- 🤖 **Ollama** per embeddings e generazione di testo (esecuzione 100% locale)
- 🗂️ **ChromaDB** come vector store per la ricerca semantica
- 📋 **games.json** come base di conoscenza strutturata

Il sistema è capace di:
- 🎮 Trovare giochi accessibili per specifiche disabilità (visiva, motoria, uditiva, cognitiva)
- 📖 Fornire dettagli su feature di accessibilità implementate
- 💡 Suggerire alternative accessibili
- 🌐 Rispondere in italiano con fonti documentate

### Esempio di utilizzo

```
 Quali giochi sono accessibili per daltonici?

 Risposta:
Forza Motorsport offre filtri avanzati per daltonismo (Tritanopia, Protanopia, 
Deuteranopia). The Last of Us Part II include una modalità Alto Contrasto che 
aiuta a distinguere nemici e alleati...

 Fonti:
  1. **Forza Motorsport**  Racing Simulation  Nativo | Accessibilità: 10/10
  2. **The Last of Us Part II**  Action-Adventure  Nativo | Accessibilità: 10/10
```

---

## ✅ Pre-requisiti

Prima di installare, assicurati di avere:

| Requisito | Versione | Descrizione |
|-----------|----------|------------|
| 🐍 **Python** | 3.11+ | Linguaggio di programmazione |
| 🤖 **Ollama** | Ultima | Runtime per modelli LLM locali |
| 🔗 **Git** | Qualsiasi | Per clonare il repository |

### Modelli Ollama richiesti

```bash
ollama pull nomic-embed-text    # ~274MB - Embeddings
ollama pull llama3:8b           # ~4.1GB - Generazione testo
```

---

##  Guida all'installazione

### 1️⃣ Installa Ollama

**🍎 macOS:**
```bash
brew install ollama
```

**🐧 Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ollama
```

**🪟 Windows:**
Scarica il file `.exe` da [ollama.ai](https://ollama.ai/download)

### 2️⃣ Scarica i modelli Ollama

```bash
ollama pull nomic-embed-text    # Embedding model
ollama pull llama3:8b           # LLM model per generazione
```

Verifica che i modelli siano stati scaricati:
```bash
ollama list
```

### 3️⃣ Clona il repository

```bash
git clone https://github.com/DriftDeV/Accessibility-Assistant.git
cd Accessibility-Assistant
```

### 4️⃣ Setup dell'ambiente Python

**Crea un virtual environment (consigliato):**

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Installa le dipendenze:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dipendenze opzionali per lo sviluppo:**

```bash
pip install -r requirements-dev.txt
```

### 5️⃣ (Opzionale) Configura variabili d'ambiente

Crea un file `.env` nella radice del progetto per personalizzare la configurazione:

```bash
# Copia il file di esempio
cp .env.example .env
```

Oppure crea manualmente `.env` con:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3:8b

# Database
GAMES_FILE=games.json
CHROMA_DB_DIR=./chroma_db

# RAG Parameters
TOP_K_RESULTS=3
MAX_TOKENS=300

# Logging
LOG_LEVEL=INFO
```

---

##  Esecuzione

### 💻 Modalità Interattiva (CLI)

**Terminale 1️⃣ - Avvia Ollama:**
```bash
ollama serve
```

**Terminale 2️⃣ - Avvia l'assistente:**
```bash
# Attiva virtual environment (se non già attivato)
source venv/bin/activate  # Linux/macOS
# oppure
.\venv\Scripts\Activate.ps1  # Windows

# Esegui il programma
python accessibility_rag.py
```

Digita le tue domande e premi Enter. Digita `esci` per terminare.

### 🌐 Interfaccia Web (Gradio)

**Terminale 1️⃣ - Avvia Ollama:**
```bash
ollama serve
```

**Terminale 2️⃣ - Avvia l'interfaccia Gradio:**
```bash
source venv/bin/activate  # Attiva venv
python app.py
```

Accedi a: **http://127.0.0.1:7860** 🎨

L'interfaccia offre:
- 💬 Chat interattiva in tempo reale
- 📚 Visualizzazione delle fonti con metadata
- 🔄 Pulsante per aggiornare lo stato del sistema
- 🗑️ Pulsante per cancellare la cronologia

### 🐍 Utilizzo Programmativo

```python
from accessibility_rag import AccessibilityAssistant, RAGConfig

# Carica configurazione da variabili d'ambiente
config = RAGConfig.from_env()

# Crea l'assistente
assistant = AccessibilityAssistant(config)

# Setup iniziale (carica il database)
assistant.setup()

# Effettua una query
result = assistant.query("Quali giochi sono accessibili per ipovedenti?")

print("Risposta:", result["answer"])
print("\nFonti:")
for source in result["sources"]:
    print(f"  - {source['name']} ({source['category']})")
```

---

##  Contribuzione

Le contribuzioni sono benvenute! Puoi aiutarci in vari modi:

### 🐛 Segnalazione di Bug

Se trovi un bug, apri una [GitHub Issue](https://github.com/DriftDeV/Accessibility-Assistant/issues) con:
- 📝 Descrizione del problema
- 📋 Passi per riprodurlo
- 🔴 Output di errore
- 🔧 Versione di Python e Ollama

### 💡 Richieste di Feature

Suggerisci nuove feature aprendo una [Discussion](https://github.com/DriftDeV/Accessibility-Assistant/discussions) con:
- 📝 Descrizione della feature
- ❓ Motivazione (perché è utile?)
- 🎯 Possibili implementazioni

### 🔀 Pull Request

1. 🍴 **Fork** il repository
2. 🌿 **Crea un branch** con nome descrittivo: `git checkout -b feature/nome-feature`
3. 💾 **Fai i commit** con messaggi chiari: `git commit -m "feat: descrizione"`
4. ✅ **Esegui i test**: `pytest`
5. 📤 **Fai push** del branch: `git push origin feature/nome-feature`
6. 📬 **Apri una Pull Request** su GitHub

**Standard di codice:**
- 🎨 Code style: [Black](https://black.readthedocs.io/)
- 🔍 Linting: [Pylint](https://pylint.pycqa.org/)
- 📌 Type hints: completi
- 📚 Docstring: formato Google

### 📊 Aggiornamento del Database JSON

Il database dei giochi è conservato in `games.json`. Per aggiungere nuovi giochi o aggiornarne i dati:

#### 📋 Struttura di un elemento

```json
{
  "id": "GAME_XXX",
  "name": "Nome del Gioco",
  "access_level": 8.5,
  "is_native": true,
  "category": "Genere",
  "platforms": ["PC", "PlayStation 5"],
  "description": "Descrizione dettagliata dell'accessibilità...",
  "features": [
    "Feature 1",
    "Feature 2 [cite: 45]"
  ],
  "accessibility_details": {
    "visual": "Dettagli per disabilità visiva...",
    "motor": "Dettagli per disabilità motoria...",
    "auditory": "Dettagli per disabilità uditiva...",
    "cognitive": "Dettagli per disabilità cognitiva..."
  },
  "source_ref": "Fonte della documentazione"
}
```

#### ⭐ Campi obbligatori

| Campo | Tipo | Descrizione |
|-------|------|------------|
| 🆔 `id` | String | ID univoco (es. `GAME_001`) |
| 🎮 `name` | String | Nome completo del gioco |
| 📊 `access_level` | Float (0-10) | Score di accessibilità |
| ✅ `is_native` | Boolean | Feature nativa vs mod/config |
| 🏷️ `category` | String | Genere del gioco |
| 🖥️ `platforms` | Array | Piattaforme disponibili |
| 📝 `description` | String | Descrizione dell'accessibilità |

#### 📌 Campi opzionali

| Campo | Tipo | Descrizione |
|-------|------|------------|
| ✨ `features` | Array | Liste di feature di accessibilità |
| ♿ `accessibility_details` | Object | Dettagli per tipo di disabilità |
| 📖 `source_ref` | String | Riferimento alla fonte |

#### 📝 Passi per aggiungere un gioco

1. 📂 **Apri** `games.json`
2. ➕ **Aggiungi un nuovo oggetto** alla fine dell'array con la struttura sopra
3. ✔️ **Assicurati che il JSON sia valido** (usa un validatore online se necessario)
4. 💾 **Salva il file**
5. 🧪 **Testa** con una query per verificare che sia indicizzato correttamente:
   ```bash
   python accessibility_rag.py
   ❓ Domanda: [Nome del nuovo gioco]
   ```

#### 💾 Esempio di aggiunta

```json
{
  "id": "GAME_XXX",
  "name": "Stardew Valley",
  "access_level": 7.0,
  "is_native": true,
  "category": "Simulation, RPG",
  "platforms": ["PC", "Nintendo Switch", "PlayStation", "Xbox"],
  "description": "Un simulatore agricolo con ottime opzioni di accessibilità...",
  "features": [
    "Controlli completamente rimappabili",
    "Testo ridimensionabile",
    "Supporto screen reader",
    "Opzioni di difficoltà personalizzabili"
  ],
  "accessibility_details": {
    "visual": "Testo grande, alte contrast mode opzionale",
    "motor": "Nessun input rapido obbligatorio, rimappabilità totale",
    "cognitive": "Pacing controllato dal giocatore, nessuna pressione temporale"
  },
  "source_ref": "Verified da comunità accessibilità"
}
```

#### 🔄 Aggiornamento dell'indice

Dopo aver aggiunto giochi a `games.json`, l'indice ChromaDB verrà rigenerato automaticamente al prossimo avvio:

```bash
# Elimina il database vecchio (opzionale)
rm -rf chroma_db

# Riavvia l'assistente
python app.py
# oppure
python accessibility_rag.py
```

---

##  Documentazione Aggiuntiva

- 🏗️ **[ARCHITECTURE.md](ARCHITECTURE.md)** - Dettagli tecnici e design del sistema
- 📖 **[CHANGELOG.md](CHANGELOG.md)** - Cronologia degli aggiornamenti
- 📋 **[SUMMARY.md](SUMMARY.md)** - Riepilogo delle feature

---

##  Licenza

Questo progetto è rilasciato sotto licenza **MIT**. Vedi [LICENSE](LICENSE) per i dettagli. 📜

---

##  Credits

Progetto sviluppato con ❤️ per rendere i videogiochi più accessibili a tutti.

**Tecnologie utilizzate:**
- 🤖 [Ollama](https://ollama.ai) - LLM locale
- 🗂️ [ChromaDB](https://www.trychroma.com/) - Vector store
- 🎨 [Gradio](https://gradio.app/) - Interfaccia web
- 🐍 [Python 3.11+](https://www.python.org/)

---

## 🚧 Roadmap & TODO

Funzionalità pianificate per le prossime versioni:

- [ ] 🌐 **Web UI Avanzata** - Migliore interfaccia Gradio con dark mode e filtri avanzati
- [ ] ♿ **Accessibilità UI** - WCAG 2.1 AA compliance per l'interfaccia stessa
- [ ] 💡 **Consigli per Developer** - Suggerimenti su come implementare l'accessibilità nei videogiochi
- [ ] 🌍 **Multi-Language Support** - Supporto per altre lingue oltre l'italiano
- [ ] 🎯 **Fine-tuning Custom** - Fine-tuning su dataset specifici dell'utente
- [ ] 📊 **Analytics Dashboard** - Statistiche sull'utilizzo e query più comuni
- [ ] 🔗 **API REST** - Esporre il sistema via API per integrazioni terze
- [ ] 🧪 **Test Expansion** - Aumentare la copertura dei test da X% a 90%+
- [ ] 📱 **Mobile App** - Applicazione mobile React Native
- [ ] 🗣️ **Voice Chat** - Supporto per input/output vocale

---

## 📞 Supporto & Contatti

Hai domande o dubbi? 🤔

- 💬 Apri una [GitHub Discussion](https://github.com/DriftDeV/Accessibility-Assistant/discussions) per domande generali
- 🐛 Segnala un bug tramite [GitHub Issues](https://github.com/DriftDeV/Accessibility-Assistant/issues) per problemi
- 📧 Contatta il team di sviluppo per collaborazioni

---

**Ultimo aggiornamento:** Novembre 2025 ✨
