# 🎮 Gradio Chat UI - Accessibility Assistant

Interfaccia web moderna per il sistema RAG di accessibilità nei videogiochi, basata su **Ollama** e **ChromaDB**.

## 🚀 Quick Start

### 1. Avvia Ollama

```bash
ollama serve
```

Ollama ascolterà su `http://localhost:11434`

### 2. Avvia la Chat UI

In un altro terminale:

```bash
python gradio_ui.py
```

Accedi a **http://localhost:7860**

## ⚙️ Configurazione

### Variabili d'Ambiente

Crea un file `.env` basato su `.env.example`:

```bash
cp .env.example .env
```

Quindi modifica i valori secondo necessità:

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=mistral

# UI Gradio
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860
GRADIO_SHARE=0

# Percorsi
GAMES_FILE=games.json
CHROMA_DB_DIR=./chroma_db

# RAG
TOP_K_RESULTS=3
MAX_TOKENS=300

# Logging
LOG_LEVEL=INFO
```

### Opzioni Gradio

| Variabile | Default | Descrizione |
|-----------|---------|-----------|
| `GRADIO_HOST` | `127.0.0.1` | Host dove ascolta Gradio |
| `GRADIO_PORT` | `7860` | Porta Gradio |
| `GRADIO_SHARE` | `0` | Genera link pubblico Gradio (`1` per abilitare) |

**Esempio: Accesso da rete locale**

```bash
GRADIO_HOST=0.0.0.0 python gradio_ui.py
```

Poi accedi da un altro PC a `http://<your-ip>:7860`

**Esempio: Condividi pubblicamente**

```bash
GRADIO_SHARE=1 python gradio_ui.py
```

Gradio genererà un link pubblico temporaneo

## 📋 Funzionalità

✅ **Chat interattiva** - Poni domande sull'accessibilità
✅ **Fonti visibili** - Vedi i giochi da cui la risposta è stata derivata
✅ **Indicatore di stato** - Sai quando il sistema è pronto
✅ **Pulsante Aggiorna** - Controlla lo stato manualmente
✅ **Cancella chat** - Pulisci la conversazione
✅ **Copy button** - Copia risposte facilmente
✅ **Responsive** - Funziona su desktop e mobile

## 🏗️ Architettura

```
┌─────────────────────────────────┐
│   Gradio Web Interface (UI)     │
│  - Chat Chatbot                 │
│  - Text Input                   │
│  - Status Indicator             │
│  - Control Buttons              │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   respond() Function            │
│  - Valida input                 │
│  - Controlla stato              │
│  - Chiama AssistantQuery        │
│  - Formatta risposta+fonti      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  AccessibilityAssistant (RAG)   │
│  - Vector Search (ChromaDB)     │
│  - Response Generation (Ollama) │
│  - Source Tracking             │
└────────────┬────────────────────┘
             │
         ┌───┴────┐
         ▼        ▼
    ChromaDB    Ollama
    (Vector)    (LLM)
```

## 🔄 Flusso di Elaborazione

### Setup Iniziale

1. **Caricamento Config** → Legge `.env` e `config.py`
2. **Thread di Background** → Avvia `initialize_assistant_in_thread()`
   - Carica `games.json`
   - Inizializza ChromaDB
   - Prepara embeddings
   - Setta `READY_EVENT` quando completo
3. **UI Pronta** → Gradio permette input, UI mostra stato

### Query Processing

1. **Utente scrive domanda** → `txt` field in Gradio
2. **Click "Invia" o Enter** → Chiama `respond(message, chat_history)`
3. **Validazione** → Check messaggio vuoto e stato ready
4. **RAG Processing**:
   - `assistant.query(message)` → Ricerca semantica
   - `VectorStore.search()` → ChromaDB ritorna top-k
   - `ResponseGenerator.generate()` → Ollama genera risposta
5. **Formatting** → `format_sources()` crea sezione fonti
6. **UI Update** → Chatbot mostra (user, assistant) + fonti

## 📊 Performance

| Operazione | Tempo |
|------------|-------|
| Setup iniziale | ~2-5 sec |
| Query ricerca | ~100 ms |
| Query LLM | ~2-3 sec |
| **Total Q&A** | ~2.5-3.5 sec |

## 🐛 Troubleshooting

### "Connection refused" a Ollama

```
❌ Error connecting to Ollama at http://localhost:11434
```

**Soluzione:**
```bash
# Verifica Ollama in esecuzione
ollama serve

# Verifica modelli
ollama list

# Se assente, scarica:
ollama pull mistral
ollama pull nomic-embed-text
```

### "Module not found: gradio"

```bash
pip install gradio>=3.40
```

### Gradio non accetta connessioni da rete locale

```bash
# Cambia host da 127.0.0.1 a 0.0.0.0
GRADIO_HOST=0.0.0.0 python gradio_ui.py
```

### ChromaDB corrotto

```bash
rm -rf ./chroma_db
python gradio_ui.py  # Ricrea il DB
```

### Lento al primo avvio

È normale! La prima volta:
- Carica `games.json` (213 giochi)
- Crea embeddings (usa Ollama)
- Salva in ChromaDB

Successivamente sarà veloce (cache).

## 📝 Esempi di Utilizzo

### Domande Sull'Accessibilità

Prova questi esempi nella UI:

1. **"Quali giochi sono accessibili ai non vedenti?"**
2. **"Cosa significa High Contrast Mode?"**
3. **"Quali giochi hanno il Text-to-Speech?"**
4. **"Quali piattaforme hanno più giochi accessibili?"**
5. **"Come funziona la Navigation Assist?"**

## 🔒 Note di Sicurezza

- **Locale per default**: Ascolta solo su `127.0.0.1`
- **Share temporaneo**: Se abiliti `GRADIO_SHARE=1`, il link scade in ~72 ore
- **No authentication**: Nessuna autenticazione by default
- **No data logging**: Nessun salvataggio automatico conversazioni

## 🎨 Personalizzazione

### Cambiare il tema

Nel file `gradio_ui.py`, modifica:

```python
theme=gr.themes.Soft()  # Cambia questo
```

Opzioni: `Soft()`, `Default()`, `Base()`, `Glass()`, `Monochrome()`

### Aggiungere i tuoi CSS

```python
with gr.Blocks(css="custom.css", theme=...) as demo:
    ...
```

### Modificare il layout

Nel file `build_ui()` puoi riorganizzare gli elementi Gradio con `gr.Row()` e `gr.Column()`

## 📚 Risorse

- [Gradio Documentation](https://www.gradio.app/)
- [Ollama](https://ollama.ai)
- [ChromaDB](https://docs.trychroma.com/)
- [RAG Pattern](https://aws.amazon.com/blogs/machine-learning/question-answering-using-retrieval-augmented-generation-with-foundation-models/)

---

**Made with ❤️ for gaming accessibility**
