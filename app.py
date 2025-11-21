
"""
Interfaccia Gradio per l'Assistente RAG Accessibilità.

Questo modulo fornisce una chat UI web per interrogare il sistema RAG
basato su Ollama per rispondere a domande sull'accessibilità nei videogiochi.

Caratteristiche:
- ✅ Setup asincrono in background (non blocca la UI)
- ✅ Chat interfaccia semplice e intuitiva
- ✅ Visualizzazione delle fonti con metadata
- ✅ Gestione errori robusto
- ✅ Logging strutturato
- ✅ Type hints completi
- ✅ Configurazione via .env

Dipendenze:
- gradio>=3.40
- chromadb>=0.4.0
- ollama>=0.0.11
- python-dotenv>=1.0.0

Utilizzo:
    python gradio_ui.py
    # Accedi a http://localhost:7860

Note:
- Ollama deve essere in esecuzione: `ollama serve`
- I modelli devono essere scaricati: `ollama pull mistral nomic-embed-text`
- Configurazione personalizzata: copia .env.example in .env e modifica i valori
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from accessibility_rag import AccessibilityAssistant, RAGConfig, setup_logging

# Logger configurato
LOGGER = logging.getLogger(__name__)

# Riferimenti a livello di modulo (evita deepcopy issues con Gradio State)
ASSISTANT: Optional[AccessibilityAssistant] = None
READY_EVENT: Optional[threading.Event] = None


def initialize_assistant_in_thread(
	cfg: RAGConfig,
) -> Tuple[AccessibilityAssistant, threading.Event]:
	"""Inizializza l'assistente in un thread separato.

	Crea l'istanza di AccessibilityAssistant e la inizializza in background
	per non bloccare l'UI di Gradio. Restituisce una tupla (assistant, ready_event)
	dove ready_event viene settato quando l'inizializzazione è completata.

	Args:
		cfg: RAGConfig con la configurazione

	Returns:
		Tupla di (assistant_instance, ready_threading_event)
	"""
	rag_config = cfg

	assistant = AccessibilityAssistant(rag_config)
	ready = threading.Event()

	def _setup() -> None:
		"""Funzione interna per il setup in thread."""
		try:
			LOGGER.info("🚀 Inizializzazione dell'assistente in background...")
			assistant.setup()
			LOGGER.info("✅ Setup completato!")
		except Exception as exc:
			LOGGER.exception("❌ Errore durante il setup: %s", exc)
		finally:
			# Setta l'evento indipendentemente dal successo/fallimento
			# per evitare blocchi infiniti
			ready.set()

	thread = threading.Thread(target=_setup, daemon=True)
	thread.start()
	return assistant, ready


def format_sources(sources: List[Dict[str, Any]]) -> str:
	"""Formatta le fonti in una stringa markdown per la chat.

	Crea una sezione "Fonti" ben formattata con i metadati disponibili
	come nome gioco, categoria, score di accessibilità, ecc.

	Args:
		sources: Lista di dizionari con i dati delle fonti

	Returns:
		Stringa markdown formattata
	"""
	if not sources:
		return ""

	lines = ["\n\n### 📚 Fonti"]

	for i, source in enumerate(sources, 1):
		name = source.get("name") or "Sconosciuto"
		category = source.get("category") or "N/A"
		score = source.get("score")
		is_native = source.get("is_native", False)

		# Badge nativo/mod
		badge = "🔹 Nativo" if is_native else "🔶 Mod/Config"

		# Riga della fonte
		line = f"{i}. **{name}** — {category} {badge}"
		if score is not None:
			line += f" | Accessibilità: {score}/10"

		lines.append(line)

	return "\n".join(lines)


def respond(
	message: str, chat_history: Optional[List[Dict[str, str]]]
) -> Tuple[List[Dict[str, str]], str]:
	"""Gestore della risposta per il messaggio dell'utente.

	Processa il messaggio attraverso il RAG assistant e aggiorna la
	cronologia della chat. Gestisce errori e stato di inizializzazione.

	Args:
		message: Messaggio dell'utente
		chat_history: Cronologia della chat (lista di dicts con role/content)

	Returns:
		Tupla di (chat_history_aggiornato, campo_di_input_vuoto)
	"""
	if chat_history is None:
		chat_history = []

	# Riferimenti globali per evitare deepcopy issues
	assistant = ASSISTANT
	ready_event = READY_EVENT

	# Validazione input
	if not message or not message.strip():
		return chat_history, ""

	# Aggiungi il messaggio dell'utente
	chat_history.append({"role": "user", "content": message})

	# Verifica stato di inizializzazione
	if ready_event is None or not ready_event.is_set():
		chat_history.append(
			{
				"role": "assistant",
				"content": "⏳ Sistema in inizializzazione... Sto caricando i dati. Riprova tra qualche istante.",
			}
		)
		return chat_history, ""

	if assistant is None:
		chat_history.append(
			{
				"role": "assistant",
				"content": "❌ Errore: Assistente non inizializzato correttamente.",
			}
		)
		return chat_history, ""

	# Elabora la query
	try:
		LOGGER.debug("Elaborando query: %s", message[:100])
		result = assistant.query(message)
		answer = result.get("answer", "")
		sources = result.get("sources", [])

		# Formatta risposta con fonti
		full_answer = answer + format_sources(sources)

		chat_history.append({"role": "assistant", "content": full_answer})

		LOGGER.debug("Query elaborata con successo")
		return chat_history, ""

	except Exception as exc:
		LOGGER.exception("Errore durante l'elaborazione: %s", exc)
		chat_history.append(
			{
				"role": "assistant",
				"content": f"❌ Errore: Si è verificato un problema. {str(exc)[:100]}",
			}
		)
		return chat_history, ""



def build_ui() -> gr.Blocks:
	"""Costruisce l'interfaccia Gradio.

	Crea una UI con:
	- Titolo e descrizione
	- Indicatore di stato
	- Chat principale
	- Campo input per le domande
	- Pulsanti di azione

	Returns:
		Oggetto gr.Blocks con l'interfaccia configurata
	"""
	with gr.Blocks(
		title="Accessibility Assistant - Chat RAG",
		theme=gr.themes.Soft(),
	) as demo:
		# Header
		gr.Markdown("# 🎮 Assistente per l'Accessibilità nei Videogiochi")
		gr.Markdown(
			"Chiedi tutto sull'accessibilità nei videogiochi. "
			"Utilizza un sistema RAG basato su **Ollama** per risposte accurate."
		)

		# Status bar
		status_label = gr.Textbox(
			value=_get_status_text(),
			label="🟢 Stato Sistema",
			interactive=False,
			lines=1,
		)

		# Chat
		chatbot = gr.Chatbot(
			label="💬 Conversazione",
			height=400,
			show_copy_button=True,
			type="messages"
		)

		# Input section
		with gr.Row():
			txt = gr.Textbox(
				show_label=False,
				placeholder="Digita la tua domanda sull'accessibilità nei videogiochi...",
				lines=1,
				scale=5,
			)
			send_btn = gr.Button("📤 Invia", scale=1)

		# Control buttons
		with gr.Row():
			refresh_btn = gr.Button("🔄 Aggiorna Stato")
			clear_btn = gr.Button("🗑️ Cancella Chat")

		# Event handlers
		send_btn.click(
			fn=respond,
			inputs=[txt, chatbot],
			outputs=[chatbot, txt],
			queue=True,
		)

		txt.submit(
			fn=respond,
			inputs=[txt, chatbot],
			outputs=[chatbot, txt],
			queue=True,
		)

		def refresh_status() -> str:
			"""Aggiorna il testo dello stato."""
			return _get_status_text()

		refresh_btn.click(
			fn=refresh_status,
			inputs=[],
			outputs=[status_label],
		)

		clear_btn.click(
			fn=lambda: [],
			inputs=[],
			outputs=[chatbot],
		)

		# Footer
		gr.Markdown(
			"---\n"
			"**Nota**: Assicurati che Ollama sia in esecuzione (`ollama serve`). "
			"Per info: [GitHub](https://github.com/DriftDeV/Accessibility-Assistant)"
		)

	return demo


def _get_status_text() -> str:
	"""Ritorna il testo dello stato formattato."""
	if READY_EVENT is None or not READY_EVENT.is_set():
		return "⏳ Inizializzazione in corso... (questa operazione può richiedere alcuni minuti)"
	return "✅ Sistema pronto! Puoi fare domande."


def main() -> None:
	"""Punto di ingresso principale.

	Carica la configurazione, inizializza l'assistente in background,
	e avvia il server Gradio.
	"""
	# Carica configurazione da variabili d'ambiente
	cfg = RAGConfig.from_env()
	logger = setup_logging(cfg.log_level)
	LOGGER.setLevel(cfg.log_level)

	LOGGER.info("=" * 70)
	LOGGER.info("Avvio Chat UI - Accessibility Assistant")
	LOGGER.info("=" * 70)
	LOGGER.info("Configurazione:")
	LOGGER.info("  - Ollama URL: %s", cfg.ollama_base_url)
	LOGGER.info("  - Embedding Model: %s", cfg.ollama_embedding_model)
	LOGGER.info("  - LLM Model: %s", cfg.ollama_llm_model)
	LOGGER.info("  - Games File: %s", cfg.games_file)
	LOGGER.info("=" * 70)

	# Inizializza l'assistente in background
	try:
		assistant, ready_event = initialize_assistant_in_thread(cfg)
		global ASSISTANT, READY_EVENT
		ASSISTANT = assistant
		READY_EVENT = ready_event
		LOGGER.info("✅ Assistente inizializzato in background")
	except Exception as e:
		LOGGER.exception("❌ Errore durante l'inizializzazione: %s", e)
		raise

	# Costruisci UI
	demo = build_ui()

	# Configurazione server (via variabili d'ambiente o default)
	host = os.getenv("GRADIO_HOST", "127.0.0.1")
	port = int(os.getenv("GRADIO_PORT", "7860"))
	share = os.getenv("GRADIO_SHARE", "0") in {"1", "true", "True"}

	LOGGER.info("🚀 Avvio Gradio Server:")
	LOGGER.info("   http://%s:%d", host, port)
	LOGGER.info("   Share: %s", share)

	# Avvia il server
	demo.launch(
		server_name=host,
		server_port=port,
		share=share,
		show_error=True,
	)


if __name__ == "__main__":
	main()
