
"""
Simple Gradio chat UI for the Accessibility RAG assistant.

Questo file fornisce una piccola interfaccia grafica per interrogare
la classe `AccessibilityAssistant` definita in `rag_assistant.py`.

Caratteristiche principali:
- Avvio in background dell'inizializzazione (indexing + LLM loading).
- Interfaccia chat semplice con visualizzazione delle fonti.
- Buone pratiche: typing, logging, separazione funzioni.

Requisiti principali (esempi):
- gradio
- sentence-transformers
- chromadb
- transformers
- torch

Esempio d'uso:
	python gradio_ui.py

Nota: l'inizializzazione può richiedere tempo (caricamento LLM, indicizzazione).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from rag_optimized_v2 import AccessibilityAssistant, Config, setup_logging


LOGGER = logging.getLogger("gradio_ui")

# Module-level references to avoid passing non-deepcopyable objects
# (e.g. threading.Event) through Gradio's `State` which may attempt
# to deepcopy values.
ASSISTANT: Optional[AccessibilityAssistant] = None
READY_EVENT: Optional[threading.Event] = None


def initialize_assistant_in_thread(cfg: Config) -> Tuple[AccessibilityAssistant, threading.Event]:
	"""Crea l'istanza dell'assistente e la inizializza in un thread separato.

	Restituisce una tupla (assistant, ready_event). L'evento viene settato
	quando l'inizializzazione è completata (o se fallisce, l'eccezione sarà
	loggata ma l'evento sarà settato comunque per evitare blocchi infiniti).
	"""

	assistant = AccessibilityAssistant(cfg)
	ready = threading.Event()

	def _setup():
		try:
			LOGGER.info("Starting assistant setup in background thread...")
			assistant.setup()
			LOGGER.info("Assistant setup completed")
		except Exception as exc:  # pragma: no cover - runtime integration
			LOGGER.exception("Assistant setup failed: %s", exc)
		finally:
			ready.set()

	th = threading.Thread(target=_setup, daemon=True)
	th.start()
	return assistant, ready


def format_sources(sources: List[Dict[str, Any]]) -> str:
	"""Formatta le fonti in una stringa compatta per mostrarle nella chat."""
	if not sources:
		return ""
	lines = ["\n\nFonti:\n"]
	for s in sources:
		name = s.get("name") or "Sconosciuto"
		score = s.get("score")
		src = s.get("source_ref") or ""
		lines.append(f"- {name}" + (f" (score: {score})" if score is not None else "") + (f" — {src}" if src else ""))
	return "\n".join(lines)


def respond(message: str, chat_history: Optional[List[Tuple[str, str]]]):
	"""Gestore chiamato quando l'utente invia un messaggio dalla UI.

	- `chat_history` è una lista di tuple (user, assistant).
	"""
	if chat_history is None:
		chat_history = []

	# Use module-level references to avoid deepcopy/pickling issues
	assistant = ASSISTANT
	ready_event = READY_EVENT

	# Messaggio vuoto -> non rispondere
	if not message or not message.strip():
		return chat_history, ""

	# Se l'assistente non è pronto segnalo all'utente
	if ready_event is None or not ready_event.is_set():
		chat_history.append(("Utente", message))
		chat_history.append(("Assistente", "Sto ancora preparando il sistema. Riprova tra qualche istante..."))
		return chat_history, ""

	# Esegui la query al RAG assistant
	try:
		res = assistant.query(message)
		answer = res.get("answer", "")
		sources = res.get("sources", [])
		answer_with_sources = answer + format_sources(sources)

		chat_history.append(("Utente", message))
		chat_history.append(("Assistente", answer_with_sources))
		return chat_history, ""
	except Exception as exc:
		LOGGER.exception("Errore durante la generazione della risposta: %s", exc)
		chat_history.append(("Utente", message))
		chat_history.append(("Assistente", "Si è verificato un errore tentando di ottenere la risposta. Controlla i log."))
		return chat_history, ""



def build_ui() -> gr.Blocks:
	"""Costruisce e ritorna l'oggetto Gradio Blocks con la chat.

	Usiamo variabili di modulo (`ASSISTANT`, `READY_EVENT`) invece di
	passare un dict tramite `gr.State`, che può tentare di deepcopyare
	oggetti non copiable come `threading.Event`.
	"""
	with gr.Blocks(title="Accessibility Assistant - Chat RAG") as demo:
		gr.Markdown("# Assistente Accessibilità — Chat UI")
		gr.Markdown(
			"Questa interfaccia usa il sistema RAG per rispondere a domande sull'accessibilità nei videogiochi."
		)

		# Status semplice
		status = gr.Textbox(value=("Pronto" if READY_EVENT and READY_EVENT.is_set() else "Inizializzazione in corso..."),
				 label="Stato", interactive=False)

		chatbot = gr.Chatbot(label="Conversazione")
		txt = gr.Textbox(show_label=False, placeholder="Scrivi la tua domanda qui e premi Invio...")
		send = gr.Button("Invia")

		# Permetti di aggiornare lo stato manualmente
		refresh = gr.Button("Aggiorna stato")

		# Azioni
		send.click(fn=respond, inputs=[txt, chatbot], outputs=[chatbot, txt])
		txt.submit(fn=respond, inputs=[txt, chatbot], outputs=[chatbot, txt])

		def refresh_status() -> str:
			return "Pronto" if READY_EVENT and READY_EVENT.is_set() else "Inizializzazione in corso..."

		refresh.click(fn=refresh_status, inputs=[], outputs=[status])

	return demo


def main() -> None:
	# Config e logging
	cfg = Config()
	logger = setup_logging(cfg.log_level)
	LOGGER.setLevel(cfg.log_level)

	# Inizializziamo l'assistente in background
	assistant, ready_event = initialize_assistant_in_thread(cfg)
	# Save into module-level references so Gradio callbacks can access them
	global ASSISTANT, READY_EVENT
	ASSISTANT = assistant
	READY_EVENT = ready_event

	# Costruzione UI
	demo = build_ui()

	# Parametri di esecuzione: HOST, PORT e SHARE via env vars (opzionali)
	host = os.getenv("GRADIO_HOST", "127.0.0.1")
	port = int(os.getenv("GRADIO_PORT", "7860"))
	share = os.getenv("GRADIO_SHARE", "0") in {"1", "true", "True"}

	LOGGER.info("Avvio Gradio su %s:%d (share=%s)", host, port, share)
	demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
	main()
