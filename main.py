import json
import os
import logging
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# --- CONFIGURAZIONE ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File JSON dei giochi
JSON_FILE_PATH = "games.json"

# Env
USE_LOCAL = os.environ.get("USE_LOCAL_MODEL", "1") in ("1", "true", "True")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "gpt2")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "games_accessibility_v1")


def load_games_data(filepath: str) -> List[dict]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Il file {filepath} non è stato trovato.")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_texts_and_metadata(games_data: List[dict]):
    texts, metadatas, ids = [], [], []
    for game in games_data:
        nativo_txt = "Supporto Nativo" if game.get("is_native") else "Non Nativo (richiede mod)"
        features_list = ", ".join(game.get("features", []))
        platforms_list = ", ".join(game.get("platforms", []))
        details = game.get("accessibility_details", {})
        visiva = details.get("visual", "Non specificato")
        motoria = details.get("motor", "Non specificato")
        uditiva = details.get("auditory", "Non specificato")
        cognitiva = details.get("cognitive", "Non specificato")

        text = (
            f"TITOLO: {game.get('name')}\n"
            f"CATEGORIA: {game.get('category','Generico')}\n"
            f"PUNTEGGIO ACCESSIBILITA': {game.get('access_level','N/A')}/10\n"
            f"TIPO: {nativo_txt}\n"
            f"PIATTAFORME: {platforms_list}\n"
            f"DESCRIZIONE: {game.get('description','')}\n"
            f"FEATURES: {features_list}\n"
            f"DETTAGLI - Visiva: {visiva} | Motoria: {motoria} | Uditiva: {uditiva} | Cognitiva: {cognitiva}"
        )

        texts.append(text)
        metadatas.append({
            "id": game.get("id"),
            "name": game.get("name"),
            "score": game.get("access_level"),
            "is_native": game.get("is_native"),
        })
        ids.append(game.get("id"))

    return texts, metadatas, ids


def build_chroma_with_local_embeddings(texts, metadatas, ids, collection_name=CHROMA_COLLECTION):
    """Calcola embedding con SentenceTransformer e popola Chroma (client locale)."""
    logger.info(f"Caricamento modello di embedding: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embed_model.encode(texts, show_progress_bar=True)

    client = chromadb.Client()
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)

    # Rimuoviamo eventuali entries con gli stessi ids per evitare duplicati
    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    collection.add(ids=ids, metadatas=metadatas, documents=texts, embeddings=embeddings.tolist())
    return client, collection, embed_model


def retrieve_documents(collection, embed_model, question, k=3):
    q_emb = embed_model.encode([question])[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    docs = []
    for docs_list in res.get("documents", []):
        # docs_list is a list of retrieved docs for the single query
        docs.extend(docs_list)
    metas = []
    for meta_list in res.get("metadatas", []):
        metas.extend(meta_list)
    return docs, metas


def answer_with_local_llm(context: str, question: str) -> str:
    """Genera la risposta usando un LLM locale (transformers pipeline)."""
    model_id = LOCAL_LLM_MODEL
    logger.info(f"Caricamento LLM locale: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    device = 0 if torch.cuda.is_available() else -1
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    prompt = (
        "Sei un assistente esperto in accessibilità nei videogiochi. "
        "Usa solo le informazioni fornite nel contesto per rispondere.\n\n"
        f"CONTESTO:\n{context}\n\n"
        "REGOLE:\n"
        "- Rispondi SOLO in Italiano.\n"
        "- Se la domanda non riguarda accessibilità o videogiochi, rispondi: 'Mi dispiace, rispondo solo a domande sull'accessibilità nei videogiochi.'\n"
        "- Specifica sempre se una funzionalità è nativa o richiede mod.\n"
        "- Fornisci il punteggio di accessibilità se disponibile.\n\n"
        f"DOMANDA: {question}\n\nRISPOSTA:\n"
    )

    out = gen(prompt, max_new_tokens=256, do_sample=False)
    text = out[0]["generated_text"]
    # Il pipeline spesso restituisce l'intero prompt+risposta: tagliamo via il prompt
    if prompt in text:
        return text.split(prompt, 1)[1].strip()
    return text.strip()


def main():
    logger.info(f"Caricamento dati da {JSON_FILE_PATH}...")
    raw = load_games_data(JSON_FILE_PATH)
    texts, metadatas, ids = prepare_texts_and_metadata(raw)
    logger.info(f"Preparati {len(texts)} documenti per l'indicizzazione.")

    if USE_LOCAL:
        client, collection, embed_model = build_chroma_with_local_embeddings(texts, metadatas, ids)

        print("\n--- ACCESSIBILITY ASSISTANT v0.1 (Locale) (Digitare 'esci' per chiudere) ---")
        while True:
            q = input("\nFai una domanda: ")
            if q.strip().lower() in ["esci", "exit", "quit"]:
                break

            docs, metas = retrieve_documents(collection, embed_model, q, k=3)
            context = "\n---\n".join(docs)
            answer = answer_with_local_llm(context, q)
            print(f"\nAI: {answer}\n")
            if metas:
                print("Fonti consultate:")
                for m in metas:
                    name = m.get("name") or m.get("id") or "Sorgente sconosciuta"
                    print(f"- {name}")
    else:
        raise RuntimeError("Modalità remota non implementata in questa versione. Imposta USE_LOCAL_MODEL=1 per usare un modello locale.")


if __name__ == "__main__":
    main()