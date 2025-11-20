"""
Esempi di utilizzo dell'Accessibility Assistant.

Questo file mostra diversi modi di usare il RAG:
1. Modalità interattiva
2. Utilizzo programmativo
3. Batch processing
4. Configurazione custom
"""

from accessibility_rag import AccessibilityAssistant, RAGConfig
from pathlib import Path


def example_interactive():
    """Esempio 1: Modalità interattiva (REPL)."""
    print("=== EXAMPLE 1: Interactive Mode ===\n")

    assistant = AccessibilityAssistant()
    assistant.setup()
    assistant.interactive_mode()


def example_programmatic():
    """Esempio 2: Utilizzo programmativo."""
    print("=== EXAMPLE 2: Programmatic Usage ===\n")

    assistant = AccessibilityAssistant()
    assistant.setup()

    # Domande di test
    questions = [
        "Quali giochi sono completamente accessibili ai non vedenti?",
        "Che cosa significa High Contrast Mode?",
        "Quali giochi supportano il daltonismo?",
    ]

    for question in questions:
        print(f"\n❓ Domanda: {question}")
        result = assistant.query(question)
        print(f"💬 Risposta: {result['answer'][:200]}...")

        if result["sources"]:
            print("📚 Fonti:")
            for source in result["sources"]:
                print(f"   - {source['name']} ({source['score']}/10)")


def example_custom_config():
    """Esempio 3: Configurazione custom."""
    print("=== EXAMPLE 3: Custom Configuration ===\n")

    # Configurazione personalizzata
    config = RAGConfig(
        games_file=Path("games.json"),
        top_k_results=5,  # Più risultati
        max_tokens=500,  # Risposte più lunghe
    )

    assistant = AccessibilityAssistant(config)
    assistant.setup()

    result = assistant.query("Quali sono i migliori giochi accessibili?")
    print(f"💬 Risposta:\n{result['answer']}\n")

    print(f"📚 Trovate {len(result['sources'])} fonti")


def example_batch_processing():
    """Esempio 4: Batch processing."""
    print("=== EXAMPLE 4: Batch Processing ===\n")

    assistant = AccessibilityAssistant()
    assistant.setup()

    questions = [
        "Quali giochi hanno il Text-to-Speech?",
        "Quali giochi sono nativamente accessibili?",
        "Cosa significa Navigation Assist?",
        "Quali piattaforme hanno più giochi accessibili?",
    ]

    results = []
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Elaborazione: {question[:50]}...")
        result = assistant.query(question)
        results.append(
            {
                "question": question,
                "answer": result["answer"],
                "num_sources": len(result["sources"]),
            }
        )

    print("\n" + "=" * 60)
    print("RISULTATI BATCH PROCESSING")
    print("=" * 60)

    for result in results:
        print(f"\nDomanda: {result['question']}")
        print(f"Risposta (prime 150 chars): {result['answer'][:150]}...")
        print(f"Fonti trovate: {result['num_sources']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "1":
            example_interactive()
        elif example == "2":
            example_programmatic()
        elif example == "3":
            example_custom_config()
        elif example == "4":
            example_batch_processing()
        else:
            print("Utilizzo: python examples.py [1|2|3|4]")
            print("  1 - Interactive Mode")
            print("  2 - Programmatic Usage")
            print("  3 - Custom Configuration")
            print("  4 - Batch Processing")
    else:
        print("Scegli un esempio (1-4):")
        print("  python examples.py 1  - Interactive Mode")
        print("  python examples.py 2  - Programmatic Usage")
        print("  python examples.py 3  - Custom Configuration")
        print("  python examples.py 4  - Batch Processing")
