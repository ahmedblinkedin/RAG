# PDF RAG Analyzer (v0 - Proof of Concept)
;
⚠️ **Ce projet est une preuve de concept technique** illustrant les concepts présentés dans l'article LinkedIn . Il ne s'agit pas d'une application production-ready.

## Objectif
Démontrer l'implémentation d'un système RAG (Retrieval-Augmented Generation) pour l'analyse de documents PDF en utilisant:
- Ollama (Llama3 + embeddings locaux)
- LangChain (traitement des documents)
- Gradio (interface web)

## Prérequis techniques
- Machine Linux/MacOS
- Python 3.10+
- GPU recommandé (bien que non obligatoire)

## Installation

### 1. Configuration Ollama
```bash
# Installer Ollama et lancer le serveur
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# Télécharger les modèles nécessaires
ollama pull llama3
ollama pull nomic-embed-text

### 2. Intalation des lib
pip install ollama langchain langchain-core langchain-community gradio gradio_pdf
