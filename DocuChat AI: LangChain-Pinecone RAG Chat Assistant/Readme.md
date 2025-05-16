# DocuChat AI

**DocuChat AI** is an AI-powered command-line assistant that lets developers interact with technical documentation through natural language queries. It uses Retrieval-Augmented Generation (RAG) with LangChain, OpenAI, and Pinecone to deliver source-backed, conversational answers, turning static docs into an interactive chat experience.

---

## Features

- Document ingestion, chunking, and embedding with OpenAI embeddings  
- Semantic search via Pinecone vector database  
- Conversational Q&A powered by LangChain and GPT  
- Command-line chat interface  
- Source-cited responses  

---

## Run Locally

1. Clone the project:
    ```bash
    git clone https://github.com/HarishWathore/MLProjects.git
    cd documentation-helper
    ```

2. Download LangChain documentation:
    ```bash
    mkdir langchain-docs
    wget -r -A.html -P langchain-docs https://api.python.langchain.com/en/latest
    ```

3. Install dependencies (using pipenv):
    ```bash
    pipenv install
    ```
4. Set up environment variables by creating a `.env` file:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_ENVIRONMENT=your_pinecone_environment
    ```    
5. Start the Streamlit server:
    ```bash
    streamlit run main.py
    ```

---

## Running Tests

Run tests with:

```bash
pipenv run pytest .
