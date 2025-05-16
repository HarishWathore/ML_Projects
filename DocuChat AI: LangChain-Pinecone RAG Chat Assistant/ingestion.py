from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    # Load documents from local path
    loader = ReadTheDocsLoader(
        r"C:\Users\hnwme\Desktop\documentation-helper\langchain-docs\api.python.langchain.com\en\latest",
        encoding="utf-8")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    # Update metadata to change source URL
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")

    # Create the vector store
    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name="langchain-doc-index"
    )

    # Define batch size
    batch_size = 100  # Adjust batch size to avoid exceeding Pinecone's size limit

    # Loop through documents in batches and upload
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Uploaded batch {i // batch_size + 1} with {len(batch)} documents")

    print("****Loading to vectorstore done ****")


if __name__ == "__main__":
    ingest_docs()
