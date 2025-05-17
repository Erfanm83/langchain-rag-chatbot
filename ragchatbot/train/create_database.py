import os
import re
import shutil
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def create_database(text_file_path: str, persist_path: str = "faiss_index"):
    """Reads a text file, splits it into chunks, and saves a FAISS vector store."""
    # Read content
    with open(text_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split using regex (lines with 5 or more dashes)
    chunks = re.split(r'\n-{5,}\n', content)

    documents = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunk = re.sub(r'^-+\n', '', chunk)
        chunk = re.sub(r'\n-+$', '', chunk)
        doc = Document(
            page_content=chunk,
            metadata={
                "source": text_file_path,
                "chunk_id": i,
                "conversation_id": i,
            }
        )
        documents.append(doc)

    print(f"Split file into {len(documents)} chunks")

    # Create vector DB
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )
    vector_db = FAISS.from_documents(documents, embeddings)

    # Persist if needed
    if persist_path:
        if os.path.exists(persist_path):
            shutil.rmtree(persist_path)
        vector_db.save_local(persist_path)
        print(f"Vector DB saved to {persist_path}")

    return vector_db
