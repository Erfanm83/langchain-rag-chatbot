from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.config import OPENAI_API_KEY
import openai
import os

openai.api_key = OPENAI_API_KEY

FAISS_PATH = "train/faiss_index"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def generate_answer(question: str) -> str:
    """Query the FAISS vector DB and return the AI’s answer."""
    
    # 1. Load embeddings & FAISS index
    embedding_fn = OpenAIEmbeddings()
    
    if not os.path.exists(FAISS_PATH):
        return "پایگاه داده پیدا نشد. لطفاً ابتدا ایندکس را بسازید."
    
    db = FAISS.load_local(FAISS_PATH, embeddings=embedding_fn, allow_dangerous_deserialization=True)

    # 2. Do a similarity search
    results = db.similarity_search_with_relevance_scores(question, k=3)
    if not results or results[0][1] < 0.7:
        return "متأسفم، قادر به یافتن پاسخ مرتبط نیستم."

    # 3. Build prompt
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context,
        question=question
    )

    # 4. Call the chat model
    model = ChatOpenAI()
    answer = model.predict(prompt)
    return answer
