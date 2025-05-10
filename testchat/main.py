import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
import re

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

# Initialize LLM
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=700,
    timeout=None,
    max_retries=2,
)

# def setup_vector_db(text_file_path):
#     """Setup vector database from a text file"""
#     # Load and chunk text file
#     loader = TextLoader(text_file_path)
#     documents = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=[
#         "----------------------------------------",  # Exact match for common separator
#         "-------------------------------------",     # Alternative length
#         "-----------------------------------------", # Alternative length
#         "\n\n----------------------------------------\n\n", # Separated sections
#         "\n----------------------------------------\n",     # Another variation
#         "\n\n",  # Double newline - typically separates message groups
#         "\n",    # Single newline - typically separates individual messages
#         " ",     # Word boundaries as last resort
#         ],
#         chunk_size=500,
#         chunk_overlap=50
#         # length_function=len
#     )
#     chunks = text_splitter.split_documents(documents)
    
#     # Create vector database
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-mpnet-base-v2"
#     )
#     vector_db = FAISS.from_documents(chunks, embeddings)
    
#     return vector_db

def setup_vector_db(text_file_path):
    """Setup vector database from a text file using custom splitting by dash separators"""
    # Read the file content
    with open(text_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content by dash separators
    # This regex matches any line consisting of 5 or more consecutive dashes
    chunks = re.split(r'\n-{5,}\n', content)
    
    # Clean up chunks and convert to LangChain Document objects
    documents = []
    for i, chunk in enumerate(chunks):
        # Skip empty chunks
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # Clean up any remaining separator lines that might be at the beginning or end
        chunk = re.sub(r'^-+\n', '', chunk)
        chunk = re.sub(r'\n-+$', '', chunk)
        
        # Create a Document object with metadata
        doc = Document(
            page_content=chunk,
            metadata={
                "source": text_file_path,
                "chunk_id": i,
                "conversation_id": i  # Assuming each separator indicates a new conversation
            }
        )
        documents.append(doc)
    
    print(f"Split file into {len(documents)} conversation chunks")
    
    # Create vector database
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_db = FAISS.from_documents(documents, embeddings)
    
    return vector_db

def test_vector_db(text_file_path):
    """Test function to see how the file gets split"""
    # Read the file content
    with open(text_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content by dash separators
    chunks = re.split(r'\n-{5,}\n', content)
    
    # Print information about the chunks
    print(f"Total number of chunks: {len(chunks)}")
    
    # Print a sample of the first few chunks
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} (Length: {len(chunk)}) ---")
        # Print just the first 150 characters if the chunk is long
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)
    
    return chunks

def get_local_content(vector_db, query, k=1):
    """Retrieve relevant content from vector database"""
    if not query:
        # For initial empty query, return empty context
        return ""
    
    # Get relevant documents from vector store
    docs = vector_db.similarity_search(query, k=k)
    
    # Combine the document contents
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def check_local_knowledge(query, context):
    """Router function to determine if we can answer from local knowledge"""
    prompt = '''Role: Sales Expert Question-Answering Assistant(ServerYar)
Task: Determine whether the system can answer the user's question based on the provided text.

Instructions:
    - Analyze the text and identify if it contains the necessary information to answer the user's question.
    - Provide a clear and concise response indicating whether the system can answer the question or not.
    - Your response should include only a single word. Nothing else, no other text, information, header/footer.
 
Output Format:
    - Answer: Yes/No

Study the below examples and based on that, respond to the last question.
 
Examples:
    Input: 
        Text: The capital of France is Paris.
        User Question: What is the capital of France?
    Expected Output:
        Answer: Yes

    Input: 
        Text: The population of the United States is over 330 million.
        User Question: What is the population of China?
    Expected Output:
        Answer: No

    Input:
        User Question: {query}
        Text: {text}'''
    
    formatted_prompt = prompt.format(text=context, query=query)
    response = llm.invoke(formatted_prompt)
    return "yes" in response.content.strip().lower()

def generate_final_answer(context, query):
    """Generate final answer using LLM"""
    messages = [
        (
            "system",
            "You are sales expert assistant for IranServer, a web hosting company. Use the provided context to answer the query accurately. Be concise but friendly and helpful.",
        ),
        ("system", f"Context: {context}"),
        ("human", query),
    ]
    response = llm.invoke(messages)
    return response.content

def process_query(query, vector_db, local_context):
    """Process user query with human operator fallback"""
    
    # First, get fresh context from the vector DB for this specific query
    retrieved_context = get_local_content(vector_db, query)

    combined_context = f"{local_context}\n\n{retrieved_context}".strip()

    print(f"\n\n\tcombined_context:\n\n {combined_context}\n\n")
    
    # Step 1: Check if we can answer from local knowledge
    can_answer_locally = check_local_knowledge(query, combined_context)
    print(f"Can answer locally: {can_answer_locally}")
    
    # Step 2: Generate answer or connect to human operator
    if can_answer_locally:
        # We can answer, generate response from the context
        answer = generate_final_answer(combined_context, query)
        return answer
    else:
        # We cannot answer confidently, connect to human operator
        return "Connected to human operator. An IranServer customer support agent will assist you shortly."

def remove_think_section(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def main():
    # Setup with text file
    text_file_path = "sample_chat.txt"
    
    # Initialize vector database
    print("Setting up vector database...")
    vector_db = setup_vector_db(text_file_path)
    
    # Get initial context
    initial_context = ""  # Empty for first interaction
    
    # Test queries
    test_queries = [
        """
        The site is in the field of clothing
        When the internet gets busy, the internets get into trouble. If I'm not mistaken, the .ir domains were open for a while and the other domains had problems
        I want the site to still be open if the internets get into trouble
        """
    ]
    
    # Process each query
    print("\nProcessing test queries:")
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        result = process_query(query, vector_db, initial_context)
        print(f"Result: {remove_think_section(result)}")
        
        # Update context for next query (in real system, this would accumulate the conversation)
        # initial_context += f"\nQ: {query}\nA: {result}\n"

if __name__ == "__main__":
    main()