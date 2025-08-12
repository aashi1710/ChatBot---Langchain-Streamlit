import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Safety check for API key
if not api_key:
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()


st.title("Hi, How can I help you today?")


query = st.text_input("Search Tab", placeholder="Type your question here...")

file_path = "mydata.txt"
if not os.path.exists(file_path):
    st.error(f"File '{file_path}' not found. Please place it in the project folder.")
    st.stop()
loader = TextLoader(file_path)
documents = loader.load()

if not documents or all(doc.page_content.strip() == "" for doc in documents):
    st.error("The file is empty or contains no readable text.")
    st.stop()


#sample_docs = [
 #   Document(page_content="LangChain is a framework for developing applications powered by language models."),
  #  Document(page_content="Gemini 1.5 Flash is a Google AI model that is optimized for fast and efficient responses."),
   # Document(page_content="Streamlit is a Python library for creating interactive web apps easily.")
#]

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Create embeddings & store in FAISS
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vectorstore = FAISS.from_documents(docs, embeddings)

# Create Gemini LLM 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=api_key  
)


#if st.button("Search"):
 #   if query.strip():
        
 #           response = llm.invoke(query)
 #           st.markdown(f"""
 #           <div style='border:2px solid #ddd; border-radius:10px; padding:15px;'>
  #              {response.content}
 #           </div>
 #           """, unsafe_allow_html=True)
        
 #   else:
  #      st.warning("Please type a question first.")
  
if st.button("Search"):
    if query.strip():
        # Retrieve relevant docs
        retrieved_docs = vectorstore.similarity_search(query, k=1)

       # st.subheader("Retrieved Context Chunks")
        #for i, doc in enumerate(retrieved_docs, 1):
         #   st.write(f"**Chunk {i}:** {doc.page_content}")

        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Create a simple RAG prompt
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful assistant. Use the following context to answer the question.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            """
        )
        prompt = prompt_template.format(context=context_text, question=query)

        # Get LLM response
        response = llm.invoke(prompt)

        # Display result
        st.markdown(f"""
        <div style='border:2px solid #ddd; border-radius:10px; padding:15px;'>
            {response.content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please type a question first.")
