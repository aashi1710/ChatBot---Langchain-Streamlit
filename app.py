import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import requests
import random

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

#Step 1: Load documents & FAISS 
loader = TextLoader("mydata.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embeddings)

#Step 2: Define Tools
def rag_query_tool(query: str) -> str:
    
    results = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

def weather_tool(city: str) -> str:
    #Fetch current weather for a city.
    api_key = weather_api_key
    if not api_key:
        return "Weather API key not found."
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    resp = requests.get(url).json()
    if resp.get("cod") != 200:
        return f"Error: {resp.get('message', 'Could not fetch weather')}"
    return f"The weather in {city} is {resp['main']['temp']}°C with {resp['weather'][0]['description']}."

def math_tool(expression: str) -> str:
    #Evaluate a math expression.
    try:
        result = eval(expression)
        return f"The result is {result}"
    except:
        return "Invalid math expression."

def joke_tool(_: str) -> str:
    #Return a random joke.
    jokes = [
        "Why don’t skeletons fight each other? They don’t have the guts.",
        "I told my computer I needed a break, and it said 'No problem — I’ll go to sleep.'",
        "Why did the math book look sad? Because it had too many problems."
    ]
    return random.choice(jokes)

#Step 3: Setup LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=google_api_key  
)

#Step 4: Wrap Tools for Agent 
tools = [
    Tool(
        name="KnowledgeBaseRAG",
        func=rag_query_tool,
        description="Useful for answering questions based on uploaded documents."
    ),
    Tool(
        name="WeatherTool",
        func=weather_tool,
        description="Get current weather info for a city. Input should be the city name."
    ),
    Tool(
        name="MathTool",
        func=math_tool,
        description="Perform math calculations. Input should be an expression like '2+2' or '10*3'."
    ),
    Tool(
        name="JokeTool",
        func=joke_tool,
        description="Tell a random joke. Input can be anything."
    )
]

#Step 5: Initialize Agent 
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

#Step 6: Streamlit UI
st.title("Hi, How can I help you today?")


user_query = st.text_input("Search Tab", placeholder="Type your question here...")

if st.button("Search") and user_query:
    with st.spinner("Thinking..."):
        response = agent.run(user_query)
    st.success(response)    
    


