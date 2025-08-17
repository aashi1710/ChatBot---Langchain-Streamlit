import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain

from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph_supervisor import create_supervisor
from langgraph.graph import StateGraph, MessagesState, END

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
    return f"The weather in {city} is {resp['main']['temp']}°C."

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


def roll_dice() -> str:
    #Rolls a dice with the given number of sides.
    return f"You rolled a {random.randint(1, 6)}"

def flip_coin() -> str:
    #Flips a coin and returns Heads or Tails.
    return random.choice(["Heads", "Tails"])

#Step 3: Setup LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=google_api_key  
)

#Step 4: Wrap tools with LangChain Tools
agent1_tools = [
    Tool(name="Dice", func=lambda _:roll_dice(), description="Rolls a six-sided dice"),
    Tool(name="Coin", func= lambda _:flip_coin(), description="Flips a coin")
]

agent2_tools = [
    Tool(name="Math", func=math_tool, description="Perform math calculations. Input should be an expression like '2+2' or '10*3'."),
    Tool(name="Joke", func=joke_tool, description="Tells a random joke")
]

agent3_tools = [
    Tool(name="RAG", func=rag_query_tool, description="Look up answers from the uploaded documents. Input must be the user's question."),
    
]

agent4_tools = [
    Tool(name="Weather", func=weather_tool, description="Get current weather info for a city. Input should contain the city name.")
]


#Step 5: Create Sub-Agents (each with its own ToolNode + LLM)
agent1 = ToolNode(tools=agent1_tools, name="Agent1")
agent2 = ToolNode(tools=agent2_tools, name="Agent2")
agent3 = ToolNode(tools=agent3_tools, name="Agent3")
agent4 = ToolNode(tools=agent4_tools, name="Agent4")


#Step 6: Supervisor Agent
supervisor = create_supervisor(
    agents=[agent1, agent2, agent3, agent4],
    model=ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=google_api_key
    ),
    prompt="""
You are a supervisor. Route user queries to the correct agent and return the response.
"""
)

# Step 7: LangGraph Setup

# Compile supervisor into a runnable
supervisor_runnable = supervisor.compile()

builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("supervisor", supervisor_runnable)
builder.add_node("Agent1", agent1)
builder.add_node("Agent2", agent2)
builder.add_node("Agent3", agent3)

# Define edges
builder.add_edge("supervisor", "Agent1")
builder.add_edge("supervisor", "Agent2")
builder.add_edge("supervisor", "Agent3")

builder.add_edge("Agent1", END)
builder.add_edge("Agent2", END)
builder.add_edge("Agent3", END)

# Entry point
builder.set_entry_point("supervisor")

# Compile graph
supervisor_agent = builder.compile()

#Step 6: Streamlit UI
st.title("Hi, How can I help you today?")


user_query = st.text_input("Search Tab", placeholder="Type your question here...")

if st.button("Search") and user_query:
    with st.spinner("Thinking..."):
        response = supervisor_agent.invoke({"messages": [("user", user_query)]})
    last_message = response["messages"][-1]
    st.success(last_message.content)
    


