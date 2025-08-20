import os
import json
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


# Load .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env")
    st.stop()


#LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY, temperature=0)


# Prompts
interpret_prompt = PromptTemplate(
    input_variables=["story"],
    template="""
    Convert this user story into structured JSON.

    Story: "{story}"

    Output ONLY JSON in this format:
    {{
      "function_name": "snake_case_function_name",
      "inputs": ["list input types"],
      "output": "description of output",
      "logic": "short description of how to implement"
    }}
    """
)

generate_prompt = PromptTemplate(
    input_variables=["requirements"],
    template="""
    You are a Python code generator.
    Based on the following specification, generate a runnable Python function.

    Specification:
    {requirements}

    Constraints:
    - Code must be valid Python
    - Should be self-contained and executable
    - No TODOs or placeholders
    
    """
)


def run_agent1(story: str):
    prompt = interpret_prompt.format(story=story)
    raw = llm.invoke(prompt).content
    try:
        match = re.search(r"\{.*\}", raw, re.S)
        return json.loads(match.group(0)) if match else {}
    except Exception:
        return {"function_name": "generated_function", "inputs": [], "output": "", "logic": raw}

def run_agent2(requirements: dict):
    prompt = generate_prompt.format(requirements=json.dumps(requirements, indent=2))
    raw_code = llm.invoke(prompt).content

    
    code = re.sub(r"^```(?:python)?\s*|```$", "", raw_code.strip(), flags=re.MULTILINE).strip()
    return code



# Streamlit UI
st.set_page_config(page_title="GET CODE!", page_icon="")
st.title("GET CODE!")

story = st.text_input("Enter your user story:")

if st.button("Generate Code") and story:
    st.write("Processing...")

    requirements = run_agent1(story)
    code = run_agent2(requirements)

    filename = f"{requirements.get('function_name', 'generated_code')}.py"
    Path(filename).write_text(code)

    st.subheader("Generated Python Code:")
    st.code(code, language="python")
    st.download_button("Download Python File", data=code, file_name=filename, mime="text/x-python")
