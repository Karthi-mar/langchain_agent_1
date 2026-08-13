from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tool import tool
from langgraph.checkpoint.memory import InMemorySaver

import wikipedia 

load_dotenv

@tool 
def calculator(expression: str) -> str:
    """ Evaluate a basic math expression, eg '12*(3+4)'.
    supports +,-,*,/ ,** and parentheses."""

    allowed_names = {"__builtins__":{}}
    try:
        result = eval(expression,allowed_names)
        return(result)
    except Exception as e:
        return f"Could not evaluate '{expression}' : {e}"

@tool
def wikipedia_search(query: str) -> str:
    """Look up a topic on wikipedia and returna short summary(2-3 sentences).
    use this for factual questions about people ,places , events, or cencepts .""""

    try:
        return wikipedia.summary(query, sentences=3, auto_suggest = True)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"That query is ambigous. Did you mean one of: {e.options[:5]}?"
    except wikipedia.exceptions.PageError:
        return f"No wikipedia page not foud for '{query}"
    except Exception as e:
        return f"wikipedia lookup failed: {e}"
    