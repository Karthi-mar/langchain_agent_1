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
    

SYSTEM_PROMPT = """ You are the study buddy ,a friendly and precise study assistant.
                    you have two tools:
                    - 'calculatore' - use this for any arithematic instead of computing math in your head
                    -'wikipedia_search' - use this when the userasks about a factual topic, person, place or event.

                    Always prefer using a tool over guessing. If a tool returns error, tell the user honestly instead 
                    of making up an answer, Keep answers consice, like a good study partner""""



model = _init_chat_model(
    "gemini-2.5-flash",
    model_provider = "google-genai",
    temperature = 0.3, 
)

checkpoint = InMemorySaver()  #stores the conversation state

agent = creat_agent(
    model = model
    tools = [calculator,wikipedia_search],
    system_prompt = SYSTEM_PROMPT
    checkpointer = checkpointer,
)


def main()

thread_id = "Study-buddy-session-1"
print("Study buddy is ready! type 'quit or exit'. \n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in {"quit", "exit"}:
        print("Good Bye!")
        break
    
    result = agent.invoke(
        {"messages" : [{"role": "user", "content":user_input}]},
        config = {"configurable": {"thread_id": thread_id}},
    )

    reply = result["messages"][-1].content
    print(f"Study Buddy: {reply} \n")

if __name__ == "__main__":
    main()


