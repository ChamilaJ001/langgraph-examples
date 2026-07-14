from typing import TypedDict, Annotated, Sequence
from  langchain_core.messages import BaseMessage,ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode

load_dotenv()

# This is global variable to store document content
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
@tool
def update(content:str) -> str:
    """Updates the document with provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"

@tool
def save(filename:str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """
    
    global document_content
    
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
        
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n Document has been saved to: {filename}")
        return f"Documnt has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    
tools = [update, save]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete update content.
    - If the user wants to save and finish, ypu need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
                                  """)
    
    user_input = input("\nWhat would you like to do with the document? ")
    print(f"\nUSER: {user_input}")
    user_message = HumanMessage(content=user_input)
        
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    
    response = model.invoke(all_messages)
    
    print(f"\n AI: {response.content}")
    if response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return{
        "messages": list(state["messages"]) + [user_message, response]
    }
    
def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""
    
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message
    for message in reversed(messages):
        # ... and checks if this is a TollsMessage resulting from save
        if(isinstance(message, ToolMessage) and
           "saved" in message.content.lower() and
           "document" in message.content.lower()):
            return "end" # goes to the end edge wich leads to the endpoint
        
    return "continue"

def print_message(messages):
    """Function I made to print the message in a more readble format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n TOOL RESULT: {message.content}")
            
def should_use_tools(state: AgentState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END
    
    
graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_use_tools,
    {
        "tools": "tools",
        END: END
    }
)

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():
    print("\n === DRAFTER ===")

    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])
        print("\n === DRAFTER FINISHED ===")     
        
if __name__ == "__main__":
    run_document_agent()