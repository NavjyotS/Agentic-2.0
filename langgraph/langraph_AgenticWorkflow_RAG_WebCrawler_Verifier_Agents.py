import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import operator
from typing import List
from pydantic import BaseModel , Field
from langchain.prompts import PromptTemplate
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph,END

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-1.5-flash')
output=model.invoke("hi")
print(output.content)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
len(embeddings.embed_query("hi"))

# Example: Load all .txt files from the "data" directory
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("../data2", glob="**/*.txt")

docs = loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

new_docs=text_splitter.split_documents(documents=docs)

doc_string=[doc.page_content for doc in new_docs]

doc_string

len(doc_string)

db=Chroma.from_documents(new_docs,embeddings)

retriever=db.as_retriever(search_kwargs={"k": 3})

retriever.invoke("industrial growth of usa?")

class TopicSelectionParser(BaseModel):
    Topic:str=Field(description="selected topic")
    Reasoning:str=Field(description='Reasoning behind topic selection')

from langchain.output_parsers import PydanticOutputParser

parser=PydanticOutputParser(pydantic_object=TopicSelectionParser)

parser.get_format_instructions()

Agentstate={}
Agentstate["messages"]=[]

Agentstate
Agentstate["messages"].append("hi how are you?")
Agentstate
Agentstate["messages"].append("what are you doing?")
Agentstate
Agentstate["messages"].append("i hope everything fine")
Agentstate
Agentstate["messages"][-1]
Agentstate["messages"][0]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

state={"messages":["hi"]}
state="hi"

def function_1(state:AgentState):
    
    question=state["messages"][-1]
    
    print("Question",question)
    
    template="""
    Your task is to classify the given user query into one of the following categories: [USA,Web Search,Not Related]. 
    Only respond with the category name and nothing else.

    User query: {question}
    {format_instructions}
    """
    
    prompt= PromptTemplate(
        template=template,
        input_variable=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    
    chain= prompt | model | parser
    
    response = chain.invoke({"question":question})
    
    print("Parsed response:", response)
    
    return {"messages": [response.Topic]}

state={"messages":["what is a today weather?"]}
state={"messages":["what is a GDP of usa??"]}
function_1(state)

class TopicSelectionParser(BaseModel):
    Topic:str=Field(description="selected topic")
    Reasoning:str=Field(description='Reasoning behind topic selection')

def router(state:AgentState):
    print("-> ROUTER ->")
    
    last_message=state["messages"][-1]
    print("last_message:", last_message)
    userQuery = last_message.lower()
    if "usa" in userQuery or "united states" in userQuery:
        route = "rag"
    elif  "search" in userQuery or "latest" in userQuery:
        route = "web"
    else:
        route = "llm"
    return {"route": route}

state={"messages":["search for todays weather details??"]}
router(state)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Function
def function_2(state:AgentState):
    print("-> RAG Call ->")
    
    question = state["messages"][0]
    
    prompt=PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, 
        just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:""",
        
        input_variables=['context', 'question']
    )
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    result = rag_chain.invoke(question)
    return  {"messages": [result]}

# LLM Function
def function_3(state:AgentState):
    print("-> LLM Call ->")
    question = state["messages"][0]
    
    # Normal LLM call
    complete_query = "Anwer the follow question with you knowledge of the real world. Following is the user question: " + question
    response = model.invoke(complete_query)
    return {"messages": [response.content]}

# To install: pip install tavily-python
from tavily import TavilyClient
client = TavilyClient("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")
response = client.search(
    query="What is the price of Iphone16 in ",
    search_depth="basic",  # Options: 'basic', 'advanced' (more results, slower)
    include_answer=True,      # Tavily will generate a summarized answer
    include_raw_content=False # If you just want summaries, not full HTML/text
)
print(response)

# WEb CrawlerFunction

from tavily import TavilyClient

def function_4(state:AgentState):
    print("-> WEB CRAWLER Call ->")
    question = state["messages"][0]
    #question = "When was mahatma Gandhi born and where?"
    
    os.environ['TAVILY_TOKEN']=os.getenv("TAVILY_API_KEY")
    client = TavilyClient(os.environ['TAVILY_TOKEN'])
    # Normal LLM call
    complete_query = """Search online and return a concise, up-to-date answer to the following question:"""+ question + """
    Make sure to include:
    - A short summary or direct answer
    - At least 2-3 supporting sources (URLs)
    - Mention if there is no recent or relevant information
    "Following is the user question: """
    response = client.search(
    query=complete_query,
    search_depth="basic",  # Options: 'basic', 'advanced' (more results, slower)
    include_answer=True,      # Tavily will generate a summarized answer
    include_raw_content=False # If you just want summaries, not full HTML/text
    )
    #print(response)
    #return {"messages": [response]}
    return {"messages": [response["answer"]]}

state={"messages":["what is a GDP of usa??"]}
state["messages"][0]
function_4(state)

from langgraph.graph import StateGraph,END

workflow3=StateGraph(AgentState)

workflow3.add_node("Supervisor",function_1)
workflow3.add_node("RAG",function_2)
workflow3.add_node("LLM",function_2)
workflow3.add_node("WEB",function_4)
workflow3.set_entry_point("Supervisor")
workflow3.add_conditional_edges(
    "Supervisor",
    router,
    {
        "RAG Call": "RAG",
        "TAV Call": "TAV",
    }
)
workflow3.add_edge("RAG",END)
workflow3.add_edge("TAV",END)
app=workflow3.compile()

state={"messages":["What is the weather of Delhi today?"]}
state={"messages":["What is the capital of USA?"]}
print(state)
app.invoke(state)
state={"messages":["what is a gdp of usa?"]}
app.invoke(state)
state={"messages":["can you tell me the industrial growth of world's most powerful economy?"]}
state={"messages":["can you tell me the industrial growth of world's poor economy?"]}
result=app.invoke(state)
result["messages"][-1]
