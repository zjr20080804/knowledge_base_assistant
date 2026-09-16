
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

from src.model import get_llm
from src.tool import all_tools
from src.MessageHistory import get_session_history

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever,with_history=True):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个知识库助手,请根据资料回答用户的问题,绝对不可以给出资料中没有的信息,不要自己编造答案,资料:{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    base_chain = \
    (RunnablePassthrough.assign(context=lambda x:format_docs(retriever.invoke(x["question"])))|
        prompt|
        llm|
        StrOutputParser()
    )


    if not with_history:
        return base_chain

    chain_with_history = RunnableWithMessageHistory(base_chain,
    get_session_history,input_messages_key="question",
    history_messages_key="history")

    return chain_with_history

def build_agent():
    llm = get_llm()
    agent = create_react_agent(model = llm,tools = all_tools)
    return agent
