from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_agent import build_rag_chain
from src.embedding import load_vectorstore, get_retriever

app = FastAPI(title="知识库问答 API")

# 全局变量（启动时赋值）



@app.on_event("startup")
def startup():
    """启动时加载向量库和链（只做一次）"""
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)
    app.state.rag_chain = build_rag_chain(retriever, with_history=True)
    print("✅ RAG 链已加载")


class ChatRequest(BaseModel):
    """请求格式"""
    question: str
    session_id: str = "user_001"


@app.post("/chat")
async def chat(req: ChatRequest):
    """问答接口"""
    result = await (app.state.rag_chain.ainvoke(
        {"question": req.question},
        config={"configurable": {"session_id": req.session_id}}
    ))
    return {"answer": result}