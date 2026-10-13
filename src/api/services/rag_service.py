"""
RAG 服务层

职责：
- 封装 rag_graph 的调用逻辑
- 只负责业务，不涉及 HTTP（路由层的事）
- 方便单元测试（不需要启动 FastAPI 就能测）
"""

from typing import Any

from langchain_core.runnables import Runnable


class RagService:
    """
    RAG 服务

    持有的东西：
    - graph：已编译的 LangGraph（含 checkpointer）
    - retriever：混合检索器（可选，未来做纯检索接口时用）
    """

    def __init__(self, graph: Runnable):
        """
        参数：
        - graph：由 build_rag_graph(retriever).compile(checkpointer=...) 得到的已编译图
        """
        self.graph = graph

    async def ask(
        self,
        question: str,
        thread_id: str,
        image_url: str = "",
    ) -> dict[str, Any]:
        """
        向 RAG 图提问

        参数：
        - question：用户问题
        - thread_id：会话 ID（LangGraph 的 checkpointer 只认 thread_id）
        - image_url：图片的 data URL（可选，多模态场景）

        返回：
        - {"answer": "...", "route": "knowledge" | "chat"}
        """
        # 1. 构造 LangGraph 的调用 config
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # 2. 构造初始 state（只传必要的字段）
        #    注意：messages 不传，由 checkpointer 从 Redis 恢复
        initial_state = {
            "question": question,
            "image_url": image_url,
            "context": "",
            "answer": "",
        }

        # 3. 调用图（异步版本，不阻塞事件循环）
        result = await self.graph.ainvoke(initial_state, config=config)

        # 4. 提取结果
        return {
            "answer": result.get("answer", ""),
            "route": result.get("route"),
        }