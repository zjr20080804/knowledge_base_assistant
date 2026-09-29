import os
from src.plan_execute import build_plan_graph
from src.loader_splitter import load_split
from src.embedding import create_vectorstore, load_vectorstore
from src.config import vectorstore_dir


def main():
    # 准备向量库
    chunks = load_split()
    if not os.path.exists(vectorstore_dir):
        vectorstore = create_vectorstore(chunks)
    else:
        vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 建图
    graph = build_plan_graph(retriever).compile()

    task = "对比《猫和老鼠》手游里的杰瑞和汤姆两个角色，给新手推荐一个"
    print(f"任务：{task}\n")

    result = graph.invoke({
        "task": task,
        "plan": [],
        "past_steps": [],
        "response": "",
    })

    print("=== 执行结果 ===")
    print(f"\n最终回答：{result['response']}")

    print("\n=== 执行过程 ===")
    for i, (step, res) in enumerate(result["past_steps"], 1):
        print(f"\n{i}. 步骤：{step}")
        print(f"   结果：{res[:150]}...")


if __name__ == "__main__":
    main()