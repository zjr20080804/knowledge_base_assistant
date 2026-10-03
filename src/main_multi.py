from src.multi_agent import build_multi_agent_graph


def main():
    graph = build_multi_agent_graph().compile()

    task = "分析《猫和老鼠》手游，写一段 200 字的简介"
    print(f"任务：{task}\n")

    result = graph.invoke({
        "task": task,
        "messages": [],
        "next": "",
    })

    print("=== 执行过程 ===")
    for m in result["messages"]:
        name = getattr(m, "name", None) or m.type
        print(f"\n[{name}]")
        print(m.content[:300])


if __name__ == "__main__":
    main()