import argparse
import json
import sys
import time

import httpx

DEFAULT_URL = "http://127.0.0.1:8000"


def login(client: httpx.Client, base_url: str, username: str) -> str:
    r = client.post(f"{base_url}/login", json={"username": username})
    if r.status_code != 200:
        print(f"登录失败 {r.status_code}: {r.text}")
        sys.exit(1)
    return r.json()["access_token"]


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 /chat/stream 的 SSE 输出")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--username", default="tester")
    parser.add_argument("--session", default="user_004")
    parser.add_argument("--question", default="请简单介绍一下你自己")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--raw", action="store_true", help="同时打印原始 SSE 帧")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    try:
        with httpx.Client(timeout=10.0) as client:
            token = login(client, base_url, args.username)
    except httpx.ConnectError:
        print(f"❌ 连不上 {base_url}，服务启动了吗？")
        return 2
    except httpx.TimeoutException:
        print(f"❌ 连接 {base_url} 超时")
        return 2

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "Accept": "text/event-stream",
    }
    payload = {"question": args.question, "session_id": args.session}

    print(f"POST {base_url}/chat/stream")
    print(f"session_id = {args.session}")
    print(f"question   = {args.question}")
    print("AI：", end="", flush=True)

    started = time.perf_counter()
    first_token_at = None
    deltas = 0
    done = False
    error_msg = None
    unknown = []

    try:
        with httpx.stream(
            "POST",
            f"{base_url}/chat/stream",
            json=payload,
            headers=headers,
            timeout=args.timeout,
        ) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", "replace")
                print(f"\n❌ 请求失败 {resp.status_code}: {body[:500]}")
                return 1

            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if args.raw:
                    print(f"\n[raw] {raw}")

                if raw == "[DONE]":
                    done = True
                    continue

                try:
                    frame = json.loads(raw)
                except json.JSONDecodeError:
                    unknown.append(raw)
                    continue

                ftype = frame.get("type")
                if ftype == "delta":
                    if first_token_at is None:
                        first_token_at = time.perf_counter() - started
                    deltas += 1
                    print(frame.get("content", ""), end="", flush=True)
                elif ftype == "done":
                    done = True
                elif ftype == "error":
                    error_msg = frame.get("message", "未知错误")
                    break
                else:
                    unknown.append(raw)
    except httpx.ReadTimeout:
        print(f"\n❌ 读取超时（{args.timeout}s）：模型或 Redis 可能卡住了")
        return 1
    except httpx.ConnectError:
        print(f"\n❌ 流式请求连不上 {base_url}")
        return 2

    elapsed = time.perf_counter() - started
    print()
    print("-" * 64)

    if error_msg:
        print(f"❌ 服务端返回错误帧：{error_msg}")
        return 1

    if deltas == 0:
        print("❌ 收到 0 个 delta 帧 —— 服务端没有吐出任何文字")
        print("   按这个顺序排查：")
        print("   1) 服务端重启了吗？uvicorn 默认不热加载，改完代码必须重启进程")
        print("      （看 logs/app.log 最近有没有一条新的『服务启动完成』）")
        print("   2) 日志里有没有『流式 0 帧输出』？有的话说明是服务端过滤/模型没开流")
        print("   3) src/rag_graph.py 里主对话模型是不是 get_llm(streaming=True)？")
        print("   4) 加 --raw 再跑一次，看帧格式是不是还是老的 data: [DONE]")
        return 1

    print(f"✅ 收到 {deltas} 个 delta 帧")
    print(f"   首字延迟 {first_token_at:.2f}s | 总耗时 {elapsed:.2f}s")
    if not done:
        print("⚠️ 没有收到 done 帧，连接可能被中途断开")
    if unknown:
        print(f"⚠️ 有 {len(unknown)} 个无法识别的帧，例如：{unknown[0][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
