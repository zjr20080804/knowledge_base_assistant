from langchain_core.callbacks import BaseCallbackHandler

class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self,serializable,prompts,**kwargs):
        print("LLM 开始")

    def on_llm_end(self,response,**kwargs):
        print("LLM 结束")
        token_usage = response.llm_output.get("token_usage", {})
        if token_usage:
            print(f"LLM 调用消耗 {token_usage.get('total_tokens', 0)} 个 token")
            print(f"LLM 输入消耗 {token_usage.get('prompt_tokens', 0)} 个token")
            print(f"LLM 输出消耗 {token_usage.get('completion_tokens', 0)} 个token")

    def on_llm_error(self,error,**kwargs):
        print(f"LLM 错误:", error)


