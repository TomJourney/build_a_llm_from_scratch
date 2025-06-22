from src.chapter07.test0708_p219_interact_with_ollama_module import query_model

# 测试案例-与ollma运行的模型交互
model = "llama3"
result = query_model("What do Llamas eat?", model)
print("result = ", result)
