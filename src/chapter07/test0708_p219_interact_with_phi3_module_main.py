from src.chapter07.test0708_p219_interact_with_phi3_module import query_model

# 测试案例-与phi3模型交互
model = "phi3"
result = query_model("what do llamas eat?", model)
print("result = ", result)
