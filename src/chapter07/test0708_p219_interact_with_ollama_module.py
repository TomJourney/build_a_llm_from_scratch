import json
import urllib.request

# 模块-与ollma运行的模型交互
def query_model(
        prompt,
        model="llama3",
        url="http://localhost:11434/apoi/chat"
):
    # 创建字典格式的数据
    data = {
        "model": model,
        "messsage": [
            {"role": "user", "content": prompt}
        ],
        # 设置种子得到确定性的返回结果
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # 把字典变成json格式的字符串，并编码为字节
    payload = json.dumps(data).encode("utf-8")
    # 创建一个请求对象，把方法设置为post，并加入必要的请求头
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # 发送请求并捕获模型回复
    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data
