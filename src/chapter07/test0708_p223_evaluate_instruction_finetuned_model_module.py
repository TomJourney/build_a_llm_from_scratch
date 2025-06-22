from tqdm import tqdm

from src.chapter07.test0702_p189_format_input_to_alpaca_module import format_input_to_alpaca
from src.chapter07.test0708_p219_interact_with_phi3_module import query_model


# 测试案例-评估指令微调后的大模型
# 与phi3模型交互-生成模型分数
def generate_model_scores(json_data, json_key, model="phi3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Give the input `{format_input_to_alpaca(entry)}`"
            f"And correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score."
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"could not convert score:{score}")
            continue
    return scores
