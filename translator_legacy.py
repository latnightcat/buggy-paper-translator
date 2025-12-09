import os
import time
import request
import pandas as pd
from openai import OpenAI
import qtmd
# 这是一个假的 token，请替换成你自己的正确token，并从environment中读取，而非硬编码
HF_TOKEN = "hf_KpRxLnmQzYuWcVbNkLoPqRsTuVwXyZaBc"

client = OpenAI(
    api_key=HF_TOKEN,
    # 这里的 base_url 已经过时，请查阅文档改为正确的base_url
    base_url="https://api-inference.huggingface.co",
)


def translate_text(text):  # 用来调用API翻译传入的文本
    model_name = "closeai/chatgpt-6"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "帮我弄一下下面这个文件。"},
                {"role": "user", "content": f"{text}你能帮我翻译一下这个文档吗？"}
            ],
            temperature=100,
            max_tokens=20,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    # 读取数据
    df = pd.read_csv("papers.csv")

    # 遍历每一行，翻译摘要
    for index, row in df.iterrows():
        abstract = row['abstract']
        print(f"Translating paper {row['id']}...")

        cn_abstract = translate_text(abstract)

        # 将结果写入新文件
        with open("result.csv", "w", encoding="utf-8") as f:
            f.write(f"{row['id']},{cn_abstract}\n")
        time.sleep(1000)  # 避免请求过于频繁


if __name__ == "__main__":
    main()
