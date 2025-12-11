import os
import time
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import tqdm
from huggingface_hub import InferenceClient

# 从环境变量安全读取 Token
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("【致命错误】: 环境变量 HF_TOKEN 未设置。请设置此变量以运行脚本。")
    exit(1)

client = InferenceClient(token=HF_TOKEN)

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def translate_text(text, task_type="abstract"):

    model_name = "deepseek-ai/DeepSeek-V3.2:fastest"

    if task_type == "title":
        system_prompt = (
            "你是一名具备高等学术素养的专业英译中译者，熟悉学术写作规范与专业术语使用标准。"
            "当用户输入英文论文标题时，请在充分理解原意的基础上，将其翻译为严谨、准确、凝练、且符合中文学术书写风格的标题。"
        )
    else:
        system_prompt = (
            "你是一名专业的英译中学术翻译专家，精通科技文献语言风格与学术表达规范，当用户提供英文摘要时。"
            "请在忠实原意的前提下，将其译为自然流畅、语义严谨、逻辑清晰且符合正式学术中文写作风格的译文。"
            "如遇专业术语或专有名词，请优先使用通用权威译法，必要时保留英文原词于括号中。"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请翻译以下文本: {text}"}
    ]


    try:
        response = client.chat_completion(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=512,
        )
        if response and response.choices and response.choices[0].message:
            return response.choices[0].message.content
        return None

    except Exception as e:
        print(f"\n[API 错误 ({task_type})，尝试重试]: {e}")
        raise e


def main():
    input_file = "iccv2025.csv"
    output_file = "test.csv"

    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    df = pd.read_csv(input_file)
    print(f"成功读取 {len(df)} 条数据。")

    required_cn_cols = ['title_cn', 'abstract_cn']
    for col in required_cn_cols:
        if col not in df.columns:
            df[col] = None

    if os.path.exists(output_file):
        try:
            print("发现已存在的 result.csv，正在尝试合并进度...")
            df_existing = pd.read_csv(output_file)

            for col in required_cn_cols:
                if col in df_existing.columns:
                    df[col] = df[col].combine_first(df_existing[col])
            print("进度合并完成。")
        except Exception as e:
            print(f"读取现有进度失败，将重新开始: {e}")

    remaining_indices = df[df['abstract_cn'].isnull() | df['title_cn'].isnull()].index

    if len(remaining_indices) == 0:
        print("所有摘要和标题都已翻译完成！")
        return

    total_rows = len(df)

    tqdm_iterator = tqdm.tqdm(
        remaining_indices,
        total=total_rows,
        initial=total_rows - len(remaining_indices),
        desc="Translating Papers",
    )

    for index in tqdm_iterator:
        row = df.loc[index]
        current_title = row.get('title')
        current_abstract = row.get('abstract')
        fail_marker = "[翻译失败 (5次重试失败)]"

        if pd.isna(row['title_cn']) and not pd.isna(current_title):
            try:
                title_cn_res = translate_text(current_title, task_type="title")
                df.at[index, 'title_cn'] = title_cn_res if title_cn_res else "[标题翻译失败]"
            except Exception as e:
                df.at[index, 'title_cn'] = fail_marker
                tqdm_iterator.set_description(f"错误: 标题在 {index + 1} 行失败，跳过。")
                print(f"\n[致命错误]: 标题在第 {index + 1} 行连续 5 次重试失败。")

        if pd.isna(row['abstract_cn']) and not pd.isna(current_abstract):
            try:
                cn_res = translate_text(current_abstract, task_type="abstract")
                if cn_res:
                    safe_cn_res = cn_res.replace('"', "'").replace('\n', ' ').replace(',', '，')
                    df.at[index, 'abstract_cn'] = safe_cn_res
                else:
                    df.at[index, 'abstract_cn'] = "[摘要翻译失败]"
            except Exception as e:
                df.at[index, 'abstract_cn'] = fail_marker
                tqdm_iterator.set_description(f"错误: 摘要在 {index + 1} 行失败，跳过。")
                print(f"\n[致命错误]: 摘要在第 {index + 1} 行连续 5 次重试失败。")

        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        time.sleep(3)

    print(f"\n全部完成！结果已保存在 {output_file}")

if __name__ == "__main__":
    if HF_TOKEN:
        main()