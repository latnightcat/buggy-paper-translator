import os
import time
import pandas as pd
# 引入 tenacity 和 tqdm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import tqdm

# 引入 huggingface_hub
from huggingface_hub import InferenceClient

# ================= 配置与初始化区域 =================
# 🚨 关键修正 1：从环境变量安全读取 Token
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    # 如果 Token 未设置，则打印错误并退出
    print("【致命错误】: 环境变量 HF_TOKEN 未设置。请设置此变量以运行脚本。")
    # 为了避免程序直接崩溃，这里可以返回一个非零退出码
    exit(1)

# 初始化客户端
client = InferenceClient(token=HF_TOKEN)


# 定义重试策略：等待时间指数增长 (min=4s, max=60s)，最多重试 5 次
# 🌟 关键修正 2：应用 tenacity 装饰器
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    # 捕获所有异常并重试，但会跳过已知的致命错误 (如 401/403/404)
    retry=retry_if_exception_type(Exception),
    reraise=True  # 确保在所有重试都失败后，异常会被抛出到 main 函数
)
def translate_text(text, task_type="abstract"):
    """通用的文本翻译函数，带有自动重试机制"""
    model_name = "openai/gpt-oss-120b:fastest"  # 当前选择的免费模型

    # 修正 Prompt
    if task_type == "title":

        system_prompt = "你是一位专业的学术翻译助手。请将用户提供的英文标题翻译成简洁、准确的中文标题。"
    else:
        system_prompt = "你是一位专业的学术翻译助手。请将用户提供的英文摘要翻译成流畅、学术风格的中文。"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请翻译以下文本: {text}"}
    ]

    # 尝试调用 API
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
        # 记录错误，并将异常抛出给 tenacity
        print(f"\n[API 错误 ({task_type})，尝试重试]: {e}")
        raise e


def main():
    input_file = "iccv2025.csv"
    output_file = "result.csv"

    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    # 1. 读取原始文件
    df = pd.read_csv(input_file)
    print(f"成功读取 {len(df)} 条数据。")

    # 2. 初始化翻译列
    required_cn_cols = ['title_cn', 'cn_abstract']
    for col in required_cn_cols:
        if col not in df.columns:
            df[col] = None

    # 3. 智能断点续传逻辑
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

    # 4. 计算需要翻译的索引 (只要有一个翻译列为空，就需要处理)
    # 使用位运算符 | (OR)
    remaining_indices = df[df['cn_abstract'].isnull() | df['title_cn'].isnull()].index

    if len(remaining_indices) == 0:
        print("所有摘要和标题都已翻译完成！")
        return

    total_rows = len(df)

    # 🌟 关键修正 3：使用 tqdm 包装迭代器以显示进度条
    tqdm_iterator = tqdm.tqdm(
        remaining_indices,
        total=total_rows,
        initial=total_rows - len(remaining_indices),  # 初始已完成数量
        desc="Translating Papers"
    )

    # 5. 遍历并翻译
    for index in tqdm_iterator:
        row = df.loc[index]

        current_title = row.get('title')
        current_abstract = row.get('abstract')

        # 错误标记，用于在所有重试失败后使用
        fail_marker = "[翻译失败 (5次重试失败)]"

        # 翻译标题 (如果 title_cn 为空)
        if pd.isna(row['title_cn']) and not pd.isna(current_title):
            try:
                title_cn_res = translate_text(current_title, task_type="title")
                df.at[index, 'title_cn'] = title_cn_res if title_cn_res else "[标题翻译失败]"
            except Exception as e:
                df.at[index, 'title_cn'] = fail_marker
                tqdm_iterator.set_description(f"错误: 标题在 {index + 1} 行失败，跳过。")
                # 记录错误并继续下一行
                print(f"\n[致命错误]: 标题在第 {index + 1} 行连续 5 次重试失败。")

        # 翻译摘要 (如果 cn_abstract 为空)
        if pd.isna(row['cn_abstract']) and not pd.isna(current_abstract):
            try:
                cn_res = translate_text(current_abstract, task_type="abstract")
                # 净化文本，防止CSV格式被破坏
                if cn_res:
                    safe_cn_res = cn_res.replace('"', "'").replace('\n', ' ').replace(',', '，')
                    df.at[index, 'cn_abstract'] = safe_cn_res
                else:
                    df.at[index, 'cn_abstract'] = "[摘要翻译失败]"
            except Exception as e:
                df.at[index, 'cn_abstract'] = fail_marker
                tqdm_iterator.set_description(f"错误: 摘要在 {index + 1} 行失败，跳过。")
                # 记录错误并继续下一行
                print(f"\n[致命错误]: 摘要在第 {index + 1} 行连续 5 次重试失败。")

        # 6. 实时保存
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        # 避免请求过快
        time.sleep(2)

    print(f"\n全部完成！结果已保存在 {output_file}")


if __name__ == "__main__":
    # 在程序开始时检查 Token
    if HF_TOKEN:
        main()