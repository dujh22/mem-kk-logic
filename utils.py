import argparse  # 用于解析命令行参数
import json  # 用于处理JSON数据
import os  # 用于操作系统相关功能
import numpy as np  # 用于数值计算
import pandas as pd  # 用于数据处理
import random  # 用于生成随机数
import torch  # PyTorch深度学习框架
import time  # 用于时间相关操作
import datasets  # Hugging Face数据集库

def load_jsonl(file_path):
    """
    加载JSONL文件并返回记录列表
    Args:
        file_path: JSONL文件路径
    Returns:
        包含所有记录的列表
    """
    records = []
    with open(file_path, "r") as file:
        for line in file:
            records.append(json.loads(line))  # 解析每行JSON数据并添加到列表中
    return records

def write_jsonl(output_file, data):
    """
    将数据写入JSONL文件
    Args:
        output_file: 输出文件路径
        data: 要写入的数据列表
    """
    with open(output_file, "w") as file:
        for item in data:
            json_line = json.dumps(item)  # 将数据项转换为JSON字符串
            file.write(json_line + "\n")  # 写入文件并添加换行符

def batch_decode_vllm(llm, prompts, batch_size=32):
    """
    使用vLLM进行批量解码
    Args:
        llm: vLLM模型实例
        prompts: 要处理的提示列表
        batch_size: 每批处理的提示数量
    Returns:
        生成的响应列表
    """
    from vllm import SamplingParams  # 导入vLLM采样参数类

    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]  # 获取当前批次的提示
        sampling_params = SamplingParams(max_tokens=llm.max_tokens, temperature=0)  # 设置采样参数
        outputs = llm.model.generate(
            batch_prompts, sampling_params
        )  # 生成响应
        responses = [output.outputs[0].text for output in outputs]  # 提取生成的文本
        all_responses.extend(responses)  # 将响应添加到结果列表
    return all_responses

def init_seed(seed=42):
    """
    初始化随机种子以确保结果可重现
    Args:
        seed: 随机种子值，默认为42
    """
    random.seed(seed)  # 设置Python随机数种子
    np.random.seed(seed)  # 设置NumPy随机数种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机数种子
    torch.random.manual_seed(seed)  # 设置PyTorch随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 如果可用，设置CUDA随机数种子

def load_llm(args):
    """
    根据参数加载语言模型
    Args:
        args: 包含模型配置的参数对象
    Returns:
        加载的语言模型实例
    """
    if "openai" in args.model:  # 如果是OpenAI模型
        from models.openai import ChatGPT
        llm = ChatGPT(model_path=args.model, max_tokens=args.max_token)
    elif "anthropic" in args.model:  # 如果是Anthropic模型
        from models.anthropic import Claude
        llm = Claude(model_path=args.model, max_tokens=args.max_token)
    else:  # 如果是其他模型（如Hugging Face模型）
        from models.hf import CasualLM
        llm = CasualLM(
            model_path=args.model,
            arch=args.arch,
            use_vllm=args.use_vllm,
            max_tokens=args.max_token,
        )
    return llm

def load_eval_records(args, subject):
    """
    加载评估记录
    Args:
        args: 包含配置的参数对象
        subject: 评估主题
    Returns:
        加载的数据集记录
    """
    if args.problem_type != "clean":  # 如果不是clean类型的问题
        records = datasets.load_dataset('K-and-K/perturbed-knights-and-knaves',data_files=f"{args.split}/{args.problem_type}/{subject}.jsonl")["train"] 
    else:  # 如果是clean类型的问题
        records = datasets.load_dataset('K-and-K/knights-and-knaves',data_files=f"{args.split}/{subject}.jsonl")["train"]
    return records