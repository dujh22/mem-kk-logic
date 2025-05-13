# 关于大语言模型在逻辑推理中的记忆性研究

本仓库提供了论文《On Memorization of Large Language Models in Logical Reasoning》的 PyTorch 实现。

## 简介

在本工作中，我们研究了大语言模型（LLM）在推理任务中的记忆性。

- 我们为推理任务提出了一种记忆性度量方法，并基于“骑士与骗子”（Knights and Knaves, K&K）谜题动态生成了逻辑推理基准。
- LLM 在微调后能够在训练集上取得很高的准确率，但在谜题稍作扰动后表现大幅下降，表明模型在解决这些训练谜题时严重依赖记忆。
- 另一方面，微调也能持续提升泛化性能。通过扰动测试、跨难度迁移、模型内部探查以及用错误答案微调等深入分析，表明 LLM 在记忆训练数据的同时，也学会了在 K&K 谜题上进行推理。
- 最后，我们使用基于谜题和基于模型的指标，对通过推理和通过记忆解决的谜题进行了分类。

## 🛠️ 安装

```bash
conda env create -f environment.yml
conda activate kk
```

## 📝 合成数据

### 选项1：使用 Huggingface 数据集

在评测/微调时，我们直接从 huggingface 导入数据集：

```python
import datasets
datasets.load_dataset('K-and-K/knights-and-knaves', 'test')
datasets.load_dataset('K-and-K/perturbed-knights-and-knaves', 'test')
```

### 选项2：本地生成数据

要为{2,3,4,5,6,7,8}人谜题生成 K&K 数据并划分训练/测试集，运行：

```bash
python data_prep/data_gen_kk.py
```

本地扰动数据也会被生成，数据将保存在 `data` 目录下。

此外，还可以用来生成错误答案数据和错误 CoT 数据（包括一步错误和打乱的 CoT 步骤）。

## 🤖 评测

常用评测参数：

| 参数               | 示例                                                                                                                             | 说明                                         |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `--max_token`    | `2048`                                                                                                                         | 最大 token 数。                              |
| `--split`        | `train`, `test`                                                                                                              | 选择评测用的数据集划分。                     |
| `--limit`        | `100`                                                                                                                          | 限定评测样本数量。                           |
| `--ntrain`       | `0`, `1`                                                                                                                     | 0-shot/少样本提示的演示数量。                |
| `--problem_type` | `clean`, `perturbed_statement`, `perturbed_leaf`, `random_pair`, `reorder_statement`, `uncommon_name`, `flip_role` | 问题类型，支持多种扰动。                     |
| `--eval_nppl`    | `2`,`3`,`4`,`5`,`6`,`7`,`8`                                                                                        | K&K 谜题中的人数。不设置则评测所有人数任务。 |
| `--vllm`         | `true`                                                                                                                         | 启用 VLLM 加速开源模型推理。                 |
| `--model`        | `openai/gpt-4o-mini-2024-07-18`                                                                                                | 被评测的模型，支持开源和闭源模型。           |

### 测试集评测

对每个 K&K 任务，评测全部测试样本（100 个）。

1/0-shot、有/无 CoT 下评测：

```bash
bash scripts/eval/run_test.sh
```

0-shot、无 CoT 下对两种数学级扰动类型（`perturbed_statement`, `perturbed_leaf`）评测：

```bash
bash scripts/eval/eval_test_pertub.sh
```

### 训练集评测

微调后（见“4. 微调”），在训练集上评测。

对微调后的 GPT-4o-mini 评测前 100 个样本，对开源模型评测全部样本。

0-shot、无 CoT 下评测：

```bash
bash scripts/eval/eval_train.sh
```

训练集扰动样本评测：

0-shot、无 CoT 下对 6 种扰动类型（`perturbed_statement`, `perturbed_leaf`, `random_pair`, `reorder_statement`, `uncommon_name`, `flip_role`）评测：

```bash
bash scripts/eval/eval_train_pertub.sh
```

#### 闭源模型评测

设置 API key：

```bash
export OPENAI_API_KEY='your-api-key-here'
export ANTHROPIC_API_KEY='your-api-key-here'
```

OpenAI/Anthropic 直接提示示例：

```bash
bash scripts/eval/gpt4omini_direct.sh
bash scripts/eval/claude-sonet.sh
```

CoT 提示评测：

```bash
bash scripts/eval/gpt4omini_cot.sh
```

## 🚗 微调

### 直接微调

直接在答案上微调（无 CoT）：

```bash
bash scripts/ft/ft_lm3.sh
```

### CoT 微调

在 CoT 上微调：

```bash
bash scripts/ft/ft_lm3_cot.sh
```

可在上述脚本中更改保存模型路径 `output_dir`。

#### 合并微调 adapter 和基础模型

加载微调保存的 adapter 和基础模型，然后合并保存：

```bash
bash scripts/ft/merge_adapter.sh
```

请根据需要更改脚本中的 `base_model_path`、`adapter_path`、`base_model_path`。

#### 闭源模型微调

闭源模型微调遵循 [OpenAI finetuning API](https://platform.openai.com/docs/guides/fine-tuning)。

## 🔍 探查

要探查模型内部表征，请在脚本中更新模型路径和谜题人数：

```bash
bash scripts/probe/run.sh
```

## 🗃️ 样本分类

对一致解答和非一致解答的谜题进行分类。

更新模型路径，并为每个训练样本提供一致解答与否的二元标签，然后运行：

基于谜题指标分类：

```bash
bash scripts/mem_classify/model_indicator.sh
```

基于模型指标分类：

```bash
bash scripts/mem_classify/puzzle_indicator.sh
```

## 📚 引用

如果本工作对您有帮助，请引用如下：

```bibtex
@article{xie2024memorization,
title={On Memorization of Large Language Models in Logical Reasoning}, 
author={Chulin Xie and Yangsibo Huang and Chiyuan Zhang and Da Yu and Xinyun Chen and Bill Yuchen Lin and Bo Li and Badih Ghazi and Ravi Kumar},
year={2024},
eprint={2410.23123},
archivePrefix={arXiv},
primaryClass={cs.CL},
url={https://arxiv.org/abs/2410.23123}, 
}
```

## 📖 问题

如有建议或需要帮助复现结果，请提交 issue 或 pull request，或发送邮件至 chulinx2@illinois.edu。
