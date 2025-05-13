# 导入必要的库
import copy
import os
import sys
import importlib
import pprint

# 添加当前目录到系统路径
module_path = os.path.abspath('.')
if not module_path in sys.path:
    sys.path.append(module_path)
# 导入自定义库并重新加载
import lib_kk
importlib.reload(lib_kk)
import numpy as np
import json
import os
import time
import random  # 用于生成随机数

def init_seed(seed=42):
    """
    初始化随机种子以确保结果可重现
    Args:
        seed: 随机种子值，默认为42
    """
    random.seed(seed)  # 设置Python随机数种子
    np.random.seed(seed)  # 设置NumPy随机数种子

# 初始化随机种子
init_seed(42)

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



# 将整数转换为字符串
def convert_int_to_str(data):
    return str(data)

# 合并训练数据
def combine_train_data(data_folder,file_config, output_name):
    result_records=[]
    for config in file_config:
        file_path = os.path.join(data_folder, config[0])
        records = load_jsonl(file_path)
        print(f"Loaded {len(records)} records from {file_path}")
        if config[1] < len(records):
            records = records[:config[1]]
        result_records.extend(records)
    output_file=os.path.join(data_folder, output_name)
    write_jsonl(output_file, result_records)

# 格式化解决方案文本
def format_solution_text(ans):
    # 移除"and"和句点
    gold = ans.replace(" and ", "").replace(".", "")
    gold_conditions=gold.split(",")
    reformat_gold_conditions=[]
    for condition in gold_conditions:
        # 移除首尾空格
        gold_condition=condition.strip()
        reformat_gold_conditions.append(gold_condition)

    # 格式化语句，添加编号
    formatted_statements = "\n".join([f"({i+1}) {reformat_gold_conditions[i]}" for i in range(len(reformat_gold_conditions))])
    return formatted_statements

# 生成问题
def generate_problems(n_problems, n_people, gen_perturb=True):
    '''
    n_problems: 问题数量
    n_people: 人数
    gen_perturb: 是否生成扰动问题
    '''
    problems = []
    problem_seed=1234
    start_time = time.time()
    # 创建问题采样器
    problem_sampler = lib_kk.KKProblemSampler(problem_seed, n_people=n_people)
    # 采样有效问题
    problems = problem_sampler.sample_valid_problems(n_problems)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print(f'{len(problems)} valid problems generated')
    
    if gen_perturb:
        # 生成语句扰动的问题
        start_time = time.time()
        per_stat = problem_sampler.perturb_problems(problems, perturb_type='statement', num_perturb=1)
        perturbed_problems_statement = [item for inner_list in per_stat for item in inner_list]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f'{len(perturbed_problems_statement)} perturbed (statement) problems generated')

        # 生成叶子节点扰动的问题
        start_time = time.time()
        per_stat = problem_sampler.perturb_problems(problems, perturb_type='leaf', num_perturb=1)
        perturbed_problems_leaf = [item for inner_list in per_stat for item in inner_list]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f'{len(perturbed_problems_leaf)} perturbed (leaf) problems generated')

        return problems, perturbed_problems_statement, perturbed_problems_leaf
    else:
        return problems, None, None

# 生成错误问题
def generate_wrong_problems(n_problems, n_people):
    problems = []
    problem_seed=1234
    start_time = time.time()
    # 创建问题采样器
    problem_sampler = lib_kk.KKProblemSampler(problem_seed, n_people=n_people)
    # 采样无效问题
    problems = problem_sampler.sample_invalid_problems(n_problems)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print(f'{len(problems)} valid problems with wrong answers generated')

    return problems



def generate_formatted_problem(problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair,uncommon_name=False, reorder_statement=False):
    data =[]
    problem_seed=1234
    for i in range(item_start_idx, item_start_idx+ num_samples):
        problem= problems[i]
        if problem is None:
            continue

        # 创建问题格式化器
        formatter_seed= problem_seed+i
        formatter = lib_kk.KKProblemFormatter(formatter_seed, problem)
        # 格式化问题
        formatted_problem = formatter.format_problem(random_knight_knave_pairs=random_knight_knave_pairs, 
                                            flip_knight_knave_pair=flip_knight_knave_pair, 
                                            random_names=True, random_saying_template=True,
                                            uncommon_name=uncommon_name, reorder_statement=reorder_statement)
        
        # 生成思维链
        chain_of_thoughts = lib_kk.generate_chain_of_thoughts(problem['statements'])
        # 格式化思维链（不重复声明）
        header, steps, footer = lib_kk.format_chain_of_thoughts(problem, formatted_problem, chain_of_thoughts,
                                        repeat_claim_for_assumption=False, repeat_claim_for_contradiction=False)
        
        # 格式化思维链（重复声明）
        repeat_header, repeat_steps, repeat_footer = lib_kk.format_chain_of_thoughts(problem, formatted_problem, chain_of_thoughts,
                                        repeat_claim_for_assumption=True, repeat_claim_for_contradiction=True)
        
        # 构建数据项
        item= copy.deepcopy(formatted_problem)
        item["solution_text_format"]= format_solution_text(item["solution_text"])
        item["cot_head"]=header
        item["cot_repeat_steps"]=repeat_steps
        item["cot_foot"]=footer
        item["statements"]=convert_int_to_str(problem["statements"]) # 将0/1转换为"0"/"1"以便后续JSON加载
        item["index"] = i
        
        data.append(item)
    return data


# 生成数据
def generate_data(num_samples_test, num_samples_train, num_samples_val, n_people):
    # 计算总问题数
    num_problems=num_samples_test+num_samples_train+num_samples_val

    # 生成问题及其扰动版本
    clean_problems, perturbed_problems_statement, perturbed_problems_leaf = generate_problems(num_problems, n_people, gen_perturb=True)
    problems_dict={
        "clean": clean_problems,
        "perturbed_statement": perturbed_problems_statement, # 语句扰动
        "perturbed_leaf": perturbed_problems_leaf # 叶子节点扰动
    }

    # 设置默认参数
    random_knight_knave_pairs=False
    flip_knight_knave_pair=False
    uncommon_name=False
    
    # 对每种问题类型进行处理
    for problem_type, problems in problems_dict.items():
        item_start_idx=0
        # 对每个数据集划分进行处理
        for (split, num_samples) in [("test", num_samples_test), ("train", num_samples_train), ("val", num_samples_val)]:
            if num_samples==0:
                continue 
            # 生成格式化的问题
            data= generate_formatted_problem(problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair, uncommon_name)

            # 构建配置名称
            config=f"people{n_people}_num{num_samples}"

            if random_knight_knave_pairs:
                config +="_random_pair"
            if flip_knight_knave_pair:
                config +="_flip_role"
            if uncommon_name:
                config +="_uncommon_name"
            
            # 创建输出目录并保存数据
            output_folder=f"data/{split}/{problem_type}"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f'{config}.jsonl')
            with open(output_file, 'w') as file:
                for item in data:
                    json_line = json.dumps(item)
                    file.write(json_line + '\n')
            print(f"Data has been written to {output_file}")
            item_start_idx+=num_samples

# 生成语言扰动的数据
def generate_data_language_perturb(num_samples_test, num_samples_train, num_samples_val, n_people):
    num_problems=num_samples_test+num_samples_train+num_samples_val

    # 只生成干净的问题，不生成扰动
    clean_problems, _, _ = generate_problems(num_problems, n_people, gen_perturb=False)
    problems_dict={
        "clean": clean_problems, 
    }
    # 定义不同的语言扰动类型
    perturb_list=["random_pair", "flip_role", "uncommon_name", "reorder_statement"] # 随机角色，翻转角色，不常见名字，重新排序语句

    for perturb_type in perturb_list:
        random_knight_knave_pairs=False
        flip_knight_knave_pair=False
        uncommon_name=False
        reorder_statement=False
        # 根据扰动类型设置参数
        if perturb_type=="random_pair":
            random_knight_knave_pairs=True
        elif perturb_type=="flip_role":
            flip_knight_knave_pair=True
        elif perturb_type=="uncommon_name":
            uncommon_name=True
        elif perturb_type=="reorder_statement":
            reorder_statement=True
    
        item_start_idx=0
        for (split, num_samples) in [("test", num_samples_test), ("train", num_samples_train), ("val", num_samples_val)]:
            if num_samples==0:
                continue 
            # 生成带有语言扰动的格式化问题
            data= generate_formatted_problem(clean_problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair, uncommon_name,reorder_statement)
           
            config=f"people{n_people}_num{num_samples}"

            
            output_folder=f"data/{split}/{perturb_type}"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f'{config}.jsonl')
            with open(output_file, 'w') as file:
                for item in data:
                    json_line = json.dumps(item)
                    file.write(json_line + '\n')
            print(f"Data has been written to {output_file}")
            item_start_idx+=num_samples


# 生成格式化的错误问题
def generate_formatted_wrong_problem(problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair,uncommon_name=False):
    data =[]
    problem_seed=1234
    for i in range(item_start_idx, item_start_idx+ num_samples):
        problem= problems[i]
        if problem is None:
            continue

        # 创建问题格式化器
        formatter_seed= problem_seed+i
        formatter = lib_kk.KKProblemFormatter(formatter_seed, problem)
        # 格式化问题
        formatted_problem = formatter.format_problem(random_knight_knave_pairs=random_knight_knave_pairs, 
                                            flip_knight_knave_pair=flip_knight_knave_pair, 
                                            random_names=True, random_saying_template=True,
                                            uncommon_name=uncommon_name)
        
        # 构建数据项
        item= copy.deepcopy(formatted_problem)
        item["solution_text_format"]= format_solution_text(item["solution_text"])
        item["cot_head"]="placeholder"
        item["cot_repeat_steps"]= ["placeholder"]
        item["cot_foot"]="placeholder"
        item["statements"]=convert_int_to_str(problem["statements"]) # 将0/1转换为"0"/"1"以便后续JSON加载
        item["index"] = i
        
        data.append(item)
    return data


# 生成错误数据
def generate_wrong_data(num_samples_test, num_samples_train, num_samples_val, n_people):
    num_problems=num_samples_test+num_samples_train+num_samples_val

    # 生成无效问题
    clean_problems = generate_wrong_problems(num_problems, n_people)
    problems_dict={
        "clean": clean_problems,
    }
    random_knight_knave_pairs=False
    flip_knight_knave_pair=False
    uncommon_name=False
    for problem_type, problems in problems_dict.items():
        item_start_idx=0
        for (split, num_samples) in [("test", num_samples_test), ("train", num_samples_train), ("val", num_samples_val)]:
            if num_samples==0:
                continue 
            # 生成格式化的错误问题
            data= generate_formatted_wrong_problem(problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair, uncommon_name)

            config=f"people{n_people}_num{num_samples}"

            if random_knight_knave_pairs:
                config +="_random_pair"
            if flip_knight_knave_pair:
                config +="_flip_role"
            if uncommon_name:
                config +="_uncommon_name"
            
            output_folder=f"data/wrong/{split}/{problem_type}"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f'{config}.jsonl')
            with open(output_file, 'w') as file:
                for item in data:
                    json_line = json.dumps(item)
                    file.write(json_line + '\n')
            print(f"Data has been written to {output_file}")
            item_start_idx+=num_samples




# 生成格式化的错误思维链（CoT）
def generate_formatted_wrong_cot(problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair,uncommon_name=False,  wrong_type="shuffle" ):
    data =[]
    problem_seed=1234
    for i in range(item_start_idx, item_start_idx+ num_samples):
        problem= problems[i]
        if problem is None:
            continue

        # 创建问题格式化器
        formatter_seed= problem_seed+i
        rng = np.random.default_rng(formatter_seed)
        formatter = lib_kk.KKProblemFormatter(formatter_seed, problem)
        # 格式化问题
        formatted_problem = formatter.format_problem(random_knight_knave_pairs=random_knight_knave_pairs, 
                                            flip_knight_knave_pair=flip_knight_knave_pair, 
                                            random_names=True, random_saying_template=True,
                                            uncommon_name=uncommon_name)
        
        # 生成思维链
        chain_of_thoughts = lib_kk.generate_chain_of_thoughts(problem['statements'])
        # 格式化思维链（不重复声明）
        header, steps, footer = lib_kk.format_chain_of_thoughts(problem, formatted_problem, chain_of_thoughts,
                                        repeat_claim_for_assumption=False, repeat_claim_for_contradiction=False)
        
        # 格式化思维链（重复声明）
        repeat_header, repeat_steps, repeat_footer = lib_kk.format_chain_of_thoughts(problem, formatted_problem, chain_of_thoughts,
                                        repeat_claim_for_assumption=True, repeat_claim_for_contradiction=True)
        
        # 如果是shuffle类型，则打乱思维链步骤顺序
        if wrong_type=="shuffle":
            rng.shuffle(repeat_steps)
        # 构建数据项
        item= copy.deepcopy(formatted_problem)
        item["solution_text_format"]= format_solution_text(item["solution_text"])
        item["cot_head"]=header
        item["cot_repeat_steps"]=repeat_steps
        item["cot_foot"]=footer
        item["statements"]=convert_int_to_str(problem["statements"]) # 将0/1转换为"0"/"1"以便后续JSON加载
        item["index"] = i
        
        data.append(item)

    # 如果是replace_one_step类型，则随机替换某一步骤
    if wrong_type=="replace_one_step":
        rng = np.random.default_rng(problem_seed)
        for j , item in enumerate(data): 
            wrong_step_idx=rng.integers(0, len(item["cot_repeat_steps"]))
            original_step=item["cot_repeat_steps"][wrong_step_idx]

            possible_replacements = [i for i in range(len((data))) if j != i]
            
            while True:
                replace_item_idx=  rng.choice(possible_replacements)
                replace_item = data[replace_item_idx]
                replace_step_idx=rng.integers(0, len(replace_item["cot_repeat_steps"]))
                replace_step = replace_item["cot_repeat_steps"][replace_step_idx]
                for name_idx, name in enumerate(replace_item["names"]):
                    replace_step=replace_step.replace(name, item["names"][name_idx])
                if original_step!=replace_step:
                    item["cot_repeat_steps"][wrong_step_idx]=replace_step
                    break 

    return data


# 生成错误思维链（CoT）数据
def generate_wrong_cot_data(num_samples_test, num_samples_train, num_samples_val, n_people, wrong_type="shuffle"):
    num_problems=num_samples_test+num_samples_train+num_samples_val

    # 生成干净的问题
    clean_problems, _, _ = generate_problems(num_problems, n_people, gen_perturb=False)
    problems_dict={
        "clean": clean_problems,
    }
    random_knight_knave_pairs=False
    flip_knight_knave_pair=False
    uncommon_name=False
    for problem_type, problems in problems_dict.items():
        item_start_idx=0
        for (split, num_samples) in [("test", num_samples_test), ("train", num_samples_train), ("val", num_samples_val)]:
            if num_samples==0:
                continue 
            # 生成格式化的错误思维链
            data= generate_formatted_wrong_cot(problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair, uncommon_name, wrong_type)

            config=f"people{n_people}_num{num_samples}"

            if random_knight_knave_pairs:
                config +="_random_pair"
            if flip_knight_knave_pair:
                config +="_flip_role"
            if uncommon_name:
                config +="_uncommon_name"
            
            output_folder=f"data/wrong_cot_{wrong_type}/{split}/{problem_type}"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f'{config}_wrong1.jsonl')
            with open(output_file, 'w') as file:
                for item in data:
                    json_line = json.dumps(item)
                    file.write(json_line + '\n')
            print(f"Data has been written to {output_file}")
            item_start_idx+=num_samples

#### main & leaf/statement perturbed generation 主问题和叶子节点/语句扰动
for n_people in [10]:
    generate_data(num_samples_test=10,num_samples_train=20,num_samples_val=0,
                    n_people=n_people) 

# for n_people in [3, 4,5,6,7,8]:
#     generate_data(num_samples_test=100,num_samples_train=1000,num_samples_val=0,
#                     n_people=n_people)


# #### LANAGUGE perturbation 语言扰动
# for n_people in [2]:
#     generate_data_language_perturb(num_samples_test=100,num_samples_train=200,num_samples_val=0,
#                     n_people=n_people)


# for n_people in [3, 4,5,6,7,8]:
#     generate_data_language_perturb(num_samples_test=100,num_samples_train=1000,num_samples_val=0,
#                     n_people=n_people)


# #### wrong CoT generation 
# wrong_type="replace_one_step"

# for n_people in [5]:
#     generate_wrong_cot_data(num_samples_test=100,num_samples_train=1000,num_samples_val=0,
#                     n_people=n_people,wrong_type=wrong_type)

# wrong_type="shuffle"

# for n_people in [5]:
#     generate_wrong_cot_data(num_samples_test=100,num_samples_train=1000,num_samples_val=0,
#                     n_people=n_people,wrong_type=wrong_type)


# #### wrong answer generation 
# for n_people in [2]:
#     generate_wrong_data(num_samples_test=100,num_samples_train=200,num_samples_val=0,
#                     n_people=n_people)
# for n_people in [3, 4, 5,6,7,8]:
#     generate_wrong_data(num_samples_test=100,num_samples_train=1000,num_samples_val=0,
#                     n_people=n_people)



