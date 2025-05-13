"""Knight and Knave problems.

Each person can have the following (recursive) statements:
    - assertion: (telling-truth, i), (lying, i)
    - negation: (not, statement)
    - conjunction: (and, statement1, statement2), could support more than 2
    - disjunction: (or, statement1, statement2), could support more than 2
    - implication: (->, statement1, statement2)
    - equivalence: (<=>, statement1, statement2)

Please see the unit tests at the bottom on examples of how to use each API.
"""

import copy
import enum
import itertools
import pprint
import unittest

import numpy as np


####################################################################################
# Problem Solving
####################################################################################
def find_solution(statements):
  """找出给定陈述列表的所有可能解。
  
  Args:
    statements: 一个元组，包含每个人的陈述。
    
  Returns:
    一个列表，包含所有可能的解。每个解是一个布尔值元组，表示每个人是骑士(True)还是骗子(False)。
  """
  n_people = len(statements)
  # 将每个人的陈述转换为等价形式：如果某人是骑士，则他的陈述必须为真
  single_statement = ('and',) + tuple(('<=>', ('telling-truth', i), statements[i])
                                      for i in range(len(statements)))
  # 暴力枚举所有可能的组合
  solutions = []
  for assignments in itertools.product([True, False], repeat=n_people):
    if test_satisfiability(single_statement, assignments):
      solutions.append(assignments)

  return solutions


def test_satisfiability(statement, assignments):
  """递归测试一个陈述在给定赋值下是否可满足。
  
  Args:
    statement: 要测试的陈述。
    assignments: 一个布尔值列表，表示每个人的身份(True表示骑士，False表示骗子)。
    
  Returns:
    布尔值，表示陈述是否可满足。
  """
  if statement[0] == 'telling-truth':
    return assignments[statement[1]]  # 如果陈述说某人是骑士，检查该人是否真的是骑士
  if statement[0] == 'lying':
    return not assignments[statement[1]]  # 如果陈述说某人是骗子，检查该人是否真的是骗子
  if statement[0] == 'not':
    return not test_satisfiability(statement[1], assignments)  # 否定
  if statement[0] == 'and':
    return np.all([test_satisfiability(statement[i], assignments)
                   for i in range(1, len(statement))])  # 所有子陈述都必须为真
  if statement[0] == 'or':
    return np.any([test_satisfiability(statement[i], assignments)
                   for i in range(1, len(statement))])  # 至少一个子陈述为真
  if statement[0] == '->':
    val1 = test_satisfiability(statement[1], assignments)
    val2 = test_satisfiability(statement[2], assignments)
    return (not val1) or val2  # 如果p为真则q为真，等价于(非p)或q
  if statement[0] == '<=>':
    val1 = test_satisfiability(statement[1], assignments)
    val2 = test_satisfiability(statement[2], assignments)
    return (val1 and val2) or ((not val1) and (not val2))  # p等价于q，等价于(p且q)或(非p且非q)
  raise KeyError(f'未知的陈述类型: {statement}')


####################################################################################
# Problem Sampling
####################################################################################
class KKProblemSampler:
  """骑士与骗子问题采样器。
  
  这个类用于生成随机的骑士与骗子问题。每个问题包含多个人的陈述，每个陈述可以是
  断言、否定、合取、析取、蕴含或等价的形式。

  Args:
    rand_seed: 随机数生成器的种子。
    n_people: 问题中的人数。
    depth_constraint: 每个人陈述的最大深度。深度指的是操作符（如'and'、'or'等）的
        递归层数。增加深度会增加问题的难度。目前自动格式化问题为自然语言的功能
        不支持深度大于2的陈述。
    width_constraint: 每个人陈述的最大宽度（操作符如'and'、'or'中的分支数量）。
  """

  def __init__(self, rand_seed: int, n_people: int, depth_constraint: int = 2, width_constraint: int = 2):
    self.rng = np.random.default_rng(rand_seed)  # 用于生成问题的随机数生成器
    self.rng_wrong = np.random.default_rng(rand_seed+1)  # 用于生成错误答案的随机数生成器
    self.n_people = n_people
    self.depth_constraint = depth_constraint
    self.width_constraint = width_constraint

  def sample(self):
    """采样一个骑士与骗子问题。
    
    Returns:
      一个元组，包含每个人的陈述。
    """
    statements = tuple(self._sample_statement(person_id, self.depth_constraint)
                       for person_id in range(self.n_people))
    return self._immutable_statements(statements)

  def sample_valid_problems(self, n_problems: int, max_retry: int = 1000,
                            skip_no_solution: bool = True, skip_multiple_solutions: bool = True):
    """采样有效的（有唯一解）问题。

    Args:
      n_problems: 要采样的问题数量。
      max_retry: 每个问题生成失败后的最大重试次数。
      skip_no_solution: 是否跳过没有解的问题。
      skip_multiple_solutions: 是否跳过有多个解的问题。

    Returns:
      一个问题列表，每个问题是一个字典，包含'statements'和'solution'键。
    """
    problems = []
    unique_statements = set()  # 用于去重
    for i_problem in range(n_problems):
      for _ in range(max_retry):
        statements = self.sample()
        if statements in unique_statements:
          continue  # 重复的问题，重试
        solutions = find_solution(statements)
        if len(solutions) == 0 and skip_no_solution:
          continue  # 没有解，重试
        if len(solutions) > 1 and skip_multiple_solutions:
          continue  # 有多个解，重试
        sol = solutions[0] if len(solutions) > 0 else None
        problems.append({'statements': statements, 'solution': sol,
                         'all_solutions': solutions})
        unique_statements.add(statements)
        break  # 继续生成下一个问题
      if i_problem + 1 < len(problems):
        raise RuntimeError(f'在{max_retry}次重试后仍未能生成有效问题。')
    return problems

  def sample_flipped_solution(self, solution):
    """生成一个通过翻转原解得到的错误解。
    
    Args:
      solution: 原始解，一个布尔值元组。
      
    Returns:
      一个新的解，通过随机翻转原解中的一些值得到。
    """
    length_of_solution = len(solution)
    # 随机决定要翻转多少个值（至少一个）
    num_to_perturb = self.rng_wrong.integers(1, length_of_solution)

    # 随机选择要翻转的位置
    indices_to_perturb = list(self.rng_wrong.choice(list(range(length_of_solution)), size=num_to_perturb, replace=False))
    
    # 创建新的解，翻转选中的位置的值
    perturbed_solution = tuple(
        not solution[i] if i in indices_to_perturb else solution[i]
        for i in range(length_of_solution)
    )
    return perturbed_solution

  def sample_invalid_problems(self, n_problems: int, max_retry: int = 1000,
                            skip_no_solution: bool = True, skip_multiple_solutions: bool = True):
    """采样无效的问题（通过扰动有效问题的解得到）。

    Args:
      n_problems: 要采样的问题数量。
      max_retry: 每个问题生成失败后的最大重试次数。
      skip_no_solution: 是否跳过没有解的问题。
      skip_multiple_solutions: 是否跳过有多个解的问题。

    Returns:
      一个问题列表，每个问题是一个字典，包含'statements'和'solution'键。
    """
    problems = []
    unique_statements = set()
    for i_problem in range(n_problems):
      for _ in range(max_retry):
        statements = self.sample()
        if statements in unique_statements:
          continue  # 重复的问题，重试
        solutions = find_solution(statements)
        if len(solutions) == 0 and skip_no_solution:
          continue  # 没有解，重试
        if len(solutions) > 1 and skip_multiple_solutions:
          continue  # 有多个解，重试
        sol = solutions[0] if len(solutions) > 0 else None
        ## 扰动解
        perturbed_sol = self.sample_flipped_solution(sol)
        problems.append({'statements': statements, 'solution': perturbed_sol,
                         'all_solutions': [perturbed_sol]})
        unique_statements.add(statements)
        break  # 继续生成下一个问题
      if i_problem + 1 < len(problems):
        raise RuntimeError(f'在{max_retry}次重试后仍未能生成有效问题。')
    return problems

  def perturb_problems(self, problems, max_retry: int = 1000, perturb_type: str = 'statement',
                       num_perturb: int = 1):
    """扰动问题（由这个采样器生成的问题）。

    扰动后的问题会在一个地方发生变化，并且保证有不同的解。'leaf'类型的扰动允许"小"的扰动，
    但当人数较少时，可能无法生成有效的扰动（即所有单步扰动都不能得到有效解）。
    一个潜在的解决方案是启用`allow_failure`并过滤掉无效的扰动（标记为None）。

    Args:
      problems: 由这个采样器生成的问题列表。
      max_retry: 生成替代有效问题的最大重试次数。
      perturb_type: 'leaf'表示只扰动随机叶子节点（即非复合陈述）；
          'statement'表示改变随机一个人的整个陈述。
      num_perturb: 要生成的扰动数量。注意实际返回的扰动可能少于这个数字
          （甚至可能是空列表），如果重试次数用尽的话。

    Returns:
      一个扰动后的问题列表。
    """
    return [self._perturb_problem(p, max_retry=max_retry, perturb_type=perturb_type, num_perturb=num_perturb)
            for p in problems]

  def _perturb_problem(self, problem, max_retry: int, perturb_type: str, num_perturb: int):
    """扰动单个问题。

    Args:
      problem: 要扰动的问题。
      max_retry: 最大重试次数。
      perturb_type: 扰动类型。
      num_perturb: 扰动数量。

    Returns:
      一个扰动后的问题列表。
    """
    assert len(problem['statements']) == self.n_people  # 确保参数匹配
    results_set = set()
    results_list = []
    for _ in range(max_retry):
      statements = self._copy_statements_as_mutable(problem['statements'])
      if perturb_type == 'statement':
        person = self.rng.integers(0, self.n_people)
        statements[person] = self._sample_statement(person, depth_constraint=self.depth_constraint)
      elif perturb_type == 'leaf':
        person = self.rng.integers(0, self.n_people)
        idx = person
        container = statements
        while not self._is_leaf_node(container[idx]):
          container = container[idx]
          idx = self.rng.integers(1, len(container))
        assert self._is_leaf_node(container[idx])
        # 设置depth_constraint为1，只采样新的叶子节点
        container[idx] = self._sample_statement(person, depth_constraint=1)

      statements = self._immutable_statements(statements)
      if len(set([statements, problem['statements']])) <= 1:
        continue  # 扰动与原问题相同，重试

      solutions = find_solution(statements)
      if len(solutions) != 1:
        continue  # 不是唯一解，重试

      if len(set([solutions[0], problem['solution']])) <= 1:
        continue  # 扰动后解没有变化，重试

      if statements in results_set:
        continue  # 重复的扰动，重试

      results_set.add(statements)
      results_list.append({'statements': statements, 'solution': solutions[0]})
      if len(results_list) >= num_perturb:
        break
    
    if len(results_list)==0:
      return [None]

    return results_list

  def _copy_statements_as_mutable(self, statements):
    """深度复制问题的陈述，将元组转换为（可变的）列表。

    Args:
      statements: 要复制的陈述。

    Returns:
      一个可变版本的陈述。
    """
    statements = copy.deepcopy(statements)
    def _make_mutable(x):
      if isinstance(x, tuple):
        return [_make_mutable(child) for child in x]
      return x
    return [_make_mutable(s) for s in statements]

  def _immutable_statements(self, mutable_statements):
    """将列表改回元组。

    Args:
      mutable_statements: 可变的陈述。

    Returns:
      不可变的陈述。
    """
    def _make_immutable(x):
      if isinstance(x, (list, tuple)):
        return tuple(_make_immutable(child) for child in x)
      if isinstance(x, np.str_):
        return str(x)
      if isinstance(x, np.int64):
        return int(x)
      return x
    return tuple(_make_immutable(s) for s in mutable_statements)

  def _is_leaf_node(self, statement):
    """判断一个陈述是否是叶子节点（即断言）。

    Args:
      statement: 要检查的陈述。

    Returns:
      布尔值，表示是否是叶子节点。
    """
    if statement[0] in ['telling-truth', 'lying']:
      return True
    return False

  def _sample_statement(self, person_id: int, depth_constraint: int):
    """采样一个陈述。

    Args:
      person_id: 做出陈述的人的ID。
      depth_constraint: 陈述的最大深度。

    Returns:
      一个陈述。
    """
    dice = self.rng.integers(0, 6)
    if depth_constraint == 1 or dice == 0:
      while True:
        knight_or_knave = self.rng.choice(['telling-truth', 'lying'])
        person = self.rng.integers(0, self.n_people)
        if not (knight_or_knave == 'lying' and person == person_id):
          # 避免明显不可满足的陈述
          return (knight_or_knave, person)

    if dice == 1:
      return ('not', self._sample_statement(person_id, depth_constraint-1))
    if dice in [2, 3]:
      operator = ['and', 'or'][dice-2]
      n_substatements = self.rng.integers(2, self.width_constraint+1)

      return (operator,) + self._sample_substatements(person_id, depth_constraint, n_substatements)
    if dice in [4, 5]:
      operator = ['->', '<=>'][dice-4]
      return (operator,) + self._sample_substatements(person_id, depth_constraint, 2)

  def _sample_substatements(self, person_id: int, depth_constraint: int, count: int, dedup: bool = True):
    """为操作符采样子陈述。

    Args:
      person_id: 做出陈述的人的ID。
      depth_constraint: 子陈述的最大深度。
      count: 要生成的子陈述数量。
      dedup: 如果为True，避免重复的子陈述。

    Returns:
      一个子陈述元组。
    """
    sub_statements = []
    dedup_set = set()
    while True:
      stmt = self._sample_statement(person_id, depth_constraint-1)
      if dedup:
        if stmt in dedup_set:
          continue
        dedup_set.add(stmt)

      sub_statements.append(stmt)
      if len(sub_statements) == count:
        break
    return tuple(sub_statements)


####################################################################################
# Problem Formatting in natural language
####################################################################################
COMMON_NAMES = ['Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'Ethan', 'Sophia',
                'Mason', 'Isabella', 'William', 'Mia', 'James', 'Charlotte',
                'Benjamin', 'Amelia', 'Lucas', 'Harper', 'Henry', 'Evelyn',
                'Alexander', 'Abigail', 'Michael', 'Emily', 'Daniel', 'Elizabeth',
                'Jacob', 'Sofia', 'Logan', 'Avery', 'Jackson', 'Ella', 'Sebastian',
                'Scarlett', 'Jack', 'Grace', 'Aiden', 'Chloe', 'Owen', 'Victoria',
                'Samuel', 'Riley', 'Matthew', 'Aria', 'Joseph', 'Lily', 'Luke',
                'Aurora', 'David', 'Zoey', 'Oliver', 'Penelope']
UNCOMMON_NAMES = [
    'Zephyr', 'Elowen', 'Caspian', 'Isolde', 'Osiris', 'Vesper', 'Thaddeus', 'Ondine',
    'Lysander', 'Xanthe', 'Oberon', 'Calliope', 'Leander', 'Eulalia', 'Florian', 'Forsythe',
    'Nephele', 'Peregrine', 'Ianthe', 'Lazarus', 'Elodie', 'Cillian', 'Ottoline', 'Evander',
    'Saffron', 'Caius', 'Zora', 'Cyprian', 'Amaryllis', 'Theron', 'Perdita', 'Ignatius',
    'Zephyrine', 'Balthazar', 'Melisande', 'Zinnia', 'Sylvester', 'Cosima', 'Leocadio',
    'Percival', 'Oceane', 'Evanthe', 'Zenobia', 'Eurydice', 'Quillan', 'Aeronwen',
    'Thorsten', 'Xiomara', 'Zephyrus', 'Ysolde'
]

KNIGHT_KNAVE_PAIRS = [
    # NOTE: we simply add 's' to make plural, so be careful when choosing words
    ['a pioneer', 'a laggard'],
    ['a saint', 'a sinner'],
    ['a hero', 'a villain'],
    ['an angel', 'a devil'],
    ['an altruist', 'an egoist'],
    ['a sage', 'a fool'],
]
PREFIX = ('A very special island is inhabited only by {knight}s and {knave}s. ' +
          '{Knight}s always tell the truth, and {knave}s always lie. ')
POSTFIX = 'So who is {a_knight} and who is {a_knave}?'
TEMPLATES = [  
    '{name} said that {content}.',
    '{name} told you that {content}.',
    '{name} said, "{content}."',
    '{name} stated, "{content}".',
    'According to {name}, "{content}".',
    '''In {name}'s words: "{content}".''',
    '{name} remarked, "{content}".',
    '"{content}," {name} declared.',
    '{name} was heard saying, "{content}".',
    '{name} expressed that {content}.',
    '"{content}" - {name}.',
    'As {name} put it, "{content}".',
    '{name} asserted: "{content}".',
    '"{content}," {name} mentioned.',
    '{name} commented, "{content}".',
    'In a statement by {name}: "{content}".',
    '{name} noted, "{content}".',
    '"{content}," {name} claimed.',
]


class KKProblemFormatter:
  """骑士与骗子问题格式化器。
  
  这个类用于将骑士与骗子问题转换为自然语言描述。它可以：
  1. 随机选择人名
  2. 随机选择说话模板
  3. 随机选择骑士/骗子的称谓对
  4. 重新排序陈述
  5. 生成问题的解答文本

  Args:
    rand_seed: 随机数生成器的种子。
    problem: 要格式化的问题。
  """

  def __init__(self, rand_seed, problem):
    self.rng = np.random.default_rng(rand_seed)  # 用于生成问题的随机数生成器
    self.rng_perturb = np.random.default_rng(rand_seed+1)  # 用于生成扰动的随机数生成器
    self.problem = problem

  def format_problem(self, random_names=True, random_saying_template=True,
                     random_knight_knave_pairs=False,
                     flip_knight_knave_pair=False, uncommon_name=False, reorder_statement=False):
    """格式化问题为自然语言描述。

    Args:
      random_names: 是否随机选择人名。
      random_saying_template: 是否随机选择说话模板。
      random_knight_knave_pairs: 是否随机选择骑士/骗子的称谓对。
      flip_knight_knave_pair: 是否翻转骑士/骗子的称谓对。
      uncommon_name: 是否使用不常见的人名。
      reorder_statement: 是否重新排序陈述。

    Returns:
      一个字典，包含：
      - quiz: 格式化后的问题文本
      - names: 使用的人名列表
      - knight_knave: 使用的骑士/骗子称谓
      - solution: 问题的解
      - solution_text: 格式化后的解答文本
    """
    statements = copy.deepcopy(self.problem['statements'])

    n_people = len(statements)
    names = COMMON_NAMES[:n_people]
    if random_names:
      if uncommon_name==False:
        names = list(self.rng.choice(COMMON_NAMES, size=n_people, replace=False))
      else:
        names = list(self.rng.choice(UNCOMMON_NAMES, size=n_people, replace=False))
    names = [str(x) for x in names]  # 将np.str_转换为str

    knight_knave = ['a knight', 'a knave']
    if random_knight_knave_pairs:
      knight_knave = self.rng.choice(KNIGHT_KNAVE_PAIRS) 
    knight_knave = [str(x) for x in knight_knave]  # 将np.str_转换为str

    if flip_knight_knave_pair:
      knight_knave = knight_knave[::-1]

    knight_knave = {'knight': knight_knave[0].split()[1],
                    'knave': knight_knave[1].split()[1],
                    'a_knight': knight_knave[0], 'a_knave': knight_knave[1]}
    knight_knave['Knight'] = knight_knave['knight'].capitalize()
    knight_knave['Knave'] = knight_knave['knave'].capitalize()

    # 生成问题开头
    text = PREFIX.format(**knight_knave)
    text += f'You meet {n_people} inhabitants: '
    text += ', '.join(names[:-1]) + ', and ' + names[-1] + '.'

    # 生成每个人的陈述
    text_statements=[]
    for i, stmt in enumerate(statements):
      tmpl = TEMPLATES[0]
      if random_saying_template:
        tmpl = self.rng.choice(TEMPLATES)

      content = format_statement(names, knight_knave, stmt)
      text_statements.append(' ' + tmpl.format(name=names[i], content=content))
      # text += ' ' + tmpl.format(name=names[i], content=content)
    
    # 重新排序陈述（如果需要）
    if reorder_statement:
      original_order = list(range(n_people))
      # Copy the original list
      shuffled_order = original_order.copy()

      # 打乱顺序直到与原顺序不同
      while True:
          self.rng_perturb.shuffle(shuffled_order)
          if shuffled_order != original_order:
              break
      for i in shuffled_order:
          text += text_statements[i]
    else:
      text += ''.join(text_statements)

    # 生成问题结尾
    text += ' ' + POSTFIX.format(**knight_knave)
    
    # 生成解答文本
    if self.problem['solution'] is None:
      solution_text = 'No valid solution exists.'
    else:
      solution_stmts = []
      for name, indicator in zip(names, self.problem['solution']):
        if indicator:
          solution_stmts.append(name + ' is ' + knight_knave['a_knight'])
        else:
          solution_stmts.append(name + ' is ' + knight_knave['a_knave'])
      solution_text = ', '.join(solution_stmts[:-1]) + ', and ' + solution_stmts[-1] + '.'
    return {'quiz': text, 'names': names, 'knight_knave': knight_knave,
            'solution': self.problem['solution'],
            'solution_text': solution_text}


# TODO: currently we do not support formatting of problems with depth more than
# 2. We may need to use LLM or think more about what would be the best way
# to format complicated recursive statements.
def format_knight_knave(names, knight_knave, statement, negation=False):
  """格式化骑士/骗子的断言陈述。

  Args:
    names: 人名列表。
    knight_knave: 骑士/骗子的称谓字典。
    statement: 要格式化的陈述。
    negation: 是否否定这个陈述。

  Returns:
    格式化后的文本。
  """
  assert statement[0] in ('telling-truth', 'lying')
  text = names[statement[1]] + ' is '
  if negation:
    text += 'not '
  text += {'telling-truth': knight_knave['a_knight'],
           'lying': knight_knave['a_knave']}[statement[0]]
  return text


def format_statement(names, knight_knave, statement):
  """格式化一个陈述为自然语言。

  Args:
    names: 人名列表。
    knight_knave: 骑士/骗子的称谓字典。
    statement: 要格式化的陈述。

  Returns:
    格式化后的文本。
  """
  if statement[0] == 'not':
    return format_knight_knave(names, knight_knave, statement[1], negation=True)
  if statement[0] in ['and', 'or']:
    text = (' ' + statement[0] + ' ').join(
        format_knight_knave(names, knight_knave, sub_stmt) for sub_stmt in statement[1:])
    return text
  if statement[0] == '->':
    return ('If ' + format_knight_knave(names, knight_knave, statement[1]) + ' then ' +
            format_knight_knave(names, knight_knave, statement[2]))
  if statement[0] == '<=>':
    return (format_knight_knave(names, knight_knave, statement[1]) + ' if and only if ' +
            format_knight_knave(names, knight_knave, statement[2]))
  return format_knight_knave(names, knight_knave, statement)


####################################################################################
# Chain of Thoughts
####################################################################################
def generate_chain_of_thoughts(statements, dynamic_person_order: bool = True):
  """生成解决问题的推理步骤。

  这个函数通过考虑每个人是否说谎以及是否会导致矛盾来生成推理步骤。
  它使用回溯算法来尝试不同的可能性，直到找到解或确定无解。

  Args:
    statements: 骑士与骗子问题的陈述。
    dynamic_person_order: 如果为False，将始终按照原始顺序检查每个人。
        如果为True，将使用更"自然"的顺序。例如，如果person1提到了person5和person4，
        那么引擎将先检查person5和person4，而不是检查person2。

  Returns:
    一个推理步骤列表，每个步骤是一个元组，包含步骤类型和相关信息。
  """
  n_people = len(statements)
  tape = []  # 记录推理步骤
  assignments = [None] * n_people  # 每个人的身份赋值
  options = {p: [False, True] for p in range(n_people)}  # 每个人可能的身份
  persons_to_consider = tuple(range(n_people))  # 待考虑的人的顺序
  p_cursor = 0  # 当前考虑的人的索引
  while True:
    if p_cursor >= n_people:
      tape.append(('success', {'assignments': tuple(assignments)}))
      break

    if not options[persons_to_consider[p_cursor]]:
      exhausted = []
      while p_cursor >= 0 and not options[persons_to_consider[p_cursor]]:
        options[persons_to_consider[p_cursor]] = [False, True]
        assignments[persons_to_consider[p_cursor]] = None
        exhausted.append(persons_to_consider[p_cursor])
        p_cursor -= 1
      if p_cursor >= 0:
        tape.append(('reconsider', {'person': persons_to_consider[p_cursor], 'exhausted': exhausted}))
      else:
        # 已经尝试了所有可能性
        tape.append(('fail',))
        break

    person = persons_to_consider[p_cursor]
    assignments[person] = options[person].pop()
    result, stmt_id = can_be_falsified_v2(statements, assignments)
    if result:
      tape.append(('proposal', {'person': person, 'assignment': assignments[person],
                                'outcome': 'ok'}))
      # 根据当前陈述中提到的人重新排序待考虑的人
      mentioned_people = _find_mentioned_people(statements[person])
      p_cursor += 1
      persons_to_consider = persons_to_consider[:p_cursor] + _reorder_people_sequence(
          persons_to_consider[p_cursor:], mentioned_people)
    else:
      tape.append(('proposal', {'person': person, 'assignment': assignments[person],
                                'outcome': 'conflict', 'conflict_statement': (stmt_id, assignments[stmt_id])}))
  return tape


def _find_mentioned_people(statement):
  """找出陈述中提到的人的ID。

  Args:
    statement: 要检查的陈述。

  Returns:
    一个列表，包含陈述中提到的人的ID。
  """
  if statement[0] in ['lying', 'telling-truth']:
    return [statement[1]]
  if statement[0] in ['not', 'and', 'or', '->', '<=>']:
    return sum([_find_mentioned_people(s) for s in statement[1:]], [])
  raise KeyError(f'未知的陈述类型: {statement}')


def _reorder_people_sequence(remaining_people, mentioned_people):
  """重新排序待考虑的人，将被提到的人移到前面。

  Args:
    remaining_people: 待考虑的人的顺序。
    mentioned_people: 被提到的人。

  Returns:
    重新排序后的人的顺序。
  """
  # 去重并保持顺序
  set_uniq_mention = set()
  list_uniq_mention = []
  for p in mentioned_people:
    if p not in set_uniq_mention:
      set_uniq_mention.add(p)
      list_uniq_mention.append(p)

  for p in reversed(mentioned_people):
    if not p in remaining_people:
      continue
    idx = remaining_people.index(p)
    remaining_people = (p,) + remaining_people[:idx] + remaining_people[idx+1:]
  return remaining_people


def can_be_falsified_v2(statements, assignments):
  """测试部分赋值是否可被证伪（版本2）。

  这个版本枚举所有可能的剩余赋值。这比v1效率低，但v1在检测自相矛盾的陈述时
  可能会有问题，比如当某人的赋值还未确定时，无法轻易检测出类似
  `('<=>', ('lying', 4), ('telling-truth', 4))`这样的自相矛盾陈述。

  Args:
    statements: 问题的陈述。
    assignments: 部分赋值。

  Returns:
    一个元组(是否可被证伪, 导致矛盾的陈述的ID)。
  """
  n_people = len(statements)
  remap = [i for i, x in enumerate(assignments) if x is None]
  n_unassigned = len(remap)

  for p_idx in range(n_people):
    if assignments[p_idx] is None:
      continue
    p_statement = statements[p_idx]
    if not assignments[p_idx]:
      p_statement = ('not', p_statement)
    has_solution = False

    for proposal in itertools.product([True, False], repeat=n_unassigned):
      new_assignments = copy.copy(assignments)
      for i, x in zip(remap, proposal):
        new_assignments[i] = x
      if test_satisfiability(p_statement, new_assignments):
        has_solution = True
        break
    if not has_solution:
      return (False, p_idx)  # 这个人的陈述无法被满足

  return (True, None)


class TruthOrWhatever(enum.Enum):
  """真值枚举类。
  
  这个枚举类用于表示一个陈述的真值状态：
  - FALSE: 假
  - TRUE: 真
  - WHATEVER: 未确定（可能是真也可能是假）
  """
  FALSE = 0
  TRUE = 1
  WHATEVER = 2

  @classmethod
  def from_bool(cls, val: bool):
    """从布尔值创建枚举值。

    Args:
      val: 布尔值。

    Returns:
      对应的枚举值。
    """
    if val:
      return cls.TRUE
    else:
      return cls.FALSE

  def f_not(self):
    """逻辑非运算。

    Returns:
      非运算后的枚举值。
    """
    if self == self.TRUE:
      return self.FALSE
    if self == self.FALSE:
      return self.TRUE
    return self.WHATEVER

  def f_and(self, other):
    """逻辑与运算。

    Args:
      other: 另一个枚举值。

    Returns:
      与运算后的枚举值。
    """
    if self == self.WHATEVER or other == self.WHATEVER:
      return self.WHATEVER
    if self == self.TRUE:
      return self.from_bool(other == self.TRUE)
    return self.FALSE

  def f_or(self, other):
    """逻辑或运算。

    Args:
      other: 另一个枚举值。

    Returns:
      或运算后的枚举值。
    """
    if self == self.WHATEVER or other == self.WHATEVER:
      return self.WHATEVER
    if self == self.FALSE:
      return self.from_bool(other == self.TRUE)
    return self.TRUE


def can_be_falsified(statements, assignments):
  """测试（部分）赋值是否可被证伪（版本1）。

  这个版本使用三值逻辑（真、假、未确定）来测试部分赋值是否可被证伪。
  它比v2更高效，但在某些特殊情况下可能无法正确检测自相矛盾的陈述。

  Args:
    statements: 问题的陈述。
    assignments: 部分赋值。

  Returns:
    一个元组(是否可被证伪, 导致矛盾的陈述的ID)。
  """
  def _test(stmt) -> TruthOrWhatever:
    """递归测试一个陈述的真值。

    Args:
      stmt: 要测试的陈述。

    Returns:
      陈述的真值状态。
    """
    if stmt[0] in ['telling-truth', 'lying'] and assignments[stmt[1]] is None:
      return TruthOrWhatever.WHATEVER
    if stmt[0] == 'telling-truth':
      return TruthOrWhatever.from_bool(assignments[stmt[1]] is True)
    if stmt[0] == 'lying':
        return TruthOrWhatever.from_bool(assignments[stmt[1]] is False)
    if stmt[0] == 'not':
      return _test(stmt[1]).f_not()
    if stmt[0] == 'and':
      val = _test(stmt[1])
      for sub_stmt in stmt[2:]:
        val = val.f_and(_test(sub_stmt))
      return val
    if stmt[0] == 'or':
      val = _test(stmt[1])
      for sub_stmt in stmt[2:]:
        val = val.f_or(_test(sub_stmt))
      return val
    if stmt[0] == '->':
      val1 = _test(stmt[1])
      val2 = _test(stmt[2])
      return val1.f_not().f_or(val2)
    if stmt[0] == '<=>':
      val1 = _test(stmt[1])
      val2 = _test(stmt[2])
      return val1.f_and(val2).f_or(val1.f_not().f_and(val2.f_not()))
    raise KeyError(f'未知的陈述类型: {stmt}')

  for i, (stmt, assmt) in enumerate(zip(statements, assignments)):
    if assmt is None:
      # 这个人的陈述不重要
      continue
    if assmt and _test(stmt) == TruthOrWhatever.FALSE:
      return (False, i)
    if not assmt and _test(stmt) == TruthOrWhatever.TRUE:
      return (False, i)
  return (True, None)


def format_chain_of_thoughts(problem, formatted_problem, tape,
                             repeat_claim_for_assumption: bool = True,
                             repeat_claim_for_contradiction: bool = False):
  """将生成的推理步骤格式化为自然语言。

  重复陈述可以使文本更自然，但也会增加需要处理的标记数量。

  Args:
    problem: 骑士与骗子问题。
    formatted_problem: 格式化后的问题结果。
    tape: 生成的推理步骤。
    repeat_claim_for_assumption: 是否在假设某人是骑士或骗子后重复他们的陈述。
    repeat_claim_for_contradiction: 是否在发现矛盾时重复导致矛盾的陈述。

  Returns:
    一个元组(header, [step1, step2, ...], footer)。
    header是推理的开始说明。
    steps是推理步骤列表。
    footer是推理的结论（成功或失败）。
    注意最终解不包含在footer中。如果需要，可以在这里附加problem['solution_text']。
  """
  format_dict = copy.copy(formatted_problem['knight_knave'])
  n_person = len(problem['statements'])
  for p in range(n_person):
    format_dict[f'P{p}'] = formatted_problem['names'][p]

  header = "Let's think step by step, by considering whether each person is lying and if that leads to contradiction."
  steps = []
  for step in tape[:-1]:  # 最后一步是失败/成功
    if step[0] == 'proposal':
      t_person = '{P' + str(step[1]['person']) + '}'
      t_assignment = '{a_knight}' if step[1]['assignment'] else '{a_knave}'
      if step[1]['outcome'] == 'ok':
        text = 'Assume ' + t_person + ' is ' + t_assignment + '.'
        if repeat_claim_for_assumption:
          t_claim = format_statement(formatted_problem['names'], formatted_problem['knight_knave'],
                                     problem['statements'][step[1]['person']])
          text += ' No contradiction is found in their '
          if not step[1]['assignment']:
            text += 'false '
          text += 'claim that ' + t_claim + '.'
      elif step[1]['outcome'] == 'conflict':
        conflict_p, conflict_assignment = step[1]['conflict_statement']
        text = t_person + ' cannot be ' + t_assignment + ', because this would contradict the '
        if not conflict_assignment:
          text += 'false '
        text += 'claim of '
        if conflict_p == step[1]['person']:
          text += 'their own'
        else:
          text += '{P' + str(conflict_p) + '}'
        if repeat_claim_for_contradiction:
          t_claim = format_statement(formatted_problem['names'], formatted_problem['knight_knave'],
                                     problem['statements'][conflict_p])
          text += ' that ' + t_claim + '.'
        else:
          text += '.'
      else:
        raise KeyError(f'Unknown outcome for CoT step: {step}')
      steps.append(text)
    elif step[0] == 'reconsider':
      text = 'We have exhausted all possibilities for '
      t_exhausted = ['{P' + str(p_idx) + '}' for p_idx in step[1]['exhausted']]
      assert len(t_exhausted) > 0
      if len(t_exhausted) == 1:
        text += t_exhausted[0]
      elif len(t_exhausted) == 2:
        text += ' and '.join(t_exhausted)
      else:
        t_exhausted[-1] = 'and ' + t_exhausted[-1]
        text += ', '.join(t_exhausted)
      text += ', so let us go back and reconsider {P' + str(step[1]['person']) + '}.'
      steps.append(text)
    else:
      raise KeyError(f'Unknown CoT step: {step}')

  if tape[-1][0] == 'success':
    footer = 'This leads to a feasible solution.'
  elif tape[-1][0] == 'fail':
    footer = 'All the configurations lead to contradictions.'
  else:
    raise KeyError(f'Expect success or fail, but get {tape[-1]}')

  steps = [x.format(**format_dict) for x in steps]
  return (header, steps, footer)


####################################################################################
# Unit Testing
####################################################################################
class TestKK(unittest.TestCase):
  """骑士与骗子问题的单元测试类。
  
  这个类包含了所有主要功能的单元测试：
  1. 问题求解
  2. 问题生成
  3. 问题格式化
  4. 问题扰动
  5. 推理链生成
  """

  def test_find_solution(self):
    """测试问题求解功能。"""
    statements = (
        ('lying', 1),
        ('and', ('telling-truth', 0), ('telling-truth', 1))
    )
    sol = find_solution(statements)
    self.assertEqual(sol, [(True, False)])

  def test_sample_problems(self):
    """测试问题生成功能。"""
    n_people = 3
    n_problems = 5
    problem_sampler = KKProblemSampler(1234, n_people=n_people)
    problems = problem_sampler.sample_valid_problems(n_problems)
    self.assertEqual(len(problems), n_problems)
    for problem in problems:
      self.assertEqual(set(problem.keys()), set(['statements', 'solution', 'all_solutions']))
      self.assertEqual(len(problem['statements']), n_people)

  def test_format_problems(self):
    """测试问题格式化功能。"""
    problem_sampler = KKProblemSampler(1234, n_people=3)
    problems = problem_sampler.sample_valid_problems(20, skip_no_solution=False)

    for problem in problems:
      formatter = KKProblemFormatter(rand_seed=1234, problem=problem)
      formatted_results = formatter.format_problem()
      self.assertIn('quiz', formatted_results)
      self.assertIn('names', formatted_results)
      self.assertIn('solution', formatted_results)
      self.assertIn('solution_text', formatted_results)
      if problem['solution'] is None:
        self.assertEqual(formatted_results['solution_text'], 'No valid solution exists.')

  def test_perturb_problems(self):
    """测试问题扰动功能。"""
    n_people = 4
    n_perturb = 3
    problem_sampler = KKProblemSampler(1234, n_people=n_people)
    problems = problem_sampler.sample_valid_problems(5)
    for perturb_type in ['statement', 'leaf']:
      perturbed_problems = problem_sampler.perturb_problems(problems, perturb_type=perturb_type, num_perturb=n_perturb)
      self.assertEqual(len(problems), len(perturbed_problems))
      for p1, p2_list in zip(problems, perturbed_problems):
        self.assertEqual(len(p2_list), n_perturb)  # 注意这个测试可能会失败，特别是当人数较少时
        self.assertNotEqual(p1['solution'], p2_list[0]['solution'])
        n_stmt_diff = 0
        for s1, s2 in zip(p1['statements'], p2_list[0]['statements']):
          if s1 != s2:
            n_stmt_diff += 1
        self.assertEqual(n_stmt_diff, 1)  # 恰好有一个陈述不同

  def test_chain_of_thoughts(self):
    """测试推理链生成功能。"""
    n_people = 5
    n_problems = 120
    problem_sampler = KKProblemSampler(1234, n_people=n_people)
    problems = problem_sampler.sample_valid_problems(n_problems, skip_no_solution=False)
    for p in problems:
      for dynamic_person_order in [False, True]:
        tape = generate_chain_of_thoughts(p['statements'], dynamic_person_order=dynamic_person_order)
        if p['solution'] is None:
          self.assertTupleEqual(tape[-1], ('fail',))
        else:
          self.assertEqual(tape[-1][0], ('success'))
          self.assertTupleEqual(tape[-1][1]['assignments'], p['solution'])

  def test_chain_of_thoughts_regression(self):
    """测试推理链生成的回归测试。
    
    注意：正确答案不是唯一的，当推理链生成器代码改变时可能会改变。
    所以这个测试失败不一定意味着代码有错误。如果代码改变并验证为正确，
    可以更新这个测试的目标输出。
    """
    statements = (('and', ('telling-truth', 2), ('lying', 3)),
                  ('telling-truth', 2),
                  ('<=>', ('lying', 4), ('telling-truth', 4)),
                  ('and', ('lying', 2), ('lying', 4)),
                  ('lying', 2))
    expected_tape = [
        ('proposal', {'person': 0, 'assignment': True, 'outcome': 'ok'}),
        ('proposal',
          {'person': 2,
          'assignment': True,
          'outcome': 'conflict',
          'conflict_statement': (2, True)}),
        ('proposal',
          {'person': 2,
          'assignment': False,
          'outcome': 'conflict',
          'conflict_statement': (0, True)}),
        ('reconsider', {'person': 0, 'exhausted': [2]}),
        ('proposal', {'person': 0, 'assignment': False, 'outcome': 'ok'}),
        ('proposal',
          {'person': 2,
          'assignment': True,
          'outcome': 'conflict',
          'conflict_statement': (2, True)}),
        ('proposal', {'person': 2, 'assignment': False, 'outcome': 'ok'}),
        ('proposal', {'person': 4, 'assignment': True, 'outcome': 'ok'}),
        ('proposal',
          {'person': 3,
          'assignment': True,
          'outcome': 'conflict',
          'conflict_statement': (3, True)}),
        ('proposal', {'person': 3, 'assignment': False, 'outcome': 'ok'}),
        ('proposal',
          {'person': 1,
          'assignment': True,
          'outcome': 'conflict',
          'conflict_statement': (1, True)}),
        ('proposal', {'person': 1, 'assignment': False, 'outcome': 'ok'}),
        ('success', {'assignments': (False, False, False, False, True)})
    ]
    tape = generate_chain_of_thoughts(statements, dynamic_person_order=True)
    self.assertEqual(tape, expected_tape)

def test_chain_of_thoughts_format_regression(self):
  """测试推理链格式化功能的回归测试。
  
  注意：正确答案不是唯一的，当推理链生成器代码改变时可能会改变。
  所以这个测试失败不一定意味着代码有错误。如果代码改变并验证为正确，
  可以更新这个测试的目标输出。
  """
  problem = {
      'statements': (('and', ('telling-truth', 2), ('lying', 3)),
                     ('telling-truth', 2),
                     ('<=>', ('lying', 4), ('telling-truth', 4)),
                     ('and', ('lying', 2), ('lying', 4)),
                     ('lying', 2)),
      'solution': (False, False, False, False, True),
      'all_slutions': [(False, False, False, False, True)]
  }
  chain_of_thoughts = generate_chain_of_thoughts(problem['statements'])
  formatted_problem = {'knight_knave': {'Knave': 'Knave',
                    'Knight': 'Knight',
                    'a_knave': 'a knave',
                    'a_knight': 'a knight',
                    'knave': 'knave',
                    'knight': 'knight'},
   'names': ['David', 'Zoey', 'Alexander', 'Aurora', 'Isabella'],
   'quiz': 'A very special island is inhabited only by knights and knaves. '
           'Knights always tell the truth, and knaves always lie. You meet 5 '
           'inhabitants: David, Zoey, Alexander, Aurora, and Isabella. In '
           'David\'s words: "Alexander is a knight and Aurora is a knave". Zoey '
           'expressed that Alexander is a knight. Alexander said, "Isabella is a '
           'knave if and only if Isabella is a knight." Aurora commented, '
           '"Alexander is a knave and Isabella is a knave". According to '
           'Isabella, "Alexander is a knave". So who is a knight and who is a '
           'knave?',
   'solution': (False, False, False, False, True),
   'solution_text': 'David is a knave, Zoey is a knave, Alexander is a knave, '
                    'Aurora is a knave, and Isabella is a knight.'}
  cot_format = format_chain_of_thoughts(problem, formatted_problem, chain_of_thoughts,
                                        repeat_claim_for_assumption=True,
                                        repeat_claim_for_contradiction=True)
  expected_cot = ('Let us think step by step, by considering whether each person is lying and if that leads to contradiction.',
   ['Assume David is a knight. No contradiction is found in their claim that Alexander is a knight and Aurora is a knave.',
    'Alexander cannot be a knight, because this would contradict the claim of their own.',
    'Alexander cannot be a knave, because this would contradict the claim of David.',
    'We have exhausted all possibilities for Alexander, so let us go back and reconsider David.',
    'Assume David is a knave. No contradiction is found in their false claim that Alexander is a knight and Aurora is a knave.',
    'Alexander cannot be a knight, because this would contradict the claim of their own.',
    'Assume Alexander is a knave. No contradiction is found in their false claim that Isabella is a knave if and only if Isabella is a knight.',
    'Assume Isabella is a knight. No contradiction is found in their claim that Alexander is a knave.',
    'Aurora cannot be a knight, because this would contradict the claim of their own.',
    'Assume Aurora is a knave. No contradiction is found in their false claim that Alexander is a knave and Isabella is a knave.',
    'Zoey cannot be a knight, because this would contradict the claim of their own.',
    'Assume Zoey is a knave. No contradiction is found in their false claim that Alexander is a knight.'],
   'This leads to a feasible solution.')
  self.assertEqual(cot_format, expected_cot)

  cot_format = format_chain_of_thoughts(problem, formatted_problem, chain_of_thoughts,
                                        repeat_claim_for_assumption=False,
                                        repeat_claim_for_contradiction=False)
  expected_cot = ('Let us think step by step, by considering whether each person is lying and if that leads to contradiction.',
   ['Assume David is a knight.',
    'Alexander cannot be a knight, because this would contradict the claim of their own.',
    'Alexander cannot be a knave, because this would contradict the claim of David.',
    'We have exhausted all possibilities for Alexander, so let us go back and reconsider David.',
    'Assume David is a knave.',
    'Alexander cannot be a knight, because this would contradict the claim of their own.',
    'Assume Alexander is a knave.',
    'Assume Isabella is a knight.',
    'Aurora cannot be a knight, because this would contradict the claim of their own.',
    'Assume Aurora is a knave.',
    'Zoey cannot be a knight, because this would contradict the claim of their own.',
    'Assume Zoey is a knave.'],
   'This leads to a feasible solution.')
  self.assertEqual(cot_format, expected_cot)


if __name__ == '__main__':
  unittest.main() # 运行测试
