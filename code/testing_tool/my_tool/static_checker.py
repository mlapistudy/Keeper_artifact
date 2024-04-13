import change_code
import ast


ASYNC_APIs = ["long_running_recognize", #google
              "describe_entities_detection_job", "describe_dominant_language_detection_job", "describe_key_phrases_detection_job","describe_sentiment_detection_job" # aws
              ] 

PARALLEL_APIs = ["annotate_image", "batch_annotate_images", "annotate_text"]

# List of parallelism-related APIs
# https://wiki.python.org/moin/ParallelProcessing
PARALLEL_LIBs = ["threading", "multiprocessing", "concurrent", "asyncio", "_thread", 
                 "dispy", "forkmap", "forkfun", "joblib", "pprocess", "processing", "PyCSP", "PyMP ", "Ray", "torcpy", "VecPy"]

CONST_APIs = ["synthesize_speech", # google stt
              "analyze_sentiment", "classify_text", "analyze_entities", "analyze_entity_sentiment", "analyze_syntax" # google language
              ]

def find_import_libs(line):
  if line.strip().startswith("import "):
    # line = line[line.find("import")+6:]
    if " as " in line:
      line = line[line.rfind(" as ")+3:]
    return line
  if line.strip().startswith("from ") and " import " in line:
    # line = line[line.find("import")+6:]
    if " as " in line:
      line = line[line.rfind(" as ")+3:]
    return line
  return ""

def get_ml_api(filename):
  used_ml_api = []
  ml_api_to_input = []
  with open(filename, 'r', encoding='utf8') as file_obj:
    text = file_obj.read()
  for line in text.split("\n")[::-1]:
    if line.startswith("# used_ml_api:"):
      used_ml_api = line.replace("# used_ml_api:","").strip().split(", ")
    if line.startswith("# ml_api_to_input:"):
      ml_api_to_input = eval(line.replace("# ml_api_to_input:","").strip())    
  return used_ml_api, ml_api_to_input

def misuse_async_api(input_file, constraint_file):
  content_line_by_line = change_code.read_wholefile(input_file, preprocess=False).split("\n")
  content_line_by_line = [x for x in content_line_by_line if not x.strip().startswith("@")]
  contains_parallel_lib = False
  call_async_apis = False
  bug_line_no = []
  for no, line in enumerate(content_line_by_line):
    tmp = find_import_libs(line)
    if len(tmp)>0:
      vars = change_code.get_varnames(tmp)
      for var in vars:
        if var.name in PARALLEL_LIBs:
          contains_parallel_lib = True
          return []

  used_ml_api, ml_api_to_input = get_ml_api(constraint_file)
  for ml_api in ASYNC_APIs:
    if ml_api in used_ml_api:
      
      for no, line in enumerate(content_line_by_line):
        line_without_space = line.replace(" ","")
        for ml_api in ASYNC_APIs:
          if "."+ml_api+"(" in line_without_space:
            call_async_apis = True
            bug_line_no.append(no)
  return bug_line_no

  
def misuse_parallel_api(input_file, constraint_file):
  content_line_by_line = change_code.read_wholefile(input_file, preprocess=False).split("\n")
  content_line_by_line = [x for x in content_line_by_line if not x.strip().startswith("@")]
  contains_parallel_lib = False
  bug_line_no = []
  for no, line in enumerate(content_line_by_line):
    tmp = find_import_libs(line)
    if len(tmp)>0:
      vars = change_code.get_varnames(tmp)
      for var in vars:
        if var.name in PARALLEL_LIBs:
          contains_parallel_lib = True
          return []
        
  used_ml_api, ml_api_to_input = get_ml_api(constraint_file)
  if len(used_ml_api) == 1:
    return []
  target_apis = {} #input: {api}
  for api in used_ml_api:
    if api not in PARALLEL_APIs:
      input_var = None
      for tmp in ml_api_to_input:
        if len(tmp)!=2:
          continue
        if tmp[0] == api:
          input_var = tmp[1]
          break
      if input_var is None:
        continue
      if not input_var in target_apis.keys():
        target_apis[input_var] = set()
      target_apis[input_var].add(api)

  for input_var, invoked_apis in target_apis.items():
    if len(invoked_apis)>1:
      ml_api = list(invoked_apis)[0]
      for no, line in enumerate(content_line_by_line):
        line_without_space = line.replace(" ","")
        if "."+ml_api+"(" in line_without_space:
          bug_line_no.append(no)
          break
  
  return bug_line_no

# the input file must be preprossed by change_code.py
def get_dependent_values(node):
  # print(node)
  right_values = []
  if isinstance(node, ast.Assign):
    right_values = get_dependent_values(node.value)
  elif isinstance(node, ast.Tuple):
    for x in node.elts:
      right_values += get_dependent_values(x)
  elif isinstance(node, ast.Call):
    for x in node.args:
      right_values += get_dependent_values(x)
    for x in node.keywords:
      right_values += get_dependent_values(x.value)
  elif isinstance(node, ast.BinOp):
    right_values = get_dependent_values(node.left) + get_dependent_values(node.right)
  elif isinstance(node, ast.BoolOp):
    for x in node.values:
      right_values += get_dependent_values(x)
  elif isinstance(node, ast.Compare):
    for x in [node.left] + node.comparators:
      right_values += get_dependent_values(x)
  elif isinstance(node, ast.UnaryOp):
    right_values = get_dependent_values(x.operand)
  elif isinstance(node, ast.NamedExpr):
    right_values = get_dependent_values(node.value)
  elif isinstance(node, ast.Attribute):
    x = node
    while isinstance(x, ast.Attribute):
      x = x.value
    if isinstance(x, ast.Name):
      right_values = [x]
  elif isinstance(node, ast.Constant) or isinstance(node, ast.Name):
    right_values = [node]
  else:
    pass
  return right_values

def misuse_constant_input(input_file, constraint_file):
  def check_one_node(node, loop_level):
    # print("============")
    # print(node)
    if isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
      for x in node.names:
        constant_set.add(x.name)
      return
    if isinstance(node, ast.Module) or isinstance(node, ast.FunctionDef) or isinstance(node, ast.With):
      for x in node.body:
        check_one_node(x, loop_level)
    elif isinstance(node, ast.For) or isinstance(node, ast.While):
      for x in node.body:
        check_one_node(x, loop_level+1)
      for x in node.orelse:
        check_one_node(x, loop_level+1)

    elif isinstance(node, ast.Assign):
      left_vars = []
      for target in node.targets:
        if isinstance(target, ast.Tuple):
          left_vars.extend(target.elts)
        else:
          left_vars.append(target)
      # print(left_vars[0].lineno, node)
      right_values = get_dependent_values(node)
      flags = [(isinstance(x, ast.Constant) or (isinstance(x, ast.Name) and x.id in constant_set)) for x in right_values]
      # print(right_values, flags)
      if all(flags):
        for lv in left_vars:
          if isinstance(lv, ast.Name):
            constant_values.append((lv.id, lv.lineno, loop_level))
            constant_set.add(lv.id)
      else:
        for lv in left_vars:
          if isinstance(lv, ast.Name) and lv.id in constant_set:
            constant_set.remove(lv.id)
    else:
      try:
        body = node.body
      except:
        return
      for x in body:
        check_one_node(x, loop_level)

  content = change_code.read_wholefile(input_file, preprocess=False)
  constant_values = []
  constant_set = set()
  bug_line_no = []
  root = ast.parse(content)
  check_one_node(root, 0)
  
  content_line_by_line = content.split("\n")
  for no, line in enumerate(content_line_by_line):
    line_without_space = line.replace(" ","")
    for ml_api in CONST_APIs:
      if "."+ml_api+"(" in line_without_space:
        for value in constant_values:
          name, lineno, loop_level = value
          if loop_level>0 and lineno==no+1:
            bug_line_no.append(no)
  return bug_line_no


    

if __name__ == '__main__':
  pass
  # print(misuse_async_api("../cmd_example/static_example.py", "../cmd_example/static_example.py"))
  # print(misuse_constant_input("../cmd_example/static_example2.py", "../cmd_example/static_example3.py"))
  # print(misuse_parallel_api("../cmd_example/static_example3.py", "../cmd_example/static_example3.py"))