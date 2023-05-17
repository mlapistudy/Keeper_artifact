import os, sys
import inflect
import pandas as pd
import math
import logging
import subprocess
import time as time

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_SRC = os.path.join(dir_path, "language_src", "sentiment_data.csv")
SENTIMENT_SRC = os.path.join(dir_path, "language_src", "sentiment140.csv")
SENTIMENT_RAW_SRC = os.path.join(dir_path, "language_src", "sentiment140", "training.csv")
CACHE_SRC = os.path.join(dir_path, "language_src", "_cache")
if not os.path.exists(CACHE_SRC):
    os.mkdir(CACHE_SRC)
sys.path.append(os.path.dirname(dir_path))
from global_vars import PYTHON_OHTER, log_level
logging.basicConfig(level=log_level)


SCORE_LIST_RANGE = 2 # result list contains med+-n scores
MAGNITUDE_LIST_RANGE = 5 # similar as above
MAG_INDEX = .08
NUM = 5 # classify text
CHAR_LIMIT = 50 # classify text

def round_up(n, decimals=0): 
  multiplier = 10 ** decimals 
  return math.ceil(n * multiplier) / multiplier
def round_down(n, decimals=0): 
  multiplier = 10 ** decimals 
  return math.floor(n * multiplier) / multiplier

# bucket score to 0.1 (BITS=1)
# magnitude<0  -> no constraint
def analyze_sentiment(score, magnitude, max_num=5, BITS=1):
  # print(str((score, magnitude)))
  if max_num <= 0:
    max_num = 1
  if score > 1 or score < -1:
    raise ValueError("Score not in range +-1")

  score_l = round_down(score, BITS)
  score_r = round_up(score, BITS)
  if score_l == score_r:
    score_l -= 10**(-BITS) / 2
    score_r += 10**(-BITS) / 2
    score_l = round_down(score_l, BITS+1)
    score_r = round_up(score_r, BITS+1)
  if magnitude >= 0:
    mag_l = round_down(magnitude, BITS)
    mag_r = round_up(magnitude, BITS)
    if mag_l == mag_r:
      mag_l -= 10**(-BITS) / 2
      mag_r += 10**(-BITS) / 2
      mag_l = round_down(mag_l, BITS+1)
      mag_r = round_up(mag_r, BITS+1)
  else: # no constraint
    mag_l = -1
    mag_r = 100
  
  # print(str((score_l, score_r, mag_l, mag_r)))
  return sentiment_within_range(score_l, score_r, mag_l, mag_r, max_num=max_num, BITS=BITS)


def analyze_sentiment_score(score, max_num=5, BITS=1):
  if max_num <= 0:
    max_num = 1
  if score > 1 or score < -1:
    raise ValueError("Score not in range +-1")

  score_l = round_down(score, BITS)
  score_r = round_up(score, BITS)
  if score_l == score_r:
    score_l -= 10**(-BITS) / 2
    score_r += 10**(-BITS) / 2
    score_l = round_down(score_l, BITS+1)
    score_r = round_up(score_r, BITS+1)

  result = []
  max_mag = 3.0
  slice = max_mag/min(max_num,10)
  num_pre = max(1, int(max_num/min(max_num,10)))
  # print(str((max_num, slice, num_pre)))
  mag = 0
  while mag+slice <= max_mag:
    text = sentiment_within_range(score_l, score_r, mag, mag+slice, max_num=num_pre, BITS=BITS)
    result += text
#    result += text1+text2
    mag += slice
    # print(str((score_l, score_r, mag, mag+slice)))
    # print(len(text))
  return list(set(result))

# this version does not persue exact reverse of API
# instead, it is looking into human understanding of positive or negative
# is_positive: whether we want a positive text or not
def analyze_sentiment_without_hard_limit(is_positive, max_num=5):
  if max_num <= 0:
    max_num = 1
  if not os.path.exists(SENTIMENT_RAW_SRC):
    if is_positive:
      return analyze_sentiment(0.8, -1, max_num=max_num)
    return analyze_sentiment(-0.8, -1, max_num=max_num)
  
  result = []
  df_in = pd.read_csv(SENTIMENT_RAW_SRC, header=None, usecols=[0,5], encoding='latin-1')
  # df_out = pd.DataFrame(columns=["score", "magnitude", "text"])
  record_num = len(df_in)
  max_num = min(max_num, record_num//2)
  if is_positive:
    for index, row in df_in.tail(max_num).iterrows():
      result.append(row[5])
  else:
    for index, row in df_in.head(max_num).iterrows():
      result.append(row[5])

  return result


def sentiment_within_range(score_l, score_r, mag_l, mag_r, max_num, BITS):
  df = pd.read_csv(SENTIMENT_SRC, encoding='latin-1')
  # print((score_l, score_r, mag_l, mag_r))
  # find text with preferred score
  mag_to_text = {}
  for index, row in df.iterrows():
    if row["score"]>=score_l and row["score"]<=score_r:
      mag = round(row["magnitude"], BITS+1)
      # print(row["score"])
      if mag in mag_to_text.keys():
        mag_to_text[mag].append(row["text"])
      else:
        mag_to_text[mag] = [row["text"]]


  # finding suitable text
  possible_mags = list(mag_to_text.keys())
  possible_mags.sort()

  def get_text_from_mag(low, high):
    result = []
    for mag in possible_mags:
      if mag>=low and mag<=high:
        result = result + mag_to_text[mag]
    return result

  target_text = get_text_from_mag(mag_l, mag_r)
  if len(target_text) >= max_num:
    return target_text[:max_num]

  for mag in possible_mags:
    if mag>=mag_l: # already included
      break
    # compose severak text
    text1 = mag_to_text[mag]
    text2 = get_text_from_mag(mag_l-mag, mag_r-mag)
    for t1 in text1:
      for t2 in text2:
        if t1==t2:
          continue
        if t1.strip().endswith(".") or t1.strip().endswith("?") or t1.strip().endswith("!") or t1.strip().endswith(";"):
          target_text.append(t1 + " " + t2)
        else:
          target_text.append(t1 + ". " + t2)
    if len(target_text) >= max_num:
      return target_text[:max_num]

  # if still not sufficient, add multiplier
  for mag in possible_mags:
    if mag >= mag_l:
      break
    mult = 2
    while mag*mult<=mag_r:
      text1 = mag_to_text[mag]
      new_mag = round(mag*mult, BITS+1)
      if new_mag>=mag_l: # do not need extra text
        text2 = [" "]
      else:
        text2 = get_text_from_mag(mag_l-new_mag, mag_r-new_mag)
      for t1 in text1:
        for t2 in text2:
          if t1==t2:
            continue
          if t1.strip().endswith(".") or t1.strip().endswith("?") or t1.strip().endswith("!") or t1.strip().endswith(";"):
            target_text.append(" ".join([t1]*mult) + " " + t2)
          else:
            target_text.append(". ".join([t1]*mult) + ". " + t2)
      mult+=1
      if len(target_text) >= max_num:
        return target_text[:max_num]

  return target_text[:max_num]

def get_text_label_variant(keyword, find_child=True, max_num=5):
  def split_phrase(word):
    phrases = [x.strip().lower() for x in word.split("&")]
    if len(phrases)>=2: # actually at most 1 &
      A = len(phrases[0].split())
      B = len(phrases[1].split())
      if A>B:
        phrases = [phrases[0]]
      elif A<B:
        phrases = [phrases[1]]
    p = inflect.engine()
    for i in range(len(phrases)):
      if p.singular_noun(phrases[i]):
        phrases[i] = p.singular_noun(phrases[i])
    return phrases

  variants = []
  phrases = split_phrase(keyword)
  for phrase in phrases:
    variants.append(phrase)
  if find_child:
    childrens = find_direct_children(keyword)
    for child in childrens:
      phrases = split_phrase(child)
      for phrase in phrases:
        if not phrase in variants:
          variants.append(phrase)
  if max_num>0:
    return variants[:max_num]
  return variants

def classify_text(keyword, max_num=3):

  logging.info(f"classify_text (reverse API): getting reverse results for keyword {keyword}")
  if len(keyword)==0:
    logging.info(f"classify_text (reverse API): returnning non-sense words")
    return ["Non sense text to aviod too few tokens (words) to process. Non sense text to aviod too few tokens (words) to process. Non sense text to aviod too few tokens (words) to process."]

  keyword_list = [x.strip() for x in keyword.split("/")]
  keyword_list = [x for x in keyword_list if len(x)>0]
  if len(keyword_list)==0:
    return []
  keyword = keyword_list[-1]

  def run_command(command):
    print(command)
    proc = subprocess.Popen(command, shell=True)
    proc.wait()

  filename = os.path.join(CACHE_SRC, keyword)
  if os.path.exists(filename):
    with open(filename, 'r', encoding='utf8') as file_obj:
        text = file_obj.read()
    ret = eval(text)
    if len(ret)>= max_num:
      ret = [x[:4000].replace("\n"," ").replace("\r"," ") for x in ret]
      logging.info(f"classify_text (reverse API): got reverse result from file {filename}")
      ret2 = []
      for x in ret:
        if x.lower().startswith("microsoft collects data from you"):
            continue
        if x.lower().startswith("britannica") or x.lower().startswith("dictionary"):
            continue
        ret2.append(x)
      return ret2[:max_num]

  # Invoking this in a seperate script because running this in the
  # solve_multi.py environment would produce some unexpected error with the wikipedia package
  # So, invoke the script to produce the desired cache file, from which to read the result
  filename = os.path.join(CACHE_SRC, keyword)  
  if max_num>10:
    bound = max_num//5
    bound = min(bound,10)
    variants = get_text_label_variant(keyword, find_child=True, max_num=bound-1)
  else:
    variants = get_text_label_variant(keyword, find_child=False)
  print(">>>>> "+str(variants))
  num = max_num//len(variants)
  all_results = []
  for i, phrase in enumerate(variants):
    filename2 = os.path.join(CACHE_SRC, "tmp")
    dir_name = os.path.dirname(os.path.realpath(__file__))
    command = f"cd {dir_name}; /usr/bin/python3 search_text.py --keyword='{phrase}' --filename='{filename2}' --max_num={num}"
    run_command(command)

    if os.path.exists(filename2):
      with open(filename2, 'r', encoding='utf8') as file_obj:
        text = file_obj.read()
      ret = eval(text)
      ret = [x[:4000].replace("\n"," ").replace("\r"," ") for x in ret]
      all_results += ret
  
  ret = all_results
  ret = [x[:4000].replace("\n"," ").replace("\r"," ") for x in ret]
  ret2 = []
  for x in ret:
    if x.lower().startswith("microsoft collects data from you"):
        continue
    if x.lower().startswith("britannica") or x.lower().startswith("dictionary"):
        continue
    ret2.append(x)
  with open(filename, 'w', encoding='utf8') as file_obj:
    file_obj.write(str(ret2))
  return ret2[:max_num]
  


def analyze_entities(type, name=None, force_new=False, max_num=1):
  """ Given either type or name of a detected entity, generate a sentence
      that contains that entity

  arguments:
  type - type of entity
  name - name of entity, if name is specified, generate text using name
  force_new - whether to generate and rewrite cached results
  max_num - max number of texts to return
  """

  if max_num < 1:
    max_num = 1

  res_list = []
  if name != None:
    for i in range(max_num):
      filename = os.path.join(CACHE_SRC, "analyze_entities_" + name + "_{}.txt".format(i))
      # Checking cache folder first
      if force_new == False:
        if os.path.exists(filename):
          with open(filename, "r", encoding='utf8') as fd:
            res_list.append(fd.read())
          continue

      res = generate_text(name, num_outputs=max_num - i)
      for each_res in res:
        filename = os.path.join(CACHE_SRC, "analyze_entities_" + name + "_{}.txt".format(i))
        with open(filename, "w", encoding='utf8') as fd:
          fd.write(each_res)
        i += 1
      res_list += res
      break
    return res_list

  special_line = None
  for i in range(max_num):
    filename = os.path.join(CACHE_SRC, "analyze_entities_" + type + "_{}.txt".format(i))
    # Checking cache folder first
    if force_new == False:
      if os.path.exists(filename):
        with open(filename, "r", encoding='utf8') as fd:
          res_list.append(fd.read())
        continue
    
    if special_line == None:
      file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "language_src", "entity_type.csv")
      with open(file_path, "r") as fd:
        for line in fd.readlines():
          line = line.strip("\n")
          if line.split(",")[0] == type:
            special_line = line
            break
      if special_line == None:
        raise ValueError("analyze_entities: cannot find matching type {} in entity_type.csv".format(type))
    
    # Found special line
    # NOTE: now only uses the first entity because that is guarateened to succeed
    res = generate_text(special_line.split(",")[1], num_outputs=max_num - i)
    for each_res in res:
      # print(len(res))
      filename = os.path.join(CACHE_SRC, "analyze_entities_" + type + "_{}.txt".format(i))
      with open(filename, "w", encoding='utf8') as fd:
        fd.write(each_res)
      i += 1
    res_list += res
    break

  return res_list


    
def analyze_syntax(tag, force_new=False, max_num=1):
  if max_num < 1:
    max_num = 1

  res_list = []

  special_line = None
  for i in range(max_num):
    filename = os.path.join(CACHE_SRC, "analyze_syntax_" + tag + "_{}.txt".format(i))
    # Checking cache folder first
    if force_new == False:
      if os.path.exists(filename):
        with open(filename, "r", encoding='utf8') as fd:
          res_list.append(fd.read())
        continue
    
    if special_line == None:
      file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "language_src", "syntax_type.csv")
      with open(file_path, "r") as fd:
        for line in fd.readlines():
          line = line.strip("\n")
          if line.split(",")[0] == tag:
            special_line = line
            break
      if special_line == None:
        raise ValueError("analyze_entities: cannot find matching tag {} in entity_type.csv".format(tag))
    
    # Found special line
    # NOTE: now only uses the first entity because that is guarateened to succeed
    res = generate_text(special_line.split(",")[1], num_outputs=max_num - i)
    for each_res in res:
      # print(len(res))
      filename = os.path.join(CACHE_SRC, "analyze_syntax_" + tag + "_{}.txt".format(i))
      with open(filename, "w", encoding='utf8') as fd:
        fd.write(each_res)
      i += 1
    res_list += res
    break

  return res_list


def generate_text(text, min_length = 30, max_length = 50, num_outputs = 1):
  import transformers
  import tensorflow as tf
  import sys

  GPT2Tokenizer = transformers.GPT2Tokenizer
  TFGPT2LMHeadModel = transformers.TFGPT2LMHeadModel

  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

  # add the EOS token as PAD token to avoid warnings
  model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

  # set seed to reproduce results. Feel free to change the seed though to get different results
  # tf.random.set_seed(0)

  # encode context the generation is conditioned on
  input_ids = tokenizer.encode(text, return_tensors='tf')

  # deactivate top_k sampling and sample only from 92% most likely words
  sample_outputs = model.generate(
      input_ids,
      do_sample=True,
      min_length=min_length,
      max_length=max_length,
      num_return_sequences=num_outputs,
      top_p=0.92, 
      top_k=0
  )

  res = []
  for i, sample_output in enumerate(sample_outputs):
    res.append(tokenizer.decode(sample_output, skip_special_tokens=True))

  return res

if __name__ == "__main__":
  pass
  logging.basicConfig(level=logging.INFO)
  classify_text("food & drink")
  
