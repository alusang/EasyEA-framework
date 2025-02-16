from llm2vec import LLM2Vec
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
import pickle
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model = PeftModel.from_pretrained(model,"McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

att_values_left = {}
att_values_right = {}

with open('rel', 'r', encoding='utf-8') as txf:
    lines = txf.readlines()
    total_lines = len(lines)
    half_lines = total_lines // 2

    for i, line in enumerate(tqdm(lines[:half_lines], desc="Processing left entities")):
        id, info = line.strip().split('\t')
        # these two code is needed when input file is att.txt or rel.txt
        info = eval(info)
        info = info[0]
        att_values_left[id] = info

    for i, line in enumerate(tqdm(lines[half_lines:], desc="Processing right entities")):
        id, info = line.strip().split('\t')
        info = eval(info)
        info = info[0]
        att_values_right[id] = info

keys_left = list(att_values_left.keys())
embed_left = l2v.encode(list(att_values_left.values()))
value_left = np.array(embed_left)

keys_right = list(att_values_right.keys())
embed_right = l2v.encode(list(att_values_right.values()))
value_right = np.array(embed_right)

for i, key in enumerate(keys_left):
    att_values_left[key] = value_left[i]
for i, key in enumerate(keys_right):
    att_values_right[key] = value_right[i]

with open('rel_dict_left.pkl', 'wb') as f:
    pickle.dump(att_values_left, f)
with open('rel_dict_right.pkl', 'wb') as f:
    pickle.dump(att_values_right, f)

embed_left = torch.nn.functional.normalize(torch.tensor(embed_left).clone().detach(), p=2, dim=1)
embed_right = torch.nn.functional.normalize(torch.tensor(embed_right).clone().detach(), p=2, dim=1)

cos_sim = torch.mm(embed_left, embed_right.transpose(0, 1)).numpy()


def calculate_metrics_with_mapping(sim_matrix, left_keys, right_keys, ground_truth, k_list=[1, 10,20,30, 50]):
    hits = {k: 0 for k in k_list}
    mrr = 0
    num_queries = sim_matrix.shape[0]

    ground_truth_dict = {left: right for left, right in ground_truth}

    for i in range(num_queries):
        left_id = left_keys[i]
        if left_id not in ground_truth_dict:
            continue
        true_right_id = ground_truth_dict[left_id]

        ranks = np.argsort(-sim_matrix[i])
        sorted_right_ids = [right_keys[j] for j in ranks]

        for k in k_list:
            if true_right_id in sorted_right_ids[:k]:
                hits[k] += 1

        if true_right_id in sorted_right_ids:
            rank_index = sorted_right_ids.index(true_right_id)
            mrr += 1 / (rank_index + 1)

    for k in k_list:
        hits[k] /= num_queries
    mrr /= num_queries

    return hits, mrr


ground_truth = []
with open('test', 'r', encoding='utf-8') as f:
    for line in f:
        left_id, right_id = line.strip().split('\t')
        ground_truth.append((left_id, right_id))

hits, mrr = calculate_metrics_with_mapping(cos_sim, keys_left, keys_right, ground_truth)
print(f"Hits@1: {hits[1]}\nHits@10: {hits[10]}\nHits@20: {hits[20]}\nHits@30: {hits[30]}\nHits@50: {hits[50]}\nMRR: {mrr}")
