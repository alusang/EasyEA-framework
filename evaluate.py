import re
import json

with open('final_results.jsonl', 'r') as f:
    lines = f.readlines()


with open('ent2id', 'r') as f:
    data = f.readlines()



ent2id = {}
for line in data:
    id, ent = line.strip().split('\t')
    ent2id[id] = ent

ans = {}
for line in lines:
    response = json.loads(line)
    value = list(response.values())[0]
    key = list(response.keys())[0]
    ans[key] = value

ground_truth = []
with open('test', 'r', encoding='utf-8') as f:
    for line in f:
        left_id, right_id = line.strip().split('\t')
        ground_truth.append((left_id, right_id))


def get_hits_1(ans, ground_truth):
    hits_1 = 0
    for key in ans:
        if (key,ans[key][0]) in ground_truth:
            hits_1 += 1
        else:
            print(key,ans[key][0])
            
    return hits_1 / len(ans)


hits_1 = get_hits_1(ans, ground_truth)
print(f"LLM Hits@1: {hits_1}\n")
