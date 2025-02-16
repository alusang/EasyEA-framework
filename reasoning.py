from tqdm import tqdm
import pickle
import json
from openai import OpenAI

API_KEY = ""
BASE_URL = ""

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

with open('incorrect_hits1_cand.pkl', 'rb') as f:
    candidates = pickle.load(f)

with open('ent2id', 'r') as f:
    data = f.readlines()

ent2id = {}
for line in data:
    entity_id, entity_name = line.strip().split('\t')
    ent2id[entity_id] = entity_name

with open('att.txt', 'r') as f:
    atts = f.readlines()

att = {}
for line in atts:
    entity_id, info = line.strip().split('\t')
    att[entity_id] = eval(info)

with open('rel.txt', 'r') as f:
    rels = f.readlines()

rel = {}
for line in rels:
    entity_id, info = line.strip().split('\t')
    rel[entity_id] = eval(info)

prompt = """
I will provide you with a source entity and 10 target entities. Your task is to select the target entity that most closely matches the source entity.
Each entity has three types of information: 
1. Name information
2. Attribute triples
3. Relationship triples

Follow this selection process:
1. Prioritize Name information as the primary criterion.
2. If Name information is ambiguous, use Attribute triples as a secondary criterion.
3. Finally, use Relationship triples as the tertiary criterion.

Once you are confident, return only the ID of the target entity you believe is the best match. Do not include any explanations, names, or other content in your response—ONLY the ID.
"""


def get_best_match(source_entity_name, candidate_entities):
    sn = ent2id[source_entity_name]
    if len(att[source_entity_name]) < 3:
        sat = att[source_entity_name]
    else:
        sat = att[source_entity_name][:3]
    if len(rel[source_entity_name]) < 3:
        srt = rel[source_entity_name]
    else:
        srt = rel[source_entity_name][:3]

    user_prompt = f"Source Entity :name: {sn}, attr_triple: {sat}, rel_triple: {srt}\nTarget Entities: \n"
    for key in candidate_entities:
        tid = key[0]
        tn = ent2id[key[0]]
        if len(att[key[0]]) < 3:
            tat = att[key[0]]
        else:
            tat = att[key[0]][:3]
        if len(rel[key[0]]) < 3:
            trt = rel[key[0]]
        else:
            trt = rel[key[0]][:3]
        user_prompt += f"id: {tid}, name: {tn}, attr_triple: {tat}, rel_triple: {trt}\n"
    try:
        response = client.chat.completions.create(
            model="llama-4-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=16,
            top_p=0.9,
            n=1
        )
        best_match_id = response.choices[0].message.content.strip()
        return best_match_id
    except Exception as e:
        print(f"Error during API call: {e}")
        return None


ans = {}
keys = list(candidates.keys())
for key in tqdm(keys):
    source_entity = key
    candidate_entities = candidates[key]
    best_match_id = get_best_match(source_entity, candidate_entities)
    if best_match_id is not None:
        ans[key] = [best_match_id]

with open("final_results.jsonl", "w") as f:
    for key, result in ans.items():
        f.write(json.dumps({key: result}) + "\n")
