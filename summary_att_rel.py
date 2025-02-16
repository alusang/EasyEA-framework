from tqdm import tqdm
import pickle
import json
from openai import OpenAI

API_KEY = ""
BASE_URL = ""

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

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
You are an expert who can provide concise explanations based on entity information. I will give you the properties of an entity in the form of a triple (subject, predicate, object). Using this information along with your general knowledge, please provide a short description of the entity.
- The explanation should be no longer than 100 words.
- Focus on summarizing the entity based on the given information and your general knowledge.
- Do not include unnecessary details or explanations beyond the entity description.
- Every word in your answer must be English.
Example:
Entity Information: (Albert Einstein, profession, Physicist), (Albert Einstein, known for, Theory of Relativity)
Explanation: Albert Einstein was a renowned physicist best known for developing the Theory of Relativity, a fundamental theory in modern physics.
Now, please summarize the following entity information and return an desctription in English:
"""


att_ans = {}
att_keys = list(att.keys())
for key in tqdm(att_keys, desc="Processing att"):
    info = att[key]
    user_prompt = f"\n{info}\n"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=256,
            top_p=0.9,
            n=1
        )
        best_match_id = response.choices[0].message.content.strip()
        att_ans[key] = [best_match_id]
    except Exception as e:
        print(f"Error during API call for att key {key}: {e}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=256,
            top_p=0.9,
            n=1
        )
        best_match_id = response.choices[0].message.content.strip()
        att_ans[key] = [best_match_id]

with open("att_summary.jsonl", "w") as f:
    for key, result in att_ans.items():
        f.write(json.dumps({key: result}) + "\n")


rel_ans = {}
rel_keys = list(rel.keys())
for key in tqdm(rel_keys, desc="Processing rel"):
    info = rel[key]
    user_prompt = f"\n{info}\n"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=256,
            top_p=0.9,
            n=1
        )
        best_match_id = response.choices[0].message.content.strip()
        rel_ans[key] = [best_match_id]
    except Exception as e:
        print(f"Error during API call for rel key {key}: {e}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=256,
            top_p=0.9,
            n=1
        )
        best_match_id = response.choices[0].message.content.strip()
        rel_ans[key] = [best_match_id]

with open("rel_summary.jsonl", "w") as f:
    for key, result in rel_ans.items():
        f.write(json.dumps({key: result}) + "\n")
