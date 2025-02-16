from tqdm import tqdm
import pickle
import json
from openai import OpenAI

API_KEY = ""
BASE_URL = ""

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

with open('name.txt', 'r') as f:
    atts = f.readlines()

att = {}
for line in atts:
    entity_id, info = line.strip().split('\t')
    att[entity_id] = info

prompt = """
Translate the following Chinese entity names into English.
You must remember that you can only give me the English entity name and cannot return any additional information.
"""

att_ans = {}
att_keys = list(att.keys())
data_len = len(att_keys)//2
att_keys_chinese = att_keys[:data_len]
att_keys_english = att_keys[data_len:]

for key in tqdm(att_keys_chinese, desc="Processing Chinese att"):
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
        att_ans[key] = best_match_id
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
        att_ans[key] = best_match_id


for key in att_keys_english:
    att_ans[key] = att[key]

with open("name_trans.jsonl", "w") as f:
    for key, result in att_ans.items():
        f.write(json.dumps({key: result}) + "\n")
