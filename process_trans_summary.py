import json

def jsonl_to_txt(jsonl_file_path, txt_file_path):
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file, \
         open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        for line in jsonl_file:
            try:
                data = json.loads(line)
                for key, value in data.items():
                    txt_file.write(f'{key}\t{value}\n')
            except json.JSONDecodeError:
                print(f"Error decoding JSON from line: {line}")

jsonl_to_txt('rel_summary.jsonl', 'rel')
