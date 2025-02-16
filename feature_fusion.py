import os
import pickle
import numpy as np
from numpy.linalg import norm


def max_pooling_fusion(*features):
    return np.max(np.stack(features), axis=0)


def weight_fusion(*features):
    return np.average(np.stack(features), axis=0, weights=[1, 1, 1])


def concatenation_fusion(*features):
    return np.concatenate(features)


def mean_fusion(*features):
    return np.mean(np.stack(features), axis=0)

def load_data(directory, pkl_files):
    data_dict = {}
    for pkl_file in pkl_files:
        pkl_path = os.path.join(directory, pkl_file)
        with open(pkl_path, 'rb') as f:
            data_dict.update(pickle.load(f))
    return data_dict


def generate_fused_embeddings(directory, fusion_method, selected_dicts):
    loaded_data = {
        key: load_data(directory, [f'{key}_dict_left.pkl', f'{key}_dict_right.pkl'])
        for key in ['att', 'name', 'rel']
    }

    fused_embeddings = {}
    for key in loaded_data['att'].keys():
        features = [loaded_data[dict_key].get(key, []) for dict_key in selected_dicts]
        features = [f for f in features if np.size(f) > 0]
        if features:
            fused_embeddings[key] = fusion_method(*features)

    return fused_embeddings

def calculate_cosine_similarity(embed_left, embed_right):
    norm_left = norm(embed_left, axis=1, keepdims=True)
    norm_right = norm(embed_right, axis=1, keepdims=True)
    return np.dot(embed_left, embed_right.T) / (norm_left * norm_right.T)


def evaluate_predictions(sim_matrix, left_keys, right_keys, ground_truth, k_values):
    reciprocal_ranks = []
    hits = {k: 0 for k in k_values}
    ground_truth_dict = {left: right for left, right in ground_truth}

    candidates = {
        left_keys[i]: sorted(
            [(right_keys[j], sim_matrix[i, j]) for j in range(len(right_keys))],
            key=lambda x: x[1], reverse=True
        )[:10]
        for i in range(len(left_keys))
    }

    correct_predictions_at_1 = 0
    total_predictions = len(ground_truth)
    incorrect_hits1_candidates = {}

    for left_id, preds in candidates.items():
        if left_id not in ground_truth_dict:
            continue
        true_right_id = ground_truth_dict[left_id]

        rank = 0
        for i, (right_id, _) in enumerate(preds):
            if right_id == true_right_id:
                rank = i + 1
                if rank == 1:
                    correct_predictions_at_1 += 1
                break

        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
            for k in k_values:
                if rank <= k:
                    hits[k] += 1

        if rank != 1:
            incorrect_hits1_candidates[left_id] = preds

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    hits_results = {f'Hits@{k}': hits[k] / total_predictions for k in k_values}

    return {'MRR': mrr, **hits_results, 'candidates': candidates,
            'correct_predictions_at_1': correct_predictions_at_1,
            'total_predictions': total_predictions,
            'incorrect_hits1_candidates': incorrect_hits1_candidates}

def load_alignments_from_txt(file_path):
    with open(file_path, 'r') as f:
        alignments = [line.strip().split() for line in f]
    return alignments


directory = ""
fusion_method = concatenation_fusion
selected_dicts = ['att', 'rel', 'name']
embedding_dict = generate_fused_embeddings(directory, fusion_method, selected_dicts)

data_path = 'test'
dev_alignments = load_alignments_from_txt(data_path)

left_ids = [itm[0] for itm in dev_alignments if itm[0] in embedding_dict]
right_ids = [itm[1] for itm in dev_alignments if itm[1] in embedding_dict]
known_pairs = [(left, right) for left, right in dev_alignments if left in embedding_dict and right in embedding_dict]

embed_left = np.array([embedding_dict[left_id] for left_id in left_ids])
embed_right = np.array([embedding_dict[right_id] for right_id in right_ids])

cos_sim = calculate_cosine_similarity(embed_left, embed_right)

k_values = [1, 10, 20, 30]
evaluation_results = evaluate_predictions(cos_sim, left_ids, right_ids, known_pairs, k_values)

print(f"Correct Predictions: {evaluation_results['correct_predictions_at_1']}")
print(f"Total Predictions: {evaluation_results['total_predictions']}")
print(f"MRR: {evaluation_results['MRR']}")
for k in k_values:
    print(f"Hits@{k}: {evaluation_results[f'Hits@{k}']}")

with open('incorrect_hits1_cand.pkl', 'wb') as f:
    pickle.dump(evaluation_results['incorrect_hits1_candidates'], f)
