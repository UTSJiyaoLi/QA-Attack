import json
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_metric
from victim_models import ask_yes_no_question, T5_model, T5_tokenizer
from textblob import TextBlob
from seqeval.metrics import f1_score
from rouge_score import rouge_scorer
from metrics import exact_match, rouge, compute_f1, compute_bleu, evaluate_all, similarity, word_embeddings


def read_and_evaluate(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    total_scores = {
        'exact_match': 0,
        'f1_score': 0,
        'rouge1': 0,
        'rouge2': 0,
        'bleu1': 0,
        'bleu2': 0,
        'answer similarity': 0
        # Initialize other scores here if needed
    }
    num_samples = len(data)
    print(num_samples)
    for sample in data:
        # Extracting necessary details
        reference = sample['raw_sentence']
        prediction = sample['best_candidate']
        
        # Evaluate all metrics
        scores = evaluate_all(reference, prediction)
        
        # Summing up the scores for averaging later
        for key in scores:
            total_scores[key] += scores[key]
    # Computing averages
    averages = {key: total / num_samples for key, total in total_scores.items()}
    
    return averages


if __name__ == '__main__':
    # Path to your JSON file
    file_path = '/home/jiyli/Data/qa_attack/json/3000samples_attack5_words_2_combined_norm-link.json'
    average_scores = read_and_evaluate(file_path)
    print(average_scores)
