from datasets import load_metric
from victim_models import ask_yes_no_question, T5_model, T5_tokenizer
from textblob import TextBlob
from seqeval.metrics import f1_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# def grammar_score(text):
#     original_blob = TextBlob(text)
#     corrected_text = str(original_blob.correct())
#     original_words = original_blob.words
#     corrected_blob = TextBlob(corrected_text)
#     corrected_words = corrected_blob.words

#     # Calculate score based on the number of unchanged words to total words
#     unchanged_words = sum(1 for orig, corr in zip(original_words, corrected_words) if orig == corr)
#     total_words = len(original_words)
#     score = unchanged_words / total_words if total_words else 0
#     return score

def exact_match(reference, prediction):
    return int(reference.strip().lower() == prediction.strip().lower())

def compute_f1(reference, prediction):
    reference_tokens = reference.lower().split()
    prediction_tokens = prediction.lower().split()
    common_tokens = set(reference_tokens) & set(prediction_tokens)
    if not common_tokens:
        return 0.0
    # Precision and recall
    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(reference_tokens)
    return 2 * (precision * recall) / (precision + recall)

def compute_bleu(reference, prediction):
    reference = [reference.lower().split()]
    prediction = prediction.lower().split()

    # Weights for BLEU-1 (100% weight on 1-grams)
    weights_for_bleu1 = (1, 0, 0, 0)
    bleu1 = sentence_bleu(reference, prediction, weights=weights_for_bleu1)

    # Weights for BLEU-2 (weights on 1-grams and 2-grams)
    weights_for_bleu2 = (0.5, 0.5, 0, 0)
    bleu2 = sentence_bleu(reference, prediction, weights=weights_for_bleu2)

    return bleu1, bleu2

def rouge(raw_answer, predicted_answer):
    # Create a ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Score the single reference sentence
    scores = scorer.score(raw_answer, predicted_answer)
    
    # Extract precision scores for ROUGE-1 and ROUGE-2
    rouge1_precision = scores['rouge1'][0]  # Precision for ROUGE-1
    rouge2_precision = scores['rouge2'][0]  # Precision for ROUGE-2
    
    return rouge1_precision, rouge2_precision

def word_embeddings(sent_1, sent_2):
    """
    Input two sentences, and return a cosine score between them.
    """

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    # Tokenize and encode sentences
    tokens1 = tokenizer.encode_plus(sent_1, add_special_tokens=True, return_tensors='pt')
    tokens2 = tokenizer.encode_plus(sent_2, add_special_tokens=True, return_tensors='pt')

    # Obtain BERT embeddings
    with torch.no_grad():
        outputs1 = model(**tokens1)
        outputs2 = model(**tokens2)

    hidden_states1 = outputs1.hidden_states
    hidden_states2 = outputs2.hidden_states

    # Extract the final BERT embeddings (CLS token)
    sentence_embedding1 = hidden_states1[-1][:, 0, :]
    sentence_embedding2 = hidden_states2[-1][:, 0, :]

    # Calculate cosine similarity
    similarity = cosine_similarity(sentence_embedding1, sentence_embedding2)[0][0]

    # print("Cosine Similarity:", similarity)
    return similarity

def evaluate_all(reference, prediction):
    results = {}

    # Calculate Exact Match Score
    results['exact_match'] = exact_match(reference, prediction)

    # Calculate F1 Score
    results['f1_score'] = compute_f1(reference, prediction)

    # Calculate BLEU Scores
    bleu_scores = compute_bleu(reference, prediction)
    results['bleu1'] = bleu_scores[0]
    results['bleu2'] = bleu_scores[1]

    # Calculate ROUGE Scores
    rouge_scores = rouge(reference, prediction)
    results['rouge1'] = rouge_scores[0]
    results['rouge2'] = rouge_scores[1]

    # Calculate word embeddings
    results['answer similarity'] = word_embeddings(reference, prediction)
    
    return results


if __name__ == '__main__':

    reference_answer = "The capital of France is Paris"
    predicted_answer = "Paris is the capital of France"
    # print("BLEU Score:", compute_bleu(reference_answer, predicted_answer))
    # print("ROUGE Score:", rouge(reference_answer, predicted_answer))
    # print('exact match: ', exact_match(reference_answer, predicted_answer))
    # print("f1:", compute_f1(reference_answer, predicted_answer))
    print(word_embeddings(reference_answer, predicted_answer))