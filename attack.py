from victim_models import ask_yes_no_question, T5_model, T5_tokenizer
from data import DatasetLoader, load_questions_and_contexts_from_json, fetch_document_content
import argparse
from transformers import BertTokenizer, BertForMaskedLM
import torch
from bert_mlm import BertMLMGuesser
from get_candidate import mask_and_predict, generate_candidate_sentences, count_unique_words_in_text1
from rank_with_attention import ranking_with_attention
from tool import save_show_result_to_file, convert_seconds_to_hms, get_logits
from rank_with_removal import calculate_logits_importance
import time
import json
from combined import combine_ranking_scores
import warnings


warnings.filterwarnings("ignore")


def attack(args):
    if args.c == 1:
        args.single = True
    
    loader = DatasetLoader(args.dataset_name)
    loader.load_dataset()
    if args.dataset_name == 'google/boolq':
        formatted_strings = loader.get_formatted_string(split='validation')
    if args.dataset_name == 'deepmind/narrativeqa':
        formatted_strings = loader.get_formatted_string(split='test')
    data = loader.get_samples(num_samples=args.n, randomize=False)
    successful_attacks, ave_words = 0, 0
    raw_answers, all_attacked_answers, all_adversary, sentences, questions, best_candidates, results = [], [], [], [], [], [], []
    candidate_file = f'results/candidate_{args.n}sample_{args.k}words_{args.c}_{args.ranking}_{args.combination}.txt'
    text_file = f'results/{args.n}samples_attack{args.k}_words_{args.c}_{args.ranking}_{args.combination}.txt'
    print(f'\n -----------   Start attack with {args.ranking}   -------------')
    for i in range(args.n):
        print(f'\n ----- Attacking No.{i + 1} sample -----')
        if args.dataset_name == 'google/boolq':
            question, sentence = data[i][1][0], data[i][1][1]
        if args.dataset_name == 'deepmind/narrativeqa':
            question, sentence = data[i][1][0], data[i][1][1]
            print(f'question: {question}')
            print(f'sentence: {sentence}')
        if args.dataset_name == 'rajpurkar/squad' or 'rajpurkar/squad_v2':
            question, sentence, answer = data[i][1][0], data[i][1][1], data[i][2]
        
        raw_answer = ask_yes_no_question(question, sentence)
        raw_logits = get_logits(question, sentence)
        sentences.append(sentence)
        questions.append(question)
        if args.ranking == 'attention':
            _, _, top_k_context_tokens, _ = ranking_with_attention(question, sentence, top_k=args.k, rate=args.rate)
            word_to_attack = [(j[0], j[2]) for j in top_k_context_tokens]
        elif args.ranking == 'removal':
            top_k_context_tokens = calculate_logits_importance(question, sentence, top_k=args.k, rate=args.rate)
            word_to_attack = [(j[0], j[2]) for j in top_k_context_tokens]
        elif args.ranking == 'combined':
            top_k_context_tokens = combine_ranking_scores(question, sentence, combination=args.combination, top_k=args.k, rate=args.rate)
            word_to_attack = [(j[0], j[1][1]) for j in top_k_context_tokens]
        else:
            raise ValueError("Wrong ranking strategy input.")
        if args.rate:
            ave_words += len(word_to_attack)
        print(f'\n Top {args.k}/{args.rate} context tokens of sample {i + 1}: {top_k_context_tokens}')

        candidates = mask_and_predict(sentence, word_to_attack, num_of_predict=args.c, single=args.single)
        candidate_sentences = generate_candidate_sentences(sentence, word_to_attack, candidates, single=args.single)
        print(f'\n sentences: {sentence}')
        # print(f'\n ---- Generated candidate sentences for sample {i + 1} ----')
        print(f'\n candidate_sentences: {candidate_sentences[0]}')
        attacked_answers = []
        max_logit_change = float('-inf')
        best_candidate = None
        successful_attack_flag = False  # Flag to track if a successful attack has been found
        for candidate_sentence in candidate_sentences:
            attacked_answer = ask_yes_no_question(data[i][1][0], candidate_sentence)
            candidate_logits = get_logits(data[i][1][0], candidate_sentence)
            logit_change = torch.max(torch.abs(raw_logits - candidate_logits)).item()
            attacked_answers.append(attacked_answer)

            # Update best candidate if current logit change is greater
            if logit_change > max_logit_change:
                max_logit_change = logit_change
                best_candidate = candidate_sentence

            # Check if the answer changed and no successful attack has been counted yet
            if attacked_answer != raw_answer and not successful_attack_flag:
                successful_attacks += 1
                successful_attack_flag = True  # Set flag to True after counting the attack
            
            # Optionally, break if a successful attack is found (depends on whether you want to evaluate all candidates or stop after the first success)
            if successful_attack_flag:
                break

        if best_candidate:
            best_candidates.append(best_candidate)
        results.append({
            "index": i,
            "question": question,
            "answer": answer,
            "raw_sentence": sentence,
            "best_candidate": best_candidate,
            "raw_answer": raw_answer,
            "predicted_answer": attacked_answer
        })  
        print(f'\n raw_answer: {raw_answer}; attacked_answers: {attacked_answers}')
        raw_answers.append(raw_answer)
        all_attacked_answers.append(attacked_answers)
        all_adversary.append(candidate_sentences[0])
    with open(f'json/newsqa_{args.n}samples_attack{args.k}_words_{args.c}_{args.ranking}_{args.combination}.json', 'w') as f:
        json.dump(results, f, indent=4)
    success_rate = successful_attacks / args.n
    show_result = [[ques, sent, adv, ans, raw] for ques, sent, adv, ans, raw in zip(questions, sentences, all_adversary, all_attacked_answers, raw_answers)]
    save_show_result_to_file(show_result, text_file)
    print(f'\n Successful attacks: {successful_attacks}, Total samples: {args.n}, Success rate: {success_rate:.2f}, Average words attack: {(ave_words/args.n):.2f}')
    return 'Finished!'


if __name__ == "__main__":

    start_time = time.time()
    # top_k and rate must have one "None" value.
    parser = argparse.ArgumentParser(description="Parameters in attacking process.")
    parser.add_argument('--dataset_name', type=str, default='rajpurkar/squad', required=False, help='dataset name')
    parser.add_argument('--k', type=int, default=5, required=False, help='top k words been attacked')
    parser.add_argument('--c', type=int, default=2, required=False, help='number of candidates predicted from bert mlm')
    parser.add_argument('--n', type=int, default=10, required=False, help='number of test/validation data attacked')
    parser.add_argument('--ranking', type=str, default='combined', required=False, choices=['combined', 'attention', 'removal'], help='ranking strategy')
    parser.add_argument('--rate', type=float, default=None, required=False, help='percentage of words been attacked')
    parser.add_argument('--mode', type=str, default='yn', required=False, help='attacking mode regarding to question & answer type')    
    parser.add_argument('--single', type=str, default=False, required=False, help='only choose one candidate from BERT MLM')
    parser.add_argument('--combination', type=str, default='norm-link', required=False, choices=['norm-add', 'norm-link'], help='type of combination: norm-add, norm-link')
    args = parser.parse_args()
    print(attack(args))
    print(f'Arguements: {vars(args)}')
    end_time = time.time()
    total_time = end_time - start_time
    print(convert_seconds_to_hms(total_time))

# nohup python -u attack.py > logs/output_3000_5_2_combined_norm-link.txt 2>&1 &