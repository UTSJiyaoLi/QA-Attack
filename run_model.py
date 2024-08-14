import torch
from victim_models import T5_model, T5_tokenizer

# This file can leverage and process the attention scores in 
# four (Abstractive, Extractive, Multiple Choice, Yes/No) QA tasks.


def clean_token(token):
    # Remove the leading underscore (▁) from SentencePiece tokens
    return token.replace('▁', '')


# Extractive Usage
def ask_extractive_question_with_attention(context, question, **generator_args):
    input_string = f"{question} \\n {context}"
    input_ids = T5_tokenizer.encode(input_string, return_tensors="pt")
    encoder_outputs = T5_model.encoder(input_ids, output_attentions=True)
    attentions = encoder_outputs.attentions
    res = T5_model.generate(input_ids, **generator_args)
    answer = T5_tokenizer.batch_decode(res, skip_special_tokens=True)
    
    tokens = T5_tokenizer.convert_ids_to_tokens(input_ids[0])

    averaged_attentions = [layer_attention.mean(dim=1) for layer_attention in attentions]
    normalized_attentions = [torch.nn.functional.softmax(layer_attention, dim=-1) for layer_attention in averaged_attentions]
    overall_attention = sum(normalized_attentions) / len(normalized_attentions)
    attention_scores = overall_attention[0].mean(dim=0).detach().numpy()

    filtered_attention_scores = []
    filtered_tokens = []
    special_tokens = ['</s>', '<unk>', 'n', '.', ',', '?']
    for token, score in zip(tokens, attention_scores):
        if token not in special_tokens:
            filtered_tokens.append(clean_token(token))
            filtered_attention_scores.append(score)

    print("\nAveraged and normalized attention scores for each token (excluding special and single-character tokens):")
    for token, score in zip(filtered_tokens, filtered_attention_scores):
        print(f"Token: {token}, Attention Score: {score}")

    attention_data = {
        'input_tokens': filtered_tokens,
        'attention_scores': filtered_attention_scores
    }
    
    return answer[0], attention_data


# Abstractive Usage
def ask_abstractive_question_with_attention(context, question, **generator_args):
    input_string = f"{question} \\n {context}"
    input_ids = T5_tokenizer.encode(input_string, return_tensors="pt")
    encoder_outputs = T5_model.encoder(input_ids, output_attentions=True)
    attentions = encoder_outputs.attentions
    res = T5_model.generate(input_ids, **generator_args)
    answer = T5_tokenizer.batch_decode(res, skip_special_tokens=True)
    
    tokens = T5_tokenizer.convert_ids_to_tokens(input_ids[0])

    averaged_attentions = [layer_attention.mean(dim=1) for layer_attention in attentions]
    normalized_attentions = [torch.nn.functional.softmax(layer_attention, dim=-1) for layer_attention in averaged_attentions]
    overall_attention = sum(normalized_attentions) / len(normalized_attentions)
    attention_scores = overall_attention[0].mean(dim=0).detach().numpy()

    filtered_attention_scores = []
    filtered_tokens = []
    for token, score in zip(tokens, attention_scores):
        if token not in ['</s>', '<unk>', 'n', '.', ',', '?','']:
            filtered_tokens.append(clean_token(token))
            filtered_attention_scores.append(score)

    print("\nAveraged and normalized attention scores for each token (excluding special and single-character tokens):")
    for token, score in zip(filtered_tokens, filtered_attention_scores):
        print(f"Token: {token}, Attention Score: {score}")

    attention_data = {
        'input_tokens': filtered_tokens,
        'attention_scores': filtered_attention_scores
    }
    
    return answer[0], attention_data



# Multiple Choice Usage
def ask_multiple_choice_question_with_attention(context, question, choices, **generator_args):
    
    choice_strings = " \\n ".join([f"{i}. {choice}" for i, choice in enumerate(choices, 1)])
    input_string = f"{question} \\n {choice_strings} \\n {context}"
    input_ids = T5_tokenizer.encode(input_string, return_tensors="pt")
    encoder_outputs = T5_model.encoder(input_ids, output_attentions=True)
    attentions = encoder_outputs.attentions
    res = T5_model.generate(input_ids, **generator_args)
    answer = T5_tokenizer.batch_decode(res, skip_special_tokens=True)
    
    tokens = T5_tokenizer.convert_ids_to_tokens(input_ids[0])

    averaged_attentions = [layer_attention.mean(dim=1) for layer_attention in attentions]
    normalized_attentions = [torch.nn.functional.softmax(layer_attention, dim=-1) for layer_attention in averaged_attentions]
    overall_attention = sum(normalized_attentions) / len(normalized_attentions)
    attention_scores = overall_attention[0].mean(dim=0).detach().numpy()

    filtered_attention_scores = []
    filtered_tokens = []
    for token, score in zip(tokens, attention_scores):
        if token not in ['</s>', '<unk>', 'n', '.', ',', '?']:
            filtered_tokens.append(clean_token(token))
            filtered_attention_scores.append(score)

    print("\nAveraged and normalized attention scores for each token (excluding special and single-character tokens):")
    for token, score in zip(filtered_tokens, filtered_attention_scores):
        print(f"Token: {token}, Attention Score: {score}")

    attention_data = {
        'input_tokens': filtered_tokens,
        'attention_scores': filtered_attention_scores
    }
    
    return answer[0], attention_data


# Yes/No Question Usage
def ask_yes_no_question_with_attention(context, question, **generator_args):
    input_string = f"{question} \\n {context}"
    input_ids = T5_tokenizer.encode(input_string, return_tensors="pt")
    encoder_outputs = T5_model.encoder(input_ids, output_attentions=True)
    attentions = encoder_outputs.attentions
    res = T5_model.generate(input_ids, **generator_args)
    answer = T5_tokenizer.batch_decode(res, skip_special_tokens=True)
    
    tokens = T5_tokenizer.convert_ids_to_tokens(input_ids[0])

    averaged_attentions = [layer_attention.mean(dim=1) for layer_attention in attentions]
    normalized_attentions = [torch.nn.functional.softmax(layer_attention, dim=-1) for layer_attention in averaged_attentions]
    overall_attention = sum(normalized_attentions) / len(normalized_attentions)
    attention_scores = overall_attention[0].mean(dim=0).detach().numpy()

    filtered_attention_scores = []
    filtered_tokens = []
    for token, score in zip(tokens, attention_scores):
        if token not in ['</s>', '<unk>', 'n', '.', ',', '?']:
            filtered_tokens.append(clean_token(token))
            filtered_attention_scores.append(score)

    print("\nAveraged and normalized attention scores for each token (excluding special and single-character tokens):")
    for token, score in zip(filtered_tokens, filtered_attention_scores):
        print(f"Token: {token}, Attention Score: {score}")

    attention_data = {
        'input_tokens': filtered_tokens,
        'attention_scores': filtered_attention_scores
    }
    
    return answer[0], attention_data


if __name__ == '__main__':

    # YES/NO Example usage
    context = "Paris is the capital of France. It is known for its art, fashion, and culture."
    question = "Is Paris the capital of France?"
    answer, attention_data = ask_yes_no_question_with_attention(context, question)
    print('Context: ', context)
    print('Question: ', question)
    print("Answer:", answer)
    print("Input Tokens:", attention_data['input_tokens'])
    print("Attention Scores:", attention_data['attention_scores'])

    # Multiple Choice Example usage
    # context = "Paris is the capital of France. It is known for its art, fashion, and culture."
    # question = "Which city is the capital of France?"
    # choices = ["Berlin", "Madrid", "Paris", "Rome"]
    # answer, attention_data = ask_multiple_choice_question_with_attention(context, question, choices)
    # print('Context: ', context)
    # print('Question: ', question)
    # print("Answer:", answer)
    # print("Input Tokens:", attention_data['input_tokens'])
    # print("Attention Scores:", attention_data['attention_scores'])

    # Abstractive Example usage
    # context = "Paris is the capital of France. It is known for its art, fashion, and culture."
    # question = "Describe the capital of France."
    # answer, attention_data = ask_abstractive_question_with_attention(context, question)
    # print('Context: ', context)
    # print('Question: ', question)
    # print("Answer:", answer)
    # print("Input Tokens:", attention_data['input_tokens'])
    # print("Attention Scores:", attention_data['attention_scores'])

    # Extractive Example Usage
    # context = "Paris is the capital of France. It is known for its art, fashion, and culture."
    # question = "What is the capital of France?"
    # answer, attention_data = ask_extractive_question_with_attention(context, question)
    # print('Context: ', context)
    # print('Question: ', question)
    # print("Answer:", answer)
    # print("Input Tokens:", attention_data['input_tokens'])
    # print("Attention Scores:", attention_data['attention_scores'])
