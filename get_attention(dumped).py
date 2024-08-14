import torch
from victim_models import T5_model, T5_tokenizer

# This code was initially used to investgate the attention score from qa model,
# will not be used in the future, dumped.
def get_attention(input_string, **generator_args):
    # This function outputs the attention scores corresponding 
    # the input tokes( without filtering special tolens)

    input_ids = T5_tokenizer.encode(input_string, return_tensors="pt")
    # Get encoder outputs with attention
    encoder_outputs = T5_model.encoder(input_ids, output_attentions=True)
    attentions = encoder_outputs.attentions
    # Generate output sequences
    res = T5_model.generate(input_ids, **generator_args)
    decoded_output = T5_tokenizer.batch_decode(res, skip_special_tokens=True)
    
    # Tokenize input to get token mappings
    tokens = T5_tokenizer.convert_ids_to_tokens(input_ids[0])

    # Average attention weights across heads for each layer
    averaged_attentions = [layer_attention.mean(dim=1) for layer_attention in attentions]

    # Sum the attention weights to normalize them (each token's attention sum to 1)
    normalized_attentions = [torch.nn.functional.softmax(layer_attention, dim=-1) for layer_attention in averaged_attentions]

    # Sum normalized attentions across all layers
    overall_attention = sum(normalized_attentions) / len(normalized_attentions)
    
    # Extract attention scores for each token in the input
    attention_scores = overall_attention[0].mean(dim=0).detach().numpy()

    print("\nAveraged and normalized attention scores for each token:")
    for i, token in enumerate(tokens):
        print(f"Token: {token}, Attention Score: {attention_scores[i]}")

    # Prepare attention data
    attention_data = {
        'input_tokens': tokens,
        'attention_scores': attention_scores
    }
    return decoded_output, attention_data


if __name__ == '__main__':

    # Example usage
    input_string = "What is the capital of France?"
    output, attention_data = get_attention(input_string)
    print("Output:", output)
    print("Input Tokens:", attention_data['input_tokens'])
    print("Attention Scores:", attention_data['attention_scores'])
