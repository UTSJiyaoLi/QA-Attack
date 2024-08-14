import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "allenai/unifiedqa-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def get_prob(question, context):
    input_text = f"{question} \\n {context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids, output_scores=True, return_dict_in_generate=True)
    # Decoding the generated token to text
    decoded_output = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    # Analyzing scores for the first token generated
    scores = output.scores[0]  # Assuming the first token is the most relevant
    probs = torch.nn.functional.softmax(scores, dim=-1)
    yes_id = tokenizer.encode('yes', add_special_tokens=False)[0]
    no_id = tokenizer.encode('no', add_special_tokens=False)[0]
    yes_prob = probs[0][yes_id].item()
    no_prob = probs[0][no_id].item()

    return decoded_output, yes_prob, no_prob

# Example use case
context = "Hydroxyzine preparations require a doctor's prescription. The drug is available in two formulations, the pamoate and the dihydrochloride or hydrochloride salts. Vistaril, Equipose, Masmoran, and Paxistil are preparations of the pamoate salt, while Atarax, Alamon, Aterax, Durrax, Tran-Q, Orgatrax, Quiess, and Tranquizine are of the hydrochloride salt."
question = "is harry potter and the escape from gringotts a roller coaster ride"
decoded_answer, yes_score, no_score = get_prob(question, context)
print(f"Decoded Answer: {decoded_answer}")
print(f"Yes Probability: {yes_score}")
print(f"No Probability: {no_score}")
