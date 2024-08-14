import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "allenai/unifiedqa-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def calculate_logits_importance(question, context):
    input_text = f"{question} \\n {context}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])  # Start with pad token

    outputs = model(input_ids, decoder_input_ids=decoder_input_ids)
    baseline_logits = outputs.logits
    print(f'baseline_logits: {baseline_logits}')
    importance_scores = []

    words = context.split()
    for i, word in enumerate(words):
        new_context = " ".join(words[:i] + ['<mask>'] + words[i+1:])
        new_input_text = f"{question} \\n {new_context}"
        new_input_ids = tokenizer(new_input_text, return_tensors='pt').input_ids

        new_outputs = model(new_input_ids, decoder_input_ids=decoder_input_ids)
        new_logits = new_outputs.logits

        # Measure change as the max absolute difference across all logits
        logit_change = torch.max(torch.abs(baseline_logits - new_logits)).item()
        importance_scores.append((word, logit_change))

    importance_scores.sort(key=lambda x: x[1], reverse=True)
    return importance_scores

# Example use
context = "All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline."
question = "does ethanol take more energy make that produces"
importance_scores = calculate_logits_importance(question, context)
print("Word Importance Scores:", importance_scores)
