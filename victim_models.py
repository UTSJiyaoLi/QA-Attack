import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings


warnings.filterwarnings("ignore")


T5_model_name = "allenai/unifiedqa-t5-small"
T5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
T5_model = T5ForConditionalGeneration.from_pretrained(T5_model_name, output_attentions=True)

def ask_yes_no_question(question, context):
    
    input_text = f"{question} \\n {context}"
    inputs = T5_tokenizer.encode(input_text, return_tensors='pt')
    outputs = T5_model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
    answer = T5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

if __name__ == "__main__":
    question = "Does ethanol take more energy to make than it produces?"
    context = ("All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, "
               "distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy "
               "input into the process compared to the energy released by burning the resulting ethanol fuel is known as the "
               "energy balance (or 'energy returned on energy invested'). Figures compiled in a 2007 report by National Geographic "
               "Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to "
               "create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is "
               "more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates "
               "are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a "
               "separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow "
               "productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns "
               "about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, "
               "after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline.")
    answer = ask_yes_no_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
