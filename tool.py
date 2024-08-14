import time
from nltk.corpus import stopwords
import nltk
import string
import re
import torch
from victim_models import ask_yes_no_question, T5_model, T5_tokenizer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Some tool functions.
translator = str.maketrans('', '', string.punctuation)
def save_two_lists_to_file(list1, list2, filename):
    with open(filename, 'w') as file:
        file.write("List 1:\n")
        for line in list1:
            file.write(line + '\n')
        file.write("\nList 2:\n")
        for line in list2:
            file.write(line + '\n')

def save_show_result_to_file(show_result, filename):
    with open(filename, 'w') as file:
        for item in show_result:
            file.write(f"Question: {item[0]}\n")
            file.write(f"Context: {item[1]}\n")
            file.write(f"Adversary: {item[2]}\n")
            file.write(f"Attacked Answers: {item[3]}\n")
            file.write(f"Raw Answer: {item[4]}\n")
            file.write("\n")

def convert_seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"Time taken: {hours} hours, {minutes} minutes, {seconds:.2f} seconds"

def clean_token(token):
    token = re.sub(r"[^\w']|(\(.*?\))", '', token)
    return token.replace('▁', '')

def is_valid_token(token):
    # Check if the token is not empty, not a stopword, not punctuation, not a single character, not a number, and not a specific unwanted token
    unwanted_tokens = {'</s>', ').', '--'}
    return token and token.lower() not in stop_words and token not in punctuation and len(token) > 1 and not token.isdigit() and token not in unwanted_tokens

def get_logits(question, context):
    input_text = f"{question} \\n {context}"
    input_ids = T5_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).input_ids
    decoder_input_ids = torch.tensor([[T5_tokenizer.pad_token_id]])  # Start with pad token
    outputs = T5_model(input_ids, decoder_input_ids=decoder_input_ids)
    baseline_logits = outputs.logits
    return outputs.logits

if __name__ == '__main__':
    # Example usage
    # list1 = [
    #     "All biomass goes through at least some of these steps:",
    #     "it needs to be grown, collected, dried, fermented, distilled, and burned."
    # ]

    # list2 = [
    #     "All of these steps require resources and an infrastructure.",
    #     "The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance."
    # ]

    context = "All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline."
    question = "does ethanol take more energy make that produces"
    # filename = "output.txt"
    # save_two_lists_to_file(list1, list2, filename)
    # print(f"Two lists of strings saved to {filename}")
    print(get_logits(question, context))
