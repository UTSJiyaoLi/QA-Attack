import torch
from victim_models import T5_model, T5_tokenizer, ask_yes_no_question
from nltk.corpus import stopwords
import nltk
import string
from tool import clean_token, is_valid_token, convert_seconds_to_hms, translator
import time

# This file can rank each word importance with removal.

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)


def calculate_logits_importance(question, context, rate=0.1, top_k=20):
    # if mode == 'yn':
# This function takes logits from outputs of the model
    input_text = f"{question} \\n {context}"
    input_ids = T5_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).input_ids
    decoder_input_ids = torch.tensor([[T5_tokenizer.pad_token_id]])  # Start with pad token

    outputs = T5_model(input_ids, decoder_input_ids=decoder_input_ids)
    baseline_logits = outputs.logits

    importance_scores = []
    words = context.split()
    cleaned_words = [clean_token(word) for word in words]  # Apply clean_token to each word individually
    for i, word in enumerate(words):
        if not is_valid_token(word):
            continue  # Skip the iteration if the word is not valid
        new_context = " ".join(cleaned_words[:i] + ['<mask>'] + cleaned_words[i+1:])
        new_input_text = f"{question} \\n {new_context}"
        new_input_ids = T5_tokenizer.encode(new_input_text, return_tensors='pt', truncation=True, max_length=512)
        new_outputs = T5_model(new_input_ids, decoder_input_ids=decoder_input_ids)
        new_logits = new_outputs.logits

        # Measure change as the max absolute difference across all logits
        logit_change = torch.max(torch.abs(baseline_logits - new_logits)).item()
        importance_scores.append((word, logit_change, i))

    importance_scores.sort(key=lambda x: x[1], reverse=True)
    max_score = importance_scores[0][1] if importance_scores else 1
    normalized_scores = [(item[0], item[1] / max_score, item[2]) for item in importance_scores]
    if rate:
        top_k_context_tokens = normalized_scores[:int(len(normalized_scores) * rate)]
    elif top_k:
        top_k_context_tokens = normalized_scores[:top_k]
    else:
        top_k_context_tokens = normalized_scores[:]
    return top_k_context_tokens
    # if mode == 'other':



if __name__ == "__main__":

    start_time = time.time()
    # context = "In the third season, Damon helps Elena in bringing his brother, Stefan, back to Mystic Falls after Stefan becomes Klaus' henchman. The arrangement transpired after a bargain for his blood that would cure Damon of the werewolf bite he had received from Tyler. At first, he is reluctant to involve Elena in the rescue attempts, employing Alaric Saltzman, Elena's guardian, instead as Klaus does not know that Elena is alive after the sacrifice which frees Klaus' hybrid side. However, Elena involves herself, desperate to find Stefan. Damon, though hesitant at first, is unable to refuse her because of his love for her. He also points out to her that she once turned back from finding Stefan since she knew Damon would be in danger, clearly showing that she also has feelings for him. He tells her that ``when (he) drag(s) (his) brother from the edge to deliver him back to (her), (he) wants her to remember the things (she) felt while he was gone.'' When Stefan finally returns to Mystic Falls, his attitude is different from that of the first and second seasons. This causes a rift between Elena and Stefan whereas the relationship between Damon and Elena becomes closer and more intimate. A still loyal Elena, however, refuses to admit her feelings for Damon. In 'Dangerous Liaisons', Elena, frustrated with her feelings for him, tells Damon that his love for her may be a problem, and that this could be causing all their troubles. This incenses Damon, causing him to revert to the uncaring and reckless Damon seen in the previous seasons. The rocky relationship between the two continues until the sexual tension hits the fan and in a moment of heated passion, Elena -- for the first time in the three seasons -- kisses Damon of her own accord. This kiss finally causes Elena to admit that she loves both brothers and realize that she must ultimately make her choice as her own ancestress, Katherine Pierce, who turned the brothers, once did. In assessment of her feelings for Damon, she states this: ``Damon just sort of snuck up on me. He got under my skin and no matter what I do, I can't shake him.'' In the season finale, a trip designed to get her to safety forces Elena to make her choice: to go to Damon and possibly see him one last time; or to go to Stefan and her friends and see them one last time. She chooses the latter when she calls Damon to tell him her decision. Damon, who is trying to stop Alaric, accepts what she says and she tells him that maybe if she had met Damon before she had met Stefan, her choice may have been different. This statement causes Damon to remember the first night he did meet Elena which was, in fact, the night her parents died - before she had met Stefan. Not wanting anyone to know he was in town and after giving her some advice about life and love, Damon compels her to forget. He remembers this as he fights Alaric and seems accepting of his death when Alaric, whose life line is tied to Elena's, suddenly collapses in his arms. Damon is grief-stricken, knowing that this means that Elena has also died and yells, ``No! You are not dead!'' A heartbroken Damon then goes to the hospital demanding to see Elena when the doctor, Meredith Fell, tells him that she gave Elena vampire blood. The last shot of the season finale episode shows Elena in transition."
    # question = "does damon and elena get together in season 3"
    # context = "The knockout stage of the 2018 FIFA World Cup was the second and final stage of the competition, following the group stage. It began on 30 June with the round of 16 and ended on 15 July with the final match, held at the Luzhniki Stadium in Moscow. The top two teams from each group (16 in total) advanced to the knockout stage to compete in a single-elimination style tournament. A third place play-off was also played between the two losing teams of the semi-finals."    
    # question = "is all of new zealand in the same time zone"
    # context = "All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline."
    # question = "does ethanol take more energy make that produces"
    context = "In the 17th century, the Kingdom of France was one of the most powerful states in Europe. Under the reign of Louis XIV, France experienced significant cultural, political, and military advancements. Louis XIV, known as the Sun King, was famous for his ambitious projects, including the construction of the Palace of Versailles, which became a symbol of absolute monarchy."
    question = "What was the main purpose of constructing the Palace of Versailles during Louis XIV's reign?"
    # context = "Hydroxyzine preparations require a doctor's prescription. The drug is available in two formulations, the pamoate and the dihydrochloride or hydrochloride salts. Vistaril, Equipose, Masmoran, and Paxistil are preparations of the pamoate salt, while Atarax, Alamon, Aterax, Durrax, Tran-Q, Orgatrax, Quiess, and Tranquizine are of the hydrochloride salt."
    # question = "is harry potter and the escape from gringotts a roller coaster ride"
    top_k_context_tokens = calculate_logits_importance(question, context, rate=None, top_k=5)
    print("\n removal Tokens: ", top_k_context_tokens)
    end_time = time.time()
    print(convert_seconds_to_hms(end_time - start_time))
    
