import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from victim_models import T5_model, T5_tokenizer
from nltk.corpus import stopwords
import nltk
import string
from tool import clean_token, is_valid_token


# Ensure you have downloaded the stopwords
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)


def ranking_with_attention(question, context, mode='yn', top_k=5, rate=0.1, **generator_args):

    if mode == 'yn':
        # Add '?' to the end of the question if it doesn't already end with one
        if not question.endswith('?'):
            question += '?'
        # get attention scores
        input_string = f"{question} \\n {context}"
        input_ids = T5_tokenizer.encode(input_string, return_tensors="pt", truncation=True, max_length=512)
        encoder_outputs = T5_model.encoder(input_ids, output_attentions=True)
        attentions = encoder_outputs.attentions
        res = T5_model.generate(input_ids, **generator_args)
        answer = T5_tokenizer.batch_decode(res, skip_special_tokens=True)
        
        tokens = T5_tokenizer.convert_ids_to_tokens(input_ids[0])
        original_tokens = tokens.copy()
        tokens = [token.replace('▁', '') for token in tokens]

        averaged_attentions = [layer_attention.mean(dim=1) for layer_attention in attentions]
        normalized_attentions = [torch.nn.functional.softmax(layer_attention, dim=-1) for layer_attention in averaged_attentions]
        overall_attention = sum(normalized_attentions) / len(normalized_attentions)
        attention_scores = overall_attention[0].mean(dim=0).detach().numpy()
        max_score = max(attention_scores) if attention_scores.size > 0 else 1  # Avoid division by zero
        attention_scores = attention_scores / max_score

        attention_data = list(zip(tokens, attention_scores))
        question_tokens = []
        context_tokens = []
        is_context = False
        for i, (token, score) in enumerate(attention_data):
            if token == '?':
                is_context = True
                continue
            if is_context:
                context_tokens.append((token, score, i - len(question.split()) - 1))
            else:
                question_tokens.append((token, score, i))
        # Filter context tokens to only include those present in context_words
        context_words = context.split()
        context_words = [clean_token(words_) for words_ in context_words]
        context_tokens = [(token, score, context_words.index(token)) for token, score, idx in context_tokens if is_valid_token(token) and token in context_words]
        context_tokens.sort(key=lambda x: x[1], reverse=True)
        if rate:
            top_k_context_tokens = context_tokens[:int(len(context_tokens) * rate)]
        elif top_k:
            top_k_context_tokens = context_tokens[:top_k]
        else:
            top_k_context_tokens = context_tokens[:]
        masked_sentences = []
        for token, score, idx in top_k_context_tokens:
            if idx < len(context_words):
                words = context_words[:]
                words[idx] = '[MASK]'
                masked_sentence = ' '.join(words)
                masked_sentences.append(masked_sentence)
        
        attention_data_dict = {
            'question_tokens': [token for token, score, idx in question_tokens],
            'question_attention_scores': [score for token, score, idx in question_tokens],
            'context_tokens': [token for token, score, idx in context_tokens],
            'context_attention_scores': [score for token, score, idx in context_tokens],
            'context_indices': [idx for token, score, idx in context_tokens]
        }
    
    return answer[0], attention_data_dict, top_k_context_tokens, masked_sentences


if __name__ == "__main__":
    
    # context = "In the third season, Damon helps Elena in bringing his brother, Stefan, back to Mystic Falls after Stefan becomes Klaus' henchman. The arrangement transpired after a bargain for his blood that would cure Damon of the werewolf bite he had received from Tyler. At first, he is reluctant to involve Elena in the rescue attempts, employing Alaric Saltzman, Elena's guardian, instead as Klaus does not know that Elena is alive after the sacrifice which frees Klaus' hybrid side. However, Elena involves herself, desperate to find Stefan. Damon, though hesitant at first, is unable to refuse her because of his love for her. He also points out to her that she once turned back from finding Stefan since she knew Damon would be in danger, clearly showing that she also has feelings for him. He tells her that ``when (he) drag(s) (his) brother from the edge to deliver him back to (her), (he) wants her to remember the things (she) felt while he was gone.'' When Stefan finally returns to Mystic Falls, his attitude is different from that of the first and second seasons. This causes a rift between Elena and Stefan whereas the relationship between Damon and Elena becomes closer and more intimate. A still loyal Elena, however, refuses to admit her feelings for Damon. In 'Dangerous Liaisons', Elena, frustrated with her feelings for him, tells Damon that his love for her may be a problem, and that this could be causing all their troubles. This incenses Damon, causing him to revert to the uncaring and reckless Damon seen in the previous seasons. The rocky relationship between the two continues until the sexual tension hits the fan and in a moment of heated passion, Elena -- for the first time in the three seasons -- kisses Damon of her own accord. This kiss finally causes Elena to admit that she loves both brothers and realize that she must ultimately make her choice as her own ancestress, Katherine Pierce, who turned the brothers, once did. In assessment of her feelings for Damon, she states this: ``Damon just sort of snuck up on me. He got under my skin and no matter what I do, I can't shake him.'' In the season finale, a trip designed to get her to safety forces Elena to make her choice: to go to Damon and possibly see him one last time; or to go to Stefan and her friends and see them one last time. She chooses the latter when she calls Damon to tell him her decision. Damon, who is trying to stop Alaric, accepts what she says and she tells him that maybe if she had met Damon before she had met Stefan, her choice may have been different. This statement causes Damon to remember the first night he did meet Elena which was, in fact, the night her parents died - before she had met Stefan. Not wanting anyone to know he was in town and after giving her some advice about life and love, Damon compels her to forget. He remembers this as he fights Alaric and seems accepting of his death when Alaric, whose life line is tied to Elena's, suddenly collapses in his arms. Damon is grief-stricken, knowing that this means that Elena has also died and yells, ``No! You are not dead!'' A heartbroken Damon then goes to the hospital demanding to see Elena when the doctor, Meredith Fell, tells him that she gave Elena vampire blood. The last shot of the season finale episode shows Elena in transition."
    # question = "does damon and elena get together in season 3"
    # context = "All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline."
    # question = "does ethanol take more energy make that produces"
    # context = "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south, bounded by Asia and Australia in the west, and the Americas in the east."
    # question = "Is the Pacific Ocean the largest ocean on Earth?"
    context = "In the 17th century, the Kingdom of France was one of the most powerful states in Europe. Under the reign of Louis XIV, France experienced significant cultural, political, and military advancements. Louis XIV, known as the Sun King, was famous for his ambitious projects, including the construction of the Palace of Versailles, which became a symbol of absolute monarchy."
    question = "What was the main purpose of constructing the Palace of Versailles during Louis XIV's reign?"
    # Answer: Yes.
    answer, attention_data, top_k_context_tokens, masked_sentences = ranking_with_attention(question, context, mode='yn', top_k=5, rate=None)
    # print("Answer:", answer)
    print(f'question_tokens: {attention_data["question_tokens"]}')
    print(f'question_attention_scores: {attention_data["question_attention_scores"]}')
    print("Context ranked Tokens:", attention_data['context_tokens'])
    print("Context ranked Attention Scores:", attention_data['context_attention_scores'])
    # print("Context Indices:", attention_data['context_indices'])
    print("attention Tokens:", top_k_context_tokens)
    # print('context: ', context)
    # print("\nMasked Sentences:")
    # for ms in masked_sentences:
    #     print(ms)
