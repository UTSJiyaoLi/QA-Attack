import torch
from transformers import BertTokenizer, BertForMaskedLM
from bert_mlm import BertMLMGuesser
from itertools import product

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# def mask_and_predict(sentence, words_to_mask, num_of_predict=5):
#     guesser = BertMLMGuesser()
#     results = {}

#     for word_to_mask, index in words_to_mask:
#         words = sentence.split()
#         words[index] = '[MASK]'
#         masked_sentence = ' '.join(words)
#         # get one more predictions in case
#         predicted_tokens = guesser.guess_masked_token(masked_sentence, num_of_predict=num_of_predict+1)

#         # Filter out the original word and subwords that might be generated
#         # predicted_tokens = [token for token in predicted_tokens if word_to_mask.lower() not in token.lower()]

#         # Filter out the original word but keep subwords
#         predicted_tokens = [token for token in predicted_tokens if token.lower() != word_to_mask.lower()]

#         # predicted_tokens = predicted_tokens[:num_of_predict]
#         results[(word_to_mask, index)] = predicted_tokens

#     return results

# To avoid exceed length 512 
def mask_and_predict(sentence, words_to_mask, num_of_predict=5, window_size=256, stride=128, single=False):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    guesser = BertMLMGuesser()
    results = {}

    for word_to_mask, index in words_to_mask:
        words = sentence.split()
        if index >= len(words):
            continue

        start_index = max(0, index - window_size // 2)
        end_index = min(len(words), start_index + window_size)
        if end_index - start_index < window_size:
            start_index = max(0, end_index - window_size)

        segments = []
        while start_index < len(words):
            segment = words[start_index:end_index]
            if index >= start_index and index < end_index:
                segments.append((segment, index - start_index))
            start_index += stride
            end_index = min(len(words), start_index + window_size)

        for segment, masked_index in segments:
            segment[masked_index] = '[MASK]'
            masked_sentence = ' '.join(segment)
            token_ids = tokenizer.encode(masked_sentence, add_special_tokens=True)
            if len(token_ids) > 512:
                continue

            predicted_tokens = guesser.guess_masked_token(masked_sentence, num_of_predict=num_of_predict + 1)
            predicted_tokens = [token for token in predicted_tokens if token.lower() != word_to_mask.lower()]
            
            if single:
                predicted_tokens = [predicted_tokens[-1]] if predicted_tokens else [] # Return only the first candidate
            if (word_to_mask, index) not in results:
                results[(word_to_mask, index)] = predicted_tokens
            else:
                results[(word_to_mask, index)].extend(predicted_tokens)
                results[(word_to_mask, index)] = list(set(results[(word_to_mask, index)]))
    return results


def count_unique_words_in_text1(context1, context2):
    # Split the contexts into sets of words
    words1 = set(context1.split())
    words2 = set(context2.split())

    # Find the unique words in context 1
    unique_words1 = words1 - words2

    # Count the number of unique words in context 1
    num_unique_words1 = len(unique_words1)

    return f"Number of unique words: {num_unique_words1}, they are{unique_words1}"


def generate_candidate_sentences(sentence, words_to_mask, candidates, single=False):
    candidate_sentences = []

    if single:
        # Generate sentences with only one candidate per masked word
        for word_pair in words_to_mask:
            words = sentence.split()
            if word_pair in candidates and candidates[word_pair]:
                candidate = candidates[word_pair][0]  # Use the first candidate
                words[word_pair[1]] = candidate
                candidate_sentence = ' '.join(words)
                candidate_sentences.append(candidate_sentence)
    else:
        # Generate all combinations of candidates for the words to mask
        candidate_combinations = product(*[candidates[word_pair] for word_pair in words_to_mask if word_pair in candidates])
        for combination in candidate_combinations:
            words = sentence.split()
            for (word_to_mask, index), candidate in zip(words_to_mask, combination):
                words[index] = candidate
            candidate_sentence = ' '.join(words)
            candidate_sentences.append(candidate_sentence)

    return candidate_sentences


if __name__ == "__main__":
    # sentence = "The knockout stage of the 2018 FIFA World Cup was the second and final stage of the competition, following the group stage. It began on 30 June with the round of 16 and ended on 15 July with the final match, held at the Luzhniki Stadium in Moscow. The top two teams from each group (16 in total) advanced to the knockout stage to compete in a single-elimination style tournament. A third place play-off was also played between the two losing teams of the semi-finals."    
    # question = "is all of new zealand in the same time zone"
    question = "is there a word with q without u"
    sentence = "Of the 71 words in this list, 67 are nouns, and most would generally be considered loanwords; the only modern-English words that contain Q not followed by U and are not borrowed from another language are qiana, qwerty, and tranq. However, all of the loanwords on this list are considered to be naturalised in English according to at least one major dictionary (see References), often because they refer to concepts or societal roles that do not have an accurate equivalent in English. For words to appear here, they must appear in their own entry in a dictionary; words which occur only as part of a longer phrase are not included."

    # sentence = "In the third season, Damon helps Elena in bringing his brother, Stefan, back to Mystic Falls after Stefan becomes Klaus' henchman. The arrangement transpired after a bargain for his blood that would cure Damon of the werewolf bite he had received from Tyler. At first, he is reluctant to involve Elena in the rescue attempts, employing Alaric Saltzman, Elena's guardian, instead as Klaus does not know that Elena is alive after the sacrifice which frees Klaus' hybrid side. However, Elena involves herself, desperate to find Stefan. Damon, though hesitant at first, is unable to refuse her because of his love for her. He also points out to her that she once turned back from finding Stefan since she knew Damon would be in danger, clearly showing that she also has feelings for him. He tells her that ``when (he) drag(s) (his) brother from the edge to deliver him back to (her), (he) wants her to remember the things (she) felt while he was gone.'' When Stefan finally returns to Mystic Falls, his attitude is different from that of the first and second seasons. This causes a rift between Elena and Stefan whereas the relationship between Damon and Elena becomes closer and more intimate. A still loyal Elena, however, refuses to admit her feelings for Damon. In 'Dangerous Liaisons', Elena, frustrated with her feelings for him, tells Damon that his love for her may be a problem, and that this could be causing all their troubles. This incenses Damon, causing him to revert to the uncaring and reckless Damon seen in the previous seasons. The rocky relationship between the two continues until the sexual tension hits the fan and in a moment of heated passion, Elena -- for the first time in the three seasons -- kisses Damon of her own accord. This kiss finally causes Elena to admit that she loves both brothers and realize that she must ultimately make her choice as her own ancestress, Katherine Pierce, who turned the brothers, once did. In assessment of her feelings for Damon, she states this: ``Damon just sort of snuck up on me. He got under my skin and no matter what I do, I can't shake him.'' In the season finale, a trip designed to get her to safety forces Elena to make her choice: to go to Damon and possibly see him one last time; or to go to Stefan and her friends and see them one last time. She chooses the latter when she calls Damon to tell him her decision. Damon, who is trying to stop Alaric, accepts what she says and she tells him that maybe if she had met Damon before she had met Stefan, her choice may have been different. This statement causes Damon to remember the first night he did meet Elena which was, in fact, the night her parents died - before she had met Stefan. Not wanting anyone to know he was in town and after giving her some advice about life and love, Damon compels her to forget. He remembers this as he fights Alaric and seems accepting of his death when Alaric, whose life line is tied to Elena's, suddenly collapses in his arms. Damon is grief-stricken, knowing that this means that Elena has also died and yells, ``No! You are not dead!'' A heartbroken Damon then goes to the hospital demanding to see Elena when the doctor, Meredith Fell, tells him that she gave Elena vampire blood. The last shot of the season finale episode shows Elena in transition."
    # question = 'does damon and elena get together in season 3'
    # sentence = "All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline."
    # sentence = "In the third season, Damon helps Elena in bringing his brother, Stefan, back to Mystic Falls after Stefan becomes Klaus' henchman. The arrangement transpired after a bargain for his blood that would cure Damon of the werewolf bite he had received from Tyler. At first, he is reluctant to involve Elena in the rescue attempts, employing Alaric Saltzman, Elena's guardian, instead as Klaus does not know that Elena is alive after the sacrifice which frees Klaus' hybrid side. However, Elena involves herself, desperate to find Stefan. Damon, though hesitant at first, is unable to refuse her because of his love for her. He also points out to her that she once turned back from finding Stefan since she knew Damon would be in danger, clearly showing that she also has feelings for him. He tells her that ``when (he) drag(s) (his) brother from the edge to deliver him back to (her), (he) wants her to remember the things (she) felt while he was gone.'' When Stefan finally returns to Mystic Falls, his attitude is different from that of the first and second seasons. This causes a rift between Elena and Stefan whereas the relationship between Damon and Elena becomes closer and more intimate. A still loyal Elena, however, refuses to admit her feelings for Damon. In 'Dangerous Liaisons', Elena, frustrated with her feelings for him, tells Damon that his love for her may be a problem, and that this could be causing all their troubles. This incenses Damon, causing him to revert to the uncaring and reckless Damon seen in the previous seasons. The rocky relationship between the two continues until the sexual tension hits the fan and in a moment of heated passion, Elena -- for the first time in the three seasons -- kisses Damon of her own accord. This kiss finally causes Elena to admit that she loves both brothers and realize that she must ultimately make her choice as her own ancestress, Katherine Pierce, who turned the brothers, once did. In assessment of her feelings for Damon, she states this: ``Damon just sort of snuck up on me. He got under my skin and no matter what I do, I can't shake him.'' In the season finale, a trip designed to get her to safety forces Elena to make her choice: to go to Damon and possibly see him one last time; or to go to Stefan and her friends and see them one last time. She chooses the latter when she calls Damon to tell him her decision. Damon, who is trying to stop Alaric, accepts what she says and she tells him that maybe if she had met Damon before she had met Stefan, her choice may have been different. This statement causes Damon to remember the first night he did meet Elena which was, in fact, the night her parents died - before she had met Stefan. Not wanting anyone to know he was in town and after giving her some advice about life and love, Damon compels her to forget. He remembers this as he fights Alaric and seems accepting of his death when Alaric, whose life line is tied to Elena's, suddenly collapses in his arms. Damon is grief-stricken, knowing that this means that Elena has also died and yells, ``No! You are not dead!'' A heartbroken Damon then goes to the hospital demanding to see Elena when the doctor, Meredith Fell, tells him that she gave Elena vampire blood. The last shot of the season finale episode shows Elena in transition."
    # words_to_mask = [('biomass', 1), ('burning', 45), ('returns', 161)]
    # words_to_mask = [('drag(s)', 139), ('third', 2), ('Katherine', 326), ('season,', 3), ('helps', 5)]
    # words_to_mask = [('less', 212), ('collected,', 15)]
    words_to_mask = [('English', 54), ('contain', 22), ('followed', 25), ('borrowed', 31), ('words', 83), ('dictionary', 61), ('considered', 15), ('modern-English', 19), ('list', 47), ('major', 60)]
    # words_to_mask = [('third', 36), ('play-off', 38), ('FIFA', 2), ('losing', 42), ('Moscow.', 22)]
    # words_to_mask = [('Coordinated', 27), ('outlying', 37), ('time', 11), ('two', 9), ('zones.', 12)]
    # words_to_mask = [('Damon', 4), ('third', 2), ('helps', 5), ('season,', 3), ('brothers,', 331)]
    num_of_predict= 1
    candidates = mask_and_predict(sentence, words_to_mask, num_of_predict=num_of_predict, single=True)
    print('candidates: ', candidates)
    candidate_sentences = generate_candidate_sentences(sentence, words_to_mask, candidates, single=True)
    print('candidate_sentences: ', candidate_sentences[0])


