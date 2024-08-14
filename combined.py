import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from victim_models import T5_model, T5_tokenizer, ask_yes_no_question
from nltk.corpus import stopwords
import nltk
import string
from tool import clean_token, is_valid_token, convert_seconds_to_hms
import time
from rank_with_attention import ranking_with_attention
from rank_with_removal import calculate_logits_importance


# This is a code for joint two ranking stratgies by adding them together
# There are three ways:
# 'norm-link', 'norm-add'
# 'norm' means normalization
# 'link' is to combine two methods to single dictionary by jointing the same word
# 'add' means add scores that match the same word

def combine_ranking_scores(question, context, top_k, combination='norm-link', rate=None, mode='yn', **generator_args):

    if combination == 'norm-link':
        # norm-link 方法：
        # 这种方法直接将两种不同排名方法中都出现的单词的分数进行相加。
        # 对于在字典中已存在的单词，直接累加其分数。
        # 每个单词的最终分数是两种方法分数的直接加和。
        # 对于在字典中不存在的单词，则使用当前分数和索引进行初始化。
        # Get scores from both methods
        removal_scores = calculate_logits_importance(question, context, rate=rate, top_k=top_k)  # Assumes returns format is (word, score, index)
        _, _, attention_scores, _ = ranking_with_attention(question, context, rate=rate, top_k=top_k)  # Assumes returns format is (word, score, index)
        combined_scores = {}
        # Append scores from both methods into a single dictionary, summing scores for the same words
        for word, score, index in removal_scores:
            if word in combined_scores:
                combined_scores[word][0] += score  # Sum scores
            else:
                combined_scores[word] = [score, index]  # Initialize word

        for word, score, index in attention_scores:
            if word in combined_scores:
                combined_scores[word][0] += score  # Sum scores
            else:
                combined_scores[word] = [score, index]  # Initialize word
        # Convert the dictionary to a list of tuples sorted by the score value in descending order
        combined_scores_list = sorted(combined_scores.items(), key=lambda x: x[1][0], reverse=True)

        # Select the top-k scores based on the rate or top_k provided
        if rate:
            sorted_combined_scores = combined_scores_list[:int(len(combined_scores_list) * rate)]
        elif top_k:
            sorted_combined_scores = combined_scores_list[:top_k]
        else:
            sorted_combined_scores = combined_scores_list  # If no rate or top_k, return all

    if combination == 'norm-add':
        # norm-add 方法：
        # 这种方法在处理时也初始化一个字典，但在处理不存在于字典中的单词时采取不同的策略，允许每种排名方法独立地贡献分数。
        # 如果在两种方法中找到某个单词，则将它们的分数相加，这一点与 norm-link 方法相似。
        # 但这种方法可能更强调在两种方法中都表现适中的单词，而不是在一种方法中表现异常出色的单词。

        # 主要差异：
        # norm-link 可能会忽略那些不在两种方法中都出现的单词，而 norm-add 则包括并对它们进行评分。
        # norm-add 通过考虑每种方法的独特条目，提供了更广阔的视角，而 norm-link 则专注于强化两种方法都同意的分数。
        removal_scores = calculate_logits_importance(question, context, top_k=None, rate=None)
        _, _, attention_scores, _ = ranking_with_attention(question, context, mode=mode, top_k=None, rate=None)
        combined_scores = {}
        # Iterate over removal scores to populate the dictionary
        for word, score, index in removal_scores:
            combined_scores[word] = [score, index]  # Store score and index as a list
        # Add attention scores to the dictionary, combine scores if word exists
        for word, score, index in attention_scores:
            if word in combined_scores:
                combined_scores[word][0] += score  # Sum up the scores
            else:
                combined_scores[word] = [score, index]
        # Convert dictionary to a sorted list of tuples by combined scores
        if rate:
            sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1][0], reverse=True)[:int(len(combined_scores) * rate)]
        elif top_k:
            sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        else:
            sorted_combined_scores = combined_scores  # If no rate or top_k, return all

    if combination == 'add':
        removal_scores = calculate_logits_importance(question, context, top_k=None)
        _, _, attention_scores, _ = ranking_with_attention(question, context, top_k=None)
        combined_scores = {}
        # Iterate over removal scores to populate the dictionary
        for word, score, index in removal_scores:
            combined_scores[word] = [score, index]  # Store score and index as a list

        # Add attention scores to the dictionary, combine scores if word exists
        for word, score, index in attention_scores:
            if word in combined_scores:
                combined_scores[word][0] += score  # Sum up the scores
            else:
                combined_scores[word] = [score, index]

        # Convert dictionary to a sorted list of tuples by combined scores
        if rate:
            sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1][0], reverse=True)[:int(len(combined_scores) * rate)]
        elif top_k:
            sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        else:
            sorted_combined_scores = combined_scores  # If no rate or top_k, return all
    
    return sorted_combined_scores


if __name__ =='__main__':

    # context = "In the third season, Damon helps Elena in bringing his brother, Stefan, back to Mystic Falls after Stefan becomes Klaus' henchman. The arrangement transpired after a bargain for his blood that would cure Damon of the werewolf bite he had received from Tyler. At first, he is reluctant to involve Elena in the rescue attempts, employing Alaric Saltzman, Elena's guardian, instead as Klaus does not know that Elena is alive after the sacrifice which frees Klaus' hybrid side. However, Elena involves herself, desperate to find Stefan. Damon, though hesitant at first, is unable to refuse her because of his love for her. He also points out to her that she once turned back from finding Stefan since she knew Damon would be in danger, clearly showing that she also has feelings for him. He tells her that ``when (he) drag(s) (his) brother from the edge to deliver him back to (her), (he) wants her to remember the things (she) felt while he was gone.'' When Stefan finally returns to Mystic Falls, his attitude is different from that of the first and second seasons. This causes a rift between Elena and Stefan whereas the relationship between Damon and Elena becomes closer and more intimate. A still loyal Elena, however, refuses to admit her feelings for Damon. In 'Dangerous Liaisons', Elena, frustrated with her feelings for him, tells Damon that his love for her may be a problem, and that this could be causing all their troubles. This incenses Damon, causing him to revert to the uncaring and reckless Damon seen in the previous seasons. The rocky relationship between the two continues until the sexual tension hits the fan and in a moment of heated passion, Elena -- for the first time in the three seasons -- kisses Damon of her own accord. This kiss finally causes Elena to admit that she loves both brothers and realize that she must ultimately make her choice as her own ancestress, Katherine Pierce, who turned the brothers, once did. In assessment of her feelings for Damon, she states this: ``Damon just sort of snuck up on me. He got under my skin and no matter what I do, I can't shake him.'' In the season finale, a trip designed to get her to safety forces Elena to make her choice: to go to Damon and possibly see him one last time; or to go to Stefan and her friends and see them one last time. She chooses the latter when she calls Damon to tell him her decision. Damon, who is trying to stop Alaric, accepts what she says and she tells him that maybe if she had met Damon before she had met Stefan, her choice may have been different. This statement causes Damon to remember the first night he did meet Elena which was, in fact, the night her parents died - before she had met Stefan. Not wanting anyone to know he was in town and after giving her some advice about life and love, Damon compels her to forget. He remembers this as he fights Alaric and seems accepting of his death when Alaric, whose life line is tied to Elena's, suddenly collapses in his arms. Damon is grief-stricken, knowing that this means that Elena has also died and yells, ``No! You are not dead!'' A heartbroken Damon then goes to the hospital demanding to see Elena when the doctor, Meredith Fell, tells him that she gave Elena vampire blood. The last shot of the season finale episode shows Elena in transition."
    # question = "does damon and elena get together in season 3"
    # context = "All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline."
    # question = "does ethanol take more energy make that produces"
    question = "can u drive in canada with us license"
    context = "Persons driving into Canada must have their vehicle's registration document and proof of insurance."

    print(combine_ranking_scores(question, context, mode='yn', combination='norm-link', rate=None, top_k=None))