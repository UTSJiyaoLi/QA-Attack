from transformers import BertTokenizer, BertForMaskedLM, logging
import torch
import warnings
# Ignore specific warning categories
warnings.filterwarnings("ignore", category=UserWarning, message="Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM")
logging.set_verbosity_error()

class BertMLMGuesser:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name, ignore_mismatched_sizes=True)

    def guess_masked_token(self, sentence, num_of_predict=5):
        # This function returns top-k predictions for the masked token '[mask]'.
        inputs = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits

        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_token_ids = predictions[0, mask_token_index].topk(num_of_predict, dim=-1).indices[0].tolist()

        predicted_tokens = [self.tokenizer.convert_ids_to_tokens(token_id) for token_id in predicted_token_ids]
        predicted_tokens = [token.replace('##', '') for token in predicted_tokens]  # Clean up subwords

        return predicted_tokens

if __name__ == "__main__":

    guesser = BertMLMGuesser()
    sentence = "The capital of France is [MASK]."
    predicted_tokens = guesser.guess_masked_token(sentence, num_of_predict=5)
    print(f"Top predictions for the masked token: {predicted_tokens}")
