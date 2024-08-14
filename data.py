from datasets import load_dataset, load_from_disk
import random
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

# Datasets we work on:
# SQuAD 1.1: https://huggingface.co/datasets/rajpurkar/squad
# SQuAD v2: https://huggingface.co/datasets/rajpurkar/squad_v2
# NarrativeQA (Abstractive QA): https://huggingface.co/datasets/deepmind/narrativeqa
# BoolQ (Yes-no QA): https://huggingface.co/datasets/google/boolq
# Newsqa: https://www.kaggle.com/datasets/fstcap/combinednewsqadatav1json/data

def load_questions_and_contexts_from_json(json_file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Print 'keys' as column titles for the top-level structure
    print("Top-Level Keys:", data.keys())
    
    # Navigate through the nested structure
    questions_and_contexts = []
    for story in data['data']:
        context = story['text']  # Assuming 'text' contains the context
        for question_data in story['questions']:
            question = question_data['q']
            answer_text = context[answer_start:answer_end]

            questions_and_contexts.append({
                'storyId': story['storyId'],
                'type': story['type'],
                'question': question,
                'context': context
            })
    
    return questions_and_contextsd_answers


def fetch_document_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises a HTTPError for bad requests
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from the webpage; you might need to adjust the tag based on the page structure
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error fetching document content: {e}")
        return ""

class DatasetLoader:
    def __init__(self, name, file_path='/home/jiyli/Data/qa_attack/datasets/newsqa-data-v1/combined-newsqa-data-v1.json', config=None):
        self.name = name
        self.config = config
        self.dataset = None
        self.cache_dir = '/home/jiyli/Data/qa_attack/huggingface_cache'
        self.file_path = file_path

    def load_dataset(self):
        if self.name == 'newsqa' and self.file_path:
            # Load JSON data if dataset is NewsQA and a file path is provided
            with open(self.file_path, 'r') as file:
                self.dataset = json.load(file)['data']
            print('NewsQA dataset loaded from JSON.')
        elif self.config:
            self.dataset = load_dataset(self.name, self.config, cache_dir=self.cache_dir, trust_remote_code=True)
        else:
            self.dataset = load_dataset(self.name, cache_dir=self.cache_dir, trust_remote_code=True)
        return 'Dataset loaded successfully!'

    def get_formatted_string(self, split='validation'):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        examples = []
        if self.name == 'newsqa':
            # Handling NewsQA data loaded from JSON
            for story_index, story in enumerate(self.dataset):
                context = story['text']
                for question_index, question_data in enumerate(story['questions']):
                    question = question_data['q']
                    answer = question_data['answers']
                    examples.append((story_index,[question, context], answer))
            return examples
        else:
            for idx, item in enumerate(self.dataset[split]):
            
                if self.name == 'google/boolq':
                    question = item.get('question', 'No question')
                    #'passage' when 'yn' mode; 'document' when 'ab' mode
                    text = item.get('passage', 'No context')
                    answer = item.get('answer', 'No answer')
                    if answer == 'false':
                        answer = 'no'
                    else:
                        answer = 'yes'
                    examples.append((idx, [question, text], answer))
                elif self.name == 'deepmind/narrativeqa':
                    question = item.get('question', 'No question').get('text', 'No question')
                    text = item.get('document', {}).get('summary', 'No context available').get('text', 'No context available')
                    answer = item.get('answers', 'No answer')
                    examples.append((idx, [question, text], answer))
                elif self.name == 'rajpurkar/squad' or 'rajpurkar/squad_v2':
                    question = item.get('question', 'No question')
                    text = item.get('context', 'No context')
                    answer = item.get('answers', 'No answer').get('text', 'No answer')
                    examples.append((idx, [question, text], answer))
                elif self.name == 'ybisk/piqa':
                    question = item.get('goal', 'No question')
                    right_answer = item.get('label') + 1
                    text = item.get(f'sol{right_answer}', 'No context')
                elif self.name == 'google-research-datasets/natural_questions':
                    html_content = example['document']['html']
                    soup = BeautifulSoup(html_content, 'html.parser')
                    text = soup.get_text()
                    question = item.get('question', 'No question').get('text', 'No question')
                # if self.name == 'lucadiliello/newsqa':
                #     question = item.get('question', 'No question')
                #     text = item.get('context', 'No context')
                #     examples.append((idx, [question, text]))
                #     print('havent ready yet!')
                # examples.append((idx, [question, text]))
            return examples
    
    def get_samples(self, split='validation', num_samples=1, randomize=False):
        formatted_strings = self.get_formatted_string(split)
        if randomize:
            samples = random.sample(formatted_strings, num_samples)
        else:
            samples = formatted_strings[:num_samples]
        return samples

if __name__ == "__main__":

    # daatsets can be used:
    # dataset_name = 'deepmind/narrativeqa', 'rajpurkar/squad', 'rajpurkar/squad_v2', 'google/boolq', 'newsqa', 
    
    # These are datasets in waitlist: 'ybisk/piqa', 'google-research-datasets/natural_questions'

    split='validation'
    loader = DatasetLoader('newsqa')
    loader.load_dataset()
    formatted_strings = loader.get_formatted_string(split=split)
    
    samples = loader.get_samples(split=split, num_samples=10, randomize=False)
    print(samples[2])



