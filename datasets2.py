from datasets import load_dataset

def get_questions_and_contexts(dataset_name, config=None, split='validation', num_samples=5, cache_dir=None):
    # Load the dataset using the specified configuration and caching options
    cache_dir='/home/jiyli/Data/qa_attack/huggingface_cache'
    dataset = load_dataset(dataset_name, config, split=split, cache_dir=cache_dir, trust_remote_code=True)
    
    # Retrieve the specified number of samples from the dataset
    samples = dataset.select(range(num_samples))
    for i in samples:
        print(i)
        print('\n')
    # Extract questions and contexts from the samples
    questions_and_contexts = []
    for sample in samples:
        if dataset_name in ['deepmind/narrativeqa']:
            # Handling for NarrativeQA which uses nested fields
            question = sample['question']['text']
            context = sample['document']['text'] if 'text' in sample['document'] else "No context available"
        elif dataset_name in ['rajpurkar/squad', 'rajpurkar/squad_v2']:
            # Standard fields for SQuAD and SQuAD v2 datasets
            question = sample['question']
            context = sample['context']
            # print(f'question length: {len(question)}')
        elif dataset_name in ['google/boolq']:
            # Handling for BoolQ dataset
            question = sample['question']
            context = sample['passage']
        else:
            # Default handling for other datasets
            question = sample.get('question', 'No question')
            context = sample.get('context', 'No context')
        
        questions_and_contexts.append((question, context))
    
    return questions_and_contexts

# Example usage
if __name__ == "__main__":
    dataset_name = 'rajpurkar/squad'  # Example dataset
    questions_contexts = get_questions_and_contexts(dataset_name, num_samples=3)
    # for question, context in questions_contexts:
    #     print("Question:", question)
    #     print("Context:", context)
    #     print("-" * 60)
