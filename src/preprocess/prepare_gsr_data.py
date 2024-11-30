from tqdm import tqdm
import json
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

relations = json.load(open('data/relations_with_triples.json'))

""" assigning relation ID """
special_tokens_rel = []
relation2semid = {}

for rel in tqdm(relations):
    rel_classes = rel.split('.')
    rel_identifier = []
    for i, rel_class in enumerate(rel_classes):
        special_token = f'<rel_{i}_{rel_class.lower()}>'
        if special_token not in special_tokens_rel:
            special_tokens_rel.append(special_token)
        rel_identifier.append(special_token)
    relation2semid[' '.join(rel_classes)] = ''.join(rel_identifier)

""" add relation IDs to tokenizer """
tokenizer = AutoTokenizer.from_pretrained("t5-base")
for relation in relation2semid:
    tokenizer.add_tokens(relation2semid[relation])

""" add special tokens  """
tokenizer.add_tokens('[relation to id]')
tokenizer.add_tokens('[query to id]')
tokenizer.add_tokens('[reasoning]')
tokenizer.add_tokens('<SEP>')
# different from the special tokens in the original paper, but do not have any impact on the performance since the token name is just a placeholder.
# [relation to id] and <SEP> is not used in this work, [query to id] refers to [index] in the paper, [reasoning] refers to [retrieval] in the paper.

tokenizer.save_pretrained('tokenizer/t5-freebase-rel-atomic')

""" creating indexing data (pseudo question) """
pseudo_questions_raw = json.load(open('data/pseudo_questions.json'))
def create_training_data_pseudo_question(prefix, dataset):
    data = []
    for sample in dataset:
        rel = sample['relation']
        input = sample['question']
        target = relation2semid[rel.replace('.', ' ')]
        data.append({
            "input": (prefix + input).lower(),
            "target": target
        })
    return data
pseudo_questions_data = create_training_data_pseudo_question('[query to id] ', pseudo_questions_raw)

""" creating retrieval data (subgraph retrieval) """
webqsp_paths = json.load(open('data/gpt_cleaned_webqsp_path.json'))
cwq_paths = json.load(open('data/gpt_cleaned_cwq_path.json'))
# shortest paths between topic entity and answer entity cleaned by gpt

webqsp_raw = load_dataset("rmanluo/RoG-webqsp", split='train')
cwq_raw = load_dataset("rmanluo/RoG-cwq", split='train')

def create_training_data_retrieval(prefix, dataset, paths):
    data = []
    for sample, path in tqdm(zip(dataset, paths)):
        input = sample['question']
        if not input.endswith('?'):
            input = input + '?'
        for p in path:
            target = []
            if not len(p):
                continue
            for rel in p:
                rel_id = relation2semid[rel.replace('.', ' ')]
                target.append(rel_id)
                
            data.append({
                "input": prefix + input.lower(),
                "target": ''.join(target)
            })
    return data

webqsp_data = create_training_data_retrieval('[reasoning] ', webqsp_raw, webqsp_paths)
cwq_data = create_training_data_retrieval('[reasoning] ', cwq_raw, cwq_paths)

""" tokenizing processed data and save as huggingface dataset """
def create_training_dataset(data):
    input_lenghts = [[x for x in tokenizer(sample['input'])['input_ids']] for sample in data]
    input_lenghts = [len(x) for x in input_lenghts]
    # take 85 percentile of max length for better utilization
    max_source_length = int(np.percentile(input_lenghts, 99.9))
    print(max_source_length)
    
    input_lenghts = [[x for x in tokenizer(sample['target'])['input_ids']] for sample in data]
    input_lenghts = [len(x) for x in input_lenghts]
    # take 85 percentile of max length for better utilization
    max_target_length = int(np.percentile(input_lenghts, 100))
    print(max_target_length)
    
    inputs = [sample['input'] for sample in data]
    targets = [sample['target'] for sample in data]

    tokenized_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=max_source_length)
    tokenized_targets = tokenizer(text_target=targets, truncation=True, padding='max_length', max_length=max_target_length)

    target_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in tokenized_targets["input_ids"]
    ]

    dataset = []
    for input_ids, attention_mask, target in zip(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], target_ids):
        dataset.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target
        })

    return dataset

pseudo_questions_dataset = create_training_dataset(pseudo_questions_data)
webqsp_dataset = create_training_dataset(webqsp_data)
cwq_dataset = create_training_dataset(cwq_data)

""" save processed dataset to disk """
def save_dataset(dataset, path):
    json.dump(dataset, open(f'{path}.json', "w"))
    dataset = load_dataset("json", data_files=f'{path}.json')
    dataset["train"].save_to_disk(path)

save_dataset(pseudo_questions_dataset, 'processed_data/pseudo_questions')
save_dataset(webqsp_dataset, 'processed_data/webqsp_train')
save_dataset(cwq_dataset, 'processed_data/cwq_train')