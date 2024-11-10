from openai import OpenAI
import json
from tqdm import tqdm
import random
import math

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="<your api key>",
)

def gen_response(prompt, model="gpt-4-turbo-preview", verbose=True):    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    
    response = chat_completion.choices[0].message.content
    
    if verbose:
        print(response)
        
    return response

rel2sample = json.load(open('../data/freebase_rel2sample.json'))
all_relations = json.load(open('../data/relations_with_triples.json'))

rel2templates_raw = {}
for rel in tqdm(all_relations):
    if rel not in rel2templates_raw:
        if rel in rel2sample:
            sampled_triple = f'({rel2sample[rel][0]}, {rel}, {rel2sample[rel][1]}'
            prompt = 'Given the Freebase relation {} and a triple example of the relation {}, generate 10 templates that can be used to ask question about the relation. Use [SUBJECT] to identify subject entity (not the one in the example).'.format(
                rel, sampled_triple)
            response = gen_response(prompt, verbose=False)
            rel2templates_raw[rel] = response
        else:
            prompt = 'Given the Freebase relation {}, generate 10 templates that can be used to ask question about the relation. Use [SUBJECT] to identify subject entity (not the one in the example).'.format(
                rel)
            response = gen_response(prompt, verbose=False)
            rel2templates_raw[rel] = response

rel2templates = {}
def decode_template(sample):
    templates = []
    lines = sample.split('\n')
    for line in lines:
        if len(line) and line[0].isdigit():
            if line[1].isdigit():
                templates.append(line[4:])
            else:
                templates.append(line[3:])
    if len(templates) != 10:
        print(sample)
        return None
    return templates
    
for rel in rel2templates_raw:
    template_raw = rel2templates_raw[rel]
    template = decode_template(template_raw)
    rel2templates[rel] = template

pseudo_questions = []

for rel in all_relations:
    if rel in rel2templates and rel in rel2sample:
        subjects = rel2sample[rel]
        if len(subjects) < 10:
            subjects = subjects * math.ceil(10 / len(subjects))
        random.shuffle(subjects)
        for template, subject in zip(rel2templates[rel], subjects[:10]):
            if '[SUBJECT]' not in template:
                continue
            pseudo_questions.append({
                'relation': rel,
                'template': template,
                'subject': subject,
                'question': template.replace('[SUBJECT]', subject['label'])
            })

json.dump(pseudo_questions, open('../data/pseudo_questions.json', 'w'))