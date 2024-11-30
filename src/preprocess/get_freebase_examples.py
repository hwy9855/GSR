from SPARQLWrapper import SPARQLWrapper, JSON
import sys
from tqdm import tqdm
import random
import json

sparql = SPARQLWrapper("http://localhost:3001/sparql")
sparql.setReturnFormat(JSON)

def execute(query):
    sparql.setQuery(query)
    return sparql.queryAndConvert()['results']['bindings']

def get_query(rel, prefix):
    query = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX owl:  <http://www.w3.org/2002/07/owl#>
                SELECT DISTINCT ?s ?o
                WHERE{{
                        ?s {prefix}:{rel} ?o.
                     }}LIMIT 1"""
    return query

rels_w_triples = []
rels_wo_triples = []

with open('data/freebase_relations.txt', 'r') as f:
    freebase_relations = f.read().splitlines()
    
for rel in tqdm(freebase_relations):
    prefix = 'ns'
    r = rel
    if '#' in rel:
        p, r = rel.split('#')
        if p == 'owl':
            prefix = 'owl'
        else:
            prefix = 'rdfs'
            
    if len(execute(get_query(r, prefix))):
        rels_w_triples.append(rel)
    else:
        rels_wo_triples.append(rel)

def get_ents_name(ents):
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT *
    WHERE{{
    VALUES ?s {{ {' '.join(['ns:' + ent for ent in ents])} }}
       ?s ns:type.object.name ?slabel.
       FILTER (lang(?slabel) = 'en')
    
    }}
    """
    res = execute(query)
    return res

def get_example(rel, prefix, k=10, verbose=False):
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl:  <http://www.w3.org/2002/07/owl#>
    SELECT DISTINCT *
    WHERE{{
       ?s {prefix}:{rel} ?o.
    
    }} LIMIT 500
    """
    res = execute(query)
    random.shuffle(res)
    subject_mids = []
    for sample in res:
        if sample['s']['value'].startswith('http://rdf.freebase.com/ns/'):
            subject_mids.append(sample['s']['value'][len('http://rdf.freebase.com/ns/'):])
    ent_name_res = get_ents_name(subject_mids)
    subjects = []
    for sample in ent_name_res[:k]:
        subjects.append({
            'mid': sample['s']['value'][len('http://rdf.freebase.com/ns/'):],
            'label': sample['slabel']['value']
        })
    return subjects

rel2subject_examples = {}
error = []

for rel in tqdm(rels_w_triples):    
    prefix = 'ns'
    r = rel
    if '#' in rel:
        p, r = rel.split('#')
        if p == 'owl':
            prefix = 'owl'
        else:
            prefix = 'rdfs'
    try:
        examples = get_example(r, prefix)
        if examples:
            rel2subject_examples[rel] = examples
    except:
        error.append(rel)

def get_ent_name(ent):
    prefix = 'http://rdf.freebase.com/ns/'
    ent = ent[len(prefix):]
    query = """PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?name
    WHERE{  
           ns:""" + ent + """ ns:type.object.name ?name.
    }"""
    try:
        ent_name = execute(query)[0]['name']['value']
    except:
        ent_name = False
    return ent_name

def get_example_triples(rel, prefix):
    query = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX owl:  <http://www.w3.org/2002/07/owl#>
                SELECT DISTINCT ?s ?o
                WHERE{{
                        ?s {prefix}:{rel} ?o.
                     }}LIMIT 1000"""
    res = execute(query)
    if not len(res):
        return False
    cnt = 0
    while cnt < 100:
        triple = random.sample(res, 1)
        if triple[0]['s']['type'] == 'uri':
            subj = get_ent_name(triple[0]['s']['value'])
        else:
            subj = triple[0]['s']['value']
        if triple[0]['o']['type'] == 'uri':
            obj = get_ent_name(triple[0]['o']['value'])
        else:
            obj = triple[0]['o']['value']
        if subj and obj:
            return subj, obj
        cnt += 1

    return False

rel2triple_example = {}
for rel in tqdm(freebase_relations):
    prefix = 'ns'
    r = rel
    if '#' in rel:
        p, r = rel.split('#')
        if p == 'owl':
            prefix = 'owl'
        else:
            prefix = 'rdfs'

    try:
        tmp = get_example_triples(r, prefix)
    except: 
        continue
    if tmp:
        rel2triple_example[rel] = tmp

json.dump(rels_w_triples, open('data/relations_with_triples.json', 'w'))
json.dump(rel2subject_examples, open('data/freebase_rel2subject_sample.json', 'w'))
json.dump(rel2triple_example, open('data/freebase_rel2triple_sample.json', 'w'))