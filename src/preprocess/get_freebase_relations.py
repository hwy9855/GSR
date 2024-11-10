from SPARQLWrapper import SPARQLWrapper, JSON
import sys
from tqdm import tqdm
import random
import json

"""  """
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

"""  """
rels_w_triples = []
rels_wo_triples = []
for rel in tqdm(sr_rels):
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

"""  """
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
    print(query)
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
    print(subject_mids)
    ent_name_res = get_ents_name(subject_mids)
    subjects = []
    for sample in ent_name_res[:k]:
        subjects.append({
            'mid': sample['s']['value'][len('http://rdf.freebase.com/ns/'):],
            'label': sample['slabel']['value']
        })
    return subjects

rels2examples = {}
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
            rels2examples[rel] = examples
    except:
        error.append(rel)

json.dump(rels_w_triples, open('../data/relations_with_triples.json', 'w'))
json.dump(rels2examples, open('../data/freebase_rel2sample.json', 'w'))