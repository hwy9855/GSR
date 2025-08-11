from transformers import AutoTokenizer
from argparse import ArgumentParser
import huggingface_hub
import json
from datasets import load_dataset
from tqdm import tqdm

PROMPT_DICT = {
    "llama2_chat": "<s>[INST]\n<<SYS>>\n{}\n<</SYS>>\n\n{}[/INST] Response: ",
    "mistral": "<s>[INST] {}\n{} [/INST] Response: "
}

def decode_predictions(pred):
    rels = pred.split('<SEP>')
    pred_path = []
    for rel in rels:
        decoded_rel = []
        rel_tokens = rel.split('>')[:-1]
        for token in rel_tokens:
            decoded_rel.append(token[7:])
        pred_path.append('.'.join(decoded_rel))
    return pred_path

def decode_predictions_wo_sep(pred):
    rels = pred.split('<rel_0')[1:]
    pred_path = []
    for rel in rels:
        decoded_rel = []
        rel_tokens = ('<rel_0' + rel).split('>')[:-1]
        for token in rel_tokens:
            decoded_rel.append(token[7:])
        pred_path.append('.'.join(decoded_rel))
    return pred_path

def get_sft_data(question, all_paths, max_input_length, tokenizer, prompt_type="llama2_chat", prompt_head=['reasoning paths', 'Reasoning Paths']):
    if not len(all_paths) and args.set == 'train':
        return None
    if not question.endswith("?"):
        question = question + "?"
    system_prompt = ""

    def get_user_prompt(paths, question, head=prompt_head):
        user_prompt = "Based on the {}, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list.\n\n{}:\n{}\n\nQuestion:\n{}".format(head[0], head[1], '\n'.join(paths), question)
        return user_prompt

    prompt = PROMPT_DICT[prompt_type].format(system_prompt, get_user_prompt(all_paths, question))

    if len(prompt.split()) > 8000 and args.set == 'train':
        return None
    
    if not len(tokenizer(prompt)['input_ids']) > max_input_length:
        return get_user_prompt(all_paths, question)
    else:
        new_paths = []
        for path in all_paths:
            prompt = PROMPT_DICT[prompt_type].format(system_prompt, get_user_prompt(new_paths + [path], question))
            if len(tokenizer(prompt)['input_ids']) > max_input_length:
                return get_user_prompt(new_paths, question)
            new_paths.append(path)

def get_text_chain(prefix, path, rels, i, ent, bidirect=True):
    if ent not in path[i]:
        return False
        
    paths = []
    for triple in path[i][ent]:
        if triple['is_obj'] or not bidirect:
            new_prefix = f"{prefix} -> {rels[i]} -> {triple['ent']}"
        else:
            new_prefix = f"{prefix} <- {rels[i]} <- {triple['ent']}"
        if i == len(rels) - 1:
            paths.append(new_prefix)
        else:
            chain = get_text_chain(new_prefix, path, rels, i+1, triple['ent'], bidirect)
            if chain:
                paths += chain

    return paths

def get_triples(prefix, path, rels, i, ent, bidirect=True):
    if ent not in path[i]:
        return False
        
    paths = []
    for triple in path[i][ent]:
        if triple['is_obj'] or not bidirect:
            new_prefix = prefix + [(ent, rels[i], triple['ent'])]
        else:
            new_prefix = prefix + [(triple['ent'], rels[i], ent)]
        if i == len(rels) - 1:
            paths += new_prefix
        else:
            chain = get_triples(new_prefix, path, rels, i+1, triple['ent'], bidirect)
            if chain:
                paths += chain

    return paths
    

def get_chain(dataset, predictions, max_paths=3):
    predicted_path_for_llm = []
    predicted_triples_for_llm = []
    cnt = 0

    for sample, preds in tqdm(zip(dataset, predictions)):
        graph = sample['graph']
        subj2triple = {}
        obj2triple = {}
        for triple in graph:
            if triple[0] not in subj2triple:
                subj2triple[triple[0]] = {}
            if triple[2] not in obj2triple:
                obj2triple[triple[2]] = {}
            if triple[1] not in subj2triple[triple[0]]:
                subj2triple[triple[0]][triple[1]] = [] 
            if triple[1] not in obj2triple[triple[2]]:
                obj2triple[triple[2]][triple[1]] = [] 
            subj2triple[triple[0]][triple[1]].append(triple[2])
            obj2triple[triple[2]][triple[1]].append(triple[0])
        predicted_paths_parsed = []
        predicted_triples = []
        for pred in preds[:10]:
            predicted_path_parsed = []
            if '<SEP>' in pred:
                predicted_path = decode_predictions(pred)
            else:
                predicted_path = decode_predictions_wo_sep(pred)
            if len(predicted_path) > 10:
                continue

            predicted_path_with_ent = []
            ents = []
            
            for ent in sample['q_entity']:
                if ent in subj2triple or ent in obj2triple:
                    ents.append(ent)
            for rel in predicted_path:
                partial_path = {}
                new_ents = []
                for ent in ents:
                    if ent in subj2triple and rel in subj2triple[ent]:
                        new_ents += subj2triple[ent][rel]
                        if ent not in partial_path:
                            partial_path[ent] = []
                        for obj in subj2triple[ent][rel]:
                            partial_path[ent].append({
                                'ent': obj,
                                'is_obj': True
                            })
                    
                    if ent in obj2triple and rel in obj2triple[ent]:
                        new_ents += obj2triple[ent][rel]
                        if ent not in partial_path:
                            partial_path[ent] = []
                        for subj in obj2triple[ent][rel]:
                            partial_path[ent].append({
                                'ent': subj,
                                'is_obj': False
                            })
                predicted_path_with_ent.append(partial_path)
                ents = list(set(new_ents))

            # print(predicted_path_with_ent, predicted_path)
            if len(predicted_path_with_ent):
                for ent in predicted_path_with_ent[0]:
                    predicted_path_parsed += get_text_chain(ent, predicted_path_with_ent, predicted_path, 0, ent)
                    predicted_triples += get_triples([], predicted_path_with_ent, predicted_path, 0, ent)
            if len(predicted_path_parsed):
                predicted_paths_parsed.append(predicted_path_parsed)
            if len(predicted_paths_parsed) == max_paths:
                break
        if len(predicted_paths_parsed) < 3:
            cnt += 1
        predicted_path_for_llm.append(predicted_paths_parsed)
        predicted_triples_for_llm.append(list(dict.fromkeys(predicted_triples)))
    print(f'{cnt} samples can not get 3 valid path')
    return predicted_path_for_llm, predicted_triples_for_llm

def process_triples(triples, connector=', '):
    processed_triples = []
    for triple in triples:
        processed_triples.append(f'{triple[0]}{connector}{triple[1]}{connector}{triple[2]}')
    return processed_triples

def gen_sft_data(args):
    webqsp_raw = load_dataset("rmanluo/RoG-webqsp", split=args.set)
    cwq_raw = load_dataset("rmanluo/RoG-cwq", split=args.set)
    webqsp_predicted_paths = json.load(open(args.webqsp_predicted_paths))
    cwq_predicted_paths = json.load(open(args.cwq_predicted_paths))
    webqsp_predicted_chains, webqsp_predicted_triples = get_chain(webqsp_raw, webqsp_predicted_paths)
    cwq_predicted_chains, cwq_predicted_triples = get_chain(cwq_raw, cwq_predicted_paths)

    tokenizer = AutoTokenizer.from_pretrained(args.llm)

    if "llama-2" in args.llm.lower():
        prompt_type = "llama2_chat"
    elif "mistral" in args.llm.lower():
        prompt_type = "mistral"
    elif "rog" in args.llm.lower():
        prompt_type = "llama2_chat"
    else:
        assert f"{args.llm} is currently not supported."

    sft_data_train = []
    for sample, predicted_chain in tqdm(zip(webqsp_raw, webqsp_predicted_chains)):
        question = sample['question']
        paths = []
        for path in predicted_chain:
            paths += path
        prompt = get_sft_data(question, paths, 4000, tokenizer, prompt_type)
        if not prompt and args.set == 'train':
            continue
        completion = '\n'.join(sample['answer'])
        sft_data_train.append({
            "prompt": prompt,
            "completion": completion
        })
        
    for sample, predicted_chain in tqdm(zip(cwq_raw, cwq_predicted_chains)):
        question = sample['question']
        paths = []
        for path in predicted_chain:
            paths += path
        prompt = get_sft_data(question, paths, 4000, tokenizer, prompt_type)
        if not prompt and args.set == 'train':
            continue
        completion = '\n'.join(sample['answer'])
        sft_data_train.append({
            "prompt": prompt,
            "completion": completion
        })

    json.dump(sft_data_train, open(args.output_dir + f'sft_{args.set}_chains.jsonl', 'w'))


    sft_data_train = []
    for sample, predicted_triples in tqdm(zip(webqsp_raw, webqsp_predicted_triples)):
        question = sample['question']
        prompt = get_sft_data(question, process_triples(predicted_triples), 4000, tokenizer, prompt_type, ['KG triples', 'KG Triples'])
        if not prompt and args.set == 'train':
            continue
        completion = '\n'.join(sample['answer'])
        sft_data_train.append({
            "prompt": prompt,
            "completion": completion
        })
        
    for sample, predicted_triples in tqdm(zip(cwq_raw, cwq_predicted_triples)):
        question = sample['question']
        prompt = get_sft_data(question, process_triples(predicted_triples), 4000, tokenizer, prompt_type, ['KG triples', 'KG Triples'])
        if not prompt and args.set == 'train':
            continue
        completion = '\n'.join(sample['answer'])
        sft_data_train.append({
            "prompt": prompt,
            "completion": completion
        })

    json.dump(sft_data_train, open(args.output_dir + f'sft_{args.set}_triples.jsonl', 'w'))



if __name__ == '__main__':
    huggingface_hub.login(token="<huggingface_api_token>")
    parser = ArgumentParser()
    parser.add_argument("--llm", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--set", default="train")
    parser.add_argument("--webqsp_predicted_paths")
    parser.add_argument("--cwq_predicted_paths")
    parser.add_argument("--output_dir", default="processed_data/")
    args = parser.parse_args()
    gen_sft_data(args)
