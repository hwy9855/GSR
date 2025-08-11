from unsloth import FastLanguageModel
import json
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_dataset
import numpy as np
import string 
import re 
import os

dtype = None
load_in_4bit = True

def process_triples(triples, connector=', '):
    processed_triples = []
    for triple in triples:
        processed_triples.append(f'{triple[0]}{connector}{triple[1]}{connector}{triple[2]}')
    return processed_triples

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.replace('<|eot_id|>', '')
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s
    
def norm_pred(pred):
    normed_pred = []
    for p in pred:
        if normalize(p) not in normed_pred:
            normed_pred.append(normalize(p))
    return normed_pred

def run_eval(preds, eval_data): 
    hits = []
    hits1 = []
    p = []
    r = []
    f1 = []
    cnt = []
    for pred_ans, sample in zip(preds, eval_data):
        hit = 0
        hit1 = 0
        cover = 0
        # norm_pred_ans = norm_pred('\n'.join(pred_ans).split('<|end_header_id|>')[-1].split('\n'))[1:]
        norm_pred_ans = norm_pred(pred_ans)
        norm_ref_ans = norm_pred(sample['completion'].split('\n'))
        cnt.append(len(norm_pred_ans))
        for ans in norm_ref_ans:
            if normalize(ans) in norm_pred_ans:
                hit = 1
                cover += 1
            if normalize(ans) in norm_pred_ans[0]:
                hit1 = 1
        if cover > 0:
            p.append(cover / len(norm_pred_ans))
            r.append(cover / len(norm_ref_ans))
            if r[-1] > 1:
                print(norm_pred_ans, sample['completion'])
            f1.append(2 * p[-1] * r[-1] / (p[-1] + r[-1]))
        else:
            p.append(0)
            r.append(0)
            f1.append(0)
        hits.append(hit)
        hits1.append(hit1)
    
    print(f'Hits:\t{np.mean(hits)}')
    print(f'Hits@1:\t{np.mean(hits1)}')
    print(f'P:\t{np.mean(p)}')
    print(f'R:\t{np.mean(r)}')
    print(f'F1:\t{np.mean(f1)}')
    print(f'Average Answers:\t{np.mean(cnt)}')
    return {
        'Hits': np.mean(hits),
        'Hits@1': np.mean(hits1),
        'P': np.mean(p),
        'R': np.mean(r),
        'F1': np.mean(f1),
        'Average Answers': np.mean(cnt),
    }
    
        
def eval(args):
    max_seq_length = args.max_seq_length
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    
    PROMPT_DICT = {
        "llama2_chat": "<s>[INST]\n<<SYS>>\n{}\n<</SYS>>\n\n{}[/INST] Response: ",
        "mistral": "<s>[INST] {}\n{} [/INST] Response: "
    }
    
    def get_sft_data(question, all_paths, max_input_length, tokenizer, prompt_type="llama2_chat"):
        # if not len(all_paths):
        #     return None
        if not question.endswith("?"):
            question = question + "?"
        system_prompt = ""
    
        def get_user_prompt(paths, question):
            if args.use_triple:
                user_prompt = "Based on the KG triples, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list.\n\KG Triples:\n{}\n\nQuestion:\n{}".format('\n'.join(paths), question)
            else:
                user_prompt = "Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list.\n\nReasoning Paths:\n{}\n\nQuestion:\n{}".format('\n'.join(paths), question)

            return user_prompt
    
        prompt = PROMPT_DICT[prompt_type].format(system_prompt, get_user_prompt(all_paths, question))
    
        if not len(tokenizer(prompt)['input_ids']) > max_input_length:
            return get_user_prompt(all_paths, question), len(tokenizer(prompt)['input_ids'])
        else:
            new_paths = []
            for path in all_paths:
                prompt = PROMPT_DICT[prompt_type].format(system_prompt, get_user_prompt(new_paths + [path], question))
                if len(tokenizer(prompt)['input_ids']) > max_input_length:
                    return get_user_prompt(new_paths, question), len(tokenizer(prompt)['input_ids'])
                new_paths.append(path)
                
    def build_eval_data(dataset, paths):
        eval_data = []
        if not paths:
            paths = [[]] * len(dataset)

        all_tokens = []
        for sample, predicted_chain in tqdm(zip(dataset, paths)):
            question = sample['question']
            if args.use_triple:
                paths = process_triples(predicted_chain)
            else:
                paths = []
                for path in predicted_chain[:3]:
                    paths += path
            prompt, tokens = get_sft_data(question, paths, 4000, tokenizer, "llama2_chat")
            all_tokens.append(tokens)
            if not prompt:
                print(paths)
                break
                continue
            completion = '\n'.join(sample['answer'])
            eval_data.append({
                "prompt": prompt,
                "completion": completion
            })
        print(np.mean(all_tokens))
        return eval_data

    if args.prompt_path:
        eval_data = json.load(open(args.prompt_path))
        if args.test_data == 'webqsp':
            eval_data = eval_data[:1628]
        elif args.test_data == 'cwq':
            eval_data = eval_data[1628:]
        else:
            assert f"{args.test_data} is currently not supported."
    else:
        test_set = load_dataset(f"rmanluo/RoG-{args.test_data}", split='test')
        if args.prediction_path:
            predicted_path = json.load(open(args.prediction_path))
        else:
            predicted_path = None
        eval_data = build_eval_data(test_set, predicted_path)

    print(eval_data[0])
    
    preds = []
    for sample in tqdm(eval_data):
        input_text = tokenizer.apply_chat_template([{'role': 'user', 'content':sample['prompt']}], tokenize=False)
        inputs = tokenizer(
        [
            input_text
        ], return_tensors = "pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens = 64, do_sample=False, use_cache = True, pad_token_id=tokenizer.eos_token_id)
        if 'llama2' in args.model_path:
            pred = tokenizer.batch_decode(outputs)[0].split('[/INST]')[-1].split('\n')
            pred_ans = []
            for p in pred:
                processed_p = p.replace('</s>', '').strip()
                if len(processed_p):
                    pred_ans.append(processed_p.strip())
        elif 'llama3' in args.model_path:
            pred = tokenizer.batch_decode(outputs)[0].split('<|end_header_id|>')[-1].split('\n')
            pred_ans = []
            for p in pred:
                processed_p = p.replace('<|eot_id|>', '').strip()
                if len(processed_p):
                    pred_ans.append(processed_p.strip())

        preds.append(pred_ans)
        if len(preds) % (len(eval_data) // 10) == 0:
            run_eval(preds, eval_data)

    evaluation = run_eval(preds, eval_data)
    
    if not os.path.exists(args.model_path + 'outputs'):
        os.makedirs(args.model_path + 'outputs')

    json.dump(preds, open(args.model_path + f'outputs/preds_{args.test_data}_' + args.prediction_path.split('/')[1], 'w'))
    json.dump(evaluation, open(args.model_path + f'outputs/evaluation_{args.test_data}_' + args.prediction_path.split('/')[1], 'w'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--test_data")
    parser.add_argument("--prediction_path")
    parser.add_argument("--prompt_path")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--use_triple", action='store_true')

    args = parser.parse_args()
    eval(args)
