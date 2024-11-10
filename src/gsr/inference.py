from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import json

def run_inference(args):
    if args.train:
        webqsp_test = load_dataset('rmanluo/RoG-webqsp', split='train')
        cwq_test = load_dataset('rmanluo/RoG-cwq', split='train')
    else:        
        webqsp_test = load_dataset('rmanluo/RoG-webqsp', split='test')
        cwq_test = load_dataset('rmanluo/RoG-cwq', split='test')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, device_map="auto").eval()

    def evaluate_model(input_text, nums_beam=args.top_k):
        input = tokenizer(input_text, return_tensors='pt')
        outputs = model.generate(input_ids=input["input_ids"].cuda(), do_sample=False, max_new_tokens=512,
                                num_beams=nums_beam, num_return_sequences=nums_beam)

        predictions = []
        for output in outputs:
            predictions.append(tokenizer.decode(output.detach().cpu().numpy(), skip_special_tokens=True))

        return predictions
    
    output_dir = args.output_dir
    if args.train:
        output_dir += 'train_'
    webqsp_predictions = []
    for sample in tqdm(webqsp_test):
        preds = evaluate_model('[reasoning] ' + sample['question'] + '?')
        webqsp_predictions.append(preds)
    json.dump(webqsp_predictions, open(output_dir + args.model_path.split('/')[1] + '_webqsp.json', 'w'))

    cwq_predictions = []
    for sample in tqdm(cwq_test):
        if 'lower' in args.model_path:
            preds = evaluate_model('[reasoning] ' + sample['question'].lower())
        else:
            preds = evaluate_model('[reasoning] ' + sample['question'])
        cwq_predictions.append(preds)
    json.dump(cwq_predictions, open(output_dir + args.model_path.split('/')[1] + '_cwq.json', 'w'))
    
    return webqsp_predictions, cwq_predictions

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def get_majority_ans(answers, topk=3):
    ans_dict = {}
    max_cnt = 0
    for answer in answers:
        if answer not in ans_dict:
            ans_dict[answer] = 0
        ans_dict[answer] += 1
        if ans_dict[answer] > max_cnt:
            max_cnt = ans_dict[answer]
    top_k_answers = []
    while len(top_k_answers) < len(ans_dict) and len(top_k_answers) < topk:
        for answer in ans_dict:
            if ans_dict[answer] == max_cnt and len(top_k_answers) < topk:
                top_k_answers.append(answer)

        max_cnt -= 1
    return top_k_answers

        
def get_f1(precision, recall):
    macro = 2 * (np.mean(precision) * np.mean(recall)) / (np.mean(precision) + np.mean(recall))
    micro = []
    for p, r in zip(precision, recall):
        if p == 0:
            micro.append(0)
        else:
            micro.append(2 * p * r / (p + r))
    return macro, np.mean(micro)


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


def get_ans(dataset, predictions, top_m=10, top_n=3):
    predicted_answers = []
    ccnt = 0

    for sample, preds in tqdm(zip(dataset, predictions)):
        graph = sample['graph']
        entity2triple = {}
        for triple in graph:
            if triple[0] not in entity2triple:
                entity2triple[triple[0]] = {}
            if triple[2] not in entity2triple:
                entity2triple[triple[2]] = {}
            if triple[1] not in entity2triple[triple[0]]:
                entity2triple[triple[0]][triple[1]] = [] 
            if triple[1] not in entity2triple[triple[2]]:
                entity2triple[triple[2]][triple[1]] = [] 
            entity2triple[triple[0]][triple[1]].append(triple[2])
            entity2triple[triple[2]][triple[1]].append(triple[0])

        predicted_ans = []
        cnt = 0
        for pred in preds[:top_m]:
            if cnt == top_n:
                break
            predicted_path = decode_predictions(pred)
            ents = [sample['q_entity']]
            for rel in predicted_path:
                objs = []
                for ent in ents[-1]:
                    if ent in entity2triple and rel in entity2triple[ent]:
                        objs += entity2triple[ent][rel]
                ents.append(list(set(objs)))

            if len(ents[-1]):
                predicted_ans += ents[-1]
                cnt += 1
        if cnt < 1:
            ccnt += 1
        predicted_answers.append(predicted_ans)
    print(f'{ccnt} samples can not get at least 1 valid path')
    return predicted_answers


def run_eval(args, rule_paths):
    for data_name in ['webqsp', 'cwq']:
        eval_data = load_dataset('rmanluo/RoG-' + data_name, split='test')
        eval_data = list(eval_data)
        if isinstance(rule_paths, str):
            predictions = json.load(open(args.predictions_path + f'_{data_name}.json'))
        else:
            predictions = rule_paths[data_name]

        answers = get_ans(eval_data, predictions, args.top_m, args.top_n)

        voted_answers = []
        for answer in answers:
            voted_answers.append(get_majority_ans(answer, 5))
        
        precision = []
        recall = []
        hits = []
        for sample, preds in tqdm(zip(eval_data, answers)):
            references = sample['answer']
            preds = list(set(preds))
            matched = 0
            preds_str = ' '.join(preds)

            for reference in references:
                if match(preds_str, reference):
                    matched += 1

            if matched > 0:
                precision.append(matched / len(preds))
                recall.append(matched / len(references))
                hits.append(1)
            else:
                precision.append(0)
                recall.append(0)
                hits.append(0)


        macro_f1, micro_f1 = get_f1(precision, recall)
        print(data_name + ' all:')
        print(f'P:\t{np.mean(precision)}')
        print(f'R:\t{np.mean(recall)}')
        print(f'Hits:\t{np.mean(hits)}')
        print(f'Macro F1:\t{macro_f1}')
        print(f'Micro F1:\t{micro_f1}')

        precision = []
        recall = []
        hits1 = []
        for sample, preds in tqdm(zip(eval_data, voted_answers)):
            references = sample['answer']
            preds = list(set(preds))
            matched = 0
            hit1 = 0
            preds_str = ' '.join(preds)

            for reference in references:
                if match(preds_str, reference):
                    matched += 1
                if preds and match(preds[0], reference):
                    hit1 = 1

            hits1.append(hit1)
            if matched > 0:
                precision.append(matched / len(preds))
                recall.append(matched / len(references))
            else:
                precision.append(0)
                recall.append(0)


        macro_f1, micro_f1 = get_f1(precision, recall)
        print(data_name + ' voted:')
        print(f'P:\t{np.mean(precision)}')
        print(f'R:\t{np.mean(recall)}')
        print(f'Hits@1:\t{np.mean(hits1)}')
        print(f'Macro F1:\t{macro_f1}')
        print(f'Micro F1:\t{micro_f1}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", default="rule_paths/")
    parser.add_argument("--top_k", default=10)
    parser.add_argument("--train", action='store_true')

    parser.add_argument("--rule_paths", default="rule_paths/")
    parser.add_argument("--top_m", default=3)
    parser.add_argument("--top_n", default=3)
    parser.add_argument("--eval_only", action='store_true')

    args = parser.parse_args()

    if not args.eval_only:
        webqsp_predictions, cwq_predictions = run_inference(args)
        if not args.train:
            run_eval(args, {
                'webqsp': webqsp_predictions,
                'cwq': cwq_predictions
            })
    else:
        run_eval(args, args.rule_paths)
    

