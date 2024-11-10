from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from argparse import ArgumentParser

import evaluate
import numpy as np

import os

DATASET_DICT = {
    'pq': 'pseudo_questions',
    'webqsp': 'webqsp_train',
    'cwq': 'cwq_train',
}

def run_finetune(args):
    train_dataset = []
    for data_name in args.finetune_data:
        assert data_name in DATASET_DICT, 'Invalid finetune data name: {}. Please specify dataset(s) from {}'.format(data_name, str(DATASET_DICT))
        train_dataset.append(load_from_disk(os.path.join(args.data_path, DATASET_DICT[data_name] + '/')))
        if data_name != 'pq' and args.low_resource < 1:
            train_dataset[-1] = train_dataset[-1].select(range(int(len(train_dataset[-1]) * args.low_resource)))
    train_dataset = concatenate_datasets(train_dataset)

    eval_dataset = []
    for data_name in args.eval_data:
        assert data_name in DATASET_DICT, 'Invalid evaluation data name: {}. Please specify dataset(s) from {}'.format(data_name, str(DATASET_DICT))
        eval_dataset.append(load_from_disk(os.path.join(args.data_path, DATASET_DICT[data_name] + '/')).select(range(1000)))
    eval_dataset = concatenate_datasets(eval_dataset)
    # just for checking train progress


    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.tokenizer_path, "t5-freebase-rel"))

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    
    em_metric = evaluate.load("exact_match")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        results = {}
        for i in range(len(args.eval_data)):
            result = em_metric.compute(predictions=decoded_preds[1000*i:1000*(i+1)], references=decoded_labels[1000*i:1000*(i+1)])
            results[args.eval_data[i]] = result['exact_match']
            
        return {k: round(v, 4) for k, v in results.items()}


    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    output_dir = os.path.join(args.output_dir, '-'.join(args.finetune_data))
    if args.low_resource < 1:
        output_dir += f'_{args.low_resource:.2f}'

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr, # higher learning rate
        num_train_epochs=args.num_epochs,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="tensorboard",
        eval_steps=100,
        evaluation_strategy="steps",
        predict_with_generate=True,
        bf16=True,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--data_path", default="../data")
    parser.add_argument("--tokenizer_path", default="./")
    parser.add_argument("--output_dir", default="../finetuned_models")

    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_epochs", type=int, default=20)

    parser.add_argument("--finetune_data", nargs='+', required=True)
    parser.add_argument("--eval_data", nargs='+')
    parser.add_argument("--low_resource", type=float, default=1.0)

    parser.add_argument("--resume_from_checkpoint", action="store_true")
    args = parser.parse_args()
    run_finetune(args)