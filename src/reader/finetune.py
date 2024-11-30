from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from argparse import ArgumentParser

dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
] # More models at https://huggingface.co/unsloth
    
def train(args):
    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.llm,
        max_seq_length = args.max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 42,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    # before finetune
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    inputs = tokenizer(
    [
        tokenizer.apply_chat_template([{'role': 'user', 'content': train_dataset[-1]['prompt']}],
                                      tokenize=False)
    ], return_tensors = "pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    print(tokenizer.batch_decode(outputs)[0])
    print(train_dataset[-1]['completion'])

    FastLanguageModel.for_training(model) # Enable native 2x faster inference
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        # dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        dataset_num_proc = 1,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            per_device_eval_batch_size = 1,
            gradient_accumulation_steps = 8,
            warmup_ratio = 0.03,
            num_train_epochs = args.num_epochs,
            learning_rate = args.lr,
            save_total_limit = 3,
            save_steps = 100,
            logging_steps = 25,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            optim = "adamw_8bit",
            weight_decay = 0.,
            lr_scheduler_type = "cosine",
            seed = 42,
            output_dir = args.output_dir,
        ),
    )

    trainer.train(resume_from_checkpoint = args.resume_from_checkpoint)
    model.save_pretrained(f"{args.output_dir}/lora_model")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--llm", default="unsloth/llama-2-7b-chat-bnb-4bit")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--output_dir", default="trained_model/llama2_qlora_chain")
    parser.add_argument("--train_data", default="processed_data/sft_train_chains.jsonl")
    parser.add_argument("--resume_from_checkpoint", action='store_true')
    
    args = parser.parse_args()
    train(args)