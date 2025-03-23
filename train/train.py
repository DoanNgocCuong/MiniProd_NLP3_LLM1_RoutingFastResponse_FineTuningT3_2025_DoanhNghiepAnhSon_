import os
import torch
import argparse
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with Unsloth")
    
    # Model and dataset arguments
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.2-3B-Instruct", 
                       help="Model name or path")
    parser.add_argument("--dataset", type=str, default="mlabonne/FineTome-100k", 
                       help="Dataset name or path")
    
    # Training parameters with alternative names
    parser.add_argument("--lr", "--learning_rate", type=float, default=2e-4, 
                       help="Learning rate")
    parser.add_argument("--epochs", "--num_epochs", type=float, default=1.0, 
                       help="Number of epochs")
    
    # Training parameters
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    
    # Output and saving
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--save_dir", type=str, default="lora_model", help="Directory to save model")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps (overrides epochs)")
    
    # Other parameters
    parser.add_argument("--chat_template", type=str, default="llama-3.1", help="Chat template to use")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    
    args = parser.parse_args()
    return args

def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

def main():
    args = parse_args()
    print(f"Loading model: {args.model}")
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Tạo cấu hình LoRA riêng
    lora_config = LoraConfig(
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM"  # Thêm task type cho language model
    )
    
    # Áp dụng LoRA config
    model = get_peft_model(
        model,
        peft_config=lora_config
    )
    
    # Apply chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=args.chat_template,
    )
    
    # Load and prepare dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    dataset = standardize_sharegpt(dataset)
    
    # Format the dataset
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True,
    )
    
    # Set up trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )
    
    # Only train on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    
    # Train model
    print("Starting training...")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    trainer_stats = trainer.train()
    
    # Show memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    # Save model
    print(f"Saving model to {args.save_dir}")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    
    # Test inference
    print("\nTesting inference with the trained model:")
    FastLanguageModel.for_inference(model)
    
    messages = [
        {"role": "user", "content": "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    print("\nGenerated output:")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        input_ids=inputs, 
        streamer=text_streamer, 
        max_new_tokens=64,
        use_cache=True, 
        temperature=1.5, 
        min_p=0.1
    )

if __name__ == "__main__":
    main() 