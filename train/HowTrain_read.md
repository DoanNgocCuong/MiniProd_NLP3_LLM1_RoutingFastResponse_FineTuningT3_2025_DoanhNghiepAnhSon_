# How to Train Your Language Model

## Setup Instructions

### 1. Install Required Packages
First, install all required packages:
```bash
pip install unsloth
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
```

### 2. Prepare Your Environment
Make sure you have:
- Python 3.8 or higher
- CUDA-compatible GPU
- At least 16GB GPU memory (recommended)

### 3. File Structure
Your project should have this structure:
```
project_folder/
├── dataset/
│   └── utils/
│       ├── dataProcessing.py
│       └── output_data_v2.xlsx
└── train/
    ├── train.py
    └── HowTrain_read.md (this file)
```

## Training Process

### 1. Data Preparation
- Your data should be in Excel format with these columns:
  - `system_prompt`
  - `user_input`
  - `assistant_response`
- Use `dataProcessing.py` to convert your data into the right format

### 2. Training Configuration
In `train.py`, you can configure:
- Model selection (`--model`)
- Batch size (`--batch_size`)
- Learning rate (`--lr`)
- Number of epochs (`--epochs`)
- LoRA rank (`--lora_r`)

### 3. Running Training
```bash
python train.py --model unsloth/Llama-3.2-3B-Instruct --batch_size 2 --epochs 1
```

### 4. Common Parameters
```bash
# Basic training
python train.py

# Custom training
python train.py \
  --model unsloth/Llama-3.2-3B-Instruct \
  --dataset your_dataset \
  --batch_size 4 \
  --epochs 3 \
  --save_dir your_model_output
```

## Common Issues and Solutions

### 1. ModuleNotFoundError: No module named 'trl'
Solution:
```bash
pip install trl
```

### 2. CUDA Out of Memory
Solutions:
- Reduce batch size: `--batch_size 1`
- Use 4-bit quantization (already enabled by default)
- Use a smaller model like "unsloth/Llama-3.2-1B-Instruct"

### 3. Dataset Loading Issues
Make sure:
- Your Excel file exists in the correct location
- Column names match exactly: `system_prompt`, `user_input`, `assistant_response`
- Data is properly formatted

## Monitoring Training

The script will show:
- GPU memory usage
- Training time
- Loss values
- Example outputs after training

## Saving the Model

The model will be saved in two ways:
1. LoRA adapters (smaller, for continued training)
2. Full model (for inference)

Find your saved model in the `--save_dir` directory (default: "lora_model")

## Testing the Model

After training, the script will automatically:
1. Load the trained model
2. Run a test inference
3. Show the results

## Additional Resources

- Unsloth Documentation: https://docs.unsloth.ai/
- HuggingFace TRL Documentation: https://huggingface.co/docs/trl/
- Join Unsloth Discord for help: https://discord.gg/unsloth

## Tips for Better Results

1. Data Quality
   - Clean your data
   - Make sure responses are high quality
   - Remove any irrelevant or noisy examples

2. Training Parameters
   - Start with default parameters
   - Adjust learning rate if loss is unstable
   - Increase epochs for better results

3. Model Selection
   - Smaller models train faster
   - Larger models may give better results
   - Consider your GPU memory constraints
