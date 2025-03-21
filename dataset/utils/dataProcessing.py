import pandas as pd
import json
from datasets import Dataset
import os

def process_data_for_fine_tuning(input_file, output_dir):
    """
    Process data from CSV/Excel format to HuggingFace dataset format
    
    Args:
        input_file: Path to input file with system_prompt, user_input, assistant_response columns
        output_dir: Directory to save processed data
    """
    # Make output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data from input file
    # Try different formats based on file extension
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext == '.csv':
        df = pd.read_csv(input_file)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please use CSV or Excel file.")

    # Check and fix column names
    required_cols = {
        'system_prompt': ['system_prompt'],
        'user_input': ['user_input', 'use_input'], 
        'assistant_response': ['assistant_response', 'assistant_respone']
    }
    
    # Map alternative column names to standard names
    for standard_name, alternatives in required_cols.items():
        found = False
        for alt in alternatives:
            if alt in df.columns:
                if alt != standard_name:
                    df = df.rename(columns={alt: standard_name})
                found = True
                break
        if not found:
            raise ValueError(f"Missing required column '{standard_name}'. Acceptable names are: {alternatives}")
    
    # Create data in HuggingFace format
    dataset_dict = []
    conversations_list = []
    
    for _, row in df.iterrows():
        # Create conversation data for each row
        conversation = [
            {"role": "system", "content": row['system_prompt']},
            {"role": "user", "content": row['user_input']},
            {"role": "assistant", "content": row['assistant_response']}
        ]
        
        # Store as JSON string for the single-column Excel file
        conversations_list.append(json.dumps(conversation, ensure_ascii=False))
        
        # Create entry for HuggingFace dataset
        entry = {
            "conversations": conversation
        }
        dataset_dict.append(entry)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(dataset_dict)
    
    # Save dataset in different formats
    # 1. As .jsonl file
    jsonl_path = os.path.join(output_dir, "training_data.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in dataset_dict:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 2. Save as HuggingFace dataset
    dataset.save_to_disk(os.path.join(output_dir, "hf_dataset"))
    
    # 3. Create Excel file with single "conversations" column
    conversations_df = pd.DataFrame({
        "conversations": conversations_list
    })
    conversations_df.to_excel(os.path.join(output_dir, "conversations_format.xlsx"), index=False)
    
    # 4. Also export as CSV with original format
    formatted_df = pd.DataFrame({
        "system_prompt": df["system_prompt"],
        "user_input": df["user_input"],
        "assistant_response": df["assistant_response"]
    })
    formatted_df.to_csv(os.path.join(output_dir, "formatted_data.csv"), index=False)
    
    print(f"Data processed successfully. Files saved to {output_dir}")
    print(f"Created conversations format Excel file with {len(conversations_list)} examples")
    return dataset

if __name__ == "__main__":
    # Example usage
    input_file = "output_data_v2.xlsx"  # Use the Excel file in current directory
    output_dir = "processed"
    
    try:
        dataset = process_data_for_fine_tuning(input_file, output_dir)
        print(f"Processed {len(dataset)} examples")
    except Exception as e:
        print(f"Error processing data: {e}") 