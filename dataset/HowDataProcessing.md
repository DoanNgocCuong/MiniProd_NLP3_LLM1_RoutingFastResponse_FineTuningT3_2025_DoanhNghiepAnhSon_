
# Quy Trình Xử Lý Dữ Liệu Cho Fine-tuning LLM

## 1. Cấu Trúc Dữ Liệu Yêu Cầu

### 1.1 Format Dữ Liệu Chuẩn
```python
{
    "conversations": [
        {
            "role": "system",  # Tùy chọn
            "content": "Bạn là trợ lý AI hữu ích."
        },
        {
            "role": "user",
            "content": "Câu hỏi của người dùng"
        },
        {
            "role": "assistant",
            "content": "Câu trả lời của AI"
        }
        # Có thể có nhiều lượt đối thoại tiếp theo...
    ]
}
```

### 1.2 Các Format Được Hỗ Trợ
- Format ShareGPT (from/value)
- Format Alpaca
- Format Vicuna
- Format ChatML
- Format Llama 3.1

## 2. Chuẩn Bị Dữ Liệu

### 2.1 Tạo Dataset
```python
from datasets import load_dataset

# Cách 1: Từ file local
dataset = load_dataset("json", data_files="data.json", split="train")
dataset = load_dataset("csv", data_files="data.csv", split="train")

# Cách 2: Từ Hugging Face Hub
dataset = load_dataset("tên_tài_khoản/tên_dataset", split="train")
```

### 2.2 Chuẩn Hóa Format
```python
from unsloth.chat_templates import standardize_sharegpt

# Chuyển đổi từ ShareGPT format sang format chuẩn
dataset = standardize_sharegpt(dataset)
```

### 2.3 Áp Dụng Chat Template
```python
from unsloth.chat_templates import get_chat_template

# Thiết lập template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1"  # Hoặc template khác
)

# Hàm format dữ liệu
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(
        convo,
        tokenize = False,
        add_generation_prompt = False
    ) for convo in convos]
    return {"text": texts}

# Áp dụng format
dataset = dataset.map(formatting_prompts_func, batched=True)
```

## 3. Cấu Hình Training

### 3.1 Thiết Lập SFTTrainer
```python
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer),
    args = TrainingArguments(...)
)
```

### 3.2 Cấu Hình Training Chỉ Trên Phản Hồi
```python
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

## 4. Các Tham Số Training Quan Trọng

### 4.1 Hyperparameters
- `per_device_train_batch_size`: Số mẫu/batch (mặc định: 2)
- `gradient_accumulation_steps`: Số bước tích lũy gradient (mặc định: 4)
- `learning_rate`: Tốc độ học (mặc định: 2e-4)
- `max_steps` hoặc `num_train_epochs`: Số bước/epoch training
- `weight_decay`: Hệ số weight decay (mặc định: 0.01)

### 4.2 LoRA Parameters
- `r`: Rank của LoRA (khuyến nghị: 8, 16, 32, 64, 128)
- `target_modules`: Các layer áp dụng LoRA
- `lora_alpha`: Alpha scaling (thường bằng r)
- `lora_dropout`: Dropout rate (tối ưu: 0)

## 5. Kiểm Tra Và Đánh Giá

### 5.1 Xác Nhận Format Dữ Liệu
```python
# Kiểm tra cấu trúc conversation
print(dataset[0]["conversations"])

# Kiểm tra text đã format
print(dataset[0]["text"])

# Kiểm tra labels masking
print(tokenizer.decode(trainer.train_dataset[0]["labels"]))
```

### 5.2 Theo Dõi Training
- Loss value
- Learning rate
- GPU memory usage
- Training time

## 6. Lưu Và Tải Model

### 6.1 Lưu Model
```python
# Lưu LoRA adapters
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Lưu merged model (16-bit)
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

# Lưu GGUF format
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
```

## 7. Lưu Ý Quan Trọng

1. **Chất Lượng Dữ Liệu**
   - Đảm bảo dữ liệu sạch và có chất lượng
   - Kiểm tra format nhất quán
   - Xóa các mẫu trùng lặp hoặc không phù hợp

2. **Tài Nguyên**
   - Theo dõi GPU memory usage
   - Điều chỉnh batch size phù hợp
   - Sử dụng gradient accumulation nếu cần

3. **Best Practices**
   - Bắt đầu với dataset nhỏ để test
   - Tăng dần kích thước dataset
   - Lưu checkpoints thường xuyên
   - Theo dõi metrics để điều chỉnh hyperparameters

---
Tôi sẽ giải thích chi tiết hơn về cấu trúc và xử lý dataset:

1. **Cấu Trúc Ban Đầu (Format ShareGPT)**
```python
# Format ShareGPT
{
    "conversations": [
        {"from": "system", "value": "You are an assistant"},
        {"from": "human", "value": "What is 2+2?"},
        {"from": "gpt", "value": "It's 4."}
    ]
}
```

2. **Sau Khi Chuẩn Hóa (Format HuggingFace)**
```python
# Format HuggingFace sau khi dùng standardize_sharegpt
{
    "conversations": [
        {"role": "system", "content": "You are an assistant"},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It's 4."}
    ]
}
```

3. **Các Thành Phần Trong Dataset**
```python
# Cấu trúc một mẫu trong dataset
{
    "conversations": [  # Mỗi mẫu là một list các lượt đối thoại
        {
            "role": str,      # "system", "user", hoặc "assistant"
            "content": str    # Nội dung tin nhắn
        },
        # ... có thể có nhiều lượt đối thoại
    ]
}
```

4. **Quy Tắc Về Role**:
- `system`: Chứa hướng dẫn/context chung (không bắt buộc)
- `user`: Câu hỏi/yêu cầu từ người dùng
- `assistant`: Câu trả lời từ AI

5. **Kiểm Tra Dataset**:
```python
# Xem cấu trúc của một mẫu
print(dataset[5]["conversations"])

# Đếm số mẫu
print(len(dataset))

# Xem thông tin dataset
print(dataset)
```

6. **Xử Lý Dataset**:
```python
# Bước 1: Chuẩn hóa format
dataset = standardize_sharegpt(dataset)

# Bước 2: Áp dụng template
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(
        convo,
        tokenize = False,
        add_generation_prompt = False
    ) for convo in convos]
    return {"text": texts}

# Bước 3: Map function lên toàn bộ dataset
dataset = dataset.map(formatting_prompts_func, batched=True)
```

7. **Cấu Trúc Sau Khi Format**:
```python
{
    "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
            "You are an assistant<|eot_id|>" +
            "<|start_header_id|>user<|end_header_id|>\n\n" +
            "What is 2+2?<|eot_id|>" +
            "<|start_header_id|>assistant<|end_header_id|>\n\n" +
            "It's 4.<|eot_id|>"
}
```

8. **Lưu Ý Quan Trọng**:
- Mỗi cuộc hội thoại phải có ít nhất 1 cặp user-assistant
- Thứ tự role phải hợp lý (system -> user -> assistant)
- Content không được để trống
- Đảm bảo chất lượng và tính nhất quán của dữ liệu

9. **Kiểm Tra Chất Lượng**:
```python
# Kiểm tra format
print(dataset[0]["text"])

# Kiểm tra tokenization
tokens = tokenizer(dataset[0]["text"])
print(tokens)

# Kiểm tra labels masking
print(trainer.train_dataset[0]["labels"])
```

Dataset là nền tảng quan trọng quyết định chất lượng fine-tuning. Cần đảm bảo dữ liệu sạch, đúng format và phù hợp với mục tiêu training.

---
- https://huggingface.co/datasets/mlabonne/FineTome-100k/viewer/default/train?views%5B%5D=train&row=0

```json
[ { "from": "human", "value": "Explain what boolean operators are, what they do, and provide examples of how they can be used in programming. Additionally, describe the concept of operator precedence and provide examples of how it affects the evaluation of boolean expressions. Discuss the difference between short-circuit evaluation and normal evaluation in boolean expressions and demonstrate their usage in code. \n\nFurthermore, add the requirement that the code must be written in a language that does not support short-circuit evaluation natively, forcing the test taker to implement their own logic for short-circuit evaluation.\n\nFinally, delve into the concept of truthiness and falsiness in programming languages, explaining how it affects the evaluation of boolean expressions. Add the constraint that the test taker must write code that handles cases where truthiness and falsiness are implemented differently across different programming languages." }, { "from": "gpt", "value": "Boolean operators are logical operators used in programming to manipulate boolean values. They operate on one or more boolean operands and return a boolean result. The three main boolean operators are \"AND\" (&&), \"OR\" (||), and \"NOT\" (!).\n\nThe \"AND\" operator returns true if both of its operands are true, and false otherwise. For example:\n\n```python\nx = 5\ny = 10\nresult = (x > 0) and (y < 20) # This expression evaluates to True\n```\n\nThe \"OR\" operator returns true if at least one of its operands is true, and false otherwise. For example:\n\n```python\nx = 5\ny = 10\nresult = (x > 0) or (y < 20) # This expression evaluates to True\n```\n\nThe \"NOT\" operator negates the boolean value of its operand. It returns true if the operand is false, and false if the operand is true. For example:\n\n```python\nx = 5\nresult = not (x > 10) # This expression evaluates to True\n```\n\nOperator precedence refers to the order in which operators are evaluated in an expression. It ensures that expressions are evaluated correctly. In most programming languages, logical AND has higher precedence than logical OR. For example:\n\n```python\nresult = True or False and False # This expression is evaluated as (True or (False and False)), which is True\n```\n\nShort-circuit evaluation is a behavior where the second operand of a logical operator is not evaluated if the result can be determined based on the value of the first operand. In short-circuit evaluation, if the first operand of an \"AND\" operator is false, the second operand is not evaluated because the result will always be false. Similarly, if the first operand of an \"OR\" operator is true, the second operand is not evaluated because the result will always be true.\n\nIn programming languages that support short-circuit evaluation natively, you can use it to improve performance or avoid errors. For example:\n\n```python\nif x != 0 and (y / x) > 10:\n # Perform some operation\n```\n\nIn languages without native short-circuit evaluation, you can implement your own logic to achieve the same behavior. Here's an example in pseudocode:\n\n```\nif x != 0 {\n if (y / x) > 10 {\n // Perform some operation\n }\n}\n```\n\nTruthiness and falsiness refer to how non-boolean values are evaluated in boolean contexts. In many programming languages, non-zero numbers and non-empty strings are considered truthy, while zero, empty strings, and null/None values are considered falsy.\n\nWhen evaluating boolean expressions, truthiness and falsiness come into play. For example:\n\n```python\nx = 5\nresult = x # The value of x is truthy, so result is also truthy\n```\n\nTo handle cases where truthiness and falsiness are implemented differently across programming languages, you can explicitly check the desired condition. For example:\n\n```python\nx = 5\nresult = bool(x) # Explicitly converting x to a boolean value\n```\n\nThis ensures that the result is always a boolean value, regardless of the language's truthiness and falsiness rules." } ]
```