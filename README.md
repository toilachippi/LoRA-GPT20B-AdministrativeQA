# 🧠 Fine-tuning GPT-OSS 20B (Unsloth) cho dữ liệu Thủ tục hành chính Việt Nam

Tài liệu này hướng dẫn quy trình chuẩn để:
1. Chuẩn bị và làm sạch dữ liệu hội thoại định dạng JSONL.  
2. Áp dụng `tokenizer.apply_chat_template` để chuyển dữ liệu thành chuỗi huấn luyện chuẩn.  
3. Huấn luyện mô hình bằng `SFTTrainer` (từ TRL) + Unsloth, **chỉ tính loss trên phần trả lời của assistant**.


## 📌 1. Chuẩn bị môi trường

```bash
pip install -U "transformers>=4.44" "accelerate>=0.34" "peft>=0.12" xformers
pip install -U bitsandbytes==0.43.3
pip install -U unsloth unsloth_zoo
```

> ⚠️ Tránh ép phiên bản `torch` và `triton` nếu không chắc CUDA của máy.
> `bitsandbytes` nên khớp CUDA (ví dụ Colab dùng CUDA 12.x → 0.43.3).

---

## 📊 2. Làm sạch dữ liệu hội thoại

Dữ liệu gốc có dạng mỗi dòng JSON:

```json
{
  "messages": [
    {"role": "system", "content": "You are ChatGPT, ..."},
    {"role": "user", "content": "Tôi là 1 người dân Việt Nam ..."},
    {"role": "user", "content": "Cá nhân đăng ký ... cần nộp những loại giấy tờ gì?"},
    {"role": "assistant", "content": "Theo quy định tại Khoản 3 ..."}
  ]
}
```

Ta cần:

* Loại bỏ metadata hệ thống không cần thiết.
* Giữ lại cấu trúc hội thoại chuẩn (system ngắn gọn, user, assistant).
* Tạo trường `text` bằng `tokenizer.apply_chat_template`.

```python
from datasets import load_dataset

raw_ds = load_dataset(
    'json',
    data_files="/content/drive/MyDrive/Colab Notebooks/data_tthc/qa_conversations.jsonl",
    split='train'
)

def clean_messages(ex):
    msgs = []
    for m in ex["messages"]:
        role = m.get("role", "").strip()
        content = (m.get("content") or "").strip()
        if not role or not content:
            continue
        # Bỏ các dòng meta không cần thiết
        if role == "system" and ("Knowledge cutoff" in content or "Valid channels" in content):
            continue
        if role == "system":
            content = "Bạn là trợ lý thủ tục hành chính công tại Việt Nam."
        msgs.append({"role": role, "content": content})
    return {"messages": msgs}

cleaned = raw_ds.map(clean_messages)
```

---

## ✍️ 3. Định dạng hội thoại bằng Chat Template

```python
def formatting_prompts_func(batch):
    texts = []
    for convo in batch["messages"]:
        s = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(s)
    return {"text": texts}

formatted = cleaned.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=cleaned.column_names
).filter(lambda ex: isinstance(ex["text"], str) and len(ex["text"]) > 0)

print(formatted[0]["text"][:300])
```

---

## 🧪 4. Cấu hình Collator để tính loss đúng phần assistant

Với template Unsloth, phần trả lời assistant bắt đầu bằng:

```
<|start|>assistant<|message|>
```

Dùng `DataCollatorForCompletionOnlyLM` để tự động mask phần prompt:

```python
from transformers import DataCollatorForCompletionOnlyLM

response_template = "<|start|>assistant<|message|>"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

data_collator = DataCollatorForCompletionOnlyLM(
    response_template_ids=response_template_ids,
    tokenizer=tokenizer,
)
```

---

## 🚀 5. Huấn luyện mô hình

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_train_epochs = 1,
    learning_rate = 2e-4,
    logging_steps = 1,
    optim = 'adamw_8bit',
    weight_decay = 0.01,
    lr_scheduler_type = 'linear',
    seed = 3407,
    output_dir = 'outputs',
    report_to = 'none',
    max_seq_length = 1024,
    dataset_text_field = "text",
    packing = True,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted,
    args = training_args,
    data_collator = data_collator,
)

trainer.train()
```

---

## 🧠 6. Kiểm thử mô hình sau huấn luyện

```python
test_messages = [
    {"role": "system", "content": "Bạn là trợ lý thủ tục hành chính công tại Việt Nam."},
    {"role": "user", "content": "Cá nhân đăng ký Bồi dưỡng nghiệp vụ đăng kiểm viên tàu cá cần nộp giấy tờ gì?"}
]

inputs = tokenizer.apply_chat_template(
    test_messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 📌 Ghi chú

* Nếu có nhiều lượt assistant trong một mẫu, collator sẽ tính loss ở tất cả phần trả lời.
* Nên đảm bảo các trường `messages` đã được chuẩn hoá role (`system`, `user`, `assistant`) trước khi mapping.
* Có thể tăng `num_train_epochs` hoặc `max_steps` khi huấn luyện thật.

---

## 📚 Tài liệu tham khảo

* [Unsloth Documentation](https://github.com/unslothai/unsloth)
* [TRL - SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
* [DataCollatorForCompletionOnlyLM](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorForCompletionOnlyLM)

---

## ✨ Kết quả mong đợi

Sau khi fine-tune, mô hình có khả năng:

* Hiểu ngữ cảnh hội thoại hành chính Việt Nam.
* Trả lời chính xác theo quy định pháp lý đã được cung cấp trong dữ liệu huấn luyện.
* Giữ được định dạng hội thoại chuẩn `<|start|>user|assistant|>`.


