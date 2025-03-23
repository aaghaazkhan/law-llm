# law-llm
A fine-tuned Qwen2.5-based legal language model for answering basic law-related queries. Trained on legal documents, and case laws.


---
```
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: transformers
license: apache-2.0
datasets:
- DevToAI/indian_laws_llama2_supported
language:
- en
metrics:
- perplexity
```
---

# **Qwen 2.5 3B - Indian Law QA**

## **Model Details**

* Model Name: ```aaghaazkhan/qwen2.5-3b-indian-law-instruct```
* Base Model: ```Qwen 2.5 3B```
* License: ```Apache 2.0```
* Library: ```transformers```
* Language: ```English```
* Fine-Tuned By: ```Aaghaaz Khan```
* Task: ```Question Answering, Instruction Fine-Tuning```

  
# **Model Description**
This model is fine-tuned on Indian legal QA using instruction fine-tuning. It helps answer legal queries based on Indian law.

**Prompt Format**
**Chat Format (Qwen-style)**
``` json
[
    {"role": "system", "content": "You are a legal expert."},
    {"role": "user", "content": "What is Article 21 of the Indian Constitution?"}
]
```

# **Usage**

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "aaghaazkhan/qwen2.5-3b-indian-law-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are a legal expert."},
    {"role": "user", "content": "What is Article 21 of the Indian Constitution?"}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

# **Evaluation Metrics**

```Perplexity: 0.303```

```Perplexity: 1.354```
