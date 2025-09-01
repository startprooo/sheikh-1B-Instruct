# Sheikh-1B-Instruct 🧠✨

**Sheikh-1B-Instruct** is a lightweight **instruction fine-tuned** Large Language Model (LLM), designed to demonstrate that even compact models can achieve compelling instruction-following capabilities.  

This model extends vocabulary, supports **function calling**, and integrates seamlessly with the latest **Hugging Face Transformers** library.

---

## 🚀 Features

- 🔤 **Extended Vocabulary**: Supports up to **32,768 tokens**  
- 🪄 **v3 Tokenizer** support  
- ⚡ **Function Calling** with `transformers >= 4.42.0`  
- 🔗 **Tool Calling** (with ID tracking) – compatible with **Mistral-style tool call IDs**  
- 🧪 **Instruct Fine-tuned** for improved task following  
- 🧩 **Lightweight & Flexible**: Easy to adapt for research, demos, and experimentation  

---

## 📦 Installation

Make sure you have the latest version of `transformers`:

```bash
pip install -U transformers>=4.42.0
````

Clone the repository:

```bash
git clone https://github.com/startprooo/sheikh-1B-Instruct.git
cd sheikh-1B-Instruct
```

---

## 🛠️ Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "startprooo/sheikh-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer("Explain the difference between AI and Machine Learning:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🔧 Function Calling Example

Sheikh-1B-Instruct supports **function (tool) calling**.
For a complete guide, see the [Transformers Function Calling Docs](https://huggingface.co/docs/transformers/main/en/chat_templating#function-calling).

⚠️ Note: Tool calls require **exactly 9-character alphanumeric IDs**, and both tool calls and results must be added back to the chat history.

Example snippet:

```python
messages = [
    {"role": "user", "content": "What's the weather in Dhaka?"},
    {"role": "assistant", "tool_calls": [
        {
            "id": "abc123xyz",   # 9-char tool call ID
            "type": "function",
            "function": {"name": "get_weather", "arguments": {"location": "Dhaka"}}
        }
    ]}
]
```

---

## ⚠️ Limitations

* 🛑 **No Moderation**: This demo model does not include safety guardrails
* 🧩 **Research/Experimentation Only**: Not suitable for production without additional alignment & moderation layers

---

## 📚 Resources

* [Transformers Documentation](https://huggingface.co/docs/transformers/index)
* [Function Calling Guide](https://huggingface.co/docs/transformers/main/en/chat_templating#function-calling)
* [Mistral Tool Calling](https://huggingface.co/docs/transformers/main/en/chat_templating#mistral-tool-calling)

---

## 🤝 Contributing

We welcome community contributions to improve safety, alignment, and functionality.
Feel free to open **Issues** or submit a **Pull Request**.

---

## 📜 License

Apache 2.0 – Free to use, modify, and distribute under the terms of the license.

---

### 🌟 Star this repo if you find it useful!

```
