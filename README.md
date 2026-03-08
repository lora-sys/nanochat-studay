# nano-chat-study
[NanoChatStudy](logo/logo.png)
## This project is for learning the nanochat concept

Inspired by Karpathy's amazing work on building a complete LLM pipeline.

### Reference
- **Original Project**: [karpathy/nanochat](https://github.com/karpathy/nanochat)
- **Goal**: Replicate the end-to-end training (Pretrain -> SFT -> RL) for ~$100 in 4 hours.

---
*Learning by building...*


---

*please use uv for package manager , is very fast and easy to use*
```bash
pip install uv
npm install uv
```


```bash
uv venv nanochat
source nanochat/bin/activate
```

---


---
[Basictokenizor](/tokenizer/bye.py)
*Tokenizor*

[tokenizor](/tokenizer/tokenizor.png)
```python

tokenizor  = BasicTokenizor
text = """
en/chinese/emoji/.../[]/:/,/./?/>/</|/{/}/+/_ /etc.....

"""

- encode()
-  decode()


tokenizor.train()

```

```bash
uv run tokenizor/bpe.py
```
train your basic tokenizor
---





