# nano-chat-study

<div align="center">
<image src="logo/logo.png" alt="nanochat studay">

<h3>
Nanochat study
</h3>
<div>
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


[basictokenizor](https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py)

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


---
Regex Tokenizor

This is use gpt4 spilt pattern text to chunk , to complement squense means

[minbpe](https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py)

core
```python
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


```

---


