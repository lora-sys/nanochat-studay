import regex as re
import pickle
from .base import Tokenizer, get_stats, merge
# GPT-2 和 GPT-4 的文本分割模式
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
class RegexTokenizer(Tokenizer):
    """
    基于 Regex 的 BPE Tokenizer
    
    特点：
    1. 使用 regex 预处理文本（分 chunk）
    2. 在每个 chunk 内应用 BPE
    3. 支持 special tokens（如 <|endoftext|>）
    """
    
    def __init__(self, pattern=None):
        """
        初始化 Tokenizer
        
        Args:
            pattern: 可选，自定义 regex 模式（默认 GPT-4 模式）
        """
        super().__init__()
        # 默认使用 GPT-4 的分词模式
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        
        # special tokens 支持
        self.special_tokens = {}          # str -> int
        self.inverse_special_tokens = {}  # int -> str
        
        # 训练后才会有的属性
        self.merges = {}    # (int, int) -> int
        self.vocab = {}     # int -> bytes
    
    def train(self, text, vocab_size, verbose=False):
        """
        训练 BPE Tokenizer
        
        流程：
        1. 用 regex 分割文本为 chunks
        2. 将每个 chunk 转为 bytes（作为 BPE 的基本单元）
        3. 迭代合并最高频的 pair，直到 vocab_size
        
        Args:
            text: 训练文本
            vocab_size: 目标词表大小（必须 >= 256）
            verbose: 是否打印训练过程
        """
        assert vocab_size >= 256, "vocab_size 必须至少为 256（字节数）"
        
        num_merges = vocab_size - 256
        
        # Step 1: 用 regex 分 chunk
        text_chunks = re.findall(self.compiled_pattern, text)
        if verbose:
            print(f"文本被分割为 {len(text_chunks)} 个 chunks")
        
        # Step 2: 每个 chunk 转为 bytes 列表（BPE 的基本单元）
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        
        # Step 3: 初始化 vocab（字节 -> bytes）
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        
        # Step 4: 迭代合并
        for i in range(num_merges):
            # 统计所有 chunk 中 pair 的频率
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            
            if not stats:
                break  # 没有更多 pair 可以合并
            
            # 找出最高频的 pair
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # 在所有 chunk 中替换该 pair
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            
            # 记录 merge
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) "
                      f"had {stats[pair]} occurrences")
        
        if verbose:
            total_before = sum(len(ch.encode("utf-8")) for ch in text_chunks)
            total_after = sum(len(chunk) for chunk in ids)
            compression = total_before / total_after
            print(f"\n压缩率: {compression:.2f}x ({total_before} -> {total_after} tokens)")
    
    def register_special_tokens(self, special_tokens):
        """
        注册特殊 token（如 <|endoftext|>）
        
        Args:
            special_tokens: dict, 如 {"<|endoftext|>": 100257}
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
    
    def _encode_chunk(self, text_bytes):
        """
        对单个 chunk 应用 BPE 编码
        
        使用训练时学到的 merges，按顺序合并 pair
        
        Args:
            text_bytes: bytes，单个 chunk 的字节
        
        Returns:
            list: token ids
        """
        ids = list(text_bytes)
        
        while len(ids) >= 2:
            # 找到所有可合并的 pair
            stats = get_stats(ids)
            
            # 选择 merge index 最小的 pair（即最早训练的 pair）
            # 这样可以保证编码结果与训练时一致
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break  # 没有更多可合并的 pair
            
            # 合并
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        
        return ids
    
    def encode_ordinary(self, text):
        """
        普通文本编码（忽略 special tokens）
        
        流程：
        1. 用 regex 分 chunk
        2. 对每个 chunk 应用 BPE
        3. 合并所有 chunk 的结果
        
        Args:
            text: 输入文本
        
        Returns:
            list: token ids
        """
        # 按 regex 模式分割
        text_chunks = re.findall(self.compiled_pattern, text)
        
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        
        return ids
    
    def encode(self, text, allowed_special="none_raise"):
        """
        完整编码（支持 special tokens）
        
        Args:
            text: 输入文本
            allowed_special: 
                - "all": 允许所有 special tokens
                - "none": 忽略 special tokens（当作普通文本）
                - "none_raise": 如果遇到 special token 则报错（默认）
                - set: 允许特定的 special tokens
        
        Returns:
            list: token ids
        """
        # 确定允许哪些 special tokens
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            # 检查是否包含未注册的 special tokens
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(f"遇到未允许的特殊 token: {token}")
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"不支持的 allowed_special: {allowed_special}")
        
        # 如果没有 special tokens，直接普通编码
        if not special:
            return self.encode_ordinary(text)
        
        # 用 special tokens 分割文本
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        parts = re.split(special_pattern, text)
        
        ids = []
        for part in parts:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        
        return ids
    
    def decode(self, ids):
        """
        将 token ids 解码为文本
        
        Args:
            ids: list of token ids
        
        Returns:
            str: 解码后的文本
        """
        byte_parts = []
        
        for idx in ids:
            if idx in self.vocab:
                byte_parts.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                byte_parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"未知的 token id: {idx}")
        
        text_bytes = b"".join(byte_parts)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def save(self, file_prefix):
        """
        保存 tokenizer 到文件
        
        Args:
            file_prefix: 文件前缀（将创建 .pkl 文件）
        """
        data = {
            "pattern": self.pattern,
            "merges": self.merges,
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
        }
        with open(f"{file_prefix}.pkl", "wb") as f:
            pickle.dump(data, f)
        
        if True:  # 同时保存人类可读版本用于调试
            with open(f"{file_prefix}.merges", "w", encoding="utf-8") as f:
                for (p0, p1), idx in self.merges.items():
                    f.write(f"{p0} {p1} -> {idx}\n")
            with open(f"{file_prefix}.vocab", "w", encoding="utf-8") as f:
                for idx, token in sorted(self.vocab.items()):
                    s = token.decode("utf-8", errors="replace")
                    f.write(f"[{idx}] -> {repr(s)}\n")
    
    @classmethod
    def load(cls, file_prefix):
        """
        从文件加载 tokenizer
        
        Args:
            file_prefix: 文件前缀
        
        Returns:
            RegexTokenizer: 加载后的 tokenizer
        """
        with open(f"{file_prefix}.pkl", "rb") as f:
            data = pickle.load(f)
        
        # 创建实例（不传 vocab_size）
        tokenizer = cls(pattern=data["pattern"])
        tokenizer.merges = data["merges"]
        tokenizer.vocab = data["vocab"]
        tokenizer.special_tokens = data.get("special_tokens", {})
        tokenizer.inverse_special_tokens = {v: k for k, v in tokenizer.special_tokens.items()}
        
        return tokenizer