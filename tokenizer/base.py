from abc import ABC, abstractmethod


def get_stats(ids, stats=None):
    """统计相邻 token pair 的频率"""
    if stats is None:
        stats = {}
    for pair in zip(ids, ids[1:]):
        stats[pair] = stats.get(pair, 0) + 1
    return stats


def merge(ids, pair, idx):
    """将 ids 中所有 pair 替换为 idx"""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class Tokenizer(ABC):
    """Tokenizer 基类"""
    
    def __init__(self):
        self.merges = {}
        self.vocab = {}
    
    @abstractmethod
    def train(self, text, vocab_size, verbose=False):
        pass
    
    @abstractmethod
    def encode(self, text):
        pass
    
    @abstractmethod
    def decode(self, ids):
        pass
    
    @abstractmethod
    def save(self, file_prefix):
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, file_prefix):
        pass


