from tokenizer.bpe import  RegexTokenizer
def main():
    # 测试文本
    text = """
    Hello World! This is a sample text for testing BPE tokenizer.
    Hello World! This is a sample text for testing BPE tokenizer.
    Hello World! This is a sample text for testing BPE tokenizer.
    Python programming is great for data science and machine learning.
    Python programming is great for data science and machine learning.
    Python programming is great for data science and machine learning.
    这是一个测试用的中文文本，包含多种语言。
    这是一个测试用的中文文本，包含多种语言。
    这是一个测试用的中文文本，包含多种语言。
    Numbers: 12345, 67890, 11111, 22222, 33333, 44444, 55555
    Numbers: 12345, 67890, 11111, 22222, 33333, 44444, 55555
    Numbers: 12345, 67890, 11111, 22222, 33333, 44444, 55555
    Symbols: @#$%&*()_+-=[]{}|;':\",./<>?
    Symbols: @#$%&*()_+-=[]{}|;':\",./<>?
    Symbols: @#$%&*()_+-=[]{}|;':\",./<>?
    Emails: user@example.com, admin@test.org, hello@world.net
    URLs: https://example.com, http://test.org/page.html
    Special tokens: <|endoftext|> <|fim_prefix|> <|fim_middle|>
    Special tokens: <|endoftext|> <|fim_prefix|> <|fim_middle|>
    你好，世界！这是一个测试。
    你好，世界！这是一个测试。
    你好，世界！这是一个测试。
    The quick brown fox jumps over the lazy dog.
    The quick brown fox jumps over the lazy dog.
    The quick brown fox jumps over the lazy dog.
    123 456 789 012 345 678 901 234 567 890
    123 456 789 012 345 678 901 234 567 890
    Python Java C++ JavaScript TypeScript Go Rust
    Python Java C++ JavaScript TypeScript Go Rust
    Python Java C++ JavaScript TypeScript Go Rust
    Machine learning and deep learning are subsets of artificial intelligence.
    Machine learning and deep learning are subsets of artificial intelligence.
    Machine learning and deep learning are subsets of artificial intelligence.
    Test: one two three four five six seven eight nine ten
    Test: one two three four five six seven eight nine ten
    Programming languages include Python, Java, C++, JavaScript, and Rust.
    Programming languages include Python, Java, C++, JavaScript, and Rust.
    Programming languages include Python, Java, C++, JavaScript, and Rust.
    """
    
    print("=" * 50)
    print("训练 RegexTokenizer")
    print("=" * 50)
    
    # 创建并训练
    tokenizer = RegexTokenizer()
    tokenizer.train(text, vocab_size=500, verbose=True)
    
    # 测试编码/解码
    
    ids = tokenizer.encode_ordinary(text)
    print(f"编码结果: {ids}")
    
    decoded = tokenizer.decode(ids)
    print(f"解码结果: {decoded}")
    
    # 保存
    tokenizer.save("test_tokenizer")
    print("\nTokenizer 已保存到 test_tokenizer.pkl")
if __name__ == "__main__":
    main()