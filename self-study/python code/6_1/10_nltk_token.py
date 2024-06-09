import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

# 示例文本
text = "Hello world. This is a test sentence."

# 句子分割
sentences = sent_tokenize(text)
print("句子分割:", sentences)

# 单词分割
words = word_tokenize(text)
print("单词分割:", words)
