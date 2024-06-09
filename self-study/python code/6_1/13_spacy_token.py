import spacy

# 加载spaCy的英语模型
nlp = spacy.load('en_core_web_sm')

# 分词函数
def spacy_tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

# 示例文本
text = "This is an example sentence for spaCy tokenization."

# 分词
tokens = spacy_tokenize(text)
print(tokens)
