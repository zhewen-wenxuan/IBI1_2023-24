import jieba

def jieba_tokenize(text):
    return list(jieba.cut(text))

# 示例
text = "这是一个使用jieba进行中文分词的例子。"
tokens = jieba_tokenize(text)
print(tokens)
