import jieba


# jieba全模式分词
def jieba_tokenize_all(text):
    return list(jieba.cut(text, cut_all=True))

# jieba精确模式分词
def jieba_tokenize_precise(text):
    return list(jieba.cut(text, cut_all=False))

# jieba搜索引擎模式分词
def jieba_tokenize_search(text):
    return list(jieba.cut_for_search(text))

# 示例文本
text_chinese = "欢迎你来到浙江大学爱丁堡国际学院，感谢你的参与！谢谢。"

# 分词
print("jieba全模式分词:", jieba_tokenize_all(text_chinese))
print("jieba精确模式分词:", jieba_tokenize_precise(text_chinese))
print("jieba搜索引擎模式分词:", jieba_tokenize_search(text_chinese))
