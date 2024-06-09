import hanlp

# 加载HanLP的预训练分词模型
tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

# 分词函数
def hanlp_tokenize(text):
    return tokenizer(text)

# 示例文本
text = "欢迎你来到浙江大学爱丁堡国际学院，感谢你的参与！谢谢。"

# 分词
tokens = hanlp_tokenize(text)
print("HanLP分词:", tokens)
