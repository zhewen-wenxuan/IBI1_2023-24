import torch
import torch.nn as nn
import torch.optim as optim
import math
import spacy

# Example data
train_data = [
    ("I am a student", "Ich bin ein Student"),
    ("You are a teacher", "Du bist ein Lehrer"),
    ("He is a doctor", "Er ist ein Arzt"),
    ("She is a nurse", "Sie ist eine Krankenschwester"),
]

# 加载spacy中的英语和德语模型
'''
补充：spaCy是一个先进的自然语言处理（NLP）库，广泛应用于处理和分析人类语言数据。
'''
spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

# 使用try进行异常处理，如果无法自动安装英语和德语模型的话，那么就手动代码安装后执行。
# try:
#     spacy_en = spacy.load('en_core_web_sm')
#     spacy_de = spacy.load('de_core_news_sm')
# except OSError:
#     import subprocess
#     subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
#     subprocess.run(["python", "-m", "spacy", "download", "de_core_news_sm"])
#     spacy_en = spacy.load('en_core_web_sm')
#     spacy_de = spacy.load('de_core_news_sm')

'''
tokenize函数的作用是使用spaCy库对输入文本进行分词（tokenization），即将文本拆分成一个个单独的词（token）
text 是函数的第一个参数，表示要进行分词的输入文本。
spacy_model 是函数的第二个参数，表示传入的spaCy模型实例，用于执行分词任务。

spacy_model.tokenizer(text)：调用传入的spaCy模型的分词器（tokenizer）对输入文本进行分词。
分词器会将文本分割成一个个词（token），生成一个包含多个token对象的列表。

[tok.text for tok in spacy_model.tokenizer(text)]：这是一个列表解析（list comprehension）表达式。
for tok in spacy_model.tokenizer(text)：遍历分词器生成的每个token对象。
tok.text：获取每个token对象的文本内容。
将每个token对象的文本内容收集到一个新的列表中。

为什么要分词呢？
1. 分词（tokenization）是自然语言处理（NLP）中的一个基本步骤，之所以需要进行分词，
是因为文本数据在计算机中通常以字符串的形式存在，直接处理这些字符串对大多数NLP任务来说并不方便。
2. 分词将文本分解为词或子词，使后续处理更加容易和高效。实际上就是相当于把一个整体按照某一种规律拆分开，然后进行从部分到整体进行分析或者操作。
3. 分词有助于语法和语义分析。通过分词，可以识别出文本中的单词，并进一步进行词性标注（POS tagging）、句法分析（syntax parsing）
和依存分析（dependency parsing），从而理解文本的结构和意义。
4. 统计词频（term frequency）来进行统计分析工作。
5. 信息检索和文本匹配
分词在信息检索和文本匹配中非常重要。搜索引擎需要将查询字符串和文档进行分词，以便进行索引和匹配。通过分词，可以提高检索的准确性和效率。
6. 机器学习和深度学习
分词是将文本转换为特征向量的前提。大多数机器学习和深度学习算法不能直接处理文本数据，需要将文本转换为数值表示（如词袋模型、TF-IDF、词向量等）。分词是实现这些转换的第一步。
7. 处理语言多样性
分词帮助处理语言的多样性和复杂性。例如，英语和汉语的分词方式不同，英文词汇以空格分隔，而汉语则没有显式的词边界。分词器能够根据语言特点，准确地识别出词语边界。

'''
def tokenize(text, spacy_model):
    return [tok.text for tok in spacy_model.tokenizer(text)]

'''
build_vocab函数的作用是根据输入的分词后的句子列表，构建一个词汇表（vocabulary）。词汇表是一个字典，其中每个词（token）对应一个唯一的索引。
if token not in vocab：检查当前词（token）是否已经在词汇表 vocab 中。
如果当前词不在词汇表中，将其添加到词汇表，并将当前词汇表的长度作为该词的索引。
'''
def build_vocab(tokenized_sentences):
    vocab = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

'''
encode 函数的作用是将输入的句子根据给定的词汇表（vocab）进行编码，即将句子中的每个词转换为词汇表中对应的索引。
例如，encode（我饿了要吃饭） 处理结果会变成 123156151202321351515615

vocab[token]：根据词汇表（vocab）查找当前词对应的索引。
最终将每个词的索引组成一个新的列表。
'''
def encode(sentence, vocab):
    return [vocab[token] for token in sentence]

# Tokenize and build vocabulary
train_src_sentences = [tokenize(pair[0], spacy_en) for pair in train_data]
train_tgt_sentences = [tokenize(pair[1], spacy_de) for pair in train_data]

src_vocab = build_vocab(train_src_sentences)
tgt_vocab = build_vocab(train_tgt_sentences)

input_dim = len(src_vocab)
output_dim = len(tgt_vocab)

encoded_src_sentences = [encode(sentence, src_vocab) for sentence in train_src_sentences]
encoded_tgt_sentences = [encode(sentence, tgt_vocab) for sentence in train_tgt_sentences]

'''
通过以上操作，我们成功将我们的train data转变成为一个数字化的列表，【1231156】来表示文本内容。
'''
#********************************************************************************************************************
'''
class position encoding实现了位置编码（Positional Encoding），它是Transformer模型中用于向输入数据中加入位置信息的一种技术。
1. 关于init初始化中的参数设定：
d_model 是输入向量的维度，也就是模型的隐藏层大小。
max_len 是序列的最大长度，默认为5000。

2. super() 函数用于调用父类的方法，这里调用了父类的 __init__ 方法，用于初始化父类的属性。

3. 创建一个形状为 (max_len, d_model) 的零张量 pe，用于存储位置编码。

4. position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
目的是创建一个张量，表示序列中每个位置的编码。
torch.arange(0, max_len, dtype=torch.float)：这段代码创建了一个张量，从0到max_len - 1，步长为1，数据类型为float。
这个张量表示了序列中每个位置的索引，从0开始一直到max_len - 1。
.unsqueeze(1)：这个方法在张量的维度中插入一个新维度，使其形状从 (max_len,) 变成 (max_len, 1)。
这样做是为了后续的计算方便，因为位置编码需要与输入张量的形状进行匹配，而输入张量的形状通常是 (seq_len, batch_size, d_model)，
其中 seq_len 表示序列的长度，batch_size 表示批量大小，d_model 表示每个位置的特征维度。
unsqueeze(1)的作用是在索引1的位置插入新的维度，即在(max_len,)的后面插入一个新的维度，使得形状变为(max_len, 1)。
这样，position 张量就表示了一个从0到 max_len - 1 的序列，每个位置的索引都是以浮点数的形式存储，并且添加了一个新的维度，以便后续的计算。
具体解释如下：
这样可以使得位置编码在进行加法操作时能够正确地与输入张量相加，而不会引发维度不匹配的错误。
具体来说，考虑到典型的 Transformer 模型中，输入张量的形状通常是 (seq_len, batch_size, d_model)，
其中 seq_len 表示序列长度，batch_size 表示批量大小，d_model 表示每个位置的特征维度。
在这种情况下，如果位置编码的形状为 (seq_len,)，则无法直接与输入张量相加，因为它们的形状不匹配。
但如果将位置编码的形状设计成 (seq_len, 1) 或 (1, seq_len)，则可以使用广播机制，使得位置编码在进行加法操作时自动扩展为与输入张量相匹配的形状。

5.  div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
计算位置编码中的频率值，用于生成正弦和余弦函数的周期。
torch.arange(0, d_model, 2).float()：这段代码创建了一个从0开始到d_model - 1的张量，步长为2，并将其转换为浮点型。
在位置编码中，我们使用一系列不同的频率值来生成正弦和余弦函数的周期。这些频率值控制了正弦和余弦函数的变化速度。通常情况下，这些频率值是固定的，因为它们是根据位置索引计算出来的。
这个张量实际上表示的是一系列位置索引，这些位置索引会被用作正弦和余弦函数中的参数，以便根据位置生成不同的编码值。
这些位置索引用于计算正弦和余弦函数的周期，从而生成位置编码。

(-math.log(10000.0) / d_model)：这个表达式计算了一个常数，用于控制正弦和余弦函数的周期。
在位置编码中，通常使用一个基于变量 d_model 的常数来调整周期，以确保不同位置的编码具有不同的频率。
在 Transformer 模型中的位置编码，通常采用正弦和余弦函数的组合来产生不同位置的编码。
这些函数的周期性是由一个常数控制的，而这个常数通常是基于模型的隐藏层大小 d_model 来确定的。
具体来说，这个常数的计算方式是通过对数函数的应用，以确保不同位置的编码具有不同的频率，而不是简单地线性增加。
这个常数的计算公式为 -math.log(10000.0) / d_model。这里的 10000.0 是一个经验值，它决定了正弦和余弦函数的变化周期，
而 d_model 则是模型的隐藏层大小，它影响了周期的调整程度。通过将这个常数应用到正弦和余弦函数中，可以确保位置编码在不同位置具有不同的频率，从而提供了更多的位置信息。
总之，这个常数的作用是调整正弦和余弦函数的周期，以确保位置编码能够为模型提供足够的位置信息，帮助模型更好地理解序列的结构和顺序。

torch.exp(...)：这个函数对输入的张量中的每个元素进行指数运算。在这里，它将上述步骤中计算出的频率值进行指数运算，得到了一系列正弦和余弦函数的周期值。

综合起来，div_term 张量包含了一系列频率值，用于生成正弦和余弦函数的周期。
这些频率值会影响位置编码中不同位置的编码值，使得编码值在不同位置具有不同的变化速度和幅度。这样，通过将这些频率值与位置索引相乘，就可以生成不同位置的位置编码。

6. pe[:, 0::2] = torch.sin(position * div_term)
   pe[:, 1::2] = torch.cos(position * div_term)
计算位置编码的正弦和余弦值，并将它们分别填充到位置编码张量 pe 的偶数列和奇数列中。
实际上就是我们之前的position已经是有了从0到len-1的编码，即0123456……；但为了能够更好地体现位置关系，我们需要再有一个张量div_term，
它是计算出的一个调整参数，对position进行计算修正，然后修正结果计算sin，cos值，重新保存到我们之前定义的位置编码张量中pe。
pe[:, 0::2]：这是 Python 中切片操作的一种形式，表示取出 pe 张量的所有行（:），并从第0列开始，以步长为2（2）取值，即取出所有偶数列。
这样做是为了确保每个位置的偶数索引对应的编码值是正弦值。
torch.sin(position * div_term)：这是计算正弦值的函数，其中 position * div_term 是每个位置的索引乘以频率值，从而生成每个位置的正弦函数的周期值。
这两行代码通过正弦和余弦函数计算了位置编码的值，并将它们分别填充到位置编码张量 pe 的偶数列和奇数列中，以此来构建位置编码张量。
这样，位置编码张量 pe 中的每一列都对应着不同位置的编码值，其中偶数列对应正弦值，奇数列对应余弦值，以此来表达序列中每个位置的位置信息。

7. pe = pe.unsqueeze(0).transpose(0, 1)
用于调整位置编码张量 pe 的形状，使其符合 Transformer 模型中对位置编码张量的要求。
pe.unsqueeze(0)：这个方法在张量的维度中插入一个新维度，将原始的位置编码张量 pe 的形状从 (max_len, d_model) 变成了 (1, max_len, d_model)。
这个新的维度是在索引 0 处插入的，因此现在 pe 张量具有了一个额外的批量维度，这样做是为了与输入张量的形状相匹配，
因为在 Transformer 模型中，通常将批量维度放在第一个维度上（就是第二个位置）。
transpose(0, 1)：这个方法用于交换张量的维度，这里是将原始的第一个维度（批量维度）和第二个维度（序列长度维度）交换了位置。
因此，最终位置编码张量 pe 的形状变成了 (max_len, 1, d_model)。
这样做是为了使位置编码张量的形状能够与输入张量的形状 (seq_len, batch_size, d_model) 相匹配，以便在模型的前向传播过程中能够正确地与输入张量进行加法操作。

8. self.register_buffer('pe', pe)
register_buffer 是 PyTorch 中的一个方法，用于将张量注册为模型的缓冲区。
'pe' 是注册的缓冲区的名称，你可以根据需要命名。
pe 是位置编码张量，这是我们在前面创建的用于存储位置编码的张量。

解释一个概念——缓冲区
将张量注册为模型的缓冲区有几个主要原因：

持久性: 注册为缓冲区的张量会与模型一起保存和加载。当你保存模型时，缓冲区的状态也会被保存。
这对于需要在训练和推理期间保持不变的张量非常有用，如位置编码、嵌入矩阵等。因此，它们的值在模型的保存和加载过程中不会改变，而无需显式地进行额外的处理。

模型状态: 缓冲区的张量会被视为模型的一部分，因此它们也会受到相同的设备放置和分布式设置的影响。
这样做可以确保张量的状态与模型的状态一致，不会出现设备不匹配或丢失状态的情况。

不参与参数更新: 注册为缓冲区的张量不会参与模型的参数更新过程。这意味着它们的梯度不会被计算或传播，并且在反向传播时不会影响优化器的参数更新。
这对于固定不变的张量（如位置编码、嵌入矩阵等）是很重要的，因为它们不需要在训练过程中进行更新。

清晰性和可读性: 将需要保持不变的张量注册为缓冲区可以提高代码的清晰性和可读性。
通过这种方式，可以清楚地表明哪些张量是模型的一部分，但不参与参数更新，从而更好地组织和管理模型的状态。

综上所述，将张量注册为模型的缓冲区可以简化模型的状态管理，并确保不变性和持久性，同时提高代码的可读性和清晰性。
'''

'''
def forward(self, x):
    return x + self.pe[:x.size(0), :]
self.pe[:x.size(0), :]：这里使用了 Python 中的切片操作，从位置编码张量 pe 中取出与输入张量 x 相同长度的部分。
x.size(0) 返回输入张量 x 的第一个维度的大小，通常表示序列的长度。因此，这个切片操作是为了确保位置编码与输入张量的长度相匹配，以便进行加法操作。

x + self.pe[:x.size(0), :]：这一行代码实现了位置编码的加法操作。位置编码张量 pe 中的每个位置编码都会被加到输入张量 x 中对应位置上。
这样，通过将位置编码加到输入张量中，就实现了位置信息的注入，从而为模型提供了序列的位置信息。
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

'''
1. init 参数解释：
input_dim: 输入数据的特征维度。对于序列数据，通常是词嵌入的维度，或者是其他特征的维度。
output_dim: 输出数据的特征维度。对于序列生成任务，通常是词汇表的大小，或者是要生成的序列的特征维度。
d_model: Transformer 模型中的隐藏层大小，也称为“模型维度”或“特征维度”。也就是编码器和解码器中每个位置的特征维度。默认为512。
nhead: 多头注意力机制中头的数量。在 Transformer 中，每个位置的输入会被分成多个头进行注意力计算，然后多个头的输出会被拼接起来。默认为8。
num_encoder_layers: 编码器中的层数。Transformer 模型由多个编码器层和解码器层堆叠而成。默认为6。
num_decoder_layers: 解码器中的层数。Transformer 模型由多个编码器层和解码器层堆叠而成。默认为6。
dim_feedforward: 每个位置的前馈神经网络（Feedforward Neural Network）的隐藏层大小。在 Transformer 的每个位置，输入会经过一个前馈神经网络进行变换。默认为2048。
dropout: Dropout 概率，用于防止过拟合。在 Transformer 模型中，通常在注意力计算和前馈神经网络中引入 Dropout。默认为0.1，表示10%的节点会被随机置零。

2.补充一下
关于d_model:
隐藏层大小：在Transformer模型中，输入数据会通过多个层进行处理，每个层都有一个固定大小的隐藏表示。
这个隐藏表示的大小就是d_model。在编码器和解码器的每个层中，输入的特征向量都会映射到一个固定大小的隐藏表示中，这个大小由d_model定义。

特征维度：d_model也被称为特征维度，因为它决定了每个位置的特征向量的维度。
在自然语言处理任务中，例如机器翻译，每个词可以表示为一个固定大小的特征向量，而这个特征向量的维度就是d_model。
换句话说，模型通过将输入词映射到一个d_model维的向量表示来表示每个位置的特征。

模型容量：d_model也反映了模型的容量大小，即模型可以表示的特征的丰富程度。
较大的d_model意味着模型有更多的参数和更高的表示能力，可以捕捉更复杂的特征和模式。但是，较大的d_model也会增加模型的计算成本和内存需求。

总之，d_model是Transformer模型中一个非常重要的超参数，它决定了模型中每个位置的特征表示的维度大小，反映了模型的容量大小和特征表示的丰富程度。

3. 关于dim_feedforward:

在Transformer模型中，dim_feedforward参数表示每个位置的前馈神经网络（Feedforward Neural Network）中隐藏层的大小。让我详细解释一下这个参数的含义和作用：

前馈神经网络：在Transformer的每个位置，输入数据都会通过一个前馈神经网络进行变换。
这个前馈神经网络通常由两个全连接层组成，每个全连接层之间会加上一个激活函数（通常是ReLU）。
这个前馈神经网络是Transformer模型中非常重要的一部分，它负责对输入进行非线性变换和特征提取。

隐藏层大小：dim_feedforward参数指定了前馈神经网络中隐藏层的大小。也就是说，每个全连接层的隐藏层都会包含dim_feedforward个神经元。
隐藏层的大小决定了前馈神经网络的表示能力和复杂度，较大的隐藏层可以捕捉更复杂的特征和模式，但也会增加模型的计算成本。

默认值2048：在大多数Transformer模型中，dim_feedforward的默认值通常设置为2048。
这个值是经验性地选择的，在实践中已经证明在各种任务中表现良好。然而，根据具体的任务和数据集，有时候可能需要调整这个参数的值以获得更好的性能。

总之，dim_feedforward参数控制了Transformer模型中每个位置的前馈神经网络中隐藏层的大小，它决定了前馈神经网络的表示能力和复杂度。

4. 关于super().__init__():
当我们定义一个新的神经网络模型时，通常会继承 nn.Module 类，并且在构造函数中调用 super().__init__() 方法来初始化父类。
这样做的目的是为了继承 nn.Module 类的属性和方法，使得我们的自定义模型可以像标准的PyTorch模型一样使用，包括模型参数的管理、前向传播的定义等。

5. self.src_embedding = nn.Embedding(input_dim, d_model)
   self.tgt_embedding = nn.Embedding(output_dim, d_model)
定义了两个嵌入层（Embedding），分别用于将输入和输出的符号序列（通常是单词或者token）转换为密集的连续向量表示.
nn.Embedding(input_dim, d_model)：这是一个 PyTorch 中的嵌入层（Embedding Layer）的创建方式。
它会将一个大小为 input_dim 的离散的输入空间（通常是词汇表的大小）映射到一个 d_model 维的连续向量空间中。
其中，d_model 是 Transformer 模型中的隐藏层大小，也就是每个位置的特征维度。
input_dim：表示输入空间的大小，通常是词汇表的大小或者符号的总数。
d_model：表示输出的连续向量的维度，也就是嵌入后的表示空间的维度。
self.src_embedding：将创建的嵌入层赋值给类的实例属性 self.src_embedding，以便在模型的其他方法中可以方便地引用它。

6. self.fc_out = nn.Linear(d_model, output_dim)
定义了一个线性变换层（Linear Layer），用于将Transformer模型中最后一个位置的隐藏表示（通常是输出的隐藏表示）映射到输出空间，从而生成最终的模型输出。
nn.Linear(d_model, output_dim)：这是一个PyTorch中线性变换层（Linear Layer）的创建方式。
它将一个大小为 d_model 的输入特征向量线性变换为一个大小为 output_dim 的输出特征向量。
在Transformer模型中，通常将最后一个位置的隐藏表示（或者是解码器的输出）通过一个线性变换映射到输出空间，这样就得到了模型的最终输出。

7. self.d_model = d_model
设置了模型的 d_model 属性，即隐藏层的大小。这个属性可以让模型的其他方法和属性可以方便地引用隐藏层的大小

8. self.init_weights()
调用了 init_weights() 方法。在这个方法中，通常会初始化模型的权重，以确保模型的参数处于合适的初始状态。
在大多数情况下，权重会被初始化为随机值，但也可以根据需要进行其他初始化操作，比如使用预训练的权重。
下面还会有这个函数的详细定义。

9. 关于init_weights()
initrange = 0.1：
这行代码定义了初始化权重时所使用的范围。在这里，使用了一个简单的初始范围，即权重的取值范围在 -0.1 到 0.1 之间。

self.src_embedding.weight.data.uniform_(-initrange, initrange)：
这行代码对源（输入）嵌入层的权重进行初始化。
.weight 属性表示嵌入层的权重，.data 属性表示权重的数据，.uniform_(-initrange, initrange) 是一个 PyTorch 张量的方法，用于将张量的值从均匀分布中初始化。

我们一共对于我们的输入嵌入层、目标嵌入层、以及最后的输出层进行了初始化权重，均是（-0.1,0.1）的均匀分布。

self.fc_out.bias.data.zero_()：
这行代码将输出线性变换层的偏置（bias）初始化为零。.bias 属性表示线性变换层的偏置，.zero_() 是一个 PyTorch 张量的方法，用于将张量的值全部置为零。

综上所述，init_weights() 方法通过均匀分布的方式初始化了模型的权重，并将输出线性变换层的偏置初始化为零。
这样做的目的是为了确保模型的参数处于合适的初始状态，从而有助于模型在训练过程中更快地收敛。

10. 关于forward():
定义了 TransformerModel 类中的前向传播方法 forward()，用于执行完整的 Transformer 模型的前向传播过程。
forward() 方法实现了完整的 Transformer 模型的前向传播过程，包括嵌入层的处理、位置编码的添加、Transformer 主体部分的编码解码操作以及输出的线性变换。

def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):：
这是一个前向传播方法的定义，接受多个参数作为输入，包括源序列 (src)、目标序列 (tgt)，以及可选的掩码 (src_mask, tgt_mask, memory_mask)。

src = self.src_embedding(src) * math.sqrt(self.d_model)：
这行代码将源序列 src 通过源嵌入层 self.src_embedding 进行嵌入，得到源序列的嵌入表示。
同时，乘以 math.sqrt(self.d_model) 是为了缩放嵌入向量，这是 Transformer 模型中的一个常用的技巧，有助于减缓嵌入向量的放大问题。

tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)：
类似地，这行代码对目标序列 tgt 进行嵌入。

src = self.pos_encoder(src) 和 tgt = self.pos_encoder(tgt)：
这两行代码分别将源序列和目标序列通过位置编码器 self.pos_encoder 进行位置编码。位置编码是为了为序列中的每个位置添加一些位置信息，以便模型能够利用序列的顺序信息。

output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask)：
这行代码调用了 Transformer 模型的主体部分，即 self.transformer，传入源序列、目标序列以及相应的掩码，进行编码和解码操作，得到模型的输出。

output = self.fc_out(output)：
最后，这行代码将模型的输出通过输出线性变换层 self.fc_out 进行线性变换，将模型的隐藏表示映射到输出空间，从而得到最终的模型输出。

'''
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.src_embedding = nn.Embedding(input_dim, d_model)
        self.tgt_embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout)
        self.fc_out = nn.Linear(d_model, output_dim)

        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask)
        output = self.fc_out(output)
        return output

'''
这个函数 generate_square_subsequent_mask 用于生成一个“后续掩码”（subsequent mask），这是在序列模型（例如Transformer）中使用的一个常见操作。
这个掩码确保模型在预测序列中的某个位置时，只能看到该位置之前的所有位置，而不能看到之后的位置。这个特性在自回归序列生成任务（例如文本生成）中非常重要。
1. mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
生成上三角矩阵
torch.ones(sz, sz)：生成一个大小为 (sz, sz) 的全为 1 的张量。
torch.triu(...)：获取张量的上三角部分（包括对角线），下三角部分会被填充为 0。
== 1：将张量中的值与 1 进行比较，返回一个布尔张量，其中上三角部分为 True，下三角部分为 False。
.transpose(0, 1)：将张量进行转置，即行列互换，这样得到的掩码可以用于序列中的时间步。

2. mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
转换为浮点型并应用掩码填充
mask.float()：将布尔张量转换为浮点型张量。
.masked_fill(mask == 0, float('-inf'))：将掩码中为 0 的位置填充为负无穷大（-inf），这些位置表示模型不应该关注的未来时间步。
.masked_fill(mask == 1, float(0.0))：将掩码中为 1 的位置填充为 0.0，这些位置表示模型可以关注的当前和之前的时间步。

具体介绍看课件最后的部分。
'''

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

'''
1. output_flat = output.argmax(dim=2).view(-1)
output.argmax(dim=2)：对于模型的输出张量 output，在 dim=2 这个维度上找出最大值的索引。
假设 output 的形状为 (batch_size, seq_len, vocab_size)，即每个时间步都有一个 vocab_size 大小的概率分布，这行代码会找到每个时间步上概率最大的那个类别的索引。
.view(-1)：将结果张量展平为一维张量，形状为 (batch_size * seq_len,)。这是为了方便后续计算准确性。

2. target_flat = target.reshape(-1)
target.reshape(-1)：将目标张量 target 展平为一维张量，形状为 (batch_size * seq_len,)。这样做是为了与展平后的预测结果进行比较。

3. correct = (output_flat == target_flat).sum().item()
output_flat == target_flat：比较预测结果和真实标签，得到一个布尔张量，表示每个位置上预测是否正确。
.sum()：将布尔张量中 True 的数量求和，得到正确预测的数量。
.item()：将标量张量转换为 Python 的标量数值。

4. total = target_flat.size(0)
target_flat.size(0)：获取展平后的目标张量的大小，即总的标签数量。

5. 为什么是从dim = 2这个维度找到最大值的索引呢？
是因为 output 张量的第三个维度（dim=2）代表了每个时间步的类别分布。

Transformer 输出张量的形状
在一个典型的序列到序列（seq2seq）模型中，例如 Transformer，输出张量 output 的形状通常是 (batch_size, seq_len, vocab_size)。这里每个维度的含义如下：
batch_size：批次的大小，即一次输入中有多少个序列。
seq_len：序列的长度，即每个输入序列中的时间步数。
vocab_size：词汇表的大小，即每个时间步的预测结果在词汇表中的概率分布。
每个时间步都有一个 vocab_size 大小的向量，这个向量表示该时间步上每个单词的预测概率。为了生成最终的预测结果，我们需要从这个概率分布中选出概率最大的那个单词，即找出最大值的索引。由于每个时间步的预测概率向量位于第三个维度（dim=2），因此我们需要在 dim=2 上执行 argmax 操作。
'''
def calculate_accuracy(output, target):
    output_flat = output.argmax(dim=2).view(-1)
    target_flat = target.reshape(-1)
    correct = (output_flat == target_flat).sum().item()
    total = target_flat.size(0)
    return correct / total
''''
1. src = torch.tensor(encoded_src_sentences, dtype=torch.long).transpose(0, 1)
   tgt = torch.tensor(encoded_tgt_sentences, dtype=torch.long).transpose(0, 1)
这两行代码的目的是将源语言句子和目标语言句子转换成适合模型输入的张量格式，并进行转置操作。
encoded_src_sentences 和 encoded_tgt_sentences 是经过分词和编码后的源语言和目标语言句子。它们通常是形如 [batch_size, seq_len] 的嵌套列表。
torch.tensor(..., dtype=torch.long)：将这些嵌套列表转换为 PyTorch 的长整型（long）张量。长整型张量通常用于表示词汇表中的单词索引。
.transpose(0, 1)：将张量的第0维和第1维进行转置。
对于 encoded_src_sentences 和 encoded_tgt_sentences，假设它们的形状是 [batch_size, seq_len]，转置后它们的形状会变成 [seq_len, batch_size]。
为什么要进行转置
在大多数 Transformer 模型的实现中，输入的形状要求是 [seq_len, batch_size]，而不是 [batch_size, seq_len]。转置操作是为了满足模型的输入要求。

例如：
我输入的文本处理后是：
[[1, 2, 3], [4, 5, 6]]
然后我需要经过tensor的转化，将numpy转变为tensor
tensor([[1, 2, 3],
        [4, 5, 6]], dtype=torch.long)
然后我又利用.transpose(0, 1)实现转置
tensor([[1, 4],
        [2, 5],
        [3, 6]], dtype=torch.long)
就是为了符合transformer的输入格式。

2. src_mask = generate_square_subsequent_mask(src.size(0))
   tgt_mask = generate_square_subsequent_mask(tgt.size(0))
generate_square_subsequent_mask:
这是一个我们之前设定的mask矩阵函数，用于生成正方形的后续掩码矩阵（square subsequent mask）。
这个掩码矩阵用于防止模型在处理序列时看到未来的时间步。
src.size(0) 和 tgt.size(0):
src和tgt分别代表源序列（source sequence）和目标序列（target sequence）。
src.size(0)和tgt.size(0)获取的是源序列和目标序列的长度，即时间步数。
src_mask 和 tgt_mask:
src_mask和tgt_mask分别是针对源序列和目标序列生成的掩码矩阵。

3. output = model(src, tgt, src_mask, tgt_mask)
model:
model是一个深度学习模型，通常是基于Transformer架构的模型。这个模型可以是用于机器翻译的Seq2Seq模型、语言模型或其他需要处理序列数据的模型。
我们的就是基于 Transformer 的序列到序列（Seq2Seq）模型。

src:
src是源序列，表示输入到模型的原始数据。例如，在我们的机器翻译任务中，src是英语句子。

tgt:
tgt是目标序列，表示模型要生成的目标数据。例如，在我们的机器翻译任务中，tgt是翻译后的德语句子。

src_mask:
src_mask是源序列的掩码矩阵，确保模型在处理源序列时，不会被填充的位置（如果有）影响，或者在某些情况下，可以用于特定位置的屏蔽。

tgt_mask:
tgt_mask是目标序列的掩码矩阵，确保模型在处理目标序列时，每个时间步只能看到当前及之前的时间步，不能看到未来的时间步。这对于自回归模型尤为重要，确保训练时的因果性。

补充一下：
模型的处理过程

输入处理:
源序列src和目标序列tgt被传入模型。
src_mask和tgt_mask被用于屏蔽特定的位置，确保模型只关注需要关注的部分。

编码（Encoder）阶段:
模型首先对src进行编码处理。编码器将src转换为一系列表示（表示序列），通常是高维的向量。
src_mask在这个过程中确保编码器忽略填充位置或其他不相关的位置。

解码（Decoder）阶段:
解码器接收编码器生成的表示和tgt进行处理。
在每个时间步，解码器生成目标序列的下一个元素。
tgt_mask确保解码器在每个时间步只看到当前时间步及之前的部分，防止信息泄露。

输出生成:
模型最终生成output，通常是目标序列的概率分布或直接的序列输出。
在训练阶段，output用于计算损失（比如交叉熵损失），并用于更新模型参数。
在推理阶段，output可以直接作为预测结果。
'''
def main():
    # Hyperparameters
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1

    # Initialize model
    model = TransformerModel(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers,
                             dim_feedforward, dropout)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # Example input data
    src = torch.tensor(encoded_src_sentences, dtype=torch.long).transpose(0, 1)
    tgt = torch.tensor(encoded_tgt_sentences, dtype=torch.long).transpose(0, 1)

    # Generate masks
    src_mask = generate_square_subsequent_mask(src.size(0))
    tgt_mask = generate_square_subsequent_mask(tgt.size(0))

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Model forward pass
        output = model(src, tgt, src_mask, tgt_mask)

        # Compute loss
        loss = criterion(output.reshape(-1, output_dim), tgt.reshape(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        accuracy = calculate_accuracy(output, tgt)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item()} | Accuracy: {accuracy}")

if __name__ == '__main__':
    main()
