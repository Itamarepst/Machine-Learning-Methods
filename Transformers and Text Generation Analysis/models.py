import torch
import torch.nn as nn
import torch.nn.functional as F

class MySelfAttention(nn.Module):
    """
    Self attention layer is a mechanism that allows the model to weigh the importance of different words in a sequence when processing each word. This is achieved
    by computing attention scores between each pair of tokens (e.g. words) and using those scores to create a weighted sum of the corresponding word em- beddings.
    Formally, given inputs x over a minibatch of size m, denoted as B = {x1, x2, ..., xm} where each sample xi represents a sentence of size (T, d),
    with T denoting the number of tokens and d as the embedding size. Self- Attention maps xi into query, key, and value feature spaces through a linear operation:
    Q = x i · W qT
    K = x i · W kT 
    V = x i · W vT
    where Wq, Wk, and Wv are learnable parameters of size d×d, consistent for all xi, which can be easily 
    implemented with the nn.Linear module. 
    Consequently, the resulting Q, K, and V matrices are of size [T × d].
    Next, the Self-Attention mechanism is expressed as:
    Self-Attention(xi) = softmax(QK T / √d) V
    Here, Q,K, and V represent the query, key, and value matrices of the mapped tokens, respectively,
    with d denoting the embedding size. The softmax func- tion normalizes attention scores across the entire sequence. 
    """
    def __init__(self, input_dim):
        """
        :param input_dim: The feature dimension the input tokens (d).
        """
        super(MySelfAttention, self).__init__()
        self.input_dim = input_dim
        
        # Initialize the learnable parameters 
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)


    def forward(self, x):
        """
        :param x: The input tensor of shape (T, d).
        :return: The output tensor of shape (T, d).
        """

        # compute Q, K, and V matrices
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # compute the attention scores - batch matrix multiplication
        #attention_scores = torch.matmul(Q, K.transpose(0, 1)) / (self.input_dim ** 0.5)
        attention_scores =  Q.bmm(K.transpose(1, 2)) / (self.input_dim ** 0.5)

        # apply the softmax function
        attention_scores = F.softmax(attention_scores, dim=-1)

        # compute the weighted sum of the values (bmm- batch matrix multiplication )
        out = attention_scores.bmm(V)

        return out





class MyLayerNorm(nn.Module):
    """
    Layer Normalization layer is a technique used to normalize the activations of each layer in the network. 
    It helps stabilize the training process by ensuring that the inputs to each layer have a consistent mean and variance. 
    Formally, given inputs x over a minibatch of size m, denoted as B = {x1, x2, ..., xm}, 
    where each sample xi represents a sentence of size (T,d), with T being the number of tokens and d as the embedding size. 
    Each sample xi can be viewed as containing K = T · d elements. 
    To compute the mean μi and variance σi2 for each sample xi, 
    we sum the elements and then normalize as follows:
    μi = 1/K ∑K k=1 xik
    σi2 = 1/K ∑K k=1 (xik − μi)2
    We can now normalize each sample xi:
    LayerNorm(xi) = γi (xi − μi) / √(σi2 + ε) + β
    Following this, each sample is normalized to have a zero mean and unit variance, with an added term ε for numerical stability.
    γ and β serve as learnable param- eters of size K for scaling and shifting each of the K dimensions (multiplication
    and addition are element-wise), respectively. We will use ε = 10−8
    """
    # use ε = 10−8
    EPSELON = 1e-8

    def __init__(self, input_dim):
        """
        :param input_dim: The dimension of the input (T, d).
        """
        super(MyLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(*input_dim))
        self.beta = nn.Parameter(torch.zeros(*input_dim))
        self.e= MyLayerNorm.EPSELON

    def forward(self, x):
        """
        :param x: The input tensor of shape (T, d).
        :return: The output tensor of shape (T, d).
        """

        # compute the mean and variance
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # normalize the input
        #out = self.gamma * (x - mean) / torch.sqrt(std ** 2 + self.e) + self.beta
        out = self.gamma * (x - mean) / (std +  self.e ) + self.beta

        return out

class MyTransformerBlock(nn.Module):
    """
    Transformer block.
    """
    def __init__(self, max_len, input_dim):
        super(MyTransformerBlock, self).__init__()
        self.attention = MySelfAttention(input_dim)
        self.norm1 = MyLayerNorm((max_len, input_dim))
        self.norm2 = MyLayerNorm((max_len, input_dim))
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.attention(x)
        x = self.norm1(self.dropout(out) + x)
        out = self.fc2(F.relu(self.fc1(x)))
        out = self.norm2(out + x)
        return out



class MyTransformer(nn.Module):
    """
    Transformer 
    """
    def __init__(self, vocab, max_len, num_of_blocks):
        """
        :param vocab: The vocabulary object.
        :param num_of_blocks: The number of transformer blocks.
        """
        super(MyTransformer, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.emb_dim = self.embedding.embedding_dim
        self.max_len = max_len
        self.blocks = nn.ModuleList([MyTransformerBlock(self.max_len, self.emb_dim) for _ in range(num_of_blocks)])
        self.fc = nn.Linear(self.emb_dim, 1)


    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        avg_pooling = x.mean(dim=1)
        x = self.fc(avg_pooling)
        return x

