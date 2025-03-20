import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """极简多头注意力机制实现"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        # 将q, k, v的线性变换合并到一起
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # 线性变换并分离qkv
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意力加权并拼接
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        context = context.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        context = context.reshape(batch_size, seq_len, d_model)
        
        # 输出投影
        output = self.out_proj(context)
        return output


class FeedForward(nn.Module):
    """简单的前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块: 注意力 + 前馈网络"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 注意力层 + 残差连接
        attn_output = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # 前馈网络 + 残差连接
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class MiniTransformer(nn.Module):
    """极小的Transformer模型"""
    def __init__(
        self, 
        vocab_size, 
        d_model=256, 
        num_heads=4, 
        num_layers=4, 
        d_ff=512, 
        max_seq_len=128, 
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embedding层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终输出层
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        self.apply(_init_weights)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        
        # 获取token embedding和position embedding
        token_emb = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # 通过Transformer块
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # 输出层
        x = self.norm(x)
        logits = self.fc(x)
        
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """生成文本"""
        # idx是当前上下文，形状为[batch_size, seq_len]
        for _ in range(max_new_tokens):
            # 如果序列太长，截断
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # 前向传播
            logits = self(idx_cond)
            
            # 只关注最后一个token的预测
            logits = logits[:, -1, :] / temperature
            
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 追加到当前序列
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx