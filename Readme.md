# 概要
arXiv:2410.05258

This code is Differential Attention on Pytorch.
上記の論文内で登場した、 Differential Attention　をPytorch上で実装したものとなります。
# 使い方 How to use
x = torch.ones(1, 1, 128)# [Batch, Seq, Dim]/
atn = DiT_Attention(128, 8, 0.99)/
x = atn(x, x, x)/
