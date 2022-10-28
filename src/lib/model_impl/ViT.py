import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """
    Implementing sinusoidal positional encoding described in the Attention is all you need paper
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        # [batch_size, seq_len, 1]
        pos_arr = torch.arange(start=0, step=1, end=seq_len).unsqueeze(0).repeat([batch_size, 1]).unsqueeze(2)
        # [batch_size, 1, embed_dim]
        i_arr = torch.arange(start=0, step=1, end=embed_dim).unsqueeze(0).repeat([batch_size, 1]).unsqueeze(1)
        encoding = pos_arr / torch.pow(10000, 2 * i_arr / embed_dim)
        # apply sinus to even idxs and cosine to odd
        encoding[:, 0::2, :] = torch.sin(encoding[:, 0::2, :])
        encoding[:, 1::2, :] = torch.cos(encoding[:, 1::2, :])
        # todo - hacky and slow: do all processing on GPU / compute positional encoding once and save!
        return x + encoding.cuda()


class PatchEmbedding(nn.Module):
    """
    Receives input image (N, C, W, H) (assuming W=H) and turns it into a set of PxP patches
    Each patch is mapped linearly to obtain a vector of the flattened patch

    Implements equation (1) in the paper
    """

    def __init__(self, input_dim: list, num_patches: int, embedding_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_patches = num_patches
        self.patch_size = int(input_dim[2] / (num_patches ** 0.5))
        self.embed_dim = embedding_dim
        # by choosing kernel size and stride to be same as the patch size we divide the image implicitly into the
        # desired patches
        self.linear_projection = nn.Conv2d(input_dim[1], embedding_dim, kernel_size=self.patch_size,
                                           stride=self.patch_size)

    def forward(self, x: torch.Tensor):
        # apply projection convolution and concatenate patches -> total of num_patches
        x = self.linear_projection(x)
        x = torch.reshape(x, [self.input_dim[0], self.num_patches, self.embed_dim])
        return x


class RowEmbedding(nn.Module):
    # Todo - perhaps dont use single rows but 5 pixel wide rows

    def __init__(self, input_dim: list, embedding_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        # linear is always applied to last dimension (BxCxWxH)
        # for now single projection is shared for each row!
        self.proj = nn.Linear(input_dim[-1], embedding_dim, bias=True)
        # Todo - missing positional encdoing

    def forward(self, t: torch.Tensor):
        return torch.squeeze(self.proj(t))


class SelfAttention(nn.Module):
    """
    Implements Multi-Head Self-Attention module as described in the Attention is all you need paper.
    """

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        """
        :param embedding_dim: dimension of the word embeddings, must be dividable by num_heads
        :param num_heads: number of attention heads
        """
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dim must be dividable by number of attention heads!"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads

        # mapping matrices W_Q, W_K, W_V for queries, keys, values
        # doing this in a single linear mapping for qkv is more efficient, but this notebook is mainly for readability
        self.query_mapping = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.key_mapping = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.value_mapping = nn.Linear(embedding_dim, embedding_dim, bias=True)
        # scaling factor for more stable gradients when using the softmax function -> refer to paper for details
        self.scaling_factor = math.sqrt(embedding_dim)

        self.multi_head_mapping = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, mask: torch.Tensor = None,
                masked_val: float = 1e15) -> torch.Tensor:
        batch_size, seq_len, _ = x1.shape
        q = self.query_mapping(x1)
        k = self.key_mapping(x2)
        v = self.value_mapping(x3)
        # reshape to heads format and [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = torch.reshape(q, shape=(batch_size, seq_len, self.num_heads, self.head_dim)).permute([0, 2, 1, 3])
        k = torch.reshape(k, shape=(batch_size, seq_len, self.num_heads, self.head_dim)).permute([0, 2, 1, 3])
        v = torch.reshape(v, shape=(batch_size, seq_len, self.num_heads, self.head_dim)).permute([0, 2, 1, 3])

        # score representing how much attention is paid to each word
        logits = q @ k.transpose(2, 3) * self.scaling_factor
        if mask is not None:
            logits = logits.masked_fill(mask, masked_val)
        score = F.softmax(logits, dim=-1)
        # compute weighted output from values
        weighted_v = score @ v
        weighted_v = torch.reshape(weighted_v, shape=(batch_size, seq_len, self.embedding_dim))
        output = self.multi_head_mapping(weighted_v)
        return output


class TransformerEncoderBlock(nn.Module):
    """
    Implements a single Transformer Encoder Block consisting of a Multi-Head Self-Attention module, followed by a MLP-Net
    """

    def __init__(self, input_dim: int, num_heads: int, hidden_dim: int, dropout_rate: float = 0.1,
                 activation: nn.Module = nn.ReLU) -> None:
        """
        :param input_dim: input dimension, generally the word embedding dimension
        :param num_heads: number of heads for the multi-head self-attention module
        :param hidden_dim: hidden dimension of the 2-layer MLP (input-dim -> hidden_dim -> input_dim)
        :param dropout_rate: dropout rate used throughout MLP and Encoder
        :param activation: activation function in MLP, defaults to ReLU
        """
        super().__init__()
        assert activation in [nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SELU, nn.ELU, nn.CELU]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            activation(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.attention_module = SelfAttention(input_dim, num_heads)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # residual
        x = x + self.dropout(self.attention_module(x, x, x, mask))
        x = self.layer_norm1(x)
        # second residual
        x = x + self.dropout(self.mlp(x))
        x = self.layer_norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: list, num_layers: int, embedding_dim: int, num_heads: int,
                 mlp_expansion_factor: int = 4, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.row_embedding = RowEmbedding(input_dim, embedding_dim)
        self.pos_encoding = SinusoidalPositionalEncoding()
        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, num_heads, embedding_dim * mlp_expansion_factor, dropout_rate) for _
             in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.row_embedding(x)
        x = self.pos_encoding(x)
        for block in self.encoder_blocks:
            x = block(x)
        return x


class ClassificationHead(nn.Module):
    """
    Basic Idea - classify each row to a specific class
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()

        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, t: torch.Tensor):
        return F.softmax(self.cls(t), dim=-1)


class ClsTransformer(nn.Module):

    def __init__(self, num_classes: int, embed_dim: int, input_dim: list, num_heads: int, num_layers: int):
        super().__init__()

        self.encoder = TransformerEncoder(input_dim, num_layers, embed_dim, num_heads)
        self.cls_head = ClassificationHead(num_classes, embed_dim)

    def forward(self, t: torch.Tensor):
        return self.cls_head(self.encoder(t))


def __main__():
    from src.lib.utils.utils import count_parameters
    t = torch.randn([8, 1, 160, 40])
    model = ClsTransformer(num_classes=37, embed_dim=64, input_dim=[16, 1, 160, 40], num_heads=4, num_layers=6)
    print(f"Number of parameters = {count_parameters(model)}")


if __name__ == "__main__":
    __main__()
