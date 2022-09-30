"""
Vision Transformer implementation based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Receives input image (N, C, W, H) (assuming W=H) and turns it into a set of PxP patches
    Each patch is mapped linearly to obtain a vector of the flattened patch

    Implements equation (1) in the paper
    """

    def __init__(self, input_dim, num_patches, embedding_dim):
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


class Attention(nn.Module):
    """
    Applies attention mechanism to input vector (N, attention_heads, embedding_dim)
    Basically computed attention(q, k, v) = softmax(qk^T / \sqrt{d_k})v
    where d_k is a scaling factor (refer to Attention is all you need)
    todo: any dropout layer for more stable training are missing (see paper)
    """

    def __init__(self, attention_heads, embedding_dim):
        # Todo: does not proper multi-head attention!
        super().__init__()

        self.attention_heads = attention_heads
        self.embedding_dim = embedding_dim

        self.query_mapping = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.key_mapping = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.value_mapping = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.scaling_factor = 1 / (attention_heads ** -.5)

    def forward(self, x: torch.Tensor):
        # compute query, key, values
        queries = self.query_mapping(x)
        keys = self.key_mapping(x)
        values = self.value_mapping(x)

        attention = F.softmax(queries @ keys.transpose(1, 2) * self.scaling_factor, dim=-1) @ values

        return attention


class MLP(nn.Module):
    """
    Todo: missing dropout layer (and possibly norm) for more stable training!
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        return F.gelu(self.lin2(F.gelu(self.lin1(x))))


class TransformerEncoder(nn.Module):
    """
    Implements quations (2) and (3) in the paper
    """

    def __init__(self, attention_heads, embedding_dim):
        super().__init__()

        self.mlp_ratio = 4.0

        self.norm1 = nn.LayerNorm([attention_heads, embedding_dim])
        self.attention = Attention(attention_heads, embedding_dim)
        self.norm2 = nn.LayerNorm([attention_heads, embedding_dim])
        self.mlp = MLP(embedding_dim, int(embedding_dim * self.mlp_ratio), embedding_dim)

    def forward(self, x: torch.Tensor):
        x = self.attention(self.norm1(x)) + x  # equation (2)
        x = self.mlp(self.norm2(x)) + x  # equation (3)

        return x


class ClassificationHead(nn.Module):

    def __init__(self, input_dim, num_classes, hidden_dim):
        super().__init__()
        # todo: dropout to prevent overfitting?
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        return F.softmax(self.linear2(F.relu(self.linear1(x))), dim=1)


class VisionTransformer(nn.Module):
    """
    Implements the vision transformer

    todo - class / pos tag missing
    todo - classficiation head / different heads missing
    """

    def __init__(self, input_dim, embedding_dim, num_patches, num_classes=10, L=12):
        """

        :param h: number of attention modules
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches

        self.patch_gen = PatchEmbedding(self.input_dim, self.num_patches, self.embedding_dim)
        # todo: how is this learned?
        self.class_embedding = torch.randn(self.input_dim[0], 1, self.embedding_dim)

        self.stacked_encoders = []
        for i in range(L):
            self.stacked_encoders.append(
                TransformerEncoder(self.num_patches + 1, embedding_dim)
            )
        self.stacked_encoders = nn.Sequential(*self.stacked_encoders)

        self.final_norm = nn.LayerNorm([self.num_patches + 1, self.embedding_dim])
        self.classification_head = ClassificationHead(input_dim=self.embedding_dim, num_classes=num_classes,
                                                      hidden_dim=self.embedding_dim // 2)

    def forward(self, x: torch.Tensor):
        # divide image in patches and compute linear projection to flattened patch vectors
        x = self.patch_gen(x)
        x = torch.cat([x, self.class_embedding], dim=1)
        x = self.stacked_encoders(x)
        x = self.final_norm(x)

        # todo: what is the semantic intention of only considering the class token for the classification head?
        y = x[:, -1, :]  # equation (4)

        return self.classification_head(y)


def __main__():
    dim = [16, 3, 256, 256]
    num_patches = 16
    embedding_dim = 768
    t = torch.randn(dim)

    pembed = PatchEmbedding(dim, num_patches, embedding_dim)
    attention_module = Attention(attention_heads=num_patches, embedding_dim=embedding_dim)
    transformer_enc = TransformerEncoder(attention_heads=num_patches, embedding_dim=embedding_dim)
    vision_transformer = VisionTransformer(dim, embedding_dim, num_patches)

    out1 = pembed(t)
    print(out1.shape)

    out2 = transformer_enc(out1)
    print(out2.shape)

    out_attention = attention_module(out1)
    print(out_attention.shape)

    pdf = vision_transformer(t)
    print(pdf.shape, pdf.sum(dim=1))


if __name__ == "__main__":
    __main__()
