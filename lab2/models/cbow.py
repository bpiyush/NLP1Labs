"""Bag-of-words model for sentence classification."""
import numpy as np
import torch
import torch.nn as nn


class CBOW(nn.Module):
    """A continuous bag-of-words model"""

    def __init__(self, vocab_size, embedding_dim, vocab, num_classes=5):
        super(CBOW, self).__init__()
        self.vocab = vocab

        # this is a trainable look-up table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        
        # projects sum of embeddings to a vector of size num_classes
        self.projector = nn.Linear(embedding_dim, num_classes, bias=True)

    def forward(self, inputs):
        # this is the forward pass of the neural network
        # it applies a function to the input and returns the output

        # this looks up the embeddings for each word ID in inputs
        # the result is a sequence of word embeddings
        embeds = self.embed(inputs)
        embeds = embeds.sum(1)

        # obtain the output (logits) of the neural network
        logits = self.projector(embeds)

        return logits


class DeepCBOW(nn.Module):
    """A deep continuous bag-of-words model"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
        super(DeepCBOW, self).__init__()
        self.vocab = vocab

        # network definition
        self.net = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Linear(embedding_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, inputs):
        # this is the forward pass of the neural network
        # it applies a function to the input and returns the output
        logits = self.net(inputs).sum(1)
        
        return logits


class PTDeepCBOW(DeepCBOW):
    """Pretrained word embeddings to initialize the embedding layer in DeepCBOW"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
        super(PTDeepCBOW, self).__init__(
            vocab_size, embedding_dim, hidden_dim, output_dim, vocab)
    
    def init_embedding_weights(self, pretrained_embeddings: np.ndarray, finetune_emb=False):
        """Initialize the embedding layer with the pretrained embeddings"""
        with torch.no_grad():
            self.net[0].weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if not finetune_emb:
                self.net[0].weight.requires_grad = False
