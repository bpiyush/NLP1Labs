"""Defines LSTM model."""
import math
import numpy as np
import torch
import torch.nn as nn


class MyLSTMCell(nn.Module):
    """Our own LSTM cell"""

    def __init__(self, input_size, hidden_size, bias=True):
        """Creates the weights for this LSTM"""
        super(MyLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.chunks = 4
        
        # define layers to project input x to Wx + b
        self.input_projectors = nn.Linear(input_size * self.chunks, hidden_size * self.chunks)

        # define layers to project hidden h to Wh + b
        self.hidden_projectors = nn.Linear(hidden_size * self.chunks, hidden_size * self.chunks)
        
        # define activations
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        """This is PyTorch's default initialization method"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)  

    def forward(self, input_, hx, mask=None):
        """
        input is (batch, input_size)
        hx is ((batch, hidden_size), (batch, hidden_size))
        """
        prev_h, prev_c = hx

        # project input and prev state

        input_repeated = torch.repeat_interleave(input_, self.chunks, dim=1)
        [ix, fx, gx, ox] = torch.chunk(self.input_projectors(input_repeated), chunks=self.chunks, dim=1)        
        prev_h_repeated = torch.repeat_interleave(prev_h, self.chunks, dim=1)
        [ih, fh, gh, oh] = torch.chunk(self.hidden_projectors(prev_h_repeated), chunks=self.chunks, dim=1)

        # main LSTM computation

        i = self.sigmoid(ix + ih)
        f = self.sigmoid(fx + fh)
        g = self.tanh(gx + gh)
        o = self.sigmoid(ox + oh)

        c = f * prev_c + i * g
        h = o * self.tanh(c)

        return h, c

    def __repr__(self):
        return "{}({:d}, {:d})".format(self.__class__.__name__, self.input_size, self.hidden_size)


class LSTMClassifier(nn.Module):
    """Encodes sentence with an LSTM and projects final hidden state"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
        super(LSTMClassifier, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.rnn = MyLSTMCell(embedding_dim, hidden_dim)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),  # explained later
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):

        B = x.size(0)  # batch size (this is 1 for now, i.e. 1 single example)
        T = x.size(1)  # timesteps (the number of words in the sentence)

        input_ = self.embed(x)

        # here we create initial hidden states containing zeros
        # we use a trick here so that, if input is on the GPU, then so are hx and cx
        hx = input_.new_zeros(B, self.rnn.hidden_size)
        cx = input_.new_zeros(B, self.rnn.hidden_size)

        # process input sentences one word/timestep at a time
        # input is batch-major (i.e., batch size is the first dimension)
        # so the first word(s) is (are) input_[:, 0]
        outputs = []

        for i in range(T):
            hx, cx = self.rnn(input_[:, i], (hx, cx))
            outputs.append(hx)
    
        # if we have a single example, our final LSTM state is the last hx
        if B == 1:
            final = hx
        else:
            #
            # This part is explained in next section, ignore this else-block for now.
            #
            # We processed sentences with different lengths, so some of the sentences
            # had already finished and we have been adding padding inputs to hx.
            # We select the final state based on the length of each sentence.
      
            # two lines below not needed if using LSTM from pytorch
            outputs = torch.stack(outputs, dim=0)           # [T, B, D]
            outputs = outputs.transpose(0, 1).contiguous()  # [B, T, D]

            # to be super-sure we're not accidentally indexing the wrong state
            # we zero out positions that are invalid
            pad_positions = (x == 1).unsqueeze(-1)

            outputs = outputs.contiguous()      
            outputs = outputs.masked_fill_(pad_positions, 0.)

            mask = (x != 1)  # true for valid positions [B, T]
            lengths = mask.sum(dim=1)                 # [B, 1]

            indexes = (lengths - 1) + torch.arange(B, device=x.device, dtype=x.dtype) * T
            final = outputs.view(-1, self.hidden_dim)[indexes]  # [B, D]
    
        # we use the last hidden state to classify the sentence
        logits = self.output_layer(final)
        return logits

    def init_embedding_weights(self, pretrained_embeddings: np.ndarray, finetune_emb=False):
        """Initialize the embedding layer with the pretrained embeddings"""
        with torch.no_grad():
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if not finetune_emb:
                self.embed.weight.requires_grad = False


if __name__ == "__main__":
    # Test LSTM cell
    cell = MyLSTMCell(300, 128)

    x = torch.randn((1, 300))
    hx = (torch.randn(1, 128), torch.randn(1, 128))
    h, c = cell(x, hx)

    assert h.shape == torch.Size([1, 128])
    assert c.shape == torch.Size([1, 128])