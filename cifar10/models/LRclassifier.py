import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, use_softmax=False):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.use_softmax = use_softmax

    def forward(self, x):
        outputs = self.linear(x)
        if self.use_softmax:
            output = output.softmax(dim=-1)
        return outputs
