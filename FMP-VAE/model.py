from torch import nn

class Variable_AutoEncoder(nn.Module):
    def __init__(self):
        super(Variable_AutoEncoder, self).__init__()
        # 定义编码器
        self.Encoder = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # 定义解码器
        self.Decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.Sigmoid()
        )
        self.fc_m = nn.Linear(64, 20)
        self.fc_sigma = nn.Linear(64, 20)
    def forward(self, input):
        code = input.view(input.size(0), -1)
        code = self.Encoder(code)
        # m, sigma = code.chunk(2, dim=1)
        m = self.fc_m(code)
        sigma = self.fc_sigma(code)
        e = torch.randn_like(sigma)
        c = torch.exp(sigma) * e + m
        # c = sigma * e + m
        output = self.Decoder(c)
        output = output.view(input.size(0), 1, 28, 28)
        return output, m, sigma