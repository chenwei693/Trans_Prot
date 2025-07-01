import torch
from torch import nn




class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dropout_rate=0.5):
        super(ConvResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout_rate)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out += residual
        return self.relu(out)









class Trans_Prot(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 128
        self.batch_size = 64
        self.emb_dim = 128
        # 24代表词汇表大小，emb_dim代表嵌入维度，padding_idx表示对变长序列进行0填充
        self.embedding_seq = nn.Embedding(24, self.emb_dim, padding_idx=0).cuda()
        # 定义了一个transform编码层,只在嵌入维度上计算，不改变输入张量形状
        self.encoder_layer_seq = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=8, batch_first=True).cuda()
        # 定义了一个Transformer编码器，并将它应用于序列数据。
        self.transformer_encoder_seq = nn.TransformerEncoder(self.encoder_layer_seq, num_layers=1).cuda()

        self.conv_tf = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),  # 输入通道，输出通道
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ReLU()
        ).cuda()
        self.conv_pt_1 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
        ).cuda()
        self.conv_pt_2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
        ).cuda()
        self.conv = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
        ).cuda()
        self.GRU = nn.GRU(1024, self.hidden_dim, num_layers=2, bidirectional=True, dropout=0.5).cuda()
        self.block1 = nn.Sequential(nn.Flatten(),
                                    nn.Linear(1600, 800),
                                    nn.BatchNorm1d(800),
                                    nn.Dropout(0.6),
                                    nn.LeakyReLU(),
                                    nn.Linear(800, 256),
                                    ).cuda()

        self.block2 = nn.Sequential(nn.Linear(256, 64),
                                    nn.BatchNorm1d(64),
                                    nn.Dropout(0.6),
                                    nn.ReLU(),
                                    nn.Linear(64, 2)).cuda()
        self.residual_conv_block = ConvResidualBlock(in_channels=128, mid_channels=64, out_channels=64,
                                                     dropout_rate=0.5).cuda()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, pos_embed, prot_tensor):
        x, pos_embed, prot_tensor = x.cuda(), pos_embed.cuda(), prot_tensor.cuda()
        output1 = self.embedding_seq(x) + pos_embed
        output1 = self.transformer_encoder_seq(output1)  # .permute(1, 0, 2)
        output1 = output1.permute(0, 2, 1)
        output1 = self.conv_tf(output1)
        output1 = output1.permute(0, 2, 1)

        output2, hn = self.GRU(prot_tensor)
        output2 = output2.permute(0, 2, 1)
        output2 = self.conv_pt_1(output2)
        output2 = self.conv_pt_2(output2)
        output2 = output2.permute(0, 2, 1)

        output = torch.cat([output1, output2], 2)

        output = output.permute(0, 2, 1)
        output = self.residual_conv_block(output)
        output = output.permute(0, 2, 1)

        output = self.block1(output)
        logits = self.block2(output)
        out = self.softmax(logits)
        return out, logits