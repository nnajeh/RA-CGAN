
class Encoder(nn.Module):

    def __init__(self, base_channels=64, n_classes=10):
        super().__init__()

        self.base_channels= base_channels

        # For adding class-conditional evidence
        self.shared_emb = nn.utils.spectral_norm(nn.Embedding(n_classes, 4 * base_channels*4*4))

        self.d_blocks = nn.Sequential(
            nn.Conv2d(1, base_channels, 3,1,1),

            DResidualBlock(base_channels, 2 * base_channels, downsample=True, use_preactivation=True),

            DResidualBlock(2 * base_channels, 4 * base_channels, downsample=True, use_preactivation=True),

            DResidualBlock(4 * base_channels, 4 * base_channels, downsample=True, use_preactivation=True),

        )
        self.proj_o = nn.utils.spectral_norm(nn.Linear(4 * base_channels*4*4, 128))


    def forward(self, x, y=None):
        h = x.contiguous()
        h = h.view(-1, 1, 32, 32)

        h = self.d_blocks(x)
        h = h.view(-1, 4*4*4*64)

        # Class-unconditional output
        uncond_out = self.proj_o(h)
        if y is None:
            return uncond_out

        # Class-conditional output
        cond_out = torch.sum(self.shared_emb(y) * h, dim=1, keepdim=True)
        out = uncond_out + cond_out
        return torch.tanh(out)
