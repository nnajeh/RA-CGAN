

### Encoder class similar to the discriminator
### used for image reconstruction, rapid mapping from image space to latent space
class Encoder(nn.Module):
    def __init__(self, base_channels=96, n_classes=2):
        super().__init__()

        self.shared_emb = nn.utils.spectral_norm(nn.Embedding(n_classes,16 * base_channels*4*4))

        self.d_blocks = nn.Sequential(
            nn.Conv2d(3, base_channels, 3,1,1),
            DResidualBlock(base_channels, 2 * base_channels, downsample=True, use_preactivation=True),
            DResidualBlock(2 * base_channels, 4 * base_channels, downsample=True, use_preactivation=True),
            DResidualBlock(4 * base_channels, 8 * base_channels, downsample=True, use_preactivation=True),
            DResidualBlock(8 * base_channels, 16 * base_channels, downsample=True, use_preactivation=True),
            DResidualBlock(16 * base_channels, 16 * base_channels, downsample=True, use_preactivation=True),
            )
        self.proj_o = (nn.Linear(16 * base_channels*4*4, z_dim))

    def forward(self, x, y=None):
        h = x.contiguous()
        h = h.view(-1, 3,128,128)

        #for module in self.d_blocks:
        h = self.d_blocks(h)
      #  h = torch.sum(h, dim=[2, 3])
        h = h.view(h.shape[0], -1)

        uncond_out = self.proj_o(h)
        if y is None:
            return uncond_out
          
        cond_out = torch.sum(self.shared_emb(y) * h, dim=1, keepdim=True)
        out = uncond_out + cond_out
        out = torch.tanh(out)
        return out
