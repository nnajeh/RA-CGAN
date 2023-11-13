class Generator(nn.Module):

    def __init__(self, base_channels = 64, bottom_width=4, z_dim=128, shared_dim=128, n_classes=10):
        super().__init__()

        n_chunks = 4    # 5 (generator blocks) + 1 (generator input)
        self.z_chunk_size = z_dim // n_chunks
        self.z_dim = z_dim
        self.shared_dim = shared_dim
        self.bottom_width = bottom_width

        # No spectral normalization on embeddings, which authors observe to cripple the generator
        self.shared_emb = nn.Embedding(n_classes, shared_dim)

        self.proj_z = nn.Linear(self.z_chunk_size, 4 * base_channels * bottom_width ** 2)

        # Can't use one big nn.Sequential since we are adding class+noise at each block
        self.g_blocks = nn.ModuleList([

            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 4 * base_channels, 4 * base_channels),
            ]),
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 4 * base_channels, 2 * base_channels),
            ]),
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 2 * base_channels, base_channels),
            ]),
        ])
        self.proj_o = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)),
            nn.Tanh(),
        )

    def forward(self, z, y):
        # Chunk z and concatenate to shared class embeddings
        zs = torch.split(z, self.z_chunk_size, dim=1)
        z = zs[0]
        ys = [torch.cat([(y), z], dim=1) for z in zs[1:]]

        # Project noise and reshape to feed through generator blocks
        h = self.proj_z(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Feed through generator blocks
        for idx, g_block in enumerate(self.g_blocks):
            h = g_block[0](h, ys[idx])
            #h = g_block[1](h)

        # Project to 3 RGB channels with tanh to map values to [-1, 1]
        h = self.proj_o(h)

        return h


class Discriminator(nn.Module):

    def __init__(self, base_channels=64, n_classes=10):
        super().__init__()

        self.base_channels= base_channels

        # For adding class-conditional evidence
        self.shared_emb = nn.utils.spectral_norm(nn.Embedding(n_classes, 4 * base_channels*4*4))

        self.d_blocks = nn.Sequential(
            DResidualBlock(1, base_channels, downsample=True, use_preactivation=False),

            DResidualBlock(base_channels, 2 * base_channels, downsample=True, use_preactivation=True),

            DResidualBlock(2 * base_channels, 4 * base_channels, downsample=True, use_preactivation=True),

            DResidualBlock(4 * base_channels, 4 * base_channels, downsample=False, use_preactivation=True),

            nn.ReLU(inplace=True),
        )
        self.proj_o = nn.utils.spectral_norm(nn.Linear(4 * base_channels*4*4, 1))


    #def forward(self, x, y=None):
    def extract_features(self, x):
        h = x.contiguous()
        h = h.view(-1, 1, 32, 32)

        h = self.d_blocks(x)
        h = h.view(-1, 4*4*4*64)

        return h


    def forward(self, x, y=None):
        h = self.extract_features(x)

        # Class-unconditional output
        uncond_out = self.proj_o(h)
        if y is None:
            return uncond_out

        # Class-conditional output
        cond_out = torch.sum(self.shared_emb(y) * h, dim=1, keepdim=True)
        out = uncond_out + cond_out
        out = out.view(-1)
        return out
