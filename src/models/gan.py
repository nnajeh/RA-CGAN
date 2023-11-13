
### Generator network
class Generator(nn.Module):

    def __init__(self, base_channels=96, bottom_width=4, z_dim=120, shared_dim=128, n_classes=2):
        super().__init__()

        n_chunks = 6   # 5 (generator blocks) + 1 (generator input)
        self.z_chunk_size = z_dim // n_chunks
        self.z_dim = z_dim
        self.shared_dim = shared_dim
        self.bottom_width = bottom_width

        # No spectral normalization on embeddings, which authors observe to cripple the generator
        self.shared_emb = nn.Embedding(n_classes, shared_dim)

        self.proj_z = nn.Linear(self.z_chunk_size, 16 * base_channels * bottom_width ** 2)

        # Can't use one big nn.Sequential since we are adding class+noise at each block
        self.g_blocks = nn.ModuleList([
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 16 * base_channels, 16 * base_channels),
            ]),
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 16 * base_channels, 8 * base_channels),
            ]),

            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 8 * base_channels, 4 * base_channels),
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
            nn.utils.spectral_norm(nn.Conv2d(base_channels, 3, kernel_size=1, padding=0)),
            nn.Tanh(),
        )

    def forward(self, z, y):
        # Chunk z and concatenate to shared class embeddings
        zs = torch.split(z, self.z_chunk_size, dim=1)
        z = zs[0]
        ys = [torch.cat([y, z], dim=1) for z in zs[1:]]

        # Project noise and reshape to feed through generator blocks
        h = self.proj_z(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Feed through generator blocks
        for idx, g_block in enumerate(self.g_blocks):
            h = g_block[0](h, ys[idx])
           # h = g_block[1](h)

        # Project to 3 RGB channels with tanh to map values to [-1, 1]
        # output an image
        h = self.proj_o(h)

        return h


#######################################################################################



### Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, base_channels=96, n_classes=2):
        super().__init__()

        self.shared_emb = nn.utils.spectral_norm(nn.Embedding(n_classes,16 * base_channels))

        self.d_blocks = nn.Sequential(
            DResidualBlock(3, base_channels, downsample=True, use_preactivation=False),
            DResidualBlock(base_channels, 2 * base_channels, downsample=True, use_preactivation=True),
            DResidualBlock(2 * base_channels, 4 * base_channels, downsample=True, use_preactivation=True),
            DResidualBlock(4 * base_channels, 8 * base_channels, downsample=True, use_preactivation=True),
            DResidualBlock(8 * base_channels, 16 * base_channels, downsample=True, use_preactivation=True),
            DResidualBlock(16 * base_channels, 16 * base_channels, downsample=False, use_preactivation=True),

            nn.ReLU(inplace=True)
        )
      
        self.proj_o = nn.utils.spectral_norm(nn.Linear(16 * base_channels, 1))

    def extract_features(self, x):
        h = x.contiguous()
        h = h.view(-1, 3,128,128)

        #for module in self.d_blocks:
        h = self.d_blocks(h)
        h = torch.sum(h, dim=[2, 3])
        return h

  
    def forward(self, x, y=None):
        h = self.extract_features(x)
        uncond_out = self.proj_o(h)
        
      if y is None:
            return uncond_out
          
        cond_out = torch.sum(self.shared_emb(y) * h, dim=1, keepdim=True)
        out = uncond_out + cond_out
      
        #output a score 
        out = out.view(-1)

        return out
