from models.gan import Generator, Discriminator



generator = Generator(base_channels=base_channels, bottom_width=4, z_dim=z_dim, shared_dim=shared_dim, n_classes=n_classes).to(device)
discriminator = Discriminator(base_channels=base_channels, n_classes=n_classes).to(device)

# Initialize weights orthogonally
for module in generator.modules():
    if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
        nn.init.orthogonal_(module.weight)
for module in discriminator.modules():
    if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
        nn.init.orthogonal_(module.weight)

# Initialize optimizers: pulmonary datasets
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.999), eps=1e-6)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.0, 0.999), eps=1e-6)
