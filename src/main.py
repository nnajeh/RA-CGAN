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


root = data_pat  # Replace with the actual path to your dataset
normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])

transform = transforms.Compose([transforms.Resize((128,128)),
                                               transforms.RandomResizedCrop((128), scale=(0.5, 1.0)),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize])
target_transform = None  # Example target transformation, you can customize it

dataset = ImageFolder(root, transform=transform, target_transform=target_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)



