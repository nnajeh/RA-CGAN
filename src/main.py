from models.gan import Generator, Discriminator

import os
import torch
import torch.nn as nn
from torch import autograd, optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms
from torchvision.utils import save_image

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision.datasets as dataset
import torchvision.datasets
from PIL import Image

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = True
from torch.utils.data import sampler





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


# Transform of medical data (for mnist change 128 to 32)
transform = transforms.Compose([transforms.Resize((128,128)),
                                               transforms.RandomResizedCrop((128), scale=(0.5, 1.0)),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize])
target_transform = None  # Example target transformation, you can customize it

dataset = ImageFolder(root, transform=transform, target_transform=target_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)



    generator.train()
    discriminator.train()

    d_losses = []
    g_losses = []
    for i, (real_images, labels) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        labels = labels.to(device)

        # Train discriminator
        d_optimizer.zero_grad()

        # Compute loss on real images
        real_output = discriminator(real_images, labels)
        real_loss = -torch.mean(real_output)

        # Compute loss on fake images
        z = torch.randn(batch_size, generator.z_dim, device=device)
        y_emb = generator.shared_emb(labels)
        fake_images = generator(z, y_emb)
        fake_output = discriminator(fake_images.detach(), labels)
        fake_loss = torch.mean(fake_output)

        # Compute conditional gradient penalty
        gradient_penalty = compute_gradient_penalty(D, real_images.data, fake_images.data)

        # Update discriminator
        train_d_loss = real_loss + fake_loss + gradient_penalty
        train_d_loss.backward()
        d_optimizer.step()

        # Train generator
        g_optimizer.zero_grad()

        # Compute loss on fake images
        z = torch.randn(batch_size, generator.z_dim, device=device)
        y_emb = generator.shared_emb(labels)
        fake_images = generator(z, y_emb)
        fake_output = discriminator(fake_images, labels)
        train_g_loss = -torch.mean(fake_output)

        # Update generator
        train_g_loss.backward()
        g_optimizer.step()

        d_losses.append(train_d_loss.item())
        g_losses.append(train_g_loss.item())



    d_train_loss = np.average(d_losses)
    g_train_loss = np.average(g_losses)
    train_total_g_losses.append(g_train_loss)
    train_total_d_losses.append(d_train_loss)
    epoch_len = len(str(n_epochs))
    print(f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
      f"[G_Train_Loss: {train_g_loss.item()}] "
      f"[D_Train_Loss: {train_d_loss.item()}]"
       )

    if batches_done % sample_interval ==0:
        save_image(fake_images.data[:25], f"./{batches_done:06}.png", nrow =5, normalize=True)

    batches_done += n_critic
    image_check(fake_images.cpu())

    torch.save(discriminator.state_dict(), f"./D.pth")
    torch.save(generator.state_dict(), f"./G.pth")


