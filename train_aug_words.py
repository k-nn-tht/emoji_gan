"""
Training of DCGAN network on MNIST dataset with Discriminator
and Generator imported from models.py

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_words import Discriminator, Generator, initialize_weights
import torchvision.utils as utils
from torch.utils.data import Dataset
import random

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 32
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 300
FEATURES_DISC = 64
FEATURES_GEN = 64
NUM_CLASSES = 100
EMBED_SIZE = 100

transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=25),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.02),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # More spatial jitter
    transforms.ToTensor(),
    transforms.Normalize([0.5] * CHANNELS_IMG, [0.5] * CHANNELS_IMG),
])

# comment mnist above and uncomment below if train on CelebA
class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, repeat_factor):
        self.base_dataset = base_dataset
        self.repeat_factor = repeat_factor

    def __len__(self):
        return len(self.base_dataset) * self.repeat_factor

    def __getitem__(self, idx):
        return self.base_dataset[idx % len(self.base_dataset)]

# Load the base dataset
base_dataset = datasets.ImageFolder(root="dataset", transform=transforms)
dataset = RepeatDataset(base_dataset, repeat_factor=10)
# dataset = datasets.ImageFolder(root="dataset", transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, EMBED_SIZE).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, EMBED_SIZE).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
# opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE * 0.5, betas=(0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(dataloader):
        real = real.to(device)
        labels = labels.to(device)
        real += 0.05 * torch.randn_like(real)  # Add slight noise for augmentation

        ### Train Discriminator
        # noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
        # noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        noise = torch.randn(labels.size(0), NOISE_DIM).to(device)

        fake = gen(noise, labels)

        disc_real = disc(real, labels).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.full_like(disc_real, 0.9))

        disc_fake = disc(fake.detach(), labels).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator
        output = disc(fake, labels).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        ### Logging and Saving
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} "
                f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
            )

            if epoch % 10 == 0:
                with torch.no_grad():
                    sample_labels = torch.randint(0, NUM_CLASSES, (32,)).to(device)
                    sample_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
                    fake = gen(sample_noise, sample_labels)
                    utils.save_image(
                        fake,
                        f"generated_emojis/conditional/fake_epoch{epoch}.png",
                        normalize=True,
                        nrow=8
                    )

            step += 1
