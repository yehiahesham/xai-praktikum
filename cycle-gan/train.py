# Imports
from dataset import Dataset_HZ
import sys
from utils.utils import save_checkpoint, load_checkpoint
from tqdm import tqdm

from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import config
from discriminator import Discriminator
from generator import Generator

def train():
    pass

def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE) # horses, real or fake
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE) # zebras, real or fake
    gen_Z = Generator(image_channels=3, num_residuals=9).to(config.DEVICE) # generates fake zebra imgs
    gen_H = Generator(image_channels=3, num_residuals=9).to(config.DEVICE) # generates fake horse imgs

    # Optimizer parameters for both 2 generators and 2 discriminators. We concatenated for two parameter lists.
    opt_discr = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999)
    )

    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    # Loss functions for GANS and cycle-consiştency, MSE and L1 respectively
    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_discr, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_discr, config.LEARNING_RATE
        )

        # Loading dataset
        dataset = Dataset_HZ(
            root_z=config.TRAIN_DIR+"/zebra", root_h=config.TRAIN_DIR+"/horse", transform=config.transforms
        )

        # DataLoaders
        loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )

        # These scalars are for Float64 training in PyTorch, we could remove these but not sure what would happen
        g_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()

        # Training Loop
        for epoch in range(config.NUM_EPOCHS):
            train(disc_H, disc_Z, gen_Z, gen_H, loader, opt_discr, opt_gen, L1_loss, MSE_loss, d_scaler, g_scaler)

            if config.SAVE_MODEL:
                save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
                save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
                save_checkpoint(disc_H, opt_discr, filename=config.CHECKPOINT_CRITIC_H)
                save_checkpoint(disc_Z, opt_discr, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    main()