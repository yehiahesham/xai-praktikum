"""
Implementation of Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
by
Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros paper.
"""

# Imports
from dataset import Dataset_HZ

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

def train(disc_H, disc_Z, gen_Z, gen_H, loader, opt_discr, opt_gen, L1Loss, MSELoss, d_scaler, g_scaler):
    """
    Composite training function for generators and discriminators
    :return:
    """
    # For progress bar
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators for H and Z
        with torch.cuda.amp.autocast(): # for Float64 training
            # Generate a fake horse
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = MSELoss(D_H_real, torch.ones_like(D_H_real)) # Real must be 1
            D_H_fake_loss = MSELoss(D_H_fake, torch.zeros_like(D_H_fake)) # Fake must be 0.
            D_H_loss = D_H_real_loss + D_H_fake_loss

            # Generate a fake zebra
            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_H(fake_zebra.detach())
            D_Z_real_loss = MSELoss(D_Z_real, torch.ones_like(D_Z_real))  # Real must be 1
            D_Z_fake_loss = MSELoss(D_Z_fake, torch.zeros_like(D_Z_fake))  # Fake must be 0.
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # We put losses together
            D_loss = (D_H_loss + D_Z_loss) / 2 # Dividing mentioned in the paper

        opt_discr.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_discr)
        d_scaler.update()

        # Train Generators for H and Z
        with torch.cuda.amp.autocast():
            # Adversarial Loss for generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)

            # To fool the discriminator, we use 1's for fake images.
            loss_G_H = MSELoss(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = MSELoss(D_Z_fake, torch.ones_like(D_Z_fake))


            # Cycle-Consistency Loss
            # Take generated fake horse and convert it to a zebra
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = L1Loss(zebra, cycle_zebra)
            cycle_horse_loss = L1Loss(horse, cycle_horse)

            # Identity Loss, we're not using this. See config file, LAMBDA_IDENTITY is 0.
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            # identity_zebra_loss = L1Loss(zebra, identity_zebra)
            # identity_horse_loss = L1Loss(horse, identity_horse)

            # Aggregate all losses
            G_loss = (loss_G_Z
                      + loss_G_H
                      + cycle_zebra_loss * config.LAMBDA_CYCLE
                      + cycle_horse_loss * config.LAMBDA_CYCLE
                      # + identity_zebra_loss * config.LAMBDA_IDENTITY
                      # + identity_horse_loss * config.LAMBDA_IDENTITY
                      )
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            # Save image
            if idx % 200 == 0:
                save_image(fake_horse * 0.5 * 0.5, f"saved_images/horse_{idx}.png")
                save_image(fake_zebra * 0.5 * 0.5, f"saved_images/zebra_{idx}.png")

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