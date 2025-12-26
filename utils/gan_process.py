import os
import shutil
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import matplotlib.pyplot as plt

def show_tensor_images(image_tensor, num_images=32, size=(1, 64, 64)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = torchvision.utils.make_grid(image_unflat[:num_images], nrow=4)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def training_loops(
        gen_model,
        disc_model,
        noise_dim,
        data_loader,
        criterion,
        gen_optimizer,
        disc_optimizer,
        device,
        num_epochs,
        save_path,
    ):
    writer = SummaryWriter(log_dir=os.path.join(save_path, "logs"))
    best_gen_loss = float('inf')

    gen_model.train()
    disc_model.train()

    for epoch in range(num_epochs):
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        for batch_idx, (real, _) in enumerate(tqdm(data_loader)):
            real = real.to(device)

            noise = torch.randn(real.size(0), noise_dim, 1, 1, device=device)
            fake = gen_model(noise)

            # Train Discriminator
            disc_real = disc_model(real).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc_model(fake.detach()).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            disc_optimizer.zero_grad()
            loss_disc.backward()
            disc_optimizer.step()

            # Train Generator
            disc_fake_for_gen = disc_model(fake).view(-1)
            loss_gen = criterion(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))

            gen_optimizer.zero_grad()
            loss_gen.backward()
            gen_optimizer.step()
            epoch_gen_loss += loss_gen.item()
            epoch_disc_loss += loss_disc.item()

        avg_gen_loss = epoch_gen_loss / len(data_loader)
        avg_disc_loss = epoch_disc_loss / len(data_loader)
        
        writer.add_scalar("Loss/Discriminator", avg_disc_loss, epoch)
        writer.add_scalar("Loss/Generator", avg_gen_loss, epoch)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] Loss D: {avg_disc_loss:.4f}, loss G: {avg_gen_loss:.4f}"
        )

        with torch.no_grad():
            fixed_noise = torch.randn(32, noise_dim, 1, 1).to(device)
            fake = gen_model(fixed_noise)
            img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True, value_range=(-1, 1))
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True, value_range=(-1, 1))
            writer.add_image("Real Images", img_grid_real, epoch)
            writer.add_image("Fake Images", img_grid_fake, epoch)
            torchvision.utils.save_image(fake[:32], os.path.join(save_path, f"gen_epoch_{epoch+1}.png"), normalize=True, value_range=(-1, 1))

        if avg_gen_loss < best_gen_loss:
            best_gen_loss = avg_gen_loss
            torch.save(gen_model.state_dict(), os.path.join(save_path, "best_gen.pth"))
            torch.save(disc_model.state_dict(), os.path.join(save_path, "best_disc.pth"))
            print(f"Saved Best Model (Loss: {best_gen_loss:.4f})")


            
