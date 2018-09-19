import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=config["image_shape"][0],
                      out_channels=config["discriminator_filters"],
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=config["discriminator_filters"],
                      out_channels=config["discriminator_filters"] * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(config["discriminator_filters"] * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=config["discriminator_filters"] * 2,
                      out_channels=config["discriminator_filters"] * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(config["discriminator_filters"] * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=config["discriminator_filters"] * 4,
                      out_channels=config["discriminator_filters"] * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(config["discriminator_filters"] * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=config["discriminator_filters"] * 8,
                      out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.network(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=config["latent_vector_size"],
                               out_channels=config["generator_filters"] * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(config["generator_filters"] * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=config["generator_filters"] * 8,
                               out_channels=config["generator_filters"] * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(config["generator_filters"] * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=config["generator_filters"] * 4,
                               out_channels=config["generator_filters"] * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(config["generator_filters"] * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=config["generator_filters"] * 2,
                               out_channels=config["generator_filters"],
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=config["generator_filters"],
                               out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)
