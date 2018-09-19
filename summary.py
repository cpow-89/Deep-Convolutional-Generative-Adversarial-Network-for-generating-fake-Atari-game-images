from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import numpy as np
import os


class Summary:
    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(os.path.join(".", *self.config["monitor_dir"], self.config["project_name"]))
        self.statistics = {"generator_loss": [], "discriminator_loss": [], "batch_number": 0}

    def update_statistics(self, generator_loss, discriminator_loss, batch_number):
        self.statistics["generator_loss"].append(generator_loss)
        self.statistics["discriminator_loss"].append(discriminator_loss)
        self.statistics["batch_number"] = batch_number

    def print_statistics(self):
        print("Number of processed Batches: ", self.statistics["batch_number"])
        print("Generator Loss - Mean: ", np.mean(self.statistics["generator_loss"]))
        print("Discriminator Loss - Mean: ", np.mean(self.statistics["discriminator_loss"]))

    def write_statistics(self):
        self.writer.add_scalar("gen_loss",
                               np.mean(self.statistics["generator_loss"]),
                               self.statistics["batch_number"])
        self.writer.add_scalar("dis_loss",
                               np.mean(self.statistics["discriminator_loss"]),
                               self.statistics["batch_number"])
        self.statistics["generator_loss"] = []
        self.statistics["discriminator_loss"] = []

    def write_images(self, generator_output, train_data):
        self.writer.add_image("fake", vutils.make_grid(generator_output.data[:64]), self.statistics["batch_number"])
        self.writer.add_image("real", vutils.make_grid(train_data.data[:64]), self.statistics["batch_number"])
