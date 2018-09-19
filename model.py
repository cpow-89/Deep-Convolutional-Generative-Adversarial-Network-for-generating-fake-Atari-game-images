import torch
import torch.nn as nn
import torch.optim as optim
from network import Discriminator
from network import Generator
import helper
import os
import glob


class AtariGan:
    def __init__(self, config):
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(config).to(self.device)
        self.discriminator = Discriminator(config).to(self.device)

        self.loss_func = nn.BCELoss()
        self.generator_optimizer = optim.Adam(params=self.generator.parameters(),
                                              lr=config["generator_learning_rate"],
                                              betas =config["generator_betas"])
        self.discriminator_optimizer = optim.Adam(params=self.discriminator.parameters(),
                                                  lr=config["discriminator_learning_rate"],
                                                  betas=config["discriminator_betas"])

        self.true_labels = torch.ones(config["batch_size"], dtype=torch.float32, device=self.device)
        self.fake_labels = torch.zeros(config["batch_size"], dtype=torch.float32, device=self.device)

    def generate(self, noise_input):
        return self.generator(noise_input)

    def discriminate(self, _input):
        return self.discriminator(_input)

    def train(self, gen_output_v, batch_v):
        discriminator_loss = self._train_discriminator(batch_v, gen_output_v)
        generator_loss = self._train_generator(gen_output_v)
        return generator_loss.item(), discriminator_loss.item()

    def _train_discriminator(self, batch, generator_output):
        loss = self._calc_discriminator_loss(batch, generator_output)
        self._optimize(loss, self.discriminator_optimizer)
        return loss

    def _calc_discriminator_loss(self, batch, generator_output):
        output_true = self.discriminate(batch.to(self.device))
        output_fake = self.discriminate(generator_output.detach())
        return self.loss_func(output_true, self.true_labels) + self.loss_func(output_fake, self.fake_labels)

    def _train_generator(self, generator_output):
        loss = self._calc_generator_loss(generator_output)
        self._optimize(loss, self.generator_optimizer)
        return loss

    def _calc_generator_loss(self, generator_output):
        dis_output_v = self.discriminate(generator_output)
        return self.loss_func(dis_output_v, self.true_labels)

    def _optimize(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def save(self):
        """Save the network weights"""
        save_dir = os.path.join(".", *self.config["checkpoint_dir"], self.config["project_name"])
        helper.mkdir(save_dir)
        current_date_time = helper.get_current_date_time()
        current_date_time = current_date_time.replace(" ", "__").replace("/", "_").replace(":", "_")

        torch.save(self.generator.state_dict(), os.path.join(save_dir, "generator_ckpt_" + current_date_time))
        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, "discriminator_ckpt_" + current_date_time))

    def load(self):
        """Load latest available network weights"""
        load_path = os.path.join(".", *self.config["checkpoint_dir"], self.config["project_name"], "*")
        list_of_files = glob.glob(load_path)
        list_of_generator_weights = [w for w in list_of_files if "generator" in w]
        list_of_discriminator_weights = [w for w in list_of_files if "discriminator" in w]
        latest_generator_weights = max(list_of_generator_weights, key=os.path.getctime)
        self.generator.load_state_dict(torch.load(latest_generator_weights))
        latest_generator_weights = max(list_of_discriminator_weights, key=os.path.getctime)
        self.discriminator.load_state_dict(torch.load(latest_generator_weights))
