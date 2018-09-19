import session
import os
import json
import torch
import argparse
from batch_handler import BatchHandler
from model import AtariGan


def main():

    parser = argparse.ArgumentParser(description="Run Extended Q-Learning with given config")
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        metavar="",
                        required=True,
                        help="Config file name - file must be available as .json in ./configs")

    args = parser.parse_args()

    with open(os.path.join(".", "configs", args.config), "r") as read_file:
        config = json.load(read_file)

    batch_handler = BatchHandler(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator_noise_input = torch.FloatTensor(16, 100, 1, 1).normal_(0, 1).to(device)
    atari_gan = AtariGan(config)
    session.train(atari_gan, generator_noise_input, batch_handler, config)


if __name__ == "__main__":
    main()
