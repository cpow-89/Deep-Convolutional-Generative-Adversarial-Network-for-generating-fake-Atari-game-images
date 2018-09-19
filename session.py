from summary import Summary


def train(gan, generator_noise_input, batch_handler, config):
    summary_writer = Summary(config)
    for batch_number, batch in enumerate(batch_handler.get_next_batch(config["batch_size"])):
        generator_output = gan.generate(generator_noise_input)
        generator_loss, discriminator_loss = gan.train(generator_output, batch)
        summary_writer.update_statistics(generator_loss, discriminator_loss, batch_number)

        if batch_number % config["report_batch_interval"] == 0:
            summary_writer.print_statistics()
            summary_writer.write_statistics()

        if batch_number % config["save_batch_interval"] == 0:
            gan.save()
            summary_writer.write_images(generator_output, batch)
