import torch

from InferenceInterfaces.Controllability.wgan.resnet_init import init_resnet
from InferenceInterfaces.Controllability.wgan.wgan_qc import WassersteinGanQuadraticCost


def create_wgan(parameters, device, optimizer='adam'):
    if parameters['model'] == "resnet":
        generator, discriminator = init_resnet(parameters)
    else:
        raise NotImplementedError

    if optimizer == 'adam':
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=parameters['learning_rate'], betas=parameters['betas'])
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=parameters['learning_rate'], betas=parameters['betas'])
    elif optimizer == 'rmsprop':
        optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=parameters['learning_rate'])
        optimizer_d = torch.optim.RMSprop(generator.parameters(), lr=parameters['learning_rate'])

    criterion = torch.nn.MSELoss()

    gan = WassersteinGanQuadraticCost(generator,
                                      discriminator,
                                      optimizer_g,
                                      optimizer_d,
                                      criterion=criterion,
                                      data_dimensions=parameters['data_dim'],
                                      epochs=parameters['epochs'],
                                      batch_size=parameters['batch_size'],
                                      device=device,
                                      n_max_iterations=parameters['n_max_iterations'],
                                      gamma=parameters['gamma'])

    return gan
