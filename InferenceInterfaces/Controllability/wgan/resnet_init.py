from InferenceInterfaces.Controllability.wgan.init_weights import weights_init_D
from InferenceInterfaces.Controllability.wgan.init_weights import weights_init_G
from InferenceInterfaces.Controllability.wgan.resnet_1 import ResNet_D
from InferenceInterfaces.Controllability.wgan.resnet_1 import ResNet_G


def init_resnet(parameters):
    critic = ResNet_D(parameters['data_dim'][-1], parameters['size'], nfilter=parameters['nfilter'], nfilter_max=parameters['nfilter_max'])
    generator = ResNet_G(parameters['data_dim'][-1], parameters['z_dim'], parameters['size'], nfilter=parameters['nfilter'],
                         nfilter_max=parameters['nfilter_max'])

    generator.apply(weights_init_G)
    critic.apply(weights_init_D)

    return generator, critic
