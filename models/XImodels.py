#=============================
# JointFontGAN
# Modified from https://github.com/azadis/MC-GAN
# By Yankun Xi
#=============================

def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cGAN':
        from .XIcGAN_model import cGANModel
        model = cGANModel()
    elif opt.model == 'StackGAN_cGAN':
        from .XIStackGAN_model import StackGANModel
        model = StackGANModel()
    elif opt.model == 'StackGAN_EcGAN':
        from .XIStackGAN_EcGAN_model import StackGANModel
        model = StackGANModel()
    elif opt.model == 'cycleGAN':
        from .XIcycleGAN_model import cycleGANModel
        model = cycleGANModel()
    elif opt.model == 'cycleGANplus':
        from .XIcycleGANplus_model import cycleGANplusModel
        model = cycleGANplusModel()
    elif opt.model == 'LSTMcGAN':
        from .XILSTMcGAN_model import LSTMcGANModel
        model = LSTMcGANModel()
    # elif opt.model == 'LSTMcycleGAN':
    #     from .XILSTMcycleGAN_model import LSTMcycleGANModel
    #     model = LSTMcycleGANModel()
    elif opt.model == 'EcGAN':
        from .XIEcGAN_model import EcGANModel
        model = EcGANModel()
    elif opt.model == 'EskGAN':
        from .XIEskGAN_model import EskGANModel
        model = EskGANModel()
    elif opt.model == 'Esk1GAN':
        from .XIEsk1GAN_model import Esk1GANModel
        model = Esk1GANModel()
    elif opt.model == 'skGAN':
        from .XIskGAN_model import skGANModel
        model = skGANModel()
    elif opt.model == 'sk1GAN':
        from .XIsk1GAN_model import sk1GANModel
        model = sk1GANModel()
    elif opt.model == 'skLSTMcGAN':
        from .XIskLSTMcGAN_model import skLSTMcGANModel
        model = skLSTMcGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] is working" % (model.name()))
    return model
