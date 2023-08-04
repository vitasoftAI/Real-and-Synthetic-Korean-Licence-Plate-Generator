# Import libraries
import os, torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks

def pp(var_name, var, shape = False):
    
    """
    
    This function gets variable and returns print with (or without) its shape.
    
    Parameters:
    
        var_name     - name of the variable to be printed, str;
        var          - variable, various type;
        shape        - whether or not to print the shape of the variable, bool;
        
    """
    
    if shape: print(f"{var_name} -> {var.shape}\n")        
    else: print(f"{var_name} -> {var}\n")

class BaseModel(ABC):
    
    
    """
    
    This class is an abstract base class (ABC) for models.
    
    """

    def __init__(self, opt):
        
        
        """
        
        This function initializes the BaseModel class.
        
        Parameter:
        
            opt - stores all the experiment flags; needs to be a subclass of BaseOptions;

        """
        
        self.opt, self.gpu_ids, self.isTrain = opt, opt.gpu_ids, opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        
        # Input images might have different sizes, which hurts the performance of cudnn.benchmark.
        if opt.preprocess != 'scale_width':  torch.backends.cudnn.benchmark = True
            
        # Initialize lists
        self.loss_names, self.model_names, self.visual_names, self.optimizers, self.image_paths   = [], [], [], [], []
        
        # Metric for "plateau" learning rate policy
        self.metric = 0 

    @staticmethod
    def dict_grad_hook_factory(add_func = lambda x: x):
        
        saved_dict = {}

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook
        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        
        """
        
        This function adds new model-specific options, and rewrite default values for existing options.
        
        Parameters:
        
            parser          - original option parser, parser object;
            is_train        - whether training phase or test phase, bool.
        
        Output:
        
            parser          - the modified parser, parser object.
            
        """
        
        return parser

    @abstractmethod
    def set_input(self, input):
        
        """
        
        This function unpacks input data from the dataloader and perform necessary pre-processing steps.
        
        Parameter:
        
            input          - includes the data itself and its metadata information, dictionary.
        
        """
        
        pass

    @abstractmethod
    def forward(self):
        
        """
        
        This function runs forward pass; called by both functions <optimize_parameters> and <test>.
        
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        
        """
        
        This function calculates losses, gradients, and update network weights; called in every training iteration
        
        """
        pass

    def setup(self, opt):
        
        """
        
        This function loads and print networks; create schedulers.
        
        Parameter:
        
            opt - stores all the experiment flags; needs to be a subclass of BaseOptions.
        
        """
        
        # Train
        if self.isTrain: self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        
        # Inference
        if not self.isTrain or opt.continue_train:
            load_suffix = opt.epoch
#             load_suffix = "95"
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    # Parallel computation
    def parallelize(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                print("Parallel training is on!")
                print(f"Training on {self.opt.gpu_ids} gpus")
                setattr(self, 'net' + name, torch.nn.DataParallel(net, self.opt.gpu_ids))

    def data_dependent_initialize(self, data): pass

    def eval(self):
        
        """
        
        This function makes models eval mode during test time
        
        """
        
        # Go through model names
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        
        """
        
        This function implements forward function during inference.
        
        """
        
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        
        """
        
        This function calculates additional output images for visdom and HTML visualization
        
        """
        pass

    def get_image_paths(self):
        
        """ 
        
        This function returns image paths that are used to load current data
        
        """
        
        return self.image_paths

    def update_learning_rate(self):
        
        
        """
        
        This function updates learning rates for all the networks; called at the end of every epoch
        
        """
        
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        
        """
        
        This function returns visualization images. train.py will display these images with visdom, and save the images to a HTML
        
        """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        
        
        """
        
        This function returns traning losses / errors. train.py will print out these errors on console, and save them to a file
        
        """
        
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  
                
        return errors_ret

    def save_networks(self, epoch):
        
        """ 
        
        This function saves all the networks to the disk.
        
        Parameter:
        
            epoch - current epoch; used in the file name '%s_net_%s.pth' % (epoch, name), int.
        
        """
        
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        
        """
        
        This function fixes InstanceNorm checkpoints incompatibility (prior to 0.4)
        
        Parameters:
            
            state_dict    - a state dictionary of a model;
            module        - module;
            keys          - dictionary keys, list;
            i             - index, int.
            
        """
        
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else: self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        
        """
        
        This function loads all the networks from the disk.
        
        Parameters:
        
            epoch - current epoch, int.
            
        """
        
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if self.opt.isTrain and self.opt.pretrained_name is not None:
                    load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
                else: load_dir = self.save_dir

                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location = str(self.device))
                if hasattr(state_dict, '_metadata'): del state_dict._metadata

                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        
        """
        
        This function prints the total number of parameters in the network and (if verbose) network architecture
        
        Parameters:
        
            verbose - if verbose: print the network architecture, bool.
        
        """
        
        print('---------- Networks initialized -------------')
        
        for name in self.model_names: # G, F, D
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(f"{name} model has {num_params / 1e6} M number of parameters!")
        
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad = False):
        
        """
        
        This function sets requies_grad = False for all the networks to avoid unnecessary computations
        
        Parameters:
        
            nets - a list of networks, list;
            requires_grad - whether the networks require gradients or not, bool.
            
        """
        
        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode): return {}
