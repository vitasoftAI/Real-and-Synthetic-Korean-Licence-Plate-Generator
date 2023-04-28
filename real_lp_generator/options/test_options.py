from .base_options import BaseOptions

class TestOptions(BaseOptions):
    
    """
    
    This class includes test options. It also includes shared options defined in BaseOptions.
    
    Parameter:
    
        parser  - parsed arguments, parser object.
        
    Output:
    
        parser - updated parsed arguments, parser object.
    
    """

    def initialize(self, parser):
        
        # Initialize a parser
        parser = BaseOptions.initialize(self, parser)  
        
        # Add arguments to the parser
        parser.add_argument('--results_dir', type = str, default = './results/', help = 'saves results here.')
        parser.add_argument('--phase', type = str, default = 'test', help = 'train, val, test, etc')
        
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action = 'store_true', help = 'use eval mode during test time.')
        parser.add_argument('--num_test', type = int, default = 10000, help = 'how many test images to run')
        
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size = parser.get_default('crop_size'))
        
        # Turn on eval mode
        self.isTrain = False
        
        # Return the parser
        return parser
