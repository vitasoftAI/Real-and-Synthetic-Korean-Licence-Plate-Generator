# Import libraries
import importlib
from models.base_model import BaseModel

def find_model_using_name(model_name):
    
    """
    
    This function imports the module "models/[model_name]_model.py".
    
    Parameter:
    
        model_name  - the name of a model, str.
        
    Output:
    
        model       - a model to be trained.
    
    """
    
    # Get a model filename
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def get_option_setter(model_name):
    
    
    """
    
    This function returns the static method <modify_commandline_options> of the model class.
    
    Parameter:
    
        model_name  - the name of a model, str.
    
    """
    
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

def create_model(opt):
    
    
    """
    
    This function creates a model given the options.
    
    Parameter:
    
        opt    - parsed options, parser object.

    """
    
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    
    return instance
