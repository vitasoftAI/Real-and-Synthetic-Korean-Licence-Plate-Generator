# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class GANLoss(nn.Module): 
    
    """
    
    This class gets several parameters and initialize GAN loss function.
    
    Parameters:
    
        gan_mode          - type of the loss function, str;
        target_real_label - label for a real image, int;
        target_fake_label - label for a fake image, int.
        
    Output:
    
        loss              - loss value, float.
    
    """

    def __init__(self, gan_mode, target_real_label = 1.0, target_fake_label = 0.0):
        
        super(GANLoss, self).__init__()
        
        assert gan_mode in ['lsgan', 'vanilla', 'wgangp', 'nonsaturating'], "Please choose a proper loss type for GAN loss."
        
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan': self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla': self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']: self.loss = None

    def get_target_tensor(self, prediction, target_is_real):
        
        """
        
        This function gets prediction and is real option and tensor with the ground truth label. 
        
        Parameters:
        
            prediction      - predicted value for the discrimination, tensor;
            target_is_real  - whether ground truth label is for real or fake images, bool. 
            
        Output:
        
            target_tensor   - a label tensor with ground truth label, tensor.
            
        """

        target_tensor = self.real_label if target_is_real else self.fake_label
        
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        
        """
        
        This function gets prediction and is real option and computes the loss value. 
        
        Parameters:
        
            prediction      - predicted value for the discrimination, tensor;
            target_is_real  - whether ground truth label is for real or fake images, bool. 
            
        Output:
        
            loss            - loss value, tensor float.
            
        """
        
        # Get batch size
        bs = prediction.size(0)
        
        # Compute loss based on the GAN mode
        if self.gan_mode in ['lsgan', 'vanilla']: loss = self.loss(prediction, self.get_target_tensor(prediction, target_is_real))
        elif self.gan_mode == 'wgangp': loss = -prediction.mean() if target_is_real else prediction.mean()
        elif self.gan_mode == 'nonsaturating': loss = F.softplus(-prediction).view(bs, -1).mean(dim = 1) if target_is_real else F.softplus(prediction).view(bs, -1).mean(dim = 1)
        
        return loss
    
class PatchNCELoss(nn.Module):
    
    """
    
    This class gets options and computes PatchNCE loss value.
    
    Parameters:
    
        opt  - train options, parser.
        
    Output:
    
        loss - computed loss value, tensor float;
    
    """
    
    def __init__(self, opt):
        
        super().__init__()
        self.opt = opt
        # Loss function
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        # Mask type
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        
        """
        This function gets query and key features and computes loss.
        
        Parameters:
        
            feat_q - features of query value, tensor;
            feat_k - features of key value, tensor.
            
        Output:
        
            loss   - computed loss value, tensor float.
        
        """ 
        
        # Get number of patches and dimensions
        num_patches, dim = feat_q.shape[0], feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit
        batch_dim_for_bmm = 1 if self.opt.nce_includes_all_negatives_from_minibatch else self.opt.batch_size

        # Reshape features to batch size
        feat_q, feat_k= feat_q.view(batch_dim_for_bmm, -1, dim), feat_k.view(batch_dim_for_bmm, -1, dim)
        
        # Get number of patches
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # Fill diagonal with very small numbers
        diagonal = torch.eye(npatches, device = feat_q.device, dtype = self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)
        
        # Concatenate positive and negative logits
        out = torch.cat((l_pos, l_neg), dim = 1) / self.opt.nce_T

        # Compute and return loss value
        return self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype = torch.long, device = feat_q.device))
