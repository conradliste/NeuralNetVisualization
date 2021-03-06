import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# No layer names can contain hyphens, periods, or quotation marks
# Returns an ordered dict with layer info
# The first level of keys are the class names of each layer
# The second level of keys are "num_params", "input_shape", "output_shape", "trainable", "weights", "bias"
# Sets the model t
# Params:
# model: the neural net
# init_input_size (tuple): input shape to the model not including batch size
def extract_layers(model, init_input_size, inputs=None, device=torch.device('cuda:0'), batch_size=2):
    
    if not isinstance(model, nn.Module):
        raise TypeError("model must be of typer nn.Module")
    
    def register_hook(layer):
        # Extracts information from one module layer
        #
        # Params:
        # module: an nn.Module
        # inp: batched input to network
        # out: batched output of network
        # layer_dict: order dict to store module info
        def hook(layer, inp, out):
            # Layer type is everything to the write of period in module class name
            layer_type = str(layer.__class__).split(".")[-1].replace("'", '').replace(">", "")
            layer_key = layer_type + "-" + str(len(layer_dict))
            # Store layer info
            layer_dict[layer_key] = OrderedDict()
            layer_dict[layer_key]["input_shape"] = inp[0].shape[1:]
            
            # Case for multiple outputs
            if isinstance(out, (list, tuple)):
                layer_dict[layer_key]["output_shape"] = [o.shape for o in out]
            else:
                layer_dict[layer_key]["output_shape"] = out[0].shape
        
            layer_dict[layer_key]["trainable"] = False
            layer_dict[layer_key]["weights"] = None
            layer_dict[layer_key]["bias"] = None
            num_params = 0
            # Grab weight parameter information
            if hasattr(layer, "weight") and hasattr(layer.weight, "size"):
                num_params += torch.prod(torch.LongTensor(list(layer.weight.size()))).item()
                # Store parameters
                layer_dict[layer_key]["trainable"] = layer.weight.requires_grad
                layer_dict[layer_key]["weights"] = layer.weight
            # Grab bias parameter information
            if hasattr(layer, "bias") and hasattr(layer.bias, "size"):
                num_params += torch.prod(torch.LongTensor(list(layer.bias.size()))).item()
                layer_dict[layer_key]["bias"] = layer.bias

            layer_dict[layer_key]["num_params"] = num_params
            layer_dict[layer_key]["input"] = inp
            layer_dict[layer_key]["output"] = out
            
        # Register the hook if it is just one module
        if not isinstance(layer, nn.Sequential) and not isinstance(layer, nn.ModuleList) and not layer == model:
            hooks.append(layer.register_forward_hook(hook))
    
    # Set dtype
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    
    if inputs is None:
        #Creates a list of batched data for each dimension
        x = [torch.rand(batch_size, *in_size).type(dtype).to(device=device) 
                for in_size in [init_input_size]]
    else:
        x = [inputs]

    layer_dict = OrderedDict()
    hooks = []

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    # Register hooks and perform a forward pass
    model.apply(register_hook)
    model.eval()
    model(*x)
    model.train()

    # Detach the hooks
    for h in hooks:
        h.remove()
    
    return layer_dict
    


