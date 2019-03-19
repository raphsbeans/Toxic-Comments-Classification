import torch
import pathlib

def save_model(model, name=None, PATH='saved_models/'):
    '''
    This function will save your model, with a given name
    '''
    if name is None:
        name = model.__class__.__name__ + '.txt'
    
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
    
    with open(pathlib.Path(PATH) / name, 'wb') as f:
        torch.save(model,f)
    f.close()

def load_model (name, PATH='saved_models/', cuda=True):
    '''
    This function will load a model from a path
    '''
    model = torch.load(PATH+name)
    if (cuda):
        device = torch.device("cuda")
        model.to(device)
    model.eval()
    return model