import torch

def save_model(model, name, PATH='saved_models/'):
    '''
    This function will save your model, with a given name
    '''
    with open(PATH+name, 'wb') as f:
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