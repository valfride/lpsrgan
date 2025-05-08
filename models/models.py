models = {}

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
        
    return decorator


def make(model_spec, load_model=False):
    model = models[model_spec['name']](**model_spec['args'])
   
    if load_model:
        try:
            if len(model) > 1:
                model_g = model[0]
                print('LOADED')
                
                model_g.load_state_dict(model_spec['sd'])
                model = (model_g, model[1])
        except:
            pass
        
    return model
