def create_model(model, **kwargs):
    model = eval(model)(**kwargs)
    return model
