def zeroed(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module
