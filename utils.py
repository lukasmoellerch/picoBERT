from numpy import typing as npt


# A class that supports accessing the model parameters as an object
# like parameters.embeddings.word_embeddings.weight
class Parameters:
    def __init__(self, parameters: dict[str, npt.NDArray], prefix: str = ""):
        self.params = parameters
        self.prefix = prefix

    @staticmethod
    def from_model(model):
        parameters = {}
        for name, param in model.named_parameters():
            parameters[name] = param.detach().numpy()
        return Parameters(parameters)

    def tree(self):
        tree = {}
        for name, param in self.params.items():
            parts = name.split(".")
            current = tree
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = param.shape
        return tree

    def __getitem__(self, name):
        new_prefix = self.prefix
        if new_prefix:
            new_prefix += "."
        new_prefix += str(name)
        return Parameters(self.params, prefix=new_prefix)

    def __getattr__(self, name):
        return self[name]

    @property
    def np(self):
        return self.params[self.prefix]
