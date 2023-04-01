Includes the weights object, as well as of the values of the weight.

### Weight Values
Load weights to network via:
        weights_dict = torch.load("/path/to/weights")
        model.load_state_dict(weights_dict)

### Weights Object
This is an object that contains meta information regarding the weights. To load it, do
        with open("/path/to/object.pickle", "rb") as file:
            pickled = pickle.load(file).DEFAULT

To access transformations use:
            pickled.transforms()