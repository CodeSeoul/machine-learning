from metaflow import FlowSpec, step


class PyTorchFlow(FlowSpec):

    @step
    def start(self):
        print("yee")
        self.next(self.end)

    # TODO: Train three models in parallel
    # Train one model for 1 epoch, another for 2 epochs and another for 3 epochs
    # Record the accuracy on the test set
    # After finishing, compare the results and save only the best model.
    # call train_mnist_model() located inside of models/model.py to train the model
    # it will output a model. 
    # Feel free to use TensorBoard to visualize the model's training progress
    # Feel free to create a CI / CD workflow. 
    # Get creative and go crazy :)
    # Note: metaflow.Parameter can be useful for defining parameters
    # how you use it is up to you to figure out :)
    # Refer to: https://docs.metaflow.org/metaflow/basics

    @step
    def end(self):
        """
        We can do something cool like upload the best model / results to a model registry or something
        """
        print("end")


if __name__ == "__main__":
    PyTorchFlow()
