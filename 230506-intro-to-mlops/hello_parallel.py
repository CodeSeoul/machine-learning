from metaflow import FlowSpec, step


class ParallelFlow(FlowSpec):
    @step
    def start(self):
        print("start")
        self.next(self.next_step)

    # TODO: Write three extra flows
    # So that we get the following DAG
    # For the training implementation, write print(statement)
    #         train_model_1 
    #       /               \
    # start - train_model_2 - evaluate - end
    #       \               /
    #         train_model_3     
    # 
    # Note: train_model_1-3 must all be executed in parallel. 
    # evaluate must always run after train_model_1-3 have completed.
    @step
    def end(self):
        print("I am the final step")


if __name__ == "__main__":
    test = ParallelFlow()
