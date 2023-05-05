from metaflow import FlowSpec, step


class BasicFlow(FlowSpec):
    @step
    def start(self):
        print("start")
        self.next(self.next_step)


    @step
    def next_step(self):
        print("I go next")
        self.next(self.end)
        
    @step
    def end(self):
        print("I am the final step")


if __name__ == "__main__":
    # Type in --help to get a list of possible metaflow commands
    # to run this DAG, type in python hello_metaflow.py run
    test = BasicFlow()
