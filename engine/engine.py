class Engine:
    def __init__(self, modules, signals):
        self.modules = modules
        self.signals = signals
        pass

    def run(self, data):
        # Init all module
        for module in self.modules:
            module.start(data)

        # Run till termination
        while True:
            data = self.step(data)

            if "terminate" in data and data["terminate"] == True:
                break

        # Shutdown all module
        for module in self.modules:
            module.stop(data)

        return data

    def step(self, data):
        stopped = False

        # Iterate all modules
        for module in self.modules:
            # Check if we shall terminate
            if "terminate" in data and data["terminate"] == True:
                print("Terminate on request")
                return data

            # Call the module
            call = False
            if "stopped" not in data:
                call = True

            if "stopped" in data and data["stopped"] == False:
                call = True

            if module.name == "Display":
                call = True

            if call:
                results = module.step(data)
            else:
                continue

            assert type(results) is dict, (
                "Module " + module.name + " must return a dictionary!"
            )
            # Verify results
            for signal, value in results.items():
                if signal in self.signals:
                    verifier = self.signals[signal]
                    if type(verifier) is type:
                        if type(results[signal]) is not verifier:
                            print(
                                "Cannot verify result of module",
                                module.name,
                                "on signal",
                                signal,
                            )
                            print("Expected type", verifier, "on signal", signal)

                    try:
                        verifier(results[signal])
                    except AssertionError as e:
                        print(
                            "Cannot verify result of module",
                            module.name,
                            "on signal",
                            signal,
                        )
                        print(e)
                        exit()

                data[signal] = results[signal]

        return data
