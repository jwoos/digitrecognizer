import numpy as np


def stochastic_gradient_descent(network, in_batch, out_batch):
    correct = 0

    for data, label in zip(in_batch, out_batch):
        # Forward propagation
        outputs = [data]
        for layer in network.layers:
            data = layer.forward(outputs[-1])
            outputs.append(data)

        if label[np.argmax(outputs[-1])]:
            correct += 1

        # Backward propagation
        errors = [
            network.loss(outputs[i], label, derivative=True)
        ]
        for i, layer in reversed(enumerate(network.layers)):
            if i == len(network.layers) - 1:
                continue

            i += 1
            error, weight_gradient, bias_gradient = layer.backward(outputs[i - 1], outputs[i], errors[-1])

            # update parameters
            layer.weights -= weight_gradient * network.learning_rate
            layer.biases -= bias_gradient * network.learning_rate

            errors.append(error)

        yield np.sum(network.loss(outputs[-1], label)), correct / len(in_batch)
