import numpy as np


def stochastic_gradient_descent(network, in_batch, out_batch):
    correct = 0

    for data, label in zip(in_batch, out_batch):
        # Forward propagation
        outputs = [
            np.reshape(data, (1, -1))
        ]
        for layer in network.layers:
            data = layer.forward(outputs[-1])
            outputs.append(data)

        if label[np.argmax(outputs[-1])]:
            correct += 1

        # Backward propagation
        errors = [
            network.loss(outputs[-1], np.reshape(label, (1, -1)), derivative=True)
        ]
        for i, layer in reversed(tuple(enumerate(network.layers, start=1))):
            error, weight_gradient, bias_gradient = layer.backward(
                data=outputs[i - 1],
                output=outputs[i],
                delta=errors[-1],
                previous_weight=network.layers[i - 1].weights,
            )

            # update parameters
            layer.weights -= weight_gradient * network.learning_rate
            layer.biases -= bias_gradient[0] * network.learning_rate

            errors.append(error)

        yield np.sum(network.loss(outputs[-1], label)), correct / len(in_batch)
