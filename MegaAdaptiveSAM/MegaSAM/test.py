from NN_utils import *
from data.mnist import *

training_set, test_set= mnist_data_gen()

training_set = [(i[0].flatten(), i[1]) for i in training_set]
test_set = [(i[0].flatten(), i[1]) for i in test_set]

output1 = 16
output2 = 16

MLP = nn.Sequential(
    nn.Linear(28 * 28, output1),
    nn.ReLU(),
    nn.Linear(output1, output2),
    nn.ReLU(),
    nn.Linear(output2, 10)
)

model, training_losses, training_accuracies, validation_accuracies = train_multi_model(train_data=training_set, test_data=test_set, model=MLP,
                       optim="SAM", batch_size=10, epochs=1, tracking=True)

# with MegaSam there was a problem with memory usage