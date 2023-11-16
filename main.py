import torch
import torchvision
from models.model import Model_MLP
from dataset.disfa import Disfa
from tqdm import tqdm


def normalize_each_sample(x):
    norm = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
    return norm.float()


def dict_to_list(labels) :
    return torch.tensor([ x for x in labels.values() ]).float()

if __name__ == '__main__':
    # Define device
    batch_size = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set fixed random number seed
    torch.manual_seed(42)

    transform = torchvision.transforms.Compose([normalize_each_sample])
    target_transform = torchvision.transforms.Compose([dict_to_list])

    dataset = Disfa(root='./data/DISFA', transform=transform, target_transform=target_transform)

    train_dataset = dataset.train_dataset

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Initialize the MLP
    mlp = Model_MLP(input_dim = 478*3, output_dim = 12)

    mlp = mlp.to(device)

    # Define the loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(5): # 5 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        epoch_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in tqdm(enumerate(trainloader), total=int(len(train_dataset) // batch_size)) :

            # Get inputs
            _, inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            epoch_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, epoch_loss / 500))

        
        print("Epoch {} completed! Average Loss: {}".format(epoch, epoch_loss / len(trainloader)))

        # Process is complete.
    print('Training process has finished.')