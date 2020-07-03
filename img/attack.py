import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


epsilons = [0, .15]
pretrained_model = "data/MNIST/lenet_mnist_model.pth"
use_cuda=False


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
# set evaluation mode for dropout layers
model.eval()


# FGSM attack
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def lbfgs_attack(image, target):
    pass


def test(model, device, test_loader, num_samples=5):
    adv_examples = []
    for i in range(num_samples):
        data, target = next(test_loader)
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for attacks
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # If the initial prediction is wrong, dont bother attacking, just move on
        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, .15, data_grad)
        # Re-classify the perturbed image
        output = model(perturbed_data)
        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max 
        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_examples.append((target.item(), init_pred.item(), final_pred.item(), data.squeeze().detach().numpy(), adv_ex))
    return adv_examples


test_loader = iter(test_loader)
adv_examples = test(model, device, test_loader)


cnt = 0
plt.figure(figsize=(9,9))
num_samples = len(adv_examples)
for i in range(num_samples):
    print(i)
    target, orig_pred, adv_pred, orig_data, adv_data = adv_examples[i]
    plt.subplot(num_samples, 2, 2*i+1)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.ylabel("{}".format(adv_examples[i][0]), fontsize=14)
    plt.imshow(orig_data, cmap="gray")
    plt.title("{} -> {}".format(target, orig_pred))
    plt.subplot(num_samples, 2, 2*i+2)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(adv_data, cmap="gray")
    plt.title("{} -> {}".format(orig_pred, adv_pred))
plt.show()
