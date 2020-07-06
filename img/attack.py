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



batch_size=5
# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=5, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
# set evaluation mode for dropout layers
model.eval()




for cln_data, true_label in test_loader:
    break
cln_data, true_label = cln_data.to(device), true_label.to(device)





cln_data.size()

adversary = LBFGSAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), initial_const=0.01,
    clip_min=0.0, clip_max=1.0, num_classes=10,
    targeted=False)

from scipy.optimize import fmin_l_bfgs_b

def lbfgs_attack(model, x, target):
    def _loss_func(adv, x, target):
        # adv_n_const
        adv = torch.from_numpy(adv.reshape(x.shape)).float().to(x.device).requires_grad_()
        # adv = torch.from_numpy(adv.reshape(x.shape)).float().requires_grad_()
        out = model(adv)
        loss_1 = torch.sum(F.nll_loss(out, target, reduction='none'))
        print(loss_1)
        loss_2 = torch.sum((adv-x)**2)
        print(loss_2)
        loss = loss_1 + loss_2/4
        loss.backward()
        adv_grad = adv.grad.data.numpy().flatten().astype(float)
        loss = loss.data.numpy().flatten().astype(float)
        return loss, adv_grad

    clip_min = 0 * np.ones(x.shape[:]).astype(float)
    clip_max = 1 * np.ones(x.shape[:]).astype(float)
    clip_bound = list(zip(clip_min.flatten(), clip_max.flatten()))

    adv, f, _ = fmin_l_bfgs_b(_loss_func,
                              x.clone().cpu().numpy().flatten().astype(float),
                              args=(x.clone(), target),
                              maxiter=100, bounds=clip_bound)

    adv = torch.from_numpy(adv.reshape(x.shape)).float().to(x.device)
    print(adv.shape)
    return adv





target = torch.ones_like(true_label) * 3
# adv_untargeted = adversary.perturb(cln_data, true_label)
adv_homemade = lbfgs_attack(model, cln_data, target)


adversary.targeted = True
adv_targeted = adversary.perturb(cln_data, target)

pred_cln = predict_from_logits(model(cln_data))
pred_untargeted_adv = predict_from_logits(model(adv_homemade))
pred_targeted_adv = predict_from_logits(model(adv_targeted))

print("adv_h", adv_homemade.size())
print("adv", adv_targeted.size())
print("adv_h")


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
for ii in range(batch_size):
    plt.subplot(3, batch_size, ii + 1)
    _imshow(cln_data[ii])
    plt.title("clean \n pred: {}".format(pred_cln[ii]))
    plt.subplot(3, batch_size, ii + 1 + batch_size)
    _imshow(adv_homemade[ii])
    plt.title("untargeted \n adv \n pred: {}".format(
        pred_untargeted_adv[ii]))
    # plt.subplot(3, batch_size, ii + 1 + batch_size * 2)
    # _imshow(adv_targeted[ii])
    # plt.title("targeted to 3 \n adv \n pred: {}".format(
    #     pred_targeted_adv[ii]))

plt.tight_layout()
plt.show()














# FGSM attack
def fgsm_attack(image, epsilon, data_grad):
    # get element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # generate perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # add clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# cnt = 0
# plt.figure(figsize=(9,9))
# num_samples = len(adv_examples)
# for i in range(num_samples):
#     print(i)
#     target, orig_pred, adv_pred, orig_data, adv_data = adv_examples[i]
#     plt.subplot(num_samples, 2, 2*i+1)
#     plt.xticks([], [])
#     plt.yticks([], [])
#     plt.ylabel("{}".format(adv_examples[i][0]), fontsize=14)
#     plt.imshow(orig_data, cmap="gray")
#     plt.title("{} -> {}".format(target, orig_pred), fontsize=8)
#     plt.subplot(num_samples, 2, 2*i+2)
#     plt.xticks([], [])
#     plt.yticks([], [])
#     plt.imshow(adv_data, cmap="gray")
#     plt.title("{} -> {}".format(orig_pred, adv_pred), fontsize=8)
# plt.show()
