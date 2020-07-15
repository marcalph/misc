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



for x, y in test_loader:
    break
x, y = x.to(device), y.to(device)

x.size()
from scipy.optimize import fmin_l_bfgs_b

def lbfgs_attack(model, x, y, target):

    coeff_lower_bound = x.new_zeros(5)
    coeff_upper_bound = x.new_ones(5) * 1e10
    loss_coeffs = x.new_ones(5) * 1e-2
    final_l2dists = [1e10] * 5
    final_labels = [-1] * 5
    final_advs = x.clone()

    def _update_loss_coeffs(
            labs, batch_size,
            loss_coeffs, coeff_upper_bound, coeff_lower_bound, output):
        for ii in range(5):
            _, cur_label = torch.max(output[ii], 0)
            if cur_label.item() == int(labs[ii]):
                coeff_upper_bound[ii] = min(
                    coeff_upper_bound[ii], loss_coeffs[ii])

                if coeff_upper_bound[ii] < 1e10:
                    loss_coeffs[ii] = (
                        coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
            else:
                coeff_lower_bound[ii] = max(
                    coeff_lower_bound[ii], loss_coeffs[ii])
                if coeff_upper_bound[ii] < 1e10:
                    loss_coeffs[ii] = (
                        coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
                else:
                    loss_coeffs[ii] *= 10

    def _update_if_better(
            adv_img, labs, output, dist, batch_size,
            final_l2dists, final_labels, final_advs):
        for ii in range(batch_size):
            target_label = labs[ii]
            output_logits = output[ii]
            _, output_label = torch.max(output_logits, 0)
            di = dist[ii]
            if (di < final_l2dists[ii] and
                    output_label.item() == target_label):
                final_l2dists[ii] = di
                final_labels[ii] = output_label
                final_advs[ii] = adv_img[ii]

    def _loss_func(adv, x, target, loss_coeffs):
        global loss_1
        global loss_2
        adv = torch.from_numpy(adv.reshape(x.shape)).float().to(x.device).requires_grad_()
        out = model(adv)
        loss_1 = torch.sum(loss_coeffs * F.nll_loss(out, target, reduction='none'))
        loss_2 = torch.sum((adv-x)**2)
        loss = loss_1 + loss_2
        loss.backward()
        adv_grad = adv.grad.data.numpy().flatten().astype(float)
        loss = loss.data.numpy().flatten().astype(float)
        return loss, adv_grad

    clip_min = 0 * np.ones(x.shape[:]).astype(float)
    clip_max = 1 * np.ones(x.shape[:]).astype(float)
    clip_bound = list(zip(clip_min.flatten(), clip_max.flatten()))


    for _ in range(10):
        adv, f, _ = fmin_l_bfgs_b(_loss_func,
                                x.clone().cpu().numpy().flatten().astype(float),
                                args=(x.clone(), target, loss_coeffs),
                                maxiter=100, bounds=clip_bound)

        print("loss1===",loss_1)
        print("loss2===",loss_2)
        adv = torch.from_numpy(adv.reshape(x.shape)).float()
        d = (x - adv)**2
        d = d.view(d.shape[0], -1).sum(dim=1)
        out=model(adv)
        _update_if_better(adv, target, out, d, 5, final_l2dists, final_labels, final_advs)
        # print("l2dist", d)

        _update_loss_coeffs(target, 5, loss_coeffs, coeff_upper_bound, coeff_lower_bound, out)
        print("loss coeffs", loss_coeffs)
        print(adv.shape)
    return adv


target = torch.ones_like(y) * 3
adv_homemade = lbfgs_attack(model, x, y, target)


pred_cln = model(x).detach().numpy().argmax(axis=1)
print(pred_cln, "pred_cln")
pred_adv = model(adv_homemade).detach().numpy().argmax(axis=1)
print("adv_h", adv_homemade.size())
print("adv_h")


import matplotlib.pyplot as plt
plt.figure(figsize=(6, 8))
for ii in range(batch_size):
    plt.subplot(2, batch_size, ii + 1)
    plt.imshow(x[ii].squeeze().numpy())
    plt.title("clean \n pred: {}".format(pred_cln[ii]))
    plt.subplot(2, batch_size, ii + 1+batch_size)
    plt.imshow(adv_homemade[ii].squeeze().numpy())
    plt.title("clean \n pred: {}".format(pred_adv[ii]))
    # plt.subplot(2, batch_size, ii + 1 + batch_size)
    # plt.imshow((adv_homemade[ii])
    # plt.title("targeted \n adv \n pred: {}".format(pred_adv[ii]))
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
