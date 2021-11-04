import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.model_utils import loss_fn_mse, loss_fn_ce, model_info

class F_cell(nn.Module):
    def __init__(self, input_size, output_size, sparse, inner=50, device='cuda'):
        super(F_cell, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        # print(input_size.dtype)
        self.sparse = sparse
        self.alpha = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(self.output_size), requires_grad=True)

        if self.sparse:
            self.weight_matrix_h = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        else:
            # self.weight_matrix_h = nn.Parameter(torch.zeros(output_size, output_size), requires_grad=True)

            u = torch.empty(output_size, inner).uniform_(-1.0 / math.sqrt(self.output_size), 1.0 / math.sqrt(self.output_size))
            v = torch.empty(inner, output_size).uniform_(-1.0 / math.sqrt(self.output_size), 1.0 / math.sqrt(self.output_size))
            self.u = torch.autograd.Variable(u, requires_grad=True)
            self.v = torch.autograd.Variable(v, requires_grad=True)
            # self.u = nn.Parameter(torch.zeros(output_size, inner), requires_grad=True)
            # self.v = nn.Parameter(torch.zeros(inner, output_size), requires_grad=True)
            self.weight_matrix_h = torch.matmul(self.u, self.v).to(device)
            eye_uv = torch.eye(self.output_size).to(device)
            self.weight_matrix_h = eye_uv + self.weight_matrix_h
            # print(self.weight_matrix_h.size())
            # eye_uv = torch.eye(output_size)
            # self.uv = nn.Parameter(torch.zeros(output_size, output_size), requires_grad=True)
            # self.weight_matrix_h = eye_uv + self.uv
            # self.weight_matrix_h = self.weight_matrix_h.to(device)


        self.weight_matrix_x = nn.Parameter(torch.zeros(self.input_size, self.output_size), requires_grad=True)
        self.device = device
        for name, weight in self.named_parameters():
            # print(name)
            '''
            if name == 'gradient_cell.weight_matrix_h':
                # print(name)
                nn.init.uniform_(weight, -1.0 / self.hidden_size, 1.0 / self.hidden_size)'''
            if name in ['u', 'v']:
                # print('a', name)
                nn.init.uniform_(weight, -1.0 / math.sqrt(self.output_size), 1.0 / math.sqrt(self.output_size))

    def forward(self, xt, ht):
        #  eye_uv = torch.eye(self.output_size).to(self.device)
        # self.weight_matrix_h = eye_uv + self.uv
        if self.sparse:
            phi = F.relu(ht * self.weight_matrix_h + torch.matmul(xt, self.weight_matrix_x) + self.b)
        else:
            phi = F.relu(torch.matmul(ht, self.weight_matrix_h) + torch.matmul(xt, self.weight_matrix_x) + self.b)

        ones = torch.ones_like(phi)
        zeros = torch.zeros_like(phi)
        phi_act = torch.where(phi > 0, ones, zeros)
        phi_p = torch.diag_embed(phi_act)
        eye_phi = torch.eye(self.output_size, device=self.device)
        eye_phi = eye_phi.reshape((1, self.output_size, self.output_size))
        eye_phi = eye_phi.repeat(ht.shape[0], 1, 1)

        if self.sparse:
            st = self.alpha * eye_phi - phi_p * self.weight_matrix_h
        else:
            st = self.alpha * eye_phi - torch.matmul(self.weight_matrix_h, phi_p)
        grad = torch.squeeze(torch.matmul(st, (self.alpha * ht - phi).view(phi.shape[0], phi.shape[1], 1)))
        return grad


class ODE_Vanilla(nn.Module):
    def __init__(self, sparse, eta, mu=0.5, task='GD', loss_mode='ce', hidden_size=128, nfeatures=1, device='cuda'):
        super(ODE_Vanilla, self).__init__()
        self.task = task
        self.loss_mode = loss_mode
        self.hidden_size = hidden_size
        self.input_size = nfeatures
        self.eta = nn.Parameter(torch.tensor([eta]), requires_grad=False)
        # self.eta = eta
        self.mu = nn.Parameter(torch.tensor([mu]), requires_grad=True)
        # self.mu = mu

        self.gradient_cell = F_cell(self.input_size, self.hidden_size, sparse)
        self.FC = nn.Linear(self.hidden_size, 10)
        self.device = device

        model_info(sparse, task, loss_mode, hidden_size)


        for name, weight in self.named_parameters():
            # print(name)

            if name in ['gradient_cell.b', 'gradient_cell.weight_matrix_x', 'FC.weight', 'FC.bias']:
                # print('a', name)
                nn.init.uniform_(weight, -1.0 / math.sqrt(self.hidden_size), 1.0 / math.sqrt(self.hidden_size))

    def forward(self, x, targets):
        batch_size = x.size()[0]
        hidden = torch.zeros((batch_size, self.hidden_size)).to(self.device)
        x = x.permute(1, 0, 2)

        # GD
        if self.task == 'GD':
            for i in range(x.shape[0]):
                hidden = hidden - self.eta * self.gradient_cell(x[i], hidden)

        # HB
        elif self.task == 'HB':
            hidden_t = torch.zeros((batch_size, self.hidden_size)).to(self.device)
            for i in range(x.shape[0]):
                hidden_t = self.mu * hidden_t - self.eta * self.gradient_cell(x[i], hidden)
                hidden = hidden + hidden_t

        # NAG
        elif self.task == 'NAG':
            hidden_t = torch.zeros((batch_size, self.hidden_size)).to(self.device)
            for i in range(x.shape[0]):
                hidden_t_previous = hidden_t
                hidden_t = hidden - self.eta * self.gradient_cell(x[i], hidden)
                hidden = hidden_t + self.mu * (hidden_t - hidden_t_previous)


        x = self.FC(hidden)

        if self.loss_mode == 'ce':
            return loss_fn_ce(x, targets)
        elif self.loss_mode == 'mse':
            return loss_fn_mse(x, targets)


# ce test
if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        model = ODE_Vanilla(False, eta=1e-3, task='GD', loss_mode='ce', hidden_size=128, nfeatures=1, device='cuda')
        model.to('cuda')
        img = torch.randn(3, 784, 1)
        img = img.to('cuda')
        labels = torch.empty(3, dtype=torch.long).random_(10)
        labels = labels.to('cuda')
    # print(labels.size())
    # print(labels)
        predicitons, out = model(img, labels)
        out.backward()
    # print(out)
'''
# mse test
if __name__ == '__main__':
    model = ODE_Vanilla(True, eta=1e-3, task='GD', loss_mode='mse', hidden_size=128, nfeatures=1, device='cuda')
    model.to('cuda')
    img = torch.randn(3, 784, 1)
    img = img.to('cuda')
    labels = torch.randn(3, 10)
    labels = labels.to('cuda')
    # print(labels.size())
    # print(labels)
    predicitons, out = model(img, labels)
    # print(predicitons)
    # print(out)'''
