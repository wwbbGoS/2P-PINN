import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from scipy.interpolate import griddata
from torch.utils.data import TensorDataset, DataLoader

class Burgers_PINN():
    def __init__(self):
        super(Burgers_PINN, self).__init__()
        
    def initiation(self,x_b, x_f):
        tmp = np.vstack([x_b,x_f])
        self.Xmean, self.Xstd = tmp.mean(0), tmp.std(0)
        
        #Jacobian of the PDE because of normalization
        self.Jacobian_X = 1 / self.Xstd[0]
        self.Jacobian_T = 1 / self.Xstd[1]

    def loading(self, x_u1,x_u2, y_u1,y_u2, x_f1,x_f2, net, device, nepochs, lambda_phy, noise,st):
        # # Normalize data

        self.st = st

        # x_u = np.vstack([x_u1,x_u2]) if x_u2.size != 0 else x_u1
        # x_f = np.vstack([x_f1,x_f2]) if x_f2.size != 0 else x_f1
        # y_u = np.vstack([y_u1,y_u2]) if y_u2.size != 0 else y_u1

        self.x_u1 = (x_u1 - self.Xmean) / self.Xstd
        self.x_u2 = (x_u2 - self.Xmean) / self.Xstd if x_u2.size != 0 else np.array([])

        self.x_f1 = (x_f1 - self.Xmean) / self.Xstd
        self.x_f2 = (x_f2 - self.Xmean) / self.Xstd if x_f2.size != 0 else np.array([])

        self.y_u1 = y_u1 + noise * np.std(y_u1)*np.random.randn(y_u1.shape[0], y_u1.shape[1])
        self.y_u2 = y_u2 if y_u2.size != 0 else np.array([])

        self.net = net
        
        self.net_optim = torch.optim.Adam(self.net.parameters(),
                                           lr=1e-4, 
                                           betas = (0.9, 0.999))
        
        # self.net_optim = torch.optim.LBFGS(self.net.parameters(), lr=1e-4)
        
        self.device = device
        
        # numpy to tensor
        self.train_x_u1 = torch.tensor(self.x_u1, requires_grad=True).float().to(self.device)
        self.train_y_u1 = torch.tensor(self.y_u1, requires_grad=True).float().to(self.device)
        self.train_x_f1 = torch.tensor(self.x_f1, requires_grad=True).float().to(self.device)
        
        if x_u2.size != 0:
            self.train_x_u2 = torch.tensor(self.x_u2, requires_grad=True).float().to(self.device)
            self.train_y_u2 = torch.tensor(self.y_u2, requires_grad=True).float().to(self.device)
            self.train_x_f2 = torch.tensor(self.x_f2, requires_grad=True).float().to(self.device)
        

        self.nepochs = nepochs
        self.lambda_phy = lambda_phy
        
        self.batch_size = 150
        num_workers = 4
        shuffle = True
        self.train_loader1 = DataLoader(
            list(zip(self.train_x_u1,self.train_y_u1)), batch_size=self.batch_size, shuffle=shuffle
        )
        if x_u2.size != 0:
            self.train_loader2 = DataLoader(
                list(zip(self.train_x_u2,self.train_y_u2)), batch_size=self.batch_size, shuffle=shuffle
            )
        
    def get_residual(self, X):
        # physics loss for collocation/boundary points
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        u = self.net.forward(torch.cat([x, t], dim=1))
        f = self.phy_residual(x, t, u)
        return u, f
    
    def uncertainity_estimate(self, x, num_samples=500):
        outputs = np.hstack([self.net(x).cpu().detach().numpy() for i in range(num_samples)]) 
        y_variance = outputs.var(axis=1)
        y_std = np.sqrt(y_variance)
        return y_mean, y_std
    
    def phy_residual(self, x, t, u, nu = (0.01/np.pi)):
        """ The pytorch autograd version of calculating residual """

        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = (self.Jacobian_T) * u_t + (self.Jacobian_X) * u * u_x - nu * (self.Jacobian_X ** 2) * u_xx 
        return f
    
    def train(self):
        TOT_loss = np.zeros(self.nepochs)
        MSE_loss = np.zeros(self.nepochs)
        PHY_loss = np.zeros(self.nepochs)
        

        for epoch in range(self.nepochs):
            epoch_loss = 0

            part1_error = [0,0,0]
            part2_error = [0,0,0]

            for i, (x, y) in enumerate(self.train_loader1):

                self.net_optim.zero_grad()

                y_pred, _ = self.get_residual(x)
                _, residual = self.get_residual(self.train_x_f1)

                mse_loss = torch.nn.functional.mse_loss(y, y_pred)
                physics_loss = torch.mean(residual**2)

                loss = mse_loss + self.lambda_phy*physics_loss

                loss.backward(retain_graph=True)
                self.net_optim.step()
            
                part1_error[0] += loss.detach().cpu().numpy()
                part1_error[1] += mse_loss.detach().cpu().numpy()
                part1_error[2] += physics_loss.detach().cpu().numpy()


            if self.y_u2.size != 0:
                for i, (x, y) in enumerate(self.train_loader2):
                    self.net_optim.zero_grad()
                    y_pred, _ = self.get_residual(x)
                    _, residual = self.get_residual(self.train_x_f2)

                    mse_loss = torch.nn.functional.mse_loss(y, y_pred)
                    physics_loss = torch.mean(residual**2)

                    # loss =  (1-self.st)*mse_loss + self.lambda_phy*physics_loss
                    loss =  mse_loss + self.lambda_phy*physics_loss


                    loss.backward(retain_graph=True)
                    self.net_optim.step()

                    part2_error[0] += loss.detach().cpu().numpy()
                    part2_error[1] += mse_loss.detach().cpu().numpy()
                    part2_error[2] += physics_loss.detach().cpu().numpy()
                
            

            if self.y_u2.size != 0:
                TOT_loss[epoch] = part1_error[0] / len(self.train_loader1) \
                                + part2_error[0] / len(self.train_loader2) 
                MSE_loss[epoch] = part1_error[1] / len(self.train_loader1) \
                                + part2_error[1] / len(self.train_loader2)
                PHY_loss[epoch] = part1_error[2] / len(self.train_loader1) \
                                + part2_error[2] / len(self.train_loader2)
            else:
                TOT_loss[epoch] = part1_error[0] / len(self.train_loader1)
                MSE_loss[epoch] = part1_error[1] / len(self.train_loader1) 
                PHY_loss[epoch] = part1_error[2] / len(self.train_loader1)


            if (epoch % 100 == 0):
                print(
                    "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [Total loss: %f]"
                    % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], TOT_loss[epoch])
                )