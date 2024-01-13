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

    def loading(self, x_u, y_u, x_f,ground_X,ground_U, net, device, nepochs, lambda_phy, noise):
        # # Normalize data

        self.x_f = (x_f - self.Xmean) / self.Xstd
        self.x_u = (x_u - self.Xmean) / self.Xstd
        
        self.ground_X = ground_X
        self.ground_U = ground_U

        self.y_u = y_u + noise * np.std(y_u)*np.random.randn(y_u.shape[0], y_u.shape[1])
        
        self.net = net
        
        self.net_optim = torch.optim.Adam(self.net.parameters(),
                                           lr=1e-4, 
                                           betas = (0.9, 0.999))
        
        # self.net_optim = torch.optim.LBFGS(self.net.parameters(), lr=1e-4)
        
        self.device = device
        
        # numpy to tensor
        self.train_x_u = torch.tensor(self.x_u, requires_grad=True).float().to(self.device)
        self.train_y_u = torch.tensor(self.y_u, requires_grad=True).float().to(self.device)
        self.train_x_f = torch.tensor(self.x_f, requires_grad=True).float().to(self.device)
        
        self.nepochs = nepochs
        self.lambda_phy = lambda_phy
        
        self.batch_size = 150
        num_workers = 4
        shuffle = True
        self.train_loader = DataLoader(
            list(zip(self.train_x_u,self.train_y_u)), batch_size=self.batch_size, shuffle=shuffle
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
    
    def train(self,phase1 = True):
        TOT_loss = np.zeros(self.nepochs)
        MSE_loss = np.zeros(self.nepochs)
        PHY_loss = np.zeros(self.nepochs)
        
        epoch_round = []
        error_u_list = []
        error_f_list = []

        for epoch in range(self.nepochs):
            epoch_loss = 0
            for i, (x, y) in enumerate(self.train_loader):


                self.net_optim.zero_grad()

                y_pred, _ = self.get_residual(x)
                _, residual = self.get_residual(self.train_x_f)

                mse_loss = torch.nn.functional.mse_loss(y, y_pred)
                physics_loss = torch.mean(residual**2)

                if (phase1):
                    loss = 2.0 * mse_loss + self.lambda_phy*physics_loss
                else:
                    loss = 1.0 * mse_loss + self.lambda_phy*physics_loss

                loss.backward(retain_graph=True)
                self.net_optim.step()


                TOT_loss[epoch] += loss.detach().cpu().numpy()
                MSE_loss[epoch] += mse_loss.detach().cpu().numpy()
                PHY_loss[epoch] += physics_loss.detach().cpu().numpy()


            TOT_loss[epoch] = TOT_loss[epoch] / len(self.train_loader)
            MSE_loss[epoch] = MSE_loss[epoch] / len(self.train_loader)
            PHY_loss[epoch] = PHY_loss[epoch] / len(self.train_loader)


            if (epoch % 2000 == 0):

                print(epoch / self.nepochs)

            #     epoch_round.append(epoch)
            #     # print(
            #     #     "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [Total loss: %f]"
            #     #     % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], TOT_loss[epoch])
            #     # )

            #     Xmean = self.Xmean
            #     Xstd = self.Xstd

            #     X_star_norm = (self.ground_X - Xmean) / Xstd

            #     u_pred_list = []
            #     f_pred_list = []

            #     for run in range(1):
            #         u_pred, f_pred = self.get_residual(X_star_norm)
            #         u_pred_list.append(u_pred.detach().cpu().numpy())
            #         f_pred_list.append(f_pred.detach().cpu().numpy())

            #     u_pred_arr = np.array(u_pred_list)
            #     f_pred_arr = np.array(f_pred_list)
            #     u_pred = u_pred_arr.mean(axis=0)
            #     f_pred = f_pred_arr.mean(axis=0)

            #     error_u = np.linalg.norm(self.ground_U-u_pred,2)/np.linalg.norm(self.ground_U,2)
            #     error_res = (f_pred**2).mean()

            #     error_u_list.append(error_u)
            #     error_f_list.append(error_res)

            #     ts_error = np.linalg.norm(self.ground_U[-256:]-u_pred[-256:],2)/np.linalg.norm(self.ground_U[-256:],2)

            #     if (phase1):
            #         print(
            #             "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [ts_MSE_loss: %f]"
            #             % (epoch, self.nepochs, error_u, error_res, ts_error)
            #         )
            #     else:
            #         print(
            #             "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f]"
            #             % (epoch, self.nepochs, error_u, error_res)
            #         )

        return epoch_round, error_u_list,error_f_list
