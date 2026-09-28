#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 21:47:31 2026

@author: mkanoute
"""

import numpy as np

from scipy.optimize import minimize
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler



class grad_FS_ELM:
    def __init__(self, lbda, C, activation_function, nb_neurons, classification_bool = False, params_optim = None):
        self.lbda, self.C, self.activation_function, self.nb_neurons = lbda, C, activation_function, nb_neurons
        self.classification_bool, self.params_optim = classification_bool, params_optim

    
    def def_activation_function_torch(self, x):
        if self.activation_function == "linear":
            return x
        if self.activation_function == "tanh":
            return torch.tanh(x)
        elif self.activation_function == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation_function == "relu":
            return torch.relu(x)
        elif self.activation_function == "leakyrelu":
            return torch.where(x > 0, x, 0.01 * x)
        else:
            raise ValueError(f"Activation '{self.activation}' non reconnue.")   
    
    def compute_Salpha(self, X, alpha):
        # print("aplha.size", alpha.size(0))
        diag_tmp = torch.diag(torch.cat([alpha, torch.ones(1, dtype=torch.float64, device=self.device)]))
        Xalpha = diag_tmp @ X
        X_alpha_W0 = Xalpha.T @ self.W0.T
        Salpha = self.def_activation_function_torch(X_alpha_W0)
        return Salpha, X_alpha_W0
        

        
    def get_W2_with_alpha_on_train(self, alpha_min, X_train, Y_train):
        Salpha, X_alpha_W0 = self.compute_Salpha(X_train, alpha_min)
        Salpha_T = Salpha.T
        H = Salpha.T @ Salpha
        A = H + self.lbda * torch.eye(self.W0.size(0), dtype=torch.float64, device=self.device)
        # torch.backends.cuda.preferred_linalg_library("magma")
        inv = torch.linalg.inv(A)
        W = inv @ (Salpha_T @ Y_train)
        res = inv, W, Salpha, X_alpha_W0
        return res



    

    def callback_scipy(self, intermediate_result):
        
        # print("intermediate_result", intermediate_result)
        print("self.iter", self.iter, intermediate_result["fun"])
        loss_iter = intermediate_result["fun"]
        self.iter = self.iter + 1
        self.loss_batch.append(loss_iter)

    def loss_function(self, alpha0, X0=None, Y0=None):
        X = X0 if X0 is not None else self.X_train
        Y = Y0 if Y0 is not None else self.y_train
        alpha = torch.tensor(alpha0, dtype=torch.float64, device=self.device)
        inv, W, Salpha, X_alpha_W0 = self.get_W2_with_alpha_on_train(alpha, X, Y)
        Yalpha = Salpha @ W
        if Y.shape[1] > 1:
            obj_term = torch.linalg.matrix_norm(Y - Yalpha, ord="fro" ) ** 2
            norm_on_W = torch.linalg.matrix_norm(W, ord="fro") ** 2
        else:
            obj_term = torch.linalg.vector_norm(Y - Yalpha, ord=2) ** 2
            norm_on_W = torch.linalg.vector_norm(W,  ord=2) ** 2
        res0 = 0.5 * obj_term + 0.5 * self.lbda * norm_on_W + self.C * torch.sum(alpha)
        
        res = res0.cpu().numpy() if not self.is_grad  else res0
        self.last_W = W
        return res
    
    
    
    
    def derivate_activation_function_torch(self, X_alpha_W0, Salpha):
            if self.activation_function == "linear":
                return torch.ones_like(Salpha)
            elif self.activation_function == "tanh":
                return 1 - Salpha ** 2
            elif self.activation_function == "sigmoid":
                return Salpha * (1 - Salpha)
            elif self.activation_function == "leakyrelu":
                return torch.where(X_alpha_W0 > 0, torch.ones_like(X_alpha_W0), 0.01*torch.ones_like(X_alpha_W0))
    
    def derivate_S_alpha_old_torch(self, j, X, derivate_aciv_func_res):
        # Vectorisation possible pour toutes les colonnes si nécessaire
        return (X[j, :].unsqueeze(1)  @ self.W0[:, j].unsqueeze(1).T) * derivate_aciv_func_res
    
    
    def compute_gradient_pytorch(self, X, Y, d, alpha, inv_old, W_old, Salpha_old, X_alpha_W0_old):
        grad_alpha = torch.zeros(d, device=self.device, dtype = torch.float64)
        X_torch = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        Y_torch = torch.as_tensor(Y, dtype=torch.float64, device=self.device)

        W, Salpha, X_alpha_W0 = W_old, Salpha_old, X_alpha_W0_old

        derivate_aciv_func_res = self.derivate_activation_function_torch(X_alpha_W0, Salpha)          
        Yalpha = Salpha@W
        Ydiff = Yalpha - Y
        first_part00 = Ydiff.T
        for j in range(d):
            derivate_S_val = self.derivate_S_alpha_old_torch(j, X, derivate_aciv_func_res)
            derivate_Y = derivate_S_val @ W
            derivate_L = (first_part00 @ derivate_Y) + ((first_part00@Salpha + self.lbda*W.T)@inv_old@((-2*Salpha_old.T@derivate_S_val@W) + (derivate_S_val.T@Y)))
                        
            grad_alpha[j] = torch.trace(derivate_L) + self.C
        return grad_alpha
            
    
    def project_into_feasible_region(self, x):
        return np.clip(x, 0, 1)
    
    def get_alpha_using_numerical_approx(self, dim): 
        cost_function = self.loss_function
        jac = None
        x0 = self.project_into_feasible_region(self.x0_user)
        bounds = [ [0,1] for i in range(dim)]
        alpha_min = minimize(fun = cost_function, x0 = x0, bounds = bounds, jac = jac, method = 'L-BFGS-B', options = {"eps": self.params_numerical_approx["eps"], "maxiter": self.params_numerical_approx["maxiter"], "maxfun": self.params_numerical_approx["maxfun"], "maxls": self.params_numerical_approx["maxls"], "maxcor": self.params_numerical_approx["maxcor"]})
        return alpha_min
    
    def proj_torch(self, x, min_val = 0):
        res = torch.clamp(x, min=0, max=1)
        return res
    
    def armijo_backtracking(self, f, x, g, X, Y, thresh0=1.0, beta=0.5, c=1e-4):
        thresh = thresh0
        f_x = f(x, X0 = X, Y0 = Y)
        counter  = 0
        while True:
            x_i = x - thresh * g
            x_i_bis = x_i - thresh*self.C
            x_new = torch.clamp(x_i_bis, 0.0, 1.0)  # projection
            d = x_new - x
            f_new = f(x_new, X0 = X, Y0 = Y)
            if f_new <= f_x + c * torch.dot(g.flatten(), d.flatten()):
                break
            thresh *= beta
            counter = counter + 1
        return thresh
    
    
    def get_alpha_using_gradient_approxi(self, dim, X, Y): 

        self.X_torch, self.Y_torch = X, Y
        maxiter = self.params_optim["maxiter"]
        print("self.x0_user", self.x0_user.shape)
        alpha_init = torch.tensor(self.x0_user, dtype=torch.float64, device=self.device)
        alpha = self.proj_torch(alpha_init)
        eps_grad = 1e-5  # pgtol
        eps_alpha = 1e-6 # xtol
        factr=1e7
        ftol = factr * np.finfo(float).eps
        # ftol = 1e-6
        alpha_old = alpha.clone()
        loss_old = float('inf')
        success_bool = 0
        message = "Not CONV"
        self.best_loss, self.best_alpha, self.best_W2 = float('inf'), None, None
        learning_rate = None
        for k in range(maxiter):
            loss = self.loss_function(alpha, X0 = X, Y0 = Y)
            # self.loss_batch.append(loss.item())
            print(k, loss, learning_rate)
            inv, W, Salpha, X_alpha_W0 = self.get_W2_with_alpha_on_train(alpha, X, Y)
            grad_alpha = self.compute_gradient_pytorch(X, Y, len(alpha), alpha, inv, W, Salpha, X_alpha_W0)
            learning_rate = self.armijo_backtracking(self.loss_function, alpha, grad_alpha, X, Y, thresh0=1.0, beta=0.5, c=1e-4)
            alpha_temp = alpha - learning_rate * grad_alpha                
            alpha = self.proj_torch(alpha_temp)
            if torch.max(torch.abs(grad_alpha)) < eps_grad:
                print("Arrêt : gradient projeté faible (pgtol)")
                success_bool = 1
                message = "CONV_GRAD"
                break
            
            if torch.norm(alpha - alpha_old, p = 2, dtype = torch.float64) < eps_alpha:
                print("Arrêt : variation alpha faible (xtol)")
                success_bool = 1
                message = "CONV_ALPHA"
                break
            
            if k > 10 and ((loss_old - loss )/ max(abs(loss_old), abs(loss), 1.0)) <= ftol:
                print("Arrêt : variation relative de la fonction (ftol)")
                success_bool = 1
                message = "CONV_F"
                break
            
            if learning_rate <= 1e-6 and k > 10:
                print("Learning rate too small. Stopping.")
                success_bool = -1
                message = "lr_too_small"
                break
                
            alpha_old = alpha.clone()
            loss_old = loss

        nit = k
        # self.best_W2_after_train, _, _ = self.get_W2_with_alpha_on_train_torch(X, Y, self.alpha)
        res = {"x": alpha.cpu().numpy(), "nit": nit, "nfev": 0, "message": message}
        return res
    
    
    def one_hot_encoding(self, targets, n_classes):
        return np.eye(n_classes)[targets]

    def fit(self, X, y, type_optim = "numerical_approx"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device == "cuda":
            print("********** GPU  ****************")
        dim = X.shape[1]
        output_neurons = dim+1
        W00 = np.random.randn(self.nb_neurons, output_neurons)
        self.W0  = torch.as_tensor(W00, dtype=torch.float64, device=self.device) 
        self.x0_user = np.ones(dim)
        X_app = np.concatenate((X.T, np.ones((1, X.T.shape[1]))), axis = 0)
        self.X_train, self.y_train = torch.as_tensor(X_app, dtype=torch.float64, device=self.device), torch.as_tensor(y, dtype=torch.float64, device=self.device) 
        if type_optim == "numerical_approx":
            self.is_grad = False
            alpha_min = self.get_alpha_using_numerical_approx(dim)
        elif type_optim == "grad_variant":
            self.is_grad = True
            alpha_min = self.get_alpha_using_gradient_approxi(dim, self.X_train, self.y_train)
        self.alpha_np = alpha_min["x"]
        _, self.last_W, _, _ = self.get_W2_with_alpha_on_train(torch.tensor(self.alpha_np, dtype=torch.float64, device=self.device), self.X_train, self.y_train)

        return alpha_min 
         
            
    def predict(self, X0):
         X_app = np.concatenate((X0.T, np.ones((1, X0.T.shape[1]))), axis = 0)
         X = torch.as_tensor(X_app, dtype=torch.float64, device=self.device)
         alpha = torch.tensor(self.alpha_np, dtype=torch.float64, device=self.device)
         S_alpha, _ = self.compute_Salpha(X, alpha)
         Yalpha = (S_alpha @ self.last_W)
         Yalpha_res = Yalpha.cpu().numpy() 
         return Yalpha_res
    
    