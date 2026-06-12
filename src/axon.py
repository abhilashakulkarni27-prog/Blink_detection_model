import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch as pt
import torch.nn.functional as F

class layer:
    def __init__(self,input_dimension,neurons,device="cpu"):
        std = (2.0 / input_dimension) ** 0.5
        self.weights=pt.randn(input_dimension,neurons,device=device)*std
        self.biases=pt.zeros(neurons,device=device)
        self.weights.requires_grad_()
        self.biases.requires_grad_()
    def parameters(self):
        return [self.weights, self.biases]
    def __call__(self,x):
        result=x@self.weights
        result+=self.biases
        return result

class MLP:
    def __init__(self,list_of_neurons,input_dimension,device="cpu"):
        self.list_of_neurons=list_of_neurons
        self.input_dimension=input_dimension
        layers=[]
        layers.append(layer(input_dimension,self.list_of_neurons[0],device))
        if len(self.list_of_neurons)>1:
            for i in range(1,len(self.list_of_neurons)-1):
                layers.append(layer(self.list_of_neurons[i-1],self.list_of_neurons[i],device))
            layers.append(layer(self.list_of_neurons[-2],self.list_of_neurons[-1],device))
        self.layers=layers
    def parameters(self):
        P=[]
        for i in self.layers:
            P.extend(i.parameters())
        return P
    def __call__(self,x):
        y=self.layers[0](x)
        y=pt.relu(y)
        if len(self.layers)>1:
            for i in range(1,len(self.layers)-1):
                y=self.layers[i](y)
                y=pt.relu(y)
            y=self.layers[-1](y)
        return y
class conv2d_general:
    def __init__(self,input_depth,number_of_kernals=1,kernal_size=(3,3),stride=(1,1),padding=(1,1),device="cpu"):
        self.input_depth=input_depth
        self.number_of_kernals=number_of_kernals
        self.kernal_size=kernal_size
        self.stride=stride
        self.padding=padding
        self.linear_layer=layer(self.kernal_size[0]*self.kernal_size[1]*input_depth,number_of_kernals,device)
    def parameters(self):
        return self.linear_layer.parameters()
    def __call__(self,x):

        Batch_size,depth,length,width = x.shape
    
        if depth != self.input_depth:
            raise ValueError(
                f"Input depth {depth} does not match expected depth {self.input_depth}"
            )
    
        kh, kw = self.kernal_size
        sh, sw = self.stride
        ph, pw = self.padding
    
        Lout = int((length + 2*ph - kh)/sh + 1)
        Wout = int((width + 2*pw - kw)/sw + 1)
    
        x = F.pad(x,(pw,pw,ph,ph))
    
        patches = F.unfold(
            x,
            kernel_size=(kh,kw),
            stride=(sh,sw)
        )
    
        patches = patches.transpose(1,2).contiguous()
    
        out = patches @ self.linear_layer.weights
        out = out + self.linear_layer.biases
    
        out = out.transpose(1,2).reshape(
            Batch_size,
            self.number_of_kernals,
            Lout,
            Wout
        )
    
        return out
                

class conv2D:
    def __init__(self,input_depth,number_of_kernals=1,kernal_size=3,stride=1,padding=1,device="cpu"):
        self.input_depth=input_depth
        self.number_of_kernals=number_of_kernals
        self.kernal_size=kernal_size
        self.stride=stride
        self.padding=padding
        self.linear_layer=layer(self.kernal_size*self.kernal_size*input_depth,number_of_kernals,device)
    def parameters(self):
        return self.linear_layer.parameters()
    def __call__(self,x):

        Batch_size,depth,length,width = x.shape
    
        if depth != self.input_depth:
            raise ValueError(
                f"Input depth {depth} does not match expected depth {self.input_depth}"
            )
    
        Lout = int((length + 2*self.padding - self.kernal_size)/self.stride + 1)
        Wout = int((width + 2*self.padding - self.kernal_size)/self.stride + 1)
    
        x = F.pad(x, (self.padding,self.padding,self.padding,self.padding))
    
        patches = F.unfold(
            x,
            kernel_size=self.kernal_size,
            stride=self.stride
        )
    
        patches = patches.transpose(1,2).contiguous()
    
        out = patches @ self.linear_layer.weights
        out = out + self.linear_layer.biases
    
        out = out.transpose(1,2).reshape(
            Batch_size,
            self.number_of_kernals,
            Lout,
            Wout
        )
    
        return out


class embedding_layer():
    def __init__(self,vocab_size,dimentionality=2,device="cpu"):
        self.table=pt.randn(vocab_size,dimentionality,device=device)
        self.table.requires_grad_()
    def parameters(self):
        return [self.table]
    def __call__(self,x):
        return self.table[x]

class multi_headed_attention():

    def __init__(self,embedding_dimension,head_dimension,number_of_heads,masked=False,device="cpu"):
        L=[]
        for i in range(number_of_heads):
            x=[]#query,key,value
            x.append(layer(embedding_dimension,head_dimension,device=device))
            x.append(layer(embedding_dimension,head_dimension,device=device))
            x.append(layer(embedding_dimension,head_dimension,device=device))
            L.append(x)
        self.projection_layer=layer(head_dimension*number_of_heads,embedding_dimension,device=device)
        self.Multiheaded_list=L
        self.head_dimension=head_dimension
        self.masked=masked
    def parameters(self):
        P=[]
        for i in self.Multiheaded_list:
            for j in i:
                P.extend(j.parameters())
        P.extend(self.projection_layer.parameters())
        return P
    def __call__(self,x):
        l=[]
        T=x.shape[-2]
        if self.masked:
            tril = pt.tril(pt.ones(T, T, device=x.device))
        for i in self.Multiheaded_list:
            keys=i[1](x)
            qureys=i[0](x)
            values=i[2](x)
            attention_weights=qureys@keys.transpose(-2,-1)/(self.head_dimension ** 0.5)
            if self.masked:
                attention_weights=attention_weights.masked_fill(tril==0,float('-inf'))
            attention_weights=F.softmax(attention_weights,dim=-1)
            output=attention_weights@values
            l.append(output)
        result = pt.cat(l, dim=-1)
        return self.projection_layer(result)

class normalization_layer():
    def __init__(self,embedding_dimension,device="cpu"):
        self.gamma = pt.ones(embedding_dimension,device=device)   # not the full (B,T,C) shape!
        self.beta  = pt.zeros(embedding_dimension,device=device)
        self.gamma.requires_grad_()  # ← missing!
        self.beta.requires_grad_()
    
    def parameters(self):
        return [self.gamma,self.beta]

    def __call__(self,x):
        x_normalised=(x-x.mean(dim=-1,keepdim=True))/(x.std(dim=-1, keepdim=True) + 1e-5)
        return self.gamma*x_normalised+self.beta
        
    
class attention_block():
    def __init__(self,embedding_dimension,head_dimension,number_of_heads,masked=False,device="cpu"):
        self.attention=multi_headed_attention(embedding_dimension,head_dimension,number_of_heads,masked=masked,device=device)
        self.normalisation_1=normalization_layer(embedding_dimension,device=device)
        self.MLP=MLP([4*embedding_dimension,embedding_dimension],embedding_dimension,device=device)
        self.normalisation_2=normalization_layer(embedding_dimension,device=device)
    
    def parameters(self):
        P=[]
        P.extend(self.attention.parameters())
        P.extend(self.normalisation_1.parameters())
        P.extend(self.MLP.parameters())
        P.extend(self.normalisation_2.parameters())
        return P

    def __call__(self,x):
        return self.normalisation_2(x+self.MLP(self.normalisation_1(x+self.attention(x))))


class Batch_norm_1D():
    def __init__(self, features, device="cpu", eps=1e-5, momentum=0.1):
        self.features = features
        self.eps = eps
        self.momentum = momentum
        self.mode_str = 'train' 
    
        self.gamma = pt.ones(features, device=device)
        self.beta = pt.zeros(features, device=device)
        self.gamma.requires_grad_()
        self.beta.requires_grad_()
        
        self.running_mean = pt.zeros(features, device=device)
        self.running_var = pt.ones(features, device=device) 
    
    def mode(self, mode='train'):
        self.mode_str = 'train' if mode == 'train' else 'test'

    def parameters(self):
        return [self.gamma, self.beta]

    def __call__(self, x):
        if self.mode_str == 'train':
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False) 
            
            with pt.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            x_hat = (x - mean) / pt.sqrt(var + self.eps)
        else:
            x_hat = (x - self.running_mean) / pt.sqrt(self.running_var + self.eps)
        
        return self.gamma * x_hat + self.beta
            
        
class Batch_norm_2D():
    def __init__(self, features, device="cpu", eps=1e-5, momentum=0.1):
        self.features = features
        self.eps = eps
        self.momentum = momentum
        self.mode_str = 'train' 
    
        self.gamma = pt.ones(features, device=device)
        self.beta = pt.zeros(features, device=device)
        self.gamma.requires_grad_()
        self.beta.requires_grad_()
        
        self.running_mean = pt.zeros(features, device=device)
        self.running_var = pt.ones(features, device=device) 
    
    def mode(self, mode='train'):
        self.mode_str = 'train' if mode == 'train' else 'test'

    def parameters(self):
        return [self.gamma, self.beta]

    def __call__(self, x):
        B, C, H, W = x.shape
        x= x.permute(0, 2, 3, 1).reshape(-1, C)
        if self.mode_str == 'train':
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False) 
            
            with pt.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            x_hat = (x - mean) / pt.sqrt(var + self.eps)
        else:
            x_hat = (x - self.running_mean) / pt.sqrt(self.running_var + self.eps)
        
        x_final=self.gamma * x_hat + self.beta
        x_final = x_final.reshape(B, H, W, C).contiguous().permute(0, 3, 1, 2)
        return x_final
            
        
        
