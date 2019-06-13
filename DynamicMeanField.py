import numpy as np
from GaussianIntegral import *

# CTensor[alpha][alpha'][i][j]: correlation between position alpha of channel i and position alpha' of channel j
# CTensor generated from Big correlation 2D matrix where channel and input size are merged  
# layer index starts from 0 which correponds to the input layer
class DMFNet:

    g_w = 0.8
    rho = 0.1
    sigma_b = 0.1

    def __init__(self, struct,kernel_size,phi = np.tanh,input_size = 30, weight = None):
        self.struct = struct
        self.kernel_size = kernel_size
        self.layer_len = len(struct)
        self.currentLayerIdx = 0
        self.phi = phi
        # General weight formulation: weight[layer idx][Cout][Cin] is a list containing the kernel size matrix (has only one element)
        # layer idx >= 1
        if weight == None:
            self.weight = []
            weight.append('0')
            for n in range(self.layer_len - 1):
                C_in = struct[n]
                C_out = struct[n+1]
                weightPerLayer = [[[] for j in range(C_in)] for i in range(C_out)]
                self.weight.append(weightPerLayer) 
            self.initialWeight(struct)
        else:
            self.weight = weight
        # Inner formulation: for fast computation in the iteration step
        # weight_inner[layer idx][betai][betaj]:  betai/j indicates the 2D location of kernel || weight[layer idx][beta] is a matrix of size [Cout,Cin]  
        self.weight_inner = ['0']
        for n in range(1,self.layer_len):
            weight_inner_layer = [[[] for betaj in range(self.kernel_size)] for betai in range(self.kernel_size)]
            for betai in range(self.kernel_size):
                for betaj in range(self.kernel_size):
                    C_in = struct[n-1]
                    C_out = struct[n]
                    wtmp = np.zeros(C_out,C_in)
                    for i in range(C_out):
                        for j in range(C_in):
                            wtmp[i][j] = self.weight[n][i][j][0][betai][betaj]
                    weight_inner_layer[betai][betaj].append(wtmp)
            self.weight_inner.append(weight_inner_layer)
        # Input corvariance tensor
        self.InitialInputCorvarianceTensor(self,DMFNet.rho,input_size)
        # initial delta tensor
        self.UpdateDeltaTensor(self)
        self.InitialMeanActivation(self,input_size)
        # initial bias
        self.bias = ['0']
        self.InitialBias(self)
        
    def initialWeight(self):
        # lidx: layer index     cidx:   channel idx
        for n in range(self.layer_len - 1):
            C_in = self.struct[n]
            C_out = self.struct[n+1]
            for i in range(C_out):
                for j in range(C_in):
                    # weight[layer idx][Cout][Cin]
                    self.weight[n+1][i][j].append(np.random.randn(self.kernel_size,self.kernel_size)*(np.sqrt(DMFNet.g_w/(C_in*self.kernel_size*self.kernel_size))))

    def InitialBias(self):
        for n in range(1,self.layer_len):
            self.bias.append(np.sqrt(DMFNet.sigma_b)*np.random.randn(self.struct[n],1))
    
    def InitialMeanActivation(self,input_size = 30):
        # a list of length input_size*input_size and each element is a column vector of size [input_size,1]
        self.h_mean = []
        h_mean_alpha = np.zeros(self.struct[0],1)
        for i in range(input_size*input_size):
            self.h_mean.append(h_mean_alpha)

    def InitialInputCorvarianceTensor(self,rho,input_size = 30):
        # Big matrix 
        input_channel_size = self.struct[0] 
        alphaN = input_size*input_size
        edge_size = alphaN*input_channel_size
        C = np.array([edge_size,edge_size])
        for i in range(edge_size):
            for j in range(i,edge_size):
                if i == j:
                    C[i][j] = 1
                else:
                    C[i][j] = (-rho + 2*rho*np.random.rand())/np.sqrt(input_channel_size*kernel_size*kernel_size)
        # keep it symmetric 
        for i in range(edge_size):
            for j in range(i):
                C[i][j] = C[j][i]
        # transfer it to 4D tensor format
        self.CTensor = np.array([alphaN,alphaN,input_channel_size,input_channel_size])
        for a in range(alphaN):
            for a_ in range(alphaN):
                for i in range(input_channel_size):
                    for j in range(input_channel_size):
                        idx_i = i*alphaN + a
                        idx_j = j*alphaN + a_
                        self.CTensor[a][a_][i][j] = C[idx_i][idx_j]

    def UpdateDeltaTensor(self):
        Cl = self.struct[self.currentLayerIdx]
        Clplus1 = self.struct[self.currentLayerIdx + 1]
        current_size = int(np.sqrt(self.CTensor.size(0)))
        delta_size = (current_size - self.kernel_size + 1 )*(current_size - self.kernel_size + 1 )# alpha range
        self.Delta = np.zeros(delta_size,delta_size,Clplus1,Clplus1)
        # each loop update a matrix of size [C(l+1),C(l+1)]
        for alpha in range(delta_size):
            for alpha_ in range(delta_size):
                alphai = alpha//(current_size - self.kernel_size + 1 )
                alphaj = alpha%(current_size - self.kernel_size + 1 )
                alpha_i = alpha_//(current_size - self.kernel_size + 1 )
                alpha_j = alpha_%(current_size - self.kernel_size + 1 )
                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                         for i_ in range(self.kernel_size):
                             for j_ in range(self.kernel_size):
                                c_tensor_i = (alphai+i)*current_size + alphaj + j   # shrink 2D into 1D
                                c_tensor_j = (alpha_i+i_)*current_size + alpha_j + j_
                                self.Delta[alpha][alpha_] += np.dot(np.dot(self.weight_inner[self.currentLayerIdx+1][i][j][0],self.CTensor[c_tensor_i][c_tensor_j]),self.weight_inner[self.currentLayerIdx+1][i_][j_][0].T)
    """
    def PhiCrossMult(self,x,y,A,B,C,A_,B_,C_):
        return self.phi(A*x+B+C)*self.phi(A_*x+B_+C_)
    """
    def IterateOneLayer(self):
        # 前进一层
        current_size = int(np.sqrt(self.CTensor.size(0)))
        layer_next_size = current_size - self.kernel_size + 1
        layer_next_size2 = layer_next_size*layer_next_size # shrink to 1D
        Cin = self.struct[self.currentLayerIdx]
        Cout = self.struct[self.currentLayerIdx+1]
        self.CTensorNext = np.array([layer_next_size2,layer_next_size2,Cin,Cin])
        self.h_mean_next = []
        for i in range(layer_next_size2):
            h_mean_next.append(np.zeros(self.struct[self.currentLayerIdx+1],1))
        # 将alpha 拆成2D alphai alphaj二维坐标
        for alphai in range(layer_next_size):
            for alphaj in range(layer_next_size):
                for alpha_i in range(layer_next_size):
                    for alpha_j in range(layer_next_size): # above loops for neurals positions
                        X = np.random.randn(Cout,20000)
                        for i in range(Cout):
                            x = X[i,:]
                            # second term in phi
                            tmp = 0
                            for m in range(self.kernel_size):
                                for n in range(self.kernel_size):
                                    tmp += np.dot(self.weight_inner[self.currentLayerIdx+1][m][n][0],self.h_mean[(alphai+m)*current_size+(alphaj+n)])[i][0]
                            Phi1 = self.phi(self.Delta[alpha][alpha_][i][i]*x + tmp + self.bias[self.currentLayerIdx+1][i][0])
                            # update mean h first
                            self.h_mean_next[alphai*layer_next_size + alphaj][i][0] = Phi1.sum()/20000
                        # update C matrix
                        for i in range(Cout):
                            for j in range(Cout):
                                x = X[i,:]
                                y = np.random.randn(1,20000)
                                alpha = alphai*layer_next_size + alphaj
                                alpha_ = alpha_i*layer_next_size + alpha_j
                                # second term in phi_
                                tmp_ = 0
                                for m in range(self.kernel_size):
                                    for n in range(self.kernel_size):
                                        tmp_ += np.dot(self.weight_inner[self.currentLayerIdx+1][m][n][0],self.h_mean[(alpha_i+m)*current_size+(alpha_j+n)])[j][0]
                                PHI = self.Delta[alpha][alpha_][i][j]/np.sqrt(self.Delta[alpha][alpha][i][i]*self.Delta[alpha_][alpha_][j][j])
                                A = (Phi1*self.phi(self.Delta[alpha_][alpha_][j][j]*(PHI*x+np.sqrt(1-PHI*PHI)*y) + tmp_ + self.bias[self.currentLayerIdx+1][j][0])).sum()/20000
                                B = self.h_mean_next[alpha][i][0]*self.h_mean_next[alpha_][j][0]
                                self.CTensorNext[alpha][alpha_][i][j] = A - B
        self.currentLayerIdx = self,currentLayerIdx + 1
        self.CTensor = self.CTensorNext.copy()
        self.h_mean = self.h_mean_next.copy()
        




        






