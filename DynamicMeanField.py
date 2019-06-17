import numpy as np
import matplotlib.pyplot as plt

# CTensor[alpha][alpha'][i][j]: correlation between position alpha of channel i and position alpha' of channel j
# CTensor generated from Big correlation 2D matrix where channel and input size are merged  
# layer index starts from 0 which correponds to the input layer
"""
# Net work feedward steps:
#   1. Initialize C tensor and mean activations by constructor
#   2. Extract the dimensionality list of current layer including block matrix and Big Matrix
#   3. Calculate Delta tensor
#   4. Iterate one layer ahead
#   5. repeat above steps until reaching the last layer
"""
class DMFNet:

    ############# PARAM #############
    g_w = 0.8
    rho = 0.1
    sigma_b = 0.1
    ############# PARAM #############
    
    def __init__(self, struct,kernel_size,phi = np.tanh,input_size = 30, weight = None, detailed_info = False):
        self.struct = struct
        self.kernel_size = kernel_size
        self.layer_len = len(struct)
        self.currentLayerIdx = 0
        self.phi = phi
        self.detailed_info = detailed_info
        # General weight formulation: weight[layer idx][Cout][Cin] is a list containing the kernel size matrix (has only one element)
        # layer idx >= 1
        if weight == None:
            self.weight = []
            self.weight.append('0')
            for n in range(self.layer_len - 1):
                C_in = struct[n]
                C_out = struct[n+1]
                weightPerLayer = [[[] for j in range(C_in)] for i in range(C_out)]
                self.weight.append(weightPerLayer) 
            self.InitialWeight()
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
                    wtmp = np.zeros([C_out,C_in])
                    for i in range(C_out):
                        for j in range(C_in):
                            wtmp[i][j] = self.weight[n][i][j][0][betai][betaj]
                    weight_inner_layer[betai][betaj].append(wtmp)
            self.weight_inner.append(weight_inner_layer)
        # Input corvariance tensor
        self.InitialInputCorvarianceTensor(DMFNet.rho,input_size)
        # initial delta tensor
        self.UpdateDeltaTensor()
        self.InitialMeanActivation(input_size)
        # initial bias
        self.bias = ['0']
        self.InitialBias()
        self.dimList = []   # store the dimensionality per feature map
        self.dimBigMatrix = []  # store the dimensionality of each layer's big matrix
    
    def PrintCorvarianceHeatMap(self, savePath = './RandomNet/HeatMap/'):
        # save the heat map of big matrix   ||  called before updating the dimensionality
        path = savePath + 'Layer' + str(self.currentLayerIdx) + '.jpg'
        plt.figure()
        plt.imshow(self.CBigMat)  
        plt.colorbar()
        plt.savefig(path) 

    def InitialWeight(self):
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
        if self.detailed_info:
            print('Bias initialization complete')
    
    def InitialMeanActivation(self,input_size = 30):
        # a list of length input_size*input_size and each element is a column vector of size [input_size,1]
        self.h_mean = []
        h_mean_alpha = np.zeros([self.struct[0],1])
        for i in range(input_size*input_size):
            self.h_mean.append(h_mean_alpha)
        if self.detailed_info:
            print('Mean activation complete')

    def InitialInputCorvarianceTensor(self,rho,input_size = 30):
        # Big matrix 
        input_channel_size = self.struct[0] 
        alphaN = input_size*input_size
        edge_size = alphaN*input_channel_size
        C = np.zeros([edge_size,edge_size])
        for i in range(edge_size):
            for j in range(i,edge_size):
                if i == j:
                    C[i][j] = 1
                else:
                    C[i][j] = (-rho + 2*rho*np.random.rand())/np.sqrt(input_channel_size*self.kernel_size*self.kernel_size)
        # keep it symmetric 
        for i in range(edge_size):
            for j in range(i):
                C[i][j] = C[j][i]
        # transfer it to 4D tensor format
        self.CTensor = np.zeros([alphaN,alphaN,input_channel_size,input_channel_size])
        for a in range(alphaN):
            if self.detailed_info:
                alphaNfifth = int(alphaN/5)
                if a%alphaNfifth == 0:
                    print('Corvariance Tensor assembly process: {:.0f}%'.format(100*a/alphaN))
            for a_ in range(alphaN):
                for i in range(input_channel_size):
                    for j in range(input_channel_size):
                        idx_i = i*alphaN + a
                        idx_j = j*alphaN + a_
                        self.CTensor[a][a_][i][j] = C[idx_i][idx_j]
        if self.detailed_info:
            print('Corvariance Tensor assembly complete')

    def UpdateDeltaTensor(self):
        Cl = self.struct[self.currentLayerIdx]
        Clplus1 = self.struct[self.currentLayerIdx + 1]
        current_size = int(np.sqrt(self.CTensor.shape[0]))
        delta_size = (current_size - self.kernel_size + 1 )*(current_size - self.kernel_size + 1 )# alpha range
        self.Delta = np.zeros([delta_size,delta_size,Clplus1,Clplus1])
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
        if self.detailed_info:
            print('Current layer:   {}'.format(self.currentLayerIdx))
        # 前进一层
        current_size = int(np.sqrt(self.CTensor.shape[0]))
        layer_next_size = current_size - self.kernel_size + 1
        layer_next_size2 = layer_next_size*layer_next_size # shrink to 1D
        Cin = self.struct[self.currentLayerIdx]
        Cout = self.struct[self.currentLayerIdx+1]
        self.CTensorNext = np.zeros([layer_next_size2,layer_next_size2,Cout,Cout])
        self.h_mean_next = []
        for i in range(layer_next_size2):
            self.h_mean_next.append(np.zeros([self.struct[self.currentLayerIdx+1],1]))
        # 将alpha 拆成2D alphai alphaj二维坐标
        # first 4 loops for feature map's positions of layer l+1
        for alphai in range(layer_next_size):
            if self.detailed_info:
                layer_next_sizefifth = int(layer_next_size/5)
                if alphai%layer_next_sizefifth == 0:
                    print('Iterating process: {:.0f}%'.format(100*alphai/layer_next_size))
            for alphaj in range(layer_next_size):
                # update mean h at position alpha
                alpha = alphai*layer_next_size + alphaj
                for i in range(Cout):
                    x = np.random.randn(1,20000)
                    # second term in phi
                    tmp = 0
                    for m in range(self.kernel_size):
                        for n in range(self.kernel_size):
                            # i stands for ith element in layer l+1
                            # self.weight_inner[self.currentLayerIdx+1][m][n][0]: C(l+1)*C(l) size 
                            tmp += np.dot(self.weight_inner[self.currentLayerIdx+1][m][n][0],self.h_mean[(alphai+m)*current_size+(alphaj+n)])[i][0]
                    Phi1 = self.phi(np.sqrt(self.Delta[alpha][alpha][i][i])*x + tmp + self.bias[self.currentLayerIdx+1][i][0])
                    self.h_mean_next[alpha][i][0] = Phi1.sum()/20000
        for alphai in range(layer_next_size):
            for alphaj in range(layer_next_size):
                alpha = alphai*layer_next_size + alphaj
                # calculate corvariance matrix C(alpha,alpha',i,j) 
                # 将alpha' 拆成2D alpha_i alpha_j二维坐标
                for alpha_i in range(layer_next_size):
                    for alpha_j in range(layer_next_size): # above loops for neurals positions
                        # this loop stands for alpha prime index
                        alpha_ = alpha_i*layer_next_size + alpha_j
                        for i in range(Cout):
                            # i indicates the same index of output layer l+1 with m(alpha,i)
                            x = np.random.randn(1,20000)
                            for j in range(Cout):
                                y = np.random.randn(1,20000)
                                # second term in phi_
                                tmp_ = 0
                                tmp = 0
                                for m in range(self.kernel_size):
                                    for n in range(self.kernel_size):
                                        tmp += np.dot(self.weight_inner[self.currentLayerIdx+1][m][n][0],self.h_mean[(alphai+m)*current_size+(alphaj+n)])[i][0]
                                        tmp_ += np.dot(self.weight_inner[self.currentLayerIdx+1][m][n][0],self.h_mean[(alpha_i+m)*current_size+(alpha_j+n)])[j][0]
                                PHI = self.Delta[alpha][alpha_][i][j]/np.sqrt(self.Delta[alpha][alpha][i][i]*self.Delta[alpha_][alpha_][j][j])
                                A = (self.phi(np.sqrt(self.Delta[alpha][alpha][i][i])*x + tmp + self.bias[self.currentLayerIdx+1][i][0])*self.phi(np.sqrt(self.Delta[alpha_][alpha_][j][j])*(PHI*x+np.sqrt(1-PHI*PHI)*y) + tmp_ + self.bias[self.currentLayerIdx+1][j][0])).sum()/20000
                                B = self.h_mean_next[alpha][i][0]*self.h_mean_next[alpha_][j][0]
                                self.CTensorNext[alpha][alpha_][i][j] = A - B
        self.currentLayerIdx = self.currentLayerIdx + 1
        self.CTensor = self.CTensorNext.copy()
        self.h_mean = self.h_mean_next.copy()
    
    # extract the dimensionality of current layer
    # dimList[layer index][channel index] is the normalized dimensionality in layer [layer index] at channel [channel index]
    def UpdateDimensionality(self):
        # dimensionality in each layer and the dimensionality of the Big Matrix
        # Called before iterating to the next layer
        num_current_channel = self.struct[self.currentLayerIdx]
        num_feature_map = self.CTensor.shape[0]
        dimPerLayer = []
        for c in range(num_current_channel):
            CMat = self.CTensor[:,:,c,c]
            Ctrace = np.trace(CMat)
            dim = Ctrace*Ctrace/np.trace(np.dot(CMat,CMat))/num_feature_map
            dimPerLayer.append(dim)
        self.dimList.append(dimPerLayer)
        # dimensionality of Big Matrix
        big_matrix_edge = num_current_channel*num_feature_map
        self.CBigMat = np.zeros([big_matrix_edge,big_matrix_edge])
        for i in range(num_current_channel):
            for j in range(num_current_channel):
                for alpha in range(num_feature_map):
                    for alpha_ in range(num_feature_map):
                        I = i*num_feature_map + alpha
                        J = j*num_feature_map + alpha_
                        self.CBigMat[I][J] = self.CTensor[alpha][alpha_][i][j]
        Ctrace = np.trace(self.CBigMat)
        dim = Ctrace*Ctrace/np.trace(np.dot(self.CBigMat,self.CBigMat))/big_matrix_edge
        self.dimBigMatrix.append(dim)
    
    # print out dimensionality information layer by layer
    def PrintDimInfo(self):
        print('Block matrix dimensionality:')
        for l in range(self.layer_len):
            print('Layer {}'.format(l))
            for c in range(self.struct[l]):
                print('Channel {0} has dimensionality {1} '.format(c,self.dimList[l][c]))
        print('Big Matrix dimensionality:')
        for l in range(self.layer_len):
            print('Layer {0} has total dimensionality {1}'.format(l,self.dimBigMatrix[l]))


    

            
            
            




        






