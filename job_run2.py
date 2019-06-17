from DynamicMeanField import *

# initialize random network
net_struct = [3,3]
net = DMFNet(net_struct,3,input_size = 6,detailed_info = True)
lengthOfLayers = len(net_struct)

# feed forward [lengthOfLayers-1] steps
# print out the heat graph of Corvariance if needed by adding the 'PrintCorvarianceHeatMap()' function
for l in range(lengthOfLayers-1):
    net.UpdateDimensionality()
    net.PrintCorvarianceHeatMap()
    net.UpdateDeltaTensor()
    net.IterateOneLayer()
net.UpdateDimensionality()
net.PrintCorvarianceHeatMap()

# print out the dimensionality info
net.PrintDimInfo()




