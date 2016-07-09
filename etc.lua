isRes = false
mode = "train"
modelName = "model_res_56.net"



db_dir = "/media/sda1/Study/Data/CIFAR10/cifar-10-batches-bin/"
save_dir = db_dir .. "model_save/"

trainSz = 50000
testSz = 10000
inputSz = 32
inputDim = 3
outputDim = 10
kernelSz = 3
padSz = 4
n = 9

lr = 1e-1
wDecay = 1e-4
mmt = 9e-1
batchSz = 128
testBatchSz = 2e3
iterLimit = 64e3

