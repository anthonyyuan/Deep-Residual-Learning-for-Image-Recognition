require 'torch'
require 'nn'
require 'cudnn'
require 'module/normalConv'
require 'module/normalLinear'
dofile 'etc.lua'

model = nn.Sequential()
fNum = {inputDim,16,32,64}

model:add(cudnn.normalConv(inputDim,16,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*inputDim))))
model:add(nn.SpatialBatchNormalization(16,false))
model:add(nn.ReLU(true))

prevDim = 16
subModel = nn.Sequential()

for lid = 1,6*n do
   if lid%(2*n) == 1 and lid>1 then
       iDim = fNum[math.floor(lid/(2*n))+1]
       oDim = fNum[math.floor(lid/(2*n))+2]
       str = 2
   else
       iDim = fNum[math.floor(lid/(2*n+1))+2]
       oDim = fNum[math.floor(lid/(2*n+1))+2]
       str = 1
   end
   
    subModel:add(cudnn.normalConv(iDim,oDim,kernelSz,kernelSz,str,str,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*iDim))))
    subModel:add(nn.SpatialBatchNormalization(oDim,false))
    
    if lid%2 == 0 then
        curDim = oDim
        if prevDim == curDim then
            shortCut = nn.Identity()
        else
            shortCut = nn.Sequential():add(nn.SpatialAveragePooling(1,1,2,2))
            shortCut:add(nn.Concat(2):add(nn.Identity()):add(nn.MulConstant(0)))
        end
        
        if isRes then
            concatTable = nn.ConcatTable()
            concatTable:add(subModel):add(shortCut)
            model:add(nn.Sequential():add(concatTable):add(nn.CAddTable(true)))
       else
           model:add(subModel)
       end

        prevDim = oDim
        subModel = nn.Sequential()
    end
    
    if subModel:size() > 0 then
        subModel:add(nn.ReLU(true))
    else
        model:add(nn.ReLU(true))
    end

end


model:add(nn.SpatialAveragePooling(inputSz/4,inputSz/4))

model:add(nn.View(64):setNumInputDims(3))
model:add(nn.normalLinear(64,outputDim,0,math.sqrt(2/64)))


model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

print(model)

cudnn.convert(model, cudnn)

model:cuda()
criterion:cuda()
