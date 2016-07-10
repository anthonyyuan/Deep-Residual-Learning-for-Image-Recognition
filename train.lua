require 'torch'
require 'optim'
require 'xlua'
require 'image'
dofile 'etc.lua'

params, gradParams = model:getParameters()
optimState = {
    learningRate = lr,
    learningRateDecay = 0.0,
    weightDecay = wDecay,
    momentum = mmt,
    dampening = 0,
    nesterov = true
}
optimMethod = optim.sgd
classes = {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"}
confusion = optim.ConfusionMatrix(classes)
tot_error = 0
cnt_error = 0
tot_iter = 0

function train(trainData, trainLabel)
    local time = sys.clock()
    
    tot_error = 0
    cnt_error = 0

    model:training()
    shuffle = torch.randperm(trainSz)
    
    local inputs = torch.CudaTensor(batchSz,inputDim,inputSz,inputSz)
    local targets = torch.CudaTensor(batchSz,1)

    for t = 1,trainSz,batchSz do
        
        --xlua.progress(t,trainSz)

        if t+batchSz-1 > trainSz then
            inputs = torch.CudaTensor(trainSz-t+1,inputDim,inputSz,inputSz)
            targets = torch.CudaTensor(trainSz-t+1,1)
            curBatchDim = trainSz-t+1
        else
            curBatchDim = batchSz
        end

        for i = t,math.min(t+batchSz-1,trainSz) do
            
            local input = trainData[shuffle[i]]
            local target = trainLabel[shuffle[i]]
            input = torch.reshape(input,inputDim,inputSz,inputSz)
          
            --augmentation
            padded = torch.Tensor(inputDim,(inputSz+2*padSz),(inputSz+2*padSz)):zero()
            padded[{{},{padSz+1,inputSz+padSz},{padSz+1,inputSz+padSz}}] = input
            ltX = math.random(2*padSz+1)
            ltY = math.random(2*padSz+1)
            flipFlag = math.random(2)
            if flipFlag == 1 then
                padded = image.flip(padded,3)
            end
            input = padded[{{},{ltY,ltY+inputSz-1},{ltX,ltX+inputSz-1}}]
          
            --[===[
            if t==1 and i<t+20 then
                img = torch.Tensor(3,inputSz,inputSz)
                img[1] = input[1]
                img[2] = input[2]
                img[3] = input[3]
                image.save(tostring(i) .. ".jpg",img)
            end
            --]===]
            
            inputs[i-t+1]:copy(input)
            targets[i-t+1]:copy(target)
        end
        
        if tot_iter == 32e3 or tot_iter == 48e3 then
            optimState.learningRate = optimState.learningRate/10
        end
        
             local feval = function(x)
                       if x ~= params then
                          params:copy(x)
                       end

                       gradParams:zero()
                    
                       local f = 0
                       local output = model:forward(inputs)
                       local df_do = torch.CudaTensor(curBatchDim,outputDim)
                       for i = 1,curBatchDim do
                        local err = criterion:forward(output[i], targets[i])
                          f = f + err

                          df_do[i]:copy(criterion:backward(output[i], targets[i]))
                          
                          conf_target = torch.Tensor(outputDim):zero()
                          id_ = targets[i]
                          conf_target[id_[1]] = 1
                          confusion:add(output[i], conf_target)
                       end
                       model:backward(inputs,df_do)
                       gradParams:div(curBatchDim)
                       f = f/curBatchDim
                       tot_error = tot_error + f
                       cnt_error = cnt_error + 1
                       return f,gradParams
                    end

         optimMethod(feval, params, optimState)
        
        if tot_iter % 100 == 0 then
            print("iteration: " .. tot_iter .. "/" .. tostring(iterLimit) .. " batch: " ..  t .. "/" .. trainSz .. " loss: " .. tot_error/cnt_error)
        end
        tot_iter = tot_iter + 1


        time = sys.clock() - time
        time = time / trainSz
        --print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

        --print(confusion, tot_iter)
        if tot_iter == iterLimit then
            local filename = paths.concat(save_dir, 'model.net')
            os.execute('mkdir -p ' .. sys.dirname(filename))
            print('==> saving model to '..filename)
            torch.save(filename, model)
        end

        confusion:zero()
    end

end



