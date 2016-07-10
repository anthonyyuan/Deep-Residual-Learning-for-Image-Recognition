require 'torch'
require 'xlua' 
require 'optim'


function test(testData, testLabel)
    
    if mode == "test" then
        print("model loading...")
        model = torch.load(save_dir .. modelName)
    end

    model:evaluate()

    print('==> testing:')
    
    for t = 1,testSz,testBatchSz do
        
        local input = testData[{{t,t+testBatchSz-1},{}}]
        local target = testLabel[{{t,t+testBatchSz-1}}]
        input = input:cuda()
        input = torch.reshape(input,testBatchSz,inputDim,inputSz,inputSz)
        local pred = model:forward(input)
        conf_target = torch.Tensor(testBatchSz,outputDim):zero()
        for i = 1,testBatchSz do
            conf_target[i][target[i][1]] = 1
            confusion:add(pred[i],conf_target[i])
        end
    end


    print(confusion)
    confusion:zero()
end
