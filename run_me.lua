require 'torch'
require 'cunn'
require "data.lua"
dofile "etc.lua"
    

local trainData
local trainLabel
local testData
local testLabel


torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())

dofile "data.lua"
dofile "model.lua"
dofile "train.lua"
dofile "test.lua"


if mode == "train" then
    trainData, trainLabel = load_data("train")
    testData, testLabel = load_data("test")
    fp_err = io.open("result/loss.txt","a")
    while tot_iter <= iterLimit do
        train(trainData, trainLabel)
        err = tot_error/cnt_error
        fp_err:write(err,"\n")
        test(testData, testLabel)
    end
    fp_err:close()
end


if mode == "test" then
    testData, testLabel = load_data("test")
    test(testData, testLabel)
end
