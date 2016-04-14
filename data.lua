require "torch"
require "image"
dofile "etc.lua"


function load_data(mode)
   
    local data
    local label

    if mode == "train" then
        fnum = 5
        data = torch.FloatTensor(trainSz, inputSz*inputSz*inputDim)
        label = torch.ByteTensor(trainSz, 1):zero()
    elseif mode == "test" then
        fnum = 1
        data = torch.FloatTensor(testSz, inputSz*inputSz*inputDim)
        label = torch.ByteTensor(testSz, 1):zero()
    end

    did = 1
    for fid = 1,fnum do
        
        if mode == "train" then
            fname = db_dir .. "data_batch_" .. tostring(fid) .. ".bin"
            print("trainDB loading... " .. tostring(fid) .. "/" .. tostring(fnum))
        elseif mode == "test" then
            fname = db_dir .. "test_batch.bin"
            print("testDB loading... " .. tostring(fid) .. "/" .. tostring(fnum))
        end

        local fp = torch.DiskFile(fname,"r"):binary()
        fp:seekEnd()
        local len = fp:position()-1
        local lineNum = len/(inputSz*inputSz*inputDim+1)
        fp:seek(1)

        for lid = 1,lineNum do
            label_tmp = fp:readByte() --0~9
            label_tmp = label_tmp + 1
            label[did] = label_tmp
            local red = torch.ByteTensor(fp:readByte(inputSz*inputSz)):type('torch.FloatTensor')
            local green = torch.ByteTensor(fp:readByte(inputSz*inputSz)):type('torch.FloatTensor')
            local blue = torch.ByteTensor(fp:readByte(inputSz*inputSz)):type('torch.FloatTensor')
            

            data[{{did},{1,inputSz*inputSz}}] = red
            data[{{did},{inputSz*inputSz+1,2*inputSz*inputSz}}] = green
            data[{{did},{2*inputSz*inputSz+1,3*inputSz*inputSz}}] = blue
            did = did + 1
            
            --[===[
            if did<20 then
                img = torch.Tensor(3,32,32)
                img[1] = torch.reshape(red,32,32)
                img[2] = torch.reshape(blue,32,32)
                img[3] = torch.reshape(green,32,32)
                image.save(tostring(did) .. ".jpg",img)
                print(label_tmp)
            end
            --]===]
        end
        
        fp:close()
    end

    for i=1,data:size()[1] do
        data[{{i},{1,inputSz*inputSz}}] = (data[{{i},{1,inputSz*inputSz}}] - 125.3)/63.0
        data[{{i},{inputSz*inputSz+1,2*inputSz*inputSz}}] = (data[{{i},{inputSz*inputSz+1,2*inputSz*inputSz}}] - 123.0)/62.1
        data[{{i},{2*inputSz*inputSz+1,3*inputSz*inputSz}}] = (data[{{i},{2*inputSz*inputSz+1,3*inputSz*inputSz}}] - 113.9)/66.7
    end

    return data, label
end

