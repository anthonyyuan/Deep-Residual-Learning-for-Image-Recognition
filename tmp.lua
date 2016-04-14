require "torch"
require "optim"
require 'nn'
require 'image'
require 'cunn'

a = torch.Tensor(3):fill(1)
print(torch.sum(a))
