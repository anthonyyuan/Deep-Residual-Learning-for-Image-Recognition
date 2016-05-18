#Residual network implementation

##Original Residual newtork(http://arxiv.org/pdf/1512.03385v1)
original residual network implementation with torch
20_plain, 20_res, 56_plain, 56_res are implemented using CIFAR-10
Yields the same performance as the paper

Basic residual block design and shortcut option A is used

##Result
![loss](https://cloud.githubusercontent.com/assets/13601723/15349154/a6a2b398-1d0b-11e6-9ee8-0030a2ffd40d.png)

Test error | me | paper
---------- | ----------- | -----------
20_plain | 9.5% | 
20_res | 8.8% | 8.75%
56_plain | 13.4% | 
56_res | 7% | 6.97%
