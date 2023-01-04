# DEC
Dec is a Deep embedding Cluster to find out is the low dim help the Cluster well and the reconstruction  distance between the originl data 
1. To pretrain an auto encoder  get representation of the data 
2. Use the Pretrain data set run kmeans 
3. Calculate loss the original data BEC(Binary Cross Entropy) and reconstuction data distance SSE(Error Sum of Squares )
## 1. Net AutoEncoder

``` =python 
 self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 10),
        )
```
