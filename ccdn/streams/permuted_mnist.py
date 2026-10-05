from __future__ import annotations
import torch
from torch.utils.data import DataLoader, TensorDataset

class PermutedMNISTStream:
    """Evaluator-owned stream; learners consume only (x, y) batches."""
    def __init__(self,root="./data",seed=1,permutations=20,batches_per_permutation=20,batch_size=128,synthetic=False,download=True):
        self.seed=int(seed); self.permutations=int(permutations); self.batches=int(batches_per_permutation); self.batch_size=int(batch_size); self.synthetic=synthetic
        if not synthetic:
            from torchvision.datasets import MNIST
            from torchvision.transforms import ToTensor
            self.train=MNIST(root,train=True,download=download,transform=ToTensor())
            self.test=MNIST(root,train=False,download=download,transform=ToTensor())
    def permutation(self,index):
        g=torch.Generator().manual_seed(self.seed+int(index)); return torch.randperm(784,generator=g)
    def _synthetic_dataset(self,train):
        g=torch.Generator().manual_seed(self.seed+(11 if train else 19))
        n=max(self.batch_size*self.batches,512 if not train else 0)
        x=torch.rand(n,1,28,28,generator=g); y=torch.randint(0,10,(n,),generator=g)
        return TensorDataset(x,y)
    def _loader(self,dataset,perm,shuffle=False):
        gen=torch.Generator().manual_seed(self.seed+int(perm)+100)
        return DataLoader(dataset,batch_size=self.batch_size,shuffle=shuffle,generator=gen)
    def segments(self):
        train_ds=self._synthetic_dataset(True) if self.synthetic else self.train
        test_ds=self._synthetic_dataset(False) if self.synthetic else self.test
        for k in range(self.permutations):
            p=self.permutation(k)
            def batches():
                for j,(x,y) in enumerate(self._loader(train_ds,k,True)):
                    if j>=self.batches: break
                    yield x.reshape(x.shape[0],-1)[:,p],y
            yield k,batches(),self._loader(test_ds,k,False),p

def permute_batch(x, permutation): return x.reshape(x.shape[0],-1)[:,permutation]
