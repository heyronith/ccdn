import torch
from ccdn.streams.permuted_mnist import PermutedMNISTStream

def test_permutation_determinism_and_variation():
    a=PermutedMNISTStream(seed=7,permutations=2,batches_per_permutation=1,batch_size=4,synthetic=True)
    b=PermutedMNISTStream(seed=7,permutations=2,batches_per_permutation=1,batch_size=4,synthetic=True)
    assert torch.equal(a.permutation(0),b.permutation(0))
    assert not torch.equal(a.permutation(0),a.permutation(1))

def test_batches_expose_only_features_and_labels():
    s=PermutedMNISTStream(seed=1,permutations=1,batches_per_permutation=1,batch_size=3,synthetic=True)
    _,batches,_,_=next(iter(s.segments())); batch=next(iter(batches))
    assert isinstance(batch,tuple) and len(batch)==2
    x,y=batch; assert x.shape==(3,784) and y.shape==(3,)
