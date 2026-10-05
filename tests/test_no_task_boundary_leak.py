import inspect
from ccdn.streams.permuted_mnist import PermutedMNISTStream
from ccdn.models.dense_mlp import DenseMLP
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.baselines import StaticDense,StaticSparse,SelectiveReset,ContinualBackprop,RigL
from ccdn.algorithms.ccdn_0a import CCDN0A

FORBIDDEN={"task_id","permutation_id","task_boundary","new_task"}

def test_all_baseline_training_hooks_reject_boundary_metadata():
    dense=DenseMLP(input_size=4,hidden_sizes=(3,))
    sparse=SparseMLP(input_size=4,hidden_sizes=(3,),density=.5)
    pairs=[(StaticDense(dense),dense),(StaticSparse(sparse),sparse)]
    for cls in (SelectiveReset,): pairs.append((cls(SparseMLP(input_size=4,hidden_sizes=(3,),density=.5)),None))
    for cls in (ContinualBackprop,): pairs.append((cls(DenseMLP(input_size=4,hidden_sizes=(3,))),None))
    for cls in (RigL,): pairs.append((cls(SparseMLP(input_size=4,hidden_sizes=(3,),density=.5)),None))
    pairs.append((ContinualBackprop(SparseMLP(input_size=4,hidden_sizes=(3,),density=.5)),None))
    pairs.append((CCDN0A(SparseMLP(input_size=4,hidden_sizes=(3,),density=.5)),None))
    for algorithm,_ in pairs:
        for callback in (algorithm.before_backward,algorithm.after_backward,algorithm.after_optimizer_step):
            assert not (FORBIDDEN & set(inspect.signature(callback).parameters)), (type(algorithm).__name__,callback)

def test_learner_facing_batch_is_exactly_features_and_labels():
    stream=PermutedMNISTStream(permutations=1,batches_per_permutation=1,batch_size=2,synthetic=True)
    evaluator_permutation,batches,_,_=next(iter(stream.segments()))
    batch=next(iter(batches))
    assert isinstance(evaluator_permutation,int)
    assert isinstance(batch,tuple) and len(batch)==2
    x,y=batch
    assert x.shape==(2,784) and y.shape==(2,)
