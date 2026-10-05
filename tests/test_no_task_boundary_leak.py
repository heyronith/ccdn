import inspect
from ccdn.streams.permuted_mnist import PermutedMNISTStream
from ccdn.baselines.selective_reset import SelectiveReset

def test_learner_stream_surface_and_hooks_have_no_boundary_arguments():
    assert "task_id" not in inspect.signature(PermutedMNISTStream._loader).parameters
    assert "task_boundary" not in inspect.signature(SelectiveReset.after_optimizer_step).parameters
    stream=PermutedMNISTStream(permutations=1,batches_per_permutation=1,batch_size=2,synthetic=True)
    _,batches,_,_=next(iter(stream.segments()))
    assert len(next(iter(batches)))==2
