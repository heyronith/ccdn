import pytest
import ccdn.experiments.train as train

class FakeStream:
    def __init__(self,**kwargs): self.kwargs=kwargs

def test_explicit_synthetic_mode_never_attempts_mnist(monkeypatch):
    calls=[]
    def factory(**kwargs): calls.append(kwargs); return FakeStream(**kwargs)
    monkeypatch.setattr(train,'PermutedMNISTStream',factory)
    stream,fallback,reason=train.create_stream({'synthetic':True,'allow_synthetic_fallback':False},11)
    assert stream.kwargs['synthetic'] is True and not fallback and reason is None
    assert len(calls)==1

def test_permitted_fallback_is_explicit_and_recordable(monkeypatch):
    calls=[]
    def factory(**kwargs):
        calls.append(kwargs)
        if not kwargs['synthetic']: raise OSError('MNIST unavailable')
        return FakeStream(**kwargs)
    monkeypatch.setattr(train,'PermutedMNISTStream',factory)
    stream,fallback,reason=train.create_stream({'synthetic':False,'allow_synthetic_fallback':True},11)
    assert fallback and 'MNIST unavailable' in reason and stream.kwargs['synthetic'] is True
    assert [x['synthetic'] for x in calls]==[False,True]

def test_scientific_mode_aborts_when_mnist_fails(monkeypatch):
    calls=[]
    def factory(**kwargs): calls.append(kwargs); raise OSError('MNIST unavailable')
    monkeypatch.setattr(train,'PermutedMNISTStream',factory)
    with pytest.raises(OSError,match='MNIST unavailable'):
        train.create_stream({'synthetic':False,'allow_synthetic_fallback':False},11)
    assert len(calls)==1 and calls[0]['synthetic'] is False
