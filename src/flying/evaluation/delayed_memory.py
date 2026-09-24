"""Decode past iid digits from current states; independent train/test streams."""
import numpy as np
from flying.models.ridge import RidgeDecoder

def delayed_labels(digits,lags,warmup):
    digits=np.asarray(digits); lags=np.asarray(lags,dtype=int)
    if lags.ndim!=1 or not len(lags) or np.any(lags<0) or warmup<lags.max() or warmup>=len(digits):
        raise ValueError('Invalid lags/warmup')
    return digits[np.arange(warmup,len(digits))[:,None]-lags[None,:]]

def decode_delays(train_states,test_states,train_digits,test_digits,lags,warmup,alpha=1.):
    train=delayed_labels(train_digits,lags,warmup); test=delayed_labels(test_digits,lags,warmup)
    if len(train_states)!=len(train_digits) or len(test_states)!=len(test_digits):
        raise ValueError('States must align with inputs')
    y=np.eye(10)[train].reshape(len(train),-1)
    decoder=RidgeDecoder(alpha).fit(train_states[warmup:],y)
    scores=decoder.scores(test_states[warmup:]).reshape(len(test),len(lags),10)
    predictions=scores.argmax(axis=2); truth=np.eye(10)[test]
    # R² relative to the TRAINING class-frequency predictor; can be negative.
    baseline=decoder.target_mean.reshape(len(lags),10)
    denom=((truth-baseline)**2).sum(axis=(0,2))
    r2=1-((truth-scores)**2).sum(axis=(0,2))/denom
    majority=baseline.argmax(axis=1)
    metrics=[dict(lag=int(lag),accuracy=float(np.mean(predictions[:,j]==test[:,j])),
                  frequency_baseline_accuracy=float(np.mean(majority[j]==test[:,j])),
                  r2_vs_training_frequency=float(r2[j])) for j,lag in enumerate(lags)]
    return metrics,predictions.astype(np.uint8),test.astype(np.uint8)
