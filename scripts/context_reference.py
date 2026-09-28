"""Independent scan/recount oracle. Does not import the fitted-table implementation."""
from collections import Counter
import numpy as np


def infer(sequence, weights, k, order, history):
    for n in reversed(range(min(order,len(history))+1)):
        context=list(history[-n:]) if n else []
        matches=[t for t in range(1,len(sequence)) if t>=n and list(sequence[t-n:t])==context]
        if matches:
            counts=[sum(float(weights[t-1]) for t in matches if sequence[t]==symbol) for symbol in range(k)]
            total=sum(counts)
            return counts.index(max(counts)),[value/total for value in counts],n
    raise AssertionError('No fallback')


def verify(record,k,order):
    s=record['symbols']; w=record['weights']; h=list(s[:3]); pred=[]; probs=[]; lengths=[]
    for _ in range(125):
        y,p,n=infer(s,w,k,order,h); pred.append(y); probs.append(p); lengths.append(n); h.append(y)
    assert pred==record['generated'] and lengths==record['used_context_lengths']
    np.testing.assert_array_equal(probs,record['probabilities'])
    teacher=[infer(s,w,k,order,s[:t]) for t in range(1,len(s))]
    assert [p[0] for p in teacher]==record['teacher']
    np.testing.assert_array_equal([p[1] for p in teacher],record['teacher_probabilities'])
    # Independently derive exact-context groups by pairwise target-position matching.
    contexts=sorted(set(tuple(s[max(0,t-order):t]) for t in range(3,len(s))))
    details=[]
    for context in contexts:
        labels=[s[t] for t in range(3,len(s)) if tuple(s[max(0,t-order):t])==context]
        counts=Counter(labels)
        details.append(dict(context=list(context),counts={str(y):n for y,n in sorted(counts.items())}))
    assert details==record['ambiguity_counts']
    tablekeys=set()
    for t in range(1,len(s)):
        for n in range(min(t,order)+1):tablekeys.add(tuple(s[t-n:t]))
    table=[]
    for context in sorted(tablekeys):
        n=len(context)
        counts=[sum(w[t-1] for t in range(1,len(s)) if t>=n and tuple(s[t-n:t])==context and s[t]==y) for y in range(k)]
        table.append(dict(context=list(context),counts=counts))
    assert table==record['tables']
    floor=(125-sum(max(d['counts'].values()) for d in details))/125
    assert floor==record['metrics']['ambiguity_error_floor']
    score=125
    for i in range(125):
        if pred[i]!=s[i+3]:score=i;break
    assert score==record['metrics']['exact_prefix_symbols']
    assert [a==b for a,b in zip(pred,s[3:])]==record['position_accuracy']
    return dict(**record.get('identity',{}),order=order,exact_tables=True,exact_autonomous=True,exact_teacher=True,exact_ambiguity=True)
