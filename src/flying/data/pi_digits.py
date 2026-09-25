"""π includes leading 3, never the decimal point. No external data request."""
import numpy as np


def decimal_pi_digits(length):
    """Dependency-free Gauss–Legendre AGM, with 30 guard decimal places."""
    from decimal import Decimal, localcontext
    if not isinstance(length, int) or length < 2:
        raise ValueError('length must be an integer >= 2')
    with localcontext() as context:
        context.prec=length+30
        a=Decimal(1);b=1/Decimal(2).sqrt();t=Decimal(1)/4;p=Decimal(1)
        for _ in range((length+30).bit_length()+1):
            following=(a+b)/2
            b=(a*b).sqrt();t-=p*(a-following)**2;a=following;p*=2
        s=format((a+b)**2/(4*t),'f').replace('.','')[:length]
    return np.array([int(c) for c in s],dtype=np.int64)


def pi_digits(length):
    if not isinstance(length, int) or length < 2:
        raise ValueError('length must be an integer >= 2')
    try:
        from mpmath import mp
    except ModuleNotFoundError as error:
        if error.name != 'mpmath':raise
        return decimal_pi_digits(length)
    with mp.workdps(length + 30):
        s = str(mp.pi).replace('.', '')[:length]
    return np.array([int(c) for c in s], dtype=np.int64)
