import dedalus
import numpy as np

def filter_field(field, frac=0.5):
    field.require_coeff_space()
    dom = field.domain
    ndim = len(dom.global_coeff_shape)
    local_slice = dom.dist.coeff_layout.slices(scales=dom.dealias)
    slices = []
    i = 0
    for n, btype in zip(dom.global_coeff_shape,dom.bases):
        if i == 0:
            start = int(np.ceil(frac*n))
            stop = None
            slices = [slice(start,stop),] + (dom.dim-1)*[slice(None,None),] 
        else:
            if isinstance(btype, dedalus.core.basis.Chebyshev):
                start = int(np.floor(frac*n))
                stop = -1
                 
            elif isinstance(btype, dedalus.core.basis.Fourier):
                kmax = np.floor(n/2)
                start = int(np.ceil(frac*kmax))
                stop = -start
            else:
                raise ValueError("Basis {} is not supported by filter field".format(type(btype)))
            slices = i*[slice(None,None),] + [slice(start,stop)] + (dom.dim-i-1)*[slice(None,None),]
        print(slices)
        field['c'][slices] = 0j
        i += 1
