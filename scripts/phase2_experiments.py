"""Diagnostic baselines/statistics; transcendentals never enter production code."""
from pathlib import Path
import ast

import jax
import jax.numpy as jnp
import numpy as np

from src.attention import octic_kernel, algebraic_softmax, _normalized_forward
from tests.reference_attention import kernel, attention, normalized_attention, softmax
from scripts.phase1_experiments import summary
from scripts.audit_primitives import source_audit, primitives_in, FORBIDDEN

LENGTHS=(64,128,256,512,1024,2048,4096)


def audit():
    source=Path(__file__).resolve().parents[1].joinpath('src/attention.py').read_text()
    x=jnp.ones((2,128),dtype=jnp.float32)
    graphs={name:dict(primitives_in(jax.make_jaxpr(fn)(x))) for name,fn in
            [('kernel',octic_kernel),('attention',algebraic_softmax),
             ('backward',jax.grad(lambda z:algebraic_softmax(z).sum()))]}
    violations=source_audit(source)
    violations += [{'line':n.lineno,'name':n.attr} for n in ast.walk(ast.parse(source))
                   if isinstance(n,ast.Attribute) and n.attr in {'softmax','sqrt'}]
    bad={key:sorted(set(value)&(FORBIDDEN|{'sqrt'})) for key,value in graphs.items()}
    return {'source_violations':violations,'graphs':graphs,'graph_violations':bad,
            'passed':not violations and not any(bad.values())}


def scalar_checks(seed=43):
    rng=np.random.default_rng(seed)
    x=rng.uniform(-3,3,100_000);s=np.sqrt(1+x*x)
    reciprocal=float(np.max(np.abs((s+x)*(s-x)-1)))
    contrast=float(kernel(np.array(3.))/kernel(np.array(-3.)))
    sharpness=float(kernel(np.array(2.)))
    exact=float(51841+23184*np.sqrt(5))
    return {'seed':seed,'samples':len(x),'domain':[-3,3], 'reciprocal_error':reciprocal,
            'contrast':contrast,'sharpness':sharpness,'sharpness_exact_expression':'51841 + 23184 sqrt(5)',
            'sharpness_exact_expression_error':abs(sharpness-exact),
            'historical_exact_integer_gate':sharpness==103682,
            'passed':reciprocal<=5e-14 and contrast>=1e5 and abs(sharpness-exact)<=2e-10}


def w1(p,q):
    """Token positions 0..1, equally spaced; omitted sink mass is at position 1.

    This is a probability-weighted token-position Wasserstein distance, not a
    distance between sorted probability values. Unit interval makes 5%=0.05.
    """
    return np.sum(np.abs(np.cumsum(p-q,axis=-1)[...,:-1]),axis=-1)/(p.shape[-1]-1)


def fp4_quantize(x, scaled=True):
    """Diagnostic E2M1 finite codebook; nearest value, ties toward lower magnitude.

    Scaled variant uses one positive max-abs/6 scale per vector; all-zero scale=1.
    A diagnostic software quantizer, not a TPU hardware FP4 execution claim.
    """
    levels=np.array([0,.5,1,1.5,2,3,4,6],dtype=float)
    scale=np.max(np.abs(x),axis=-1,keepdims=True)/6 if scaled else np.ones_like(x[...,:1])
    scale=np.where(scale==0,1,scale)
    y=np.abs(x)/scale
    indices=np.searchsorted((levels[1:]+levels[:-1])/2,y,side='left')
    return np.sign(x)*levels[indices]*scale


def mc_study(trials=100_000,seed=42,progress=print):
    rng=np.random.default_rng(seed);rows=[];raw={}
    for index,length in enumerate(LENGTHS):
        n=trials//len(LENGTHS)+(index<trials%len(LENGTHS)); observations=[]
        for start in range(0,n,128):
            x=rng.normal(size=(min(128,n-start),length));noise=rng.normal(0,.05,size=x.shape)
            p=attention(x);q=softmax(x)
            # H over the token coordinates as requested; sink entropy also recorded.
            entropy=-np.sum(p*np.log(p),axis=-1)/np.log(length)
            mass=p.sum(-1);sink=np.maximum(0,1-mass)
            full_entropy=entropy-np.where(sink>0,sink*np.log(np.maximum(sink,np.finfo(float).tiny)),0)/np.log(length)
            da=np.linalg.norm(attention(x+noise)-p,axis=-1)
            db=np.linalg.norm(softmax(x+noise)-q,axis=-1)
            quant=fp4_quantize(x)
            qa=np.linalg.norm(attention(quant)-p,axis=-1);qb=np.linalg.norm(softmax(quant)-q,axis=-1)
            observations.append(np.stack([entropy,full_entropy,w1(p,q),da,db,qa,qb,mass],axis=-1))
        a=np.concatenate(observations);raw[f'L{length}']=a
        stats={name:summary(a[:,i]) for i,name in enumerate(('entropy','entropy_with_sink','w1','noise_alg','noise_softmax','fp4_alg','fp4_softmax','mass'))}
        gain=stats['noise_softmax']['mean']/stats['noise_alg']['mean']
        qgain=stats['fp4_softmax']['mean']/stats['fp4_alg']['mean']
        # Preserve strict original per-trial entropy and exact finite simplex gates.
        gates={'entropy_all_trials':bool(np.all((a[:,0]>=.10)&(a[:,0]<=.95))),
               'entropy_mean_ci':stats['entropy']['ci95'][0]>=.10 and stats['entropy']['ci95'][1]<=.95,
               'w1_mean_ci':stats['w1']['ci95'][1]<=.05,'noise_100x':gain>=100,
               'fp4_100x':qgain>=100,'simplex_float64':bool(np.all(a[:,7]<=1))}
        row={'length':length,'trials':n,'statistics':stats,'noise_suppression_ratio':gain,'fp4_suppression_ratio':qgain,
             'entropy_violations':int(np.sum((a[:,0]<.10)|(a[:,0]>.95))),'gates':gates,'passed':all(gates.values())}
        rows.append(row);progress(f'L={length}: entropy={stats["entropy"]["mean"]:.4f}, W1={stats["w1"]["mean"]:.4f}, noise ratio={gain:.4f}, FP4 ratio={qgain:.4f}')
    return {'seed':seed,'trials':trials,'scores':'independent N(0,1) raw scores',
            'noise':'independent Gaussian sigma=.05 on raw scores, common noise for both arms',
            'distance':'L2 output displacement; gain=mean softmax displacement / mean algebraic displacement',
            'w1':'probability-weighted positions 0..1, sink at last position; 5% threshold=.05',
            'raw_columns':['entropy','entropy_with_sink','w1','noise_alg','noise_softmax','fp4_alg','fp4_softmax','mass'],
            'rows':rows,'passed':all(r['passed'] for r in rows)},raw


def jacobian_study(trials=10_000,seed=44,progress=print):
    rng=np.random.default_rng(seed);rows=[];raw={}
    # Full matrices, including every diagonal and off-diagonal; bounded memory.
    lengths=(2,8,16,64,128)
    for i,length in enumerate(lengths):
        n=trials//len(lengths)+(i<trials%len(lengths)); maxima=[];errors=[]
        fn=jax.jit(jax.vmap(jax.jacrev(lambda z:_normalized_forward(z,.5)[0])))
        for start in range(0,n,32):
            x=rng.normal(size=(min(32,n-start),length))
            x=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)
            jac=np.asarray(fn(jnp.asarray(x)))
            p=normalized_attention(x)
            expected=(np.eye(length)[None,:,:]-p[:,None,:])*p[:,:,None]*8/np.sqrt(1+x[:,None,:]**2)
            maxima.extend(np.max(np.abs(jac),axis=(1,2)));errors.extend(np.max(np.abs(jac-expected),axis=(1,2)))
        raw[f'L{length}']=np.stack([maxima,errors],axis=-1)
        row={'length':length,'trials':n,'maximum_entry':summary(maxima),'oracle_error':float(max(errors)),
             'passed':bool(max(maxima)<=2+1e-12 and max(errors)<=2e-12)}
        rows.append(row);progress(f'Full Jacobian L={length}: max={max(maxima):.6f}, oracle error={max(errors):.3g}')
    return {'seed':seed,'trials':trials,'coordinates':'AVN-normalized scores treated as independent inputs',
            'rows':rows,'passed':all(r['passed'] for r in rows)},raw


def sink_sweep(seed=45,trials=2048):
    rng=np.random.default_rng(seed);x=rng.normal(size=(trials,128));noise=rng.normal(0,.05,size=x.shape)
    q=softmax(x);db=np.linalg.norm(softmax(x+noise)-q,axis=-1);rows=[]
    for sink in (0.,.25,.5,.75,1.):
        p=attention(x,sink);da=np.linalg.norm(attention(x+noise,sink)-p,axis=-1)
        rows.append({'sink':sink,'noise_ratio':float(db.mean()/da.mean()),'w1':summary(w1(p,q)),
                     'passed':bool(db.mean()/da.mean()>=100 and np.mean(w1(p,q))<=.05)})
    return {'seed':seed,'trials':trials,'length':128,'rows':rows,'passed':any(r['passed'] for r in rows)}
