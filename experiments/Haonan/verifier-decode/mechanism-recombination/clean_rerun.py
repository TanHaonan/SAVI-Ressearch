"""Trustworthy clean re-run: FRESH per-instance generator (immune to the cross-T leak),
own exec-parity baseline (exec counted for EVERY rollout). p=0.7, depths x K, paired bootstrap."""
import os, sys, functools, time
MR=os.path.dirname(os.path.abspath(__file__))      # mechanism-recombination/
VD=os.path.dirname(MR)                              # verifier-decode/ (holds decode_core/)
for p_ in (MR,VD): sys.path.insert(0,p_)
from decode_core import decode as dc
from decode_core._deps.metrics.passk import paired_bootstrap
import gen_fair, arms_ext
import domain_lattice as L

D = L.LatticeDomain()
def idict(inst): return L.lattice_inst_dict(inst)
def leaf_ok(inst_d, res):
    if not res.ok or res.path is None: return bool(res.ok)
    cur=D.initial_state(inst_d)
    for m in res.path: cur=D.apply(cur,m)
    return D.is_goal(cur)
def iso_exec(gen, inst_d, target):  # exec-parity selection, exec counted for EVERY rollout
    b=dc.Budget(); s0=D.initial_state(inst_d); win=False; nr=0
    while b.exec < target and nr < 200000:
        cands=gen(s0,1,0.7,(7,nr),"chain"); b.record_sample(cands); nr+=1
        for t in cands:
            mv=D.parse_chain(t,s0)
            if mv is None: continue
            cur=s0
            for m in mv: cur=D.apply(cur,m); b.exec+=1
            if D.is_goal(cur): win=True
    return win
P=0.7; N=16; SEEDS=[1,2,3,4,5,6]; NINST=24
print(f"CLEAN (per-instance gen, canon now incl T). p={P} N={N} seeds={SEEDS} inst={NINST}",flush=True)
print(f"{'D':<4}{'K':<5}{'savi_freq':<10}{'iso_exec':<10}{'D1_exec[CI]':<26}{'D_verif':<9}{'D_merge':<9}",flush=True)
for depth in (4,8,16,24):
    insts=L.make_lattice_instances(depth, NINST, seed=1000+depth)
    for K in (8,16,32):
        a_freq=[]; a_iso=[]; a_ver=[]; a_bnm=[]
        t0=time.time()
        for seed in SEEDS:
            for inst in insts:
                d=idict(inst)
                gen=gen_fair.make_fair_generator(D, functools.partial(L.lattice_enumerate), P, depth)
                rf=dc.savi(D,gen,d,K,N,"freq",0.7,seed,verifier=False,max_depth=depth)
                rv=dc.savi(D,gen,d,K,N,"freq",0.7,seed,verifier=True,max_depth=depth)
                rb=arms_ext.beam_no_merge(D,gen,d,K,N,0.7,seed,verifier=False,max_depth=depth,edge_mode="freq")
                ok_f=leaf_ok(d,rf)
                io=iso_exec(gen,d,rf.budget.exec)
                a_freq.append(ok_f); a_iso.append(io); a_ver.append(leaf_ok(d,rv)); a_bnm.append(leaf_ok(d,rb))
        mf=sum(a_freq)/len(a_freq); mi=sum(a_iso)/len(a_iso); mv=sum(a_ver)/len(a_ver); mb=sum(a_bnm)/len(a_bnm)
        delta,lo,hi=paired_bootstrap(a_freq,a_iso,n=10000,seed=1)
        print(f"{depth:<4}{K:<5}{mf:<10.3f}{mi:<10.3f}{f'{delta:+.3f}[{lo:+.3f},{hi:+.3f}]':<26}{mv-mi:<+9.3f}{mf-mb:<+9.3f}  ({time.time()-t0:.0f}s)",flush=True)
print("DONE",flush=True)
