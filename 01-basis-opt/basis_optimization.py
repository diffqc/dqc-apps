import dqc
import pyscf
import time
import torch
import xitorch as xt
import xitorch.optimize
import pyscf.dft
import numpy as np

train_moldescs = [
    "C 0.0000 0.0000 0.0000; H 0.0000 0.0000 2.1163",  # methylidyne,
    """C 0.0000 0.0000 0.0000; H 2.0390 0.0000 0.0000; H -1.0195 -1.7658 0.0000;
       H -1.0195 1.7658 0.0000""",  # CH3
    """C 0.0000 0.0000 0.0000; H 1.1860 1.1860 1.1860; H 1.1860 -1.1860 -1.1860;
       H -1.1860 1.1860 -1.1860; H -1.1860 -1.1860 1.1860""",  # CH4
    """C 0.0000 0.0000 1.1363; C 0.0000 0.0000 -1.1363; H 0.0000 0.0000 3.1453;
       H 0.0000 0.0000 -3.1453""",  # C2H2
    """C 0.0000 0.0000 1.2652; C 0.0000 0.0000 -1.2652; H 0.0000 1.7554 2.3283;
       H 0.0000 -1.7554 2.3283; H 0.0000 1.7554 -2.3283; H 0.0000 -1.7554 -2.3283""",  # C2H4
]
test_moldescs = [
    """C 0.0000 0.0000 1.4513; C 0.0000 0.0000 -1.4513; H -1.9260 0.0000 2.1870;
        H 0.9630 1.6679 2.1870; H 0.9630 -1.6679 2.1870; H 1.9260 0.0000 -2.1870;
        H -0.9630 -1.6679 -2.1870; H -0.9630 1.6679 -2.1870""",  # C2H6
    """C 0.0000 0.0000 -2.3537; C 0.0000 0.0000 0.4035; C 0.0000 0.0000 2.6825;
        H 0.0000 0.0000 4.6780; H 0.0000 1.9776 -3.0241; H 1.7127 -0.9887 -3.0241;
        H -1.7127 -0.9887 -3.0241""",  # CH3CCH (propyne)
    """C 0.0000 0.0000 0.0000; C 0.0000 0.0000 2.4718; C 0.0000 0.0000 -2.4718;
        H 0.0000 1.7625 3.5266; H 0.0000 -1.7625 3.5266; H 1.7625 0.0000 -3.5266;
        H -1.7625 0.0000 -3.5266""",  # CH2CCH2 (allene)
    """C 0.0000 0.0000 1.6305; C 0.0000 1.2238 -0.9451; C 0.0000 -1.2238 -0.9451;
        H 0.0000 2.9754 -1.9627; H 0.0000 -2.9754 -1.9627; H 1.7299 0.0000 2.7418;
        H -1.7299 0.0000 2.7418""",  # C3H4 (cyclopropene)
    """C 0.0000 1.6376 0.0000; C 1.4182 -0.8188 0.0000; C -1.4182 -0.8188
        0.0000; H 0.0000 2.7448 1.7212; H 2.3771 -1.3723 1.7212; H -2.3771 -1.3723
        1.7212; H 0.0000 2.7448 -1.7212; H 2.3771 -1.3723 -1.7212; H -2.3771 -1.3723
        -1.7212""",  # C3H6 (Cyclopropane)
    """C 0.0000 1.1079 -0.0000; C -2.3964 -0.4962 0.0000; C 2.3964 -0.4962
        -0.0000; H 0.0000 2.3525 1.6554; H -0.0006 2.3533 -1.6550; H -4.0773 0.7071
        0.0000; H 4.0773 0.7073 -0.0000; H -2.5079 -1.7034 1.6630; H -2.5079 -1.7034
        -1.6630; H 2.5079 -1.7034 -1.6630; H 2.5080 -1.7034 1.6630""",  # C3H8 (propane)
    """C 0.0000 0.0000 3.0840; C 0.0000 0.0000 0.5669; H 0.0000 1.7272 4.1993;
        H 0.0000 -1.7272 4.1993; C 0.0000 1.4570 -1.7694; C 0.0000 -1.4570 -1.7694;
        H 1.7240 2.4678 -2.2682; H -1.7240 2.4678 -2.2682; H -1.7240 -2.4678 -2.2682;
        H 1.7240 -2.4678 -2.2682""",  # C4H6 (Methylenecyclopropane)
    """C 0.0000 1.2680 1.5320; C 0.0000 -1.2680 1.5320; C 0.0000 1.4780 -1.3272;
        C 0.0000 -1.4780 -1.3272; H 0.0000 2.6768 3.0164; H 0.0000 -2.6768 3.0164;
        H 1.6981 2.3480 -2.1227; H -1.6981 -2.3480 -2.1227; H -1.6981 2.3480 -2.1227;
        H 1.6981 -2.3480 -2.1227""",  # C4H6 (Cyclobutene)
    """C 0.0000 0.0000 0.6897; H 0.0000 0.0000 2.7836; C 0.0000 2.7454 -0.1865;
        C 2.3777 -1.3727 -0.1865; C -2.3777 -1.3727 -0.1865; H 0.0000 2.8095 -2.2546;
        H 2.4330 -1.4046 -2.2546; H -2.4330 -1.4046 -2.2546; H 1.6896 3.6991 0.5331;
        H -1.6896 3.6991 0.5331; H 2.3588 -3.3127 0.5331; H 4.0482 -0.3864 0.5331;
        H -4.0482 -0.3864 0.5331; H -2.3588 -3.3127 0.5331""",  # CH3CH(CH3)CH3 (Isobutane)
    """C 0.0000 2.6399 0.0000; C 2.2862 1.3200 0.0000; C 2.2862 -1.3200 0.0000;
        C 0.0000 -2.6399 0.0000; C -2.2862 -1.3200 0.0000; C -2.2862 1.3200 0.0000;
        H 0.0000 4.6884 0.0000; H 4.0603 2.3442 0.0000; H 4.0603 -2.3442 0.0000; H
        0.0000 -4.6884 0.0000; H -4.0603 -2.3442 0.0000; H -4.0603 2.3442 0.0000""",  # C6H6 (Benzene)
]

print("Initial values for the training data")
print("Idx, ene")
bname = "cc-pvtz"
for i in range(len(train_moldescs)):
    m0 = dqc.Mol(train_moldescs[i], basis=bname, grid=4)
    ene0 = dqc.KS(m0, xc="lda_x+lda_c_pw").run().energy()
    print(i, ene0)

class Counter:
    def __init__(self, n):
        self.n = n
        self.restart()

    def restart(self):
        self.c = 0
        self.sum_ene = 0.0
        self.idxs = np.random.permutation(self.n)

    def feed(self, ene):
        self.sum_ene += ene.detach()
        self.c += 1
        if self.c == self.n:
            print(self.sum_ene / self.n)
            self.restart()

    def idx(self):
        return self.idxs[self.c]

basis = {
    "H": dqc.loadbasis("1:" + bname),
    "C": dqc.loadbasis("6:" + bname),
}
bpacker = xt.Packer(basis)
bparams = bpacker.get_param_tensor()

counter = Counter(len(train_moldescs))

def fcn(bparams, bpacker):
    basis = bpacker.construct_from_tensor(bparams)
    m = dqc.Mol(train_moldescs[counter.idx()], basis=basis, grid=4)
    ene = dqc.KS(m, xc="lda_x+lda_c_pw").run().energy()
    counter.feed(ene)
    return ene

min_bparams = bparams
for i in range(10):
    min_bparams = xt.optimize.minimize(fcn, min_bparams, (bpacker,), method="gd",
                                       step=1e-2, maxiter=1000, verbose=True)
    print(min_bparams)

basis = bpacker.construct_from_tensor(bparams)
min_basis = bpacker.construct_from_tensor(min_bparams)
print("Idx, ene0, opt-ene")
xc = "lda_x+lda_c_pw"
for i in range(len(test_moldescs)):
    m0 = dqc.Mol(test_moldescs[i], basis=basis, grid=4)
    ene0 = dqc.KS(m0, xc=xc).run().energy()
    m = dqc.Mol(test_moldescs[i], basis=min_basis, grid=4)
    ene = dqc.KS(m, xc=xc).run().energy()
    print(i, ene0, ene)
