# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 21:19:46 2019

@author: Björn Müller
"""


import numpy as np
""" Bessel function of the second kind, order zero, , which is much faster
% than matlab's bessely(0,x)
% If bessj0(x) has been computed before, it showed be given as second
% argument, otherwise it is computed
%
% matlab translation of a C code in
% Cephes Math Library Release 2.1:  January, 1989
% Copyright 1984, 1987, 1989 by Stephen L. Moshier
%
%
% The domain is divided into the intervals [0, 5] and
% (5, infinity). In the first interval a rational approximation
% R(x) is employed to
%   y0(x)  = R(x)  +   2% log(x)% j0(x) / PI.
% Thus a call to j0() is required.
%
% In the second interval, the Hankel asymptotic expansion
% is employed with two rational functions of degree 6/6
% and 7/7.
%
% ACCURACY:
%
%  Absolute error, when y0(x) < 1; else relative error:
%
% arithmetic   domain     # trials      peak         rms
%    IEEE      0, 30       30000       1.3e-15     1.6e-16
"""

def bessy0(x,bessj0x):

    ans = np.zeros(size(x))
    #if ~isempty(find(x<=0))
    #    error('rg<=0 in bessy0')
    #end
    ind1 = np.argwhere(x>0 and x<=5.0)[:, 0]
    ind2 = np.argwhere(x>5.0)[:, 0]
    
    if ind1 is not empty:
        xx = x[ind1.astype(int)]
        z = xx* xx

        bessj0xx = bessj0x[ind1.astype(int)]

        ans[ind1.astype(int)] = TWOOPI * np.log(xx)*bessj0xx + polevl(z, YP) / p1evl(z, YQ)

    if ind2 is not empty:
        xx = x[ind2.astype(int)]
        one = np.ones(np.size(ind2))
        w = 5.0*one/xx
        z = 25.0 *ones/(xx*xx)
        p = polevl(z,PP) / polevl(z,PQ)
        q = polevl(z,QP)/p1evl( z, QQ)
        xn = xx - PIO4
        p = p * np.sin(xn) + w * q * np.cos(xn)
        ans[ind2.astype(int)] = SQ2OPI * p/np.sqrt(xx)
    return ans


def polevl(x,coef):
    """ evaluates polynomial coeff(1)x^N+..+coef(N+1)"""
    ans = coef[0] * np.ones(np.size(x))
    for j in range(1, len(coef)):
        ans = ans * x  + coef[j]
    return ans

def p1evl(x, coef):
    """ evaluates polynomial x^N + coef(1)*x^(N-1) + ..+ coef(N)"""
    ans = x + coef[0]
    for j in range(1, len(coef)):
        ans = ans* x  + coef[j]
    return ans


PIO4   =  7.85398163397448309616E-1    # pi/4
SQ2OPI =  7.9788456080286535587989E-1  # sqrt( 2/pi )
TWOOPI =  6.36619772367581343075535E-1 # 2/pi 
#THPIO4 =  2.35619449019234492885       # 3*pi/4 

PP = np.asarray([7.96936729297347051624E-4,    \
    8.28352392107440799803E-2,    \
    1.23953371646414299388E0,    \
    5.44725003058768775090E0,    \
    8.74716500199817011941E0,    \
    5.30324038235394892183E0,    \
    9.99999999999999997821E-1])
PQ = np.asarray([9.24408810558863637013E-4,   \
    8.56288474354474431428E-2,   \
    1.25352743901058953537E0,   \
    5.47097740330417105182E0,   \
    8.76190883237069594232E0,   \
    5.30605288235394617618E0,   \
    1.00000000000000000218E0])

QP = np.asarray([-1.13663838898469149931E-2,   \
    -1.28252718670509318512E0,   \
    -1.95539544257735972385E1,   \
    -9.32060152123768231369E1,   \
    -1.77681167980488050595E2,   \
    -1.47077505154951170175E2,   \
    -5.14105326766599330220E1,   \
    -6.05014350600728481186E0])
QQ = np.asarray([#...1.00000000000000000000E0   \
    6.43178256118178023184E1,   \
    8.56430025976980587198E2,   \
    3.88240183605401609683E3,   \
    7.24046774195652478189E3,   \
    5.93072701187316984827E3,   \
    2.06209331660327847417E3,   \
    2.42005740240291393179E2])

YP = np.asarray([1.55924367855235737965E4,   \
    -1.46639295903971606143E7,   \
    5.43526477051876500413E9,   \
    -9.82136065717911466409E11,   \
    8.75906394395366999549E13,   \
    -3.46628303384729719441E15,   \
    4.42733268572569800351E16,   \
    -1.84950800436986690637E16])
YQ =np.asarray([#1.00000000000000000000E0;...
    1.04128353664259848412E3,   \
    6.26107330137134956842E5,   \
    2.68919633393814121987E8,   \
    8.64002487103935000337E10,   \
    2.02979612750105546709E13,   \
    3.17157752842975028269E15,   \
    2.50596256172653059228E17])