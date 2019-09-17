# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:18:07 2019

@author: Björn Müller
"""
import numpy as np

""" Bessel function of order one of the second kind,
%  which is much faster than matlab's bessely(1,x)
% If bessj1(x) has been computed before, it showed be given as second
% argument, otherwise it is computed
%
% matlab translation of a C code in
% Cephes Math Library Release 2.1:  January, 1989
% Copyright 1984, 1987, 1989 by Stephen L. Moshier
%
%The domain is divided into the intervals [0, 8] and
% (8, infinity). In the first interval a 25 term Chebyshev
% expansion is used, and a call to j1() is required.
% In the second, the asymptotic trigonometric representation
% is employed using two rational functions of degree 5/5.
%
% ACCURACY:
%                      Absolute error:
% arithmetic   domain      # trials      peak         rms
%    IEEE      0, 30       30000       1.0e-15     1.3e-16
% (error criterion relative when |y1| > 1).
"""





def bessy1(x,bessj1x):
    ans = np.zeros(np.size(x))
    #if ~isempty(find(x<=0))
    #    error('rg<=0 in bessy0')
    #end
    set_1=set(np.argwhere(0<x)[:, 0])
    set_2=set(np.argwhere(x<=5)[:, 0])
    final_set=set_1.intersection(set_2)
    ind1=np.asarray([list(final_set)])

#    ind1 = np.argwhere(x>0 and x<=5.0)[:, 0]
    ind2 = np.argwhere(x>5.0)[:, 0]

    if len(ind1)!=0:
        xx = x[ind1.astype(int)]
        z = xx * xx
        w = xx * polevl(z, YP1) / p1evl(z, YQ1)

        bessj1xx = bessj1x[ind1.astype(int)]


        w = w + TWOOPI * ( bessj1xx * np.log(xx)  -  np.ones(np.size(ind1))/xx )
        ans[ind1.astype(int)] =w

    if len(ind2)!=0:
        xx = x[ind2.astype(int)]
        one = np.ones(np.size(ind2))
        w = 5.0 * one/xx
        z = w* w
        p = polevl(z, PP1)/polevl(z,PQ1)
        q = polevl(z, QP1)/p1evl(z,QQ1)
        xn = xx - THPIO4
        p = p * np.sin(xn) + w * q * np.cos(xn)
        ans[ind2.asytpe(int)] =  SQ2OPI *p/ np.sqrt(xx)
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

#PIO4   =  7.85398163397448309616E-1;    % pi/4
SQ2OPI =  7.9788456080286535587989E-1  # sqrt( 2/pi )
TWOOPI =  6.36619772367581343075535E-1 # 2/pi
THPIO4 =  2.35619449019234492885       # 3*pi/4

PP1 = np.asarray([7.62125616208173112003E-4,
 7.31397056940917570436E-2,
 1.12719608129684925192E0,
 5.11207951146807644818E0,
 8.42404590141772420927E0,
 5.21451598682361504063E0,
 1.00000000000000000254E0])
PQ1 = np.asarray([5.71323128072548699714E-4,
 6.88455908754495404082E-2,
 1.10514232634061696926E0,
 5.07386386128601488557E0,
 8.39985554327604159757E0,
 5.20982848682361821619E0,
 9.99999999999999997461E-1])

QP1 = np.asarray([5.10862594750176621635E-2,
 4.98213872951233449420E0,
 7.58238284132545283818E1,
 3.66779609360150777800E2,
 7.10856304998926107277E2,
 5.97489612400613639965E2,
 2.11688757100572135698E2,
 2.52070205858023719784E1])
QQ1 = np.asarray([#... 1.00000000000000000000E0
 7.42373277035675149943E1,
 1.05644886038262816351E3,
 4.98641058337653607651E3,
 9.56231892404756170795E3,
 7.99704160447350683650E3,
 2.82619278517639096600E3,
 3.36093607810698293419E2])

YP1 = np.asarray([1.26320474790178026440E9,
-6.47355876379160291031E11,
 1.14509511541823727583E14,
-8.12770255501325109621E15,
 2.02439475713594898196E17,
 -7.78877196265950026825E17])
YQ1 = np.asarray([#... 1.00000000000000000000E0
 5.94301592346128195359E2,
 2.35564092943068577943E5,
 7.34811944459721705660E7,
 1.87601316108706159478E10,
 3.88231277496238566008E12,
 6.20557727146953693363E14,
 6.87141087355300489866E16,
 3.97270608116560655612E18])
