#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:55:35 2019

@author: agaltsov
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi, ConvexHull

from pynfft.nfft import NFFT as PYNFFT
from pynfft.solver import Solver
from enum import Enum, auto

from copy import copy

from scipy.sparse.linalg import LinearOperator, cg

### Ewald sphere geometry framework for NFFT


class Rep(Enum):
    """ Enumeration specifying representation of data"""
    
    # Data defined on the whole Ewald sphere
    EwaldSphere = auto()
    
    # Data defined on pairs (indident_direction,measurement_direction)
    PairsOfDirections = auto()
    
    # Data defined in the coordinate (domain) space 
    CoordinateDomain = auto()
    

def voronoi_box(nodes):
    ### Computes the Voronoi diagram with a bounding box [-0.5,0.5)^2
    
    # Extend the set of nodes by reflecting along boundaries
    nodes_up = np.array([[x,1-y+1e-6] for x,y in nodes])
    nodes_down = np.array([[x,-1-y-1e-6] for x,y in nodes])
    nodes_right = np.array([[1-x+1e-6,y] for x,y in nodes])
    nodes_left = np.array([[-1-x-1e-6,y] for x,y in nodes])
    
    enodes = np.concatenate([nodes,nodes_up,nodes_down,nodes_left,nodes_right])
    
    # Computing the extended Voronoi diagram
    evor = Voronoi(enodes)
    
    ### Shrinking the Voronoi diagram
    regions = [evor.regions[reg] for reg in evor.point_region[:nodes.shape[0]]]
    used_vertices = np.unique([i for reg in regions for i in reg])
    regions = [[ np.where(used_vertices==i)[0][0] for i in reg] for reg in regions]
    vertices = [evor.vertices[i] for i in used_vertices]
    
    return regions, vertices


class NFFT:
    

    def __init__(self,inc_directions,meas_directions,proj,p):
        
        # Computing nodes of the Ewald sphere scaled to [-0.5,0.5)
        all_nodes = np.array([(x-y)/4  for x,Y in zip(inc_directions,meas_directions) for y in Y])
        _, self.node_indices = np.unique(all_nodes.round(decimals=6), axis=0,return_index=True) 
        self.nodes = all_nodes[self.node_indices]
        self.ewald_node_count = self.nodes.shape[0]
        
        # Computing the uniform grid surrounding the Ewald sphere
        x = np.arange(p['GRID_SHAPE'][0])/p['GRID_SHAPE'][0]-0.5
        y = np.arange(p['GRID_SHAPE'][1])/p['GRID_SHAPE'][1]-0.5
        [X,Y] = np.meshgrid(x,y)
        outer_ind = X**2 + Y**2 > 0.25
        outer_nodes = np.stack([X[outer_ind],Y[outer_ind]],axis=1)
        self.nodes = np.concatenate([self.nodes,outer_nodes])
        
        ### Computing the weights of nodes to compute Riemann sums over the Ewald sphere
        # Compute the bounded by [-0.5,0.5)^2 Voronoi diagram of the Ewald sphere
        regions, vertices = voronoi_box(self.nodes)
        
        # Physical Ewald sphere  has radius 2sqrt(E)
        # Scaling the Ewald sphere to radius 2*F.kappa*F.rho / (pi*F.N(1)) 
        # corresponds to scaling the x-domain to [-.5,.5)
        self.scaling_factor = 8 * p['WAVE_NUMBER'] * p['SUPPORT_RADIUS'] / (np.pi*p['GRID_SHAPE'][0])
        self.nodes *= self.scaling_factor
        
        # Computing areas of cells of the Voronoi diagram
        self.weights = np.array(
                [ConvexHull([vertices[i]*self.scaling_factor for i in reg]).volume for reg in regions])
        
        # Saving the patches of the Voronoi diagram
        self.patches = PatchCollection(
                [Polygon([vertices[i]*self.scaling_factor for i in reg]) for reg in regions], \
                                        edgecolors=None)
        
        #self.set_submanifold(0.5)
        #self.display(self.weights,Rep.EwaldSphereExtended)
                   
        # Initialize the NFFT 
        self.plan = PYNFFT(N=p['GRID_SHAPE'],M=self.nodes.shape[0])
        
        # NFFT Precomputations
        self.plan.x = self.nodes
        self.plan.precompute()
        
        # NFFT scaling factor
        self.nfft_factor = (4*p['SUPPORT_RADIUS']/(2*np.pi))**2 / np.prod(p['GRID_SHAPE'])
        
        # Initialize the Solver for computing the inverse NFFT
        self.solver = Solver(self.plan)
        self.solver.w = self.weights
        self.solver_num_iter = p['INFFT_NUM_ITER']
        self.solver_threshold = np.float(p['INFFT_THRESHOLD'])
        
        # Projector onto support
        self.proj = proj
        
    def forward_NFFT(self,f_hat,cutoff=True):
        """Computes the forward NFFT
        
           Parameters
           ----------
           f_hat : function on the rectangular grid
           
           
           Output
           ------
           f: function on the Ewald sphere """
               
        self.plan.f_hat = self.proj.adjoint(f_hat)
        
        f = self.plan.trafo() 
        
        if cutoff:
            f *= self.submanifold_indicator(0.5)
        
        return f
    
    def adjoint_NFFT(self,f):
        """ Computes the adjoint NFFT 
        
            Output:
            -------
            f_hat : function on the grid
        """
        
        self.plan.f = f
        f_hat = self.plan.adjoint()
        return f_hat
        
    
    def inverse_NFFT(self,f):
        """Computes the inverse NFFT
        
           Parameters
           ----------
           f : function on the Ewald sphere"""
        
        f_hat = self.adjoint_NFFT(self.weights*f)
        
        return self.proj(f_hat)
    
    
    def convert(self,x,from_rep,to_rep):
        """ Changes the representation of data between different formats
       
            Parameters
            ----------
            `from_rep`  :Rep:   initial data representation
            `to_rep`    :Rep:   target data representation
            """
        
        assert isinstance(from_rep,Rep) and isinstance(to_rep,Rep)
 
        if from_rep == Rep.PairsOfDirections and to_rep == Rep.EwaldSphere:
            y = np.zeros(self.nodes.shape[0],dtype=complex)
            y[:self.ewald_node_count] = x.flatten('F')[self.node_indices]
            return y
         
        if from_rep == Rep.CoordinateDomain and to_rep == Rep.EwaldSphere:
            y = self.nfft_factor * np.conj(self.forward_NFFT(np.conj(x)))  \
                * self.submanifold_indicator(0.5)
            return y
         
        if from_rep == Rep.EwaldSphere and to_rep == Rep.CoordinateDomain:
            return np.conj(self.inverse_NFFT(np.conj(x))) / self.nfft_factor       
       
           
    def submanifold_indicator(self,radius):
        return np.linalg.norm(self.nodes,axis=1)<=radius*self.scaling_factor       
        
        
    def display(self,f,**kwargs):
        """Display the function f on the Ewald sphere"""
                    
        # Setting colors
        self.patches.set_array(np.real(f))
        
        plt.gca().add_collection(copy(self.patches))
        plt.xlim(-0.5*self.scaling_factor,0.5*self.scaling_factor)
        plt.ylim(-0.5*self.scaling_factor,0.5*self.scaling_factor)
        plt.gcf().colorbar(self.patches)        
        
        
    def norm(self,f):
        #assert f.shape==self.nodes.shape[0]
        return np.sqrt(sum(np.abs(f*f)*self.weights))
    
    def zeros(self,**kwargs):
        return np.zeros(self.nodes.shape[0],**kwargs)
        