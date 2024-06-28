
import astra 
import numpy as np
from regpy.operators import Operator
from regpy.vecsps import UniformGridFcts

class RadonAstra2D(Operator):
    """ this class implemets the 2D radonopertor for parallel and fanflat(fanbeam) geometry 
    using the astra-toolbox, for instalation see https://astra-toolbox.com/docs/install.html
    """    
    def __init__(self,num_pix:int,num_det:int,angles:np.array,geom_type="parallel",beam_type="strip",dx=1.,source_to_origin = None,origin_to_detector=None) -> None:
        """creates radon operator for parrlelel and fanflat(fanbeam) geometry

        Parameters
        ----------
        num_pix : int 
            the data object is square and has num_pix times num_pix pixel
        num_det : int
            number of pixels on the detector
        angles : np.array
            np.array of the measurment angles in pi
        geom_type : str, optional
            beam geometry "parallel" or "fanflat", by default "parallel"
        beam_type : str, optional
            type of beam each beam is either a "line" or a "strip", by default "strip"
        dx : float, optional
            length of one detector pixel, by default 1.
        source_to_origin : _type_, optional
            distance from source to origin in the length of one Pixel from the objct only necessary for "fanflat", by default None
        origin_to_detector : _type_, optional
            distance from origin to detector center in the length of one Pixel from the objct only necessary for "fanflat", by default None
        """        
        self.num_pix = num_pix
        self.domain = UniformGridFcts(np.arange(num_pix),np.arange(num_pix))
        self.num_det = num_det
        self.dx = dx
        self.angles = angles
        self.num_angles = len(angles)
        self.codomain = UniformGridFcts(np.arange(self.num_det),angles)
        self.geom_type = geom_type
        self.beam_type = beam_type 
        self.so = source_to_origin
        self.od = origin_to_detector
        self._create_projector()
        super().__init__(self.domain,self.codomain,linear=True)
        

    def _create_projector(self):
        """computes the linear radon operator using the astra library CPU implementation and the paramters defined in init
        """        
        #parralel beam
        if self.geom_type == 'parallel':
            vol_geom = astra.create_vol_geom(self.domain.shape)
            proj_geom=astra.create_proj_geom('parallel',self.dx,self.num_det,self.angles)
            self.proj_id = astra.create_projector(self.beam_type,proj_geom,vol_geom)
            self.rec_id = astra.data2d.create("-vol", vol_geom)  
            self.sino_id = astra.data2d.create("-sino", proj_geom)

        # fanbeam geometry
        elif self.geom_type == 'fanflat':
            vol_geom = astra.create_vol_geom(self.domain.shape)
            proj_geom=astra.create_proj_geom('fanflat',self.dx,self.num_det,self.angles,self.so,self.od)
            self.proj_id = astra.create_projector(self.beam_type+'_fanflat',proj_geom,vol_geom)
            self.rec_id = astra.data2d.create("-vol", vol_geom)  
            self.sino_id = astra.data2d.create("-sino", proj_geom)

    
    def _eval(self,f, differentiate=False, adjoint_derivative = False):
        """computes teh radon froward opertor

        Parameters
        ----------
        f : np.array
            object that  the radon transform is aplied on
        differentiate : bool, optional
            does notihing inherited from operator, by default False
        adjoint_derivative : bool, optional
             does notihing inherited from operator, by default False

        Returns
        -------
        np.array
            sinogram with shape (num_det,num_angles)
        """ 
        astra.data2d.store(self.rec_id,f)
        return np.array((astra.create_sino(self.rec_id,self.proj_id)[1]).tolist()).T
    
    def _adjoint(self,g):
        """computes the adjoint of the radon transform 

        Parameters
        ----------
        g : np.array
            sinogram with shape (num_det,num_angles)

        Returns
        -------
        np.array
            element in domian
        """        
        astra.data2d.store(self.sino_id,g.T)
        return np.array(astra.create_backprojection(self.sino_id,self.proj_id)[1].tolist())

    def __del__(self):
        astra.data2d.delete(self.rec_id)
        astra.data2d.delete(self.sino_id)
        astra.projector.delete(self.proj_id)
    
    def get_matrix(self):
        matrix_id = astra.projector.matrix(self.proj_id)
        A = astra.matrix.get(matrix_id)
        astra.matrix.delete(matrix_id)
        return A






class RadonMatrixAstra2D(Operator):
    """ this class implemets the 2D radonopertor for parallel and fanflat(fanbeam) geometry 
    using the astra-toolbox, for instalation see https://astra-toolbox.com/docs/install.html
    """    
    def __init__(self,num_pix:int,num_det:int,angles:np.array,geom_type="parallel",beam_type="strip",dx=1.,source_to_origin = None,origin_to_detector=None) -> None:
        """creates radon operator for parrlelel and fanflat(fanbeam) geometry, by creating a sparse matrix

        Parameters
        ----------
        num_pix : int 
            the data object is square and has num_pix times num_pix pixel
        num_det : int
            number of pixels on the detector
        angles : np.array
            np.array of the measurment angles in pi
        geom_type : str, optional
            beam geometry "parallel" or "fanflat", by default "parallel"
        beam_type : str, optional
            type of beam each beam is either a "line" or a "strip", by default "strip"
        dx : float, optional
            length of one detector pixel, by default 1.
        source_to_origin : _type_, optional
            distance from source to origin in the length of one Pixel from the objct only necessary for "fanflat", by default None
        origin_to_detector : _type_, optional
            distance from origin to detector center in the length of one Pixel from the objct only necessary for "fanflat", by default None
        """        
        self.num_pix = num_pix
        self.domain = UniformGridFcts(np.arange(num_pix),np.arange(num_pix))
        self.num_det = num_det
        self.dx = dx
        self.angles = angles
        self.num_angles = len(angles)
        self.codomain = UniformGridFcts(np.arange(self.num_det),angles)
        self.geom_type = geom_type
        self.beam_type = beam_type 
        self.so = source_to_origin
        self.od = origin_to_detector
        self.A = None
        self.get_matrix()
        super().__init__(self.domain,self.codomain,linear=True)
        

    def _create_projector(self):
        """computes the linear radon operator using the astra library CPU implementation and the paramters defined in init
        """        
        #parralel beam
        if self.geom_type == 'parallel':
            vol_geom = astra.create_vol_geom(self.domain.shape)
            proj_geom=astra.create_proj_geom('parallel',self.dx,self.num_det,self.angles)
            self.proj_id = astra.create_projector(self.beam_type,proj_geom,vol_geom)
 

        # fanbeam geometry
        elif self.geom_type == 'fanflat':
            vol_geom = astra.create_vol_geom(self.domain.shape)
            proj_geom=astra.create_proj_geom('fanflat',self.dx,self.num_det,self.angles,self.so,self.od)
            self.proj_id = astra.create_projector(self.beam_type+'_fanflat',proj_geom,vol_geom)


    
    def _eval(self,f, differentiate=False, adjoint_derivative = False):
        """computes teh radon froward opertor

        Parameters
        ----------
        f : np.array
            object that  the radon transform is aplied on
        differentiate : bool, optional
            does notihing inherited from operator, by default False
        adjoint_derivative : bool, optional
             does notihing inherited from operator, by default False

        Returns
        -------
        np.array
            sinogram with shape (num_det,num_angles)
        """        
        return (self.A@f.flatten()).reshape(self.codomain.shape[::-1]).T
    
    def _adjoint(self,g):
        """computes the adjoint of the radon matrix

        Parameters
        ----------
        g : np.array
            sinogram with shape (num_det,num_angles)

        Returns
        -------
        np.array
            element in domian
        """        
        return (self.A.T@g.T.flatten()).reshape(self.domain.shape)
    
    def get_matrix(self):
        if self.A is None:
            self._create_projector()
            matrix_id = astra.projector.matrix(self.proj_id)
            self.A = A = astra.matrix.get(matrix_id)
            astra.matrix.delete(matrix_id)
            astra.projector.delete(self.proj_id)
        else:
            return self.A
