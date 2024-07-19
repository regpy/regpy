
import astra 
import numpy as np
from regpy.operators import Operator
from regpy.vecsps import UniformGridFcts
from scipy import sparse

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

class RadonAstra3D(Operator):
    def __init__(self,num_pix,num_det,angles,geom_type,source_to_origin=0.,origin_to_detector=0.,dp=1,affine_shift_fkts_dic = None) -> None:
        """creates 3D radon transform with the possibility of dynamic affine shifts
        !!!needs gpu (?nvdia i.e cuda?) to run

        Parameters
        ----------
        num_pix : list
            dimension of the pixel basisi of the object,
             if int is given a quader pixel base with sidelength num_pix is choosen
        num_det : list
            dimensions of the detector pixel basis,
             if int a square detector with side length num_det is choosen
        angles : np.array
            array of projection angles in $\pi$
        geom_type : str
            "parallel" or "cone"
        source_to_origin : float, optional
            distance source to origin in object pixel, by default 0
        origin_to_detector : float, optional
            distance origin to detector in object pixel, by default 0
        dp : float or list, optional
            sice of rectengular detector pixel, auare if float, by default 1
        affine_shift_fkts_dic : dict , optional
            dictionary wit entries "list_affine_mats" list of 3 x 3 matrices 
             and "list_affine_vecs" list of 3 dimensional vectors 
             both are optional and both need to be of same length as the number of angles  
             togeter the elements form an affine transformation mat @ (x,y,z) + vec,
             by default None
        """        
        if type(num_pix) is int:
            self.num_pix = [num_pix]*3
        elif len(num_pix) == 3:
            self.num_pix = num_pix
        if type(num_det) is int:
            self.num_det = [num_det,num_det]
        elif len(num_det) == 2:
            self.num_det = num_det
        try:
            assert len(dp) == 2
            self.dx = dp[0]
            self.dy = dp[1]
        except:
            self.dx = dp
            self.dy = dp

        self.angles = angles
        self.num_angles = len(angles)
        self.geom_type = geom_type
        self.so = source_to_origin
        self.od = origin_to_detector
        self.affine_shift_fkts_dic = affine_shift_fkts_dic
        self._create_geometry()
        self._norm = None
        domain = UniformGridFcts(np.arange(self.num_pix[0]),
                                 np.arange(self.num_pix[1]),
                                 np.arange(self.num_pix[2]))
        codomain = UniformGridFcts(np.arange(self.num_angles),
                                   np.arange(self.num_det[0]),
                                   np.arange(self.num_det[1]))
        super().__init__(domain,codomain,linear = True)
        

    def _create_geometry(self):
        """computes the linear radon operator using the astra library CPU implementation and the paramters defined in init

        Returns:
            scipy.sparse.csr_matrix: the radon operator in matrix form
        """        
        vol_geom = astra.create_vol_geom(*self.num_pix)
        self.rec_id = astra.data3d.create("-vol", vol_geom)  
        #parralel beam
        if self.geom_type == 'parallel':
            proj_geom=astra.create_proj_geom('parallel3d',self.dx,self.dy,
                                             self.num_det[0],self.num_det[1],
                                             self.angles)
            # add shift
            if self.affine_shift_fkts_dic:
                geom_dic = astra.geom_2vec(proj_geom)
                vecs_shifted = self._apply_shifts(geom_dic["Vectors"])
                proj_geom=astra.create_proj_geom('parallel3d_vec',
                                                 geom_dic['DetectorRowCount'],
                                                 geom_dic['DetectorColCount'],
                                                 vecs_shifted)
            self.sino_id = astra.data3d.create("-sino", proj_geom)

        # fanbeam geometry
        elif self.geom_type == 'cone':
            proj_geom= astra.create_proj_geom('cone',self.dx,self.dy,
                                             self.num_det[0],self.num_det[1],
                                             self.angles,
                                             self.so,self.od)
            # add shift only possible for the line at the moment with astra
            if self.affine_shift_fkts_dic:
                geom_dic = astra.geom_2vec(proj_geom)
                vecs_shifted = self._apply_shifts(geom_dic["Vectors"])
                proj_geom=astra.create_proj_geom('cone_vec',
                                                 geom_dic['DetectorRowCount'],
                                                 geom_dic['DetectorColCount'],
                                                 vecs_shifted)
            self.sino_id = astra.data3d.create("-sino", proj_geom)
        self.dic = astra.geom_2vec(proj_geom)
    
    def _fproject(self):
        fpalg_cfg = astra.astra_dict("FP3D_CUDA")
        fpalg_cfg["ProjectionDataId"] = self.sino_id
        fpalg_cfg["VolumeDataId"] = self.rec_id
        fpalg_id = astra.algorithm.create(fpalg_cfg)
        
        astra.algorithm.run(fpalg_id)
        astra.algorithm.delete(fpalg_id)
        
    def _bproject(self):
        bpalg_cfg = astra.astra_dict("BP3D_CUDA")
        bpalg_cfg["ProjectionDataId"] = self.sino_id
        bpalg_cfg["ReconstructionDataId"] = self.rec_id
        bpalg_id = astra.algorithm.create(bpalg_cfg)
        
        astra.algorithm.run(bpalg_id)
        astra.algorithm.delete(bpalg_id)
    
    def _eval(self,f):
        astra.data3d.store(self.rec_id,f)
        self._fproject()
        return np.swapaxes(astra.data3d.get(self.sino_id),0,1)
    
    def _adjoint(self,g):
        astra.data3d.store(self.sino_id,np.swapaxes(g,0,1))
        self._bproject()
        return astra.data3d.get(self.rec_id)

    def __del__(self):
        astra.data3d.delete(self.rec_id)
        astra.data3d.delete(self.sino_id)
        
    def _apply_shifts(self,geom_vecs):
        #ToDo add a logging option that tells if shift is used or not
        geom_vecs = np.swapaxes(geom_vecs.reshape(self.num_angles,4,3),1,2)
        geom_vecs = np.concatenate(geom_vecs)
        if "list_affine_mats" in self.affine_shift_fkts_dic.keys():
            affine_mat = sparse.block_diag(self.affine_shift_fkts_dic["list_affine_mats"])
            geom_vecs = affine_mat @ geom_vecs
        # The x-y-z-shifts are only aplied to the source_to_origin and origin_to_detector vectors
        # the detector coordinate system does not shift those are only directions for the detectoe 
        # coordinate system
        if "list_affine_vecs" in self.affine_shift_fkts_dic.keys():
            aff_vec = np.concatenate(self.affine_shift_fkts_dic["list_affine_vecs"])
            assert (aff_vec.shape[1] == 1) 
            geom_vecs[:,:2] += aff_vec
            
        geom_vecs = np.array(np.split(geom_vecs,self.num_angles))
        geom_vecs = np.swapaxes(geom_vecs,1,2).reshape(self.num_angles,12)
        return geom_vecs


