from regpy import vecsps
from regpy.util import Errors

from .base import Operator,Identity

__all__ = ["OperatorNode","Edge","OperatorGraph"]

class OperatorNode:
    """Object that represents a node in a Graph of operators and manages the input and output from the edges to the operator.

    Parameters
    ----------
    op : regpy.vecsps.Operator
        The underlying operator
    """

    def __init__(self,op):
        if not isinstance(op,Operator):
            raise TypeError(Errors.not_instance(op,Operator, add_info="An OperatorNode can only be constructed on a RegPy Operator!"))
        self.op=op
        self.N_in=len(self.op.domain.summands) if isinstance(self.op.domain,vecsps.DirectSum) else 1
        self.N_out=len(self.op.codomain.summands) if isinstance(self.op.codomain,vecsps.DirectSum) else 1
        self.input_edges=[None]*self.N_in
        self.output_edges=[]

    def __str__(self):
        return str(self.op)

    def get_free_inputs(self):
        """Calculates set of inputs with no assigned input edge.

        Returns:
            set: set of indices of inputs with no assigned input edge.
        """        
        return set([i for i in range(self.N_in) if self.input_edges[i] is None])
    
    def get_in_nodes(self):
        """Compute nodes where incoming edges come from.

        Returns:
            set: set of nodes
        """        
        return set([edge.start_node for edge in self.input_edges if edge is not None and edge.start_node is not None])
    
    def get_out_nodes(self):
        """Compute nodes where outgoing edges go to.

        Returns:
            set: set of nodes
        """       
        return set([edge.end_node for edge in self.output_edges if edge is not None and edge.end_node is not None])
    
    def combine_input(self,data_dict):
        """Combines the input from the input edges.

        Parameters
        ----------
            data_dict (dict): Dictionary with nodes as keys and computed data of that node as value

        Returns
        -------
            array-like: element in the domain of the operator
        """ 
        if any(edge is None for edge in self.input_edges):
            raise RuntimeError(Errors.runtime_error("There is an empty input edge. Not able to combine such inputs!",self,"combine_input"))
        if(self.N_in==1):
            edge=self.input_edges[0]
            return edge.pass_forward(data_dict[edge.start_node])
        else:
            return self.op.domain.join(*[edge.pass_forward(data_dict[edge.start_node]) for edge in self.input_edges])
        
    def combine_output(self,data_dict):
        """Combines the output from the output edges. This is used for the evaluation of the adjoint.

        Parameters
        ----------
            data_dict (dict): Dictionary with nodes as keys and computed data of that node as value

        Returns
        -------
            array-like: element in the codomain of the operator
        """           
        if(self.output_edges==[]):
            return self.op.codomain.zeros()
        edge=self.output_edges[0]
        data_list=edge.pass_backward(data_dict[edge.end_node])
        for i in range(1,len(self.output_edges)):
            edge=self.output_edges[i]
            new_data=edge.pass_backward(data_dict[edge.end_node])
            for i,d in enumerate(new_data):
                if(d is not None):
                    if(data_list[i] is None):
                        data_list[i]=d
                    else:
                        data_list[i]+=d
        if(self.N_out==1):
            return data_list[0]
        else:
            return self.op.codomain.join(*[d if d is not None else self.op.codomain.summands[i].zeros() for i,d in enumerate(data_list)])

class Edge:
    """Object that represents an edge in a Graph of operators and manages the data transport between operator nodes.

    Parameters
    ----------
    start_node : OperatorNode
        The node where the edge starts.
    end_node : OperatorNode
        The node where the edge ends.
    start_list : list of int
        list of output indices of the start node
    end_index : int
        index of input of the end node
    """

    def __init__(self,start_node,end_node,start_list,end_index):
        if not isinstance(start_node,OperatorNode) and start_node is not None:
            raise TypeError(Errors.type_error("The start node for an Edge in operator networks need to be either None or an OperatorNode"))
        if not isinstance(end_node,OperatorNode) and end_node is not None:
            raise TypeError(Errors.type_error("The end node for an Edge in operator networks need to be either None or an OperatorNode"))
        if not isinstance(start_list,list) or any(not isinstance(k,int) for k in start_list):
            raise TypeError(Errors.not_instance(start_list,list,add_info="The list of output indices needs to be a list of integers!"))
        if not isinstance(end_index,int):
            raise TypeError(Errors.not_instance(end_index,int,add_info="The index of the input needs to be a single integer!"))
        self.start_node=start_node
        self.end_node=end_node
        self.start_list=start_list
        self.end_index=end_index
        if(self.start_node is not None):
            if any(i<0 or i>=self.start_node.N_out for i in self.start_list):
                raise ValueError(Errors.value_error("The given list of start indices has indices out of the index set of the start node!"+"\n\t"+f"start_list = {start_list}"+"\n\t"+f"start_node.N_out = {start_node.N_out}"))
            self.start_node.output_edges.append(self)
        if(self.end_node is not None):
            if end_index<0 or end_index>=self.end_node.N_in:
                raise ValueError(Errors.value_error("The given end index is out of the bound of indices of the end node!"+"\n\t"+f"end_index = {end_index}"+"\n\t"+f"end_node.N_in = {end_node.N_in}"))
            if self.end_node.input_edges[self.end_index] is not None:
                raise ValueError(Errors.value_error(f"The input edge of the end_node is not empty for the index {end_index} you gave! It is already set to {self.end_node.input_edges[self.end_index]}"))
            self.end_node.input_edges[self.end_index]=self
            if(self.end_node.N_in==1):
                self.end_space=self.end_node.op.domain
            else:
                self.end_space=self.end_node.op.domain.summands[self.end_index]

    def construct_start_space(self):
        """Constructs vector space corresponding to input of the edge.

        Returns
        -------
            regpy.vecsps.VectorSpace: VectorSpace corresponding to combined input of this edge.
        """
        if self.start_node is None:
            raise RuntimeError(Errors.runtime_error("Constructing a start space of an edge with no start_node is not possible!",self,"construct_start_space"))
        if(self.start_node.N_out==1):
            return self.start_node.op.codomain
        if(len(self.start_list)==1):
            return self.start_node.op.codomain.summands[self.start_list[0]]
        return vecsps.DirectSum(*[self.start_node.codomain.summands[i] for i in self.start_list])

    def remove(self):
        """Removes this edge.
        """
        if(self.start_node is not None):
            self.start_node.output_edges.remove(self)
        if(self.end_node is not None):
            self.end_node.input_edges[self.end_index]=None
        self.start_node=None
        self.end_node=None

    def __str__(self):
        return f"{self.start_node}{self.start_list}-->[{self.end_index}]{self.end_node}"

    def pass_forward(self,x):
        """Passes and transforms data forwards through the edge.

        Parameters
        ----------
        x : array_like
            element of the codomain of the operator of the input node.

        Returns
        -------
        array_like 
            element of the part of the domain of the operator of the output node.
        """       
        if self.end_node is None or self.start_node is None:
            raise RuntimeError(Errors.runtime_error(f"Cannot pass data through this edge because either start node or end node are None:"+"\n\t "+f"start_node = {self.start_node}"+"\n\t "+f"end_node = {self.end_node}",self,"pass_forward"))
        if(self.start_node.N_out==1):
            if(len(self.start_list)==1):
                return x
            else:
                return self.end_space.join(*[x for _ in self.start_list])
        else:
            x_split=self.start_node.op.codomain.split(x)
            if(len(self.start_list)==1):
                return x_split[self.start_list[0]]
            else:
                return self.end_space.join(*[x_split[i] for i in self.start_list])
            

    def pass_backward(self,y):
        """Passes and transforms data backwards through the edge.

        Parameters
        ----------
        y : array-like 
            element of the part of the domain of the operator of the output node.

        Returns
        -------
        list
            List where each entry corresponds to that part of the codomain of the operator of the input node.
        """ 
        if(self.end_node.N_in>1):
            y=self.end_node.op.domain.split(y)[self.end_index]
        x_vals=[None] * self.start_node.N_out
        if(len(self.start_list)==1):
            x_vals[self.start_list[0]]=y
            return x_vals
        else:
            y_split=self.end_space.split(y)
            for i,index in enumerate(self.start_list):
                if(x_vals[index] is None):
                    x_vals[index]=y_split[i]
                else:
                    x_vals[index]+=y_split[i]
            return x_vals
        
    def __getitem__(self,index):
        if not isinstance(index,int):
            raise IndexError(Errors.indexation(index=index,obj=self,add_info="Indexation of edge only allows integers 0 or 1, for start and end node respectively!"))
        if index == 0:
            return self.start_node.op,self.start_list
        elif index == 1:
            return self.end_node.op,self.end_index
        else:
            raise IndexError(Errors.indexation(index=index,obj=self,add_info="Only index 0 for start,1 for end allowed."))
        

class OperatorGraph(Operator):
    """Operator that consists of different operators that are connected in a Graph structure.

    Parameters
    ----------
    operators : list of regpy.vecsps.Operator
        The underlying operators
    edges : list of tuple
        Tuple representing edges have the form ((input operator,[input indices]),(output operator,output index))
    calc_exec_order : bool, optional
        If True the order of calculations of the operators is computed. Else it is assumed to be the order in which
        the operators are given. Defaults to True.
    """
    def __init__(self, operators,edges,calc_exec_order=True):
        if not isinstance(operators,list) or any(not isinstance(op_i,Operator) for op_i in operators):
            raise TypeError(Errors.type_error("To construct an operator graph the operators need to be a list of proper RegPy operators"))
        # if not OperatorGraph._validate_edges_input(edges):
            # raise ValueError(Errors.value_error("The given edges do not follow the desired format of a list of ((input operator,[input indices]),(output operator,output index)). Was given: "+"\n\t"+f"edges = {edges}"))
        self.node_dict={op:OperatorNode(op) for op in operators}
        self.edges=[]
        linear=all(op.linear for op in  self.node_dict.keys())
        ed_in,ed_middle,ed_out=OperatorGraph._clean_edge_data(edges)
        self.N_in=len(ed_in)
        self.N_out=len(ed_out)
        domain=self._compute_domain(ed_in)
        self.input_op=Identity(domain,copy=False)
        self.node_dict.update({self.input_op:OperatorNode(self.input_op)})
        codomains=[]
        for edge in ed_in:
            op_end,end_index=edge[1]
            new_edge=Edge(self.node_dict[self.input_op],self.node_dict[op_end],edge[0][1],end_index)
            self.edges.append(new_edge)
        for edge in ed_middle:
            op_start,start_list=edge[0]
            op_end,end_index=edge[1]
            new_edge=Edge(self.node_dict[op_start],self.node_dict[op_end],start_list,end_index)
            self.edges.append(new_edge)
        for edge in ed_out:
            op_start,start_list=edge[0]
            new_edge=Edge(self.node_dict[op_start],None,start_list,0)
            self.edges.append(new_edge)
            codomains.append(new_edge.construct_start_space())
        codomain=vecsps.DirectSum(*codomains) if len(codomains)>1 else codomains[0]
        self.output_op=Identity(codomain,copy=False)
        self.node_dict.update({self.output_op:OperatorNode(self.output_op)})
        offset=len(ed_in)+len(ed_middle)
        for i in range(len(ed_out)):
            edge=self.edges[offset+i]
            edge.end_node=self.node_dict[self.output_op]
            edge.end_index=i
            self.node_dict[self.output_op].input_edges[i]=edge
        if(calc_exec_order):
            self.operators=self._calc_exec_order()
        else:
            self.operators=self.input_op+operators+self.output_op
        super().__init__(self.input_op.domain, self.output_op.codomain, linear)

    # @staticmethod
    # def _validate_edges_input(edges):
    #     """ Validates the input of edges determining if edges are a list of the format  
    #     ((input operator,[input indices]),(output operator,output index)). 
        
    #     Parameters
    #     ----------
    #     edges : list of tuple
    #         Tuple representing edges have the form ((input operator,[input indices]),(output operator,output index))
        
    #     Returns
    #     -------
    #     boolean
    #         Returns true if the given edge set has a valid structure.
    #     """
    #     # needs to be a list or tuple
    #     if not isinstance(edges,(tuple,list)):
    #         return False
    #     # each entry needs and in and out 
    #     elif any(not isinstance(e,tuple) or len(e)!=2 for e in edges):
    #         return False
    #     # each output needs to be (output operator,output index)
    #     elif any(len(e[1])!=2 or 
    #              not isinstance(e[1][0],Operator) or 
    #              not isinstance(e[1][1],int) for e in edges):
    #         return False
    #     # each input needs to be (input operator,[input indices])
    #     elif any(len(e[0])!=2 or 
    #              not isinstance(e[0][0],Operator) or 
    #              not isinstance(e[0][1],list) or 
    #              any(ind is not None or not isinstance(ind,int) for ind in e[0][1]) for e in edges):
    #         return False
    #     else:
    #         return True

    def _clean_edge_data(edge_data):
        """Cleans up edge data. Removes duplicates and overwrites empty inputs if necessary.
        Additionally sorts edge data into incoming, middle and outgoing.

        Parameters
        ----------
        edge_data : list 
            List of edge data in format specified in constructor

        Raises
        ------
        ValueError
            if multiple different edges are assigned to same input of operator

        Returns
        -------
        ed_in (list)
            list of data for incoming edges

        ed_middle (list)
            list of data for middle edges

        ed_out (list)
            list of data for outgoing edges 
        """        
        ed_dict={}
        ed_in=[]
        ed_middle=[]
        ed_out=[]
        for ed in edge_data:
            if(ed[0][0]==None):#input edges
                if(ed[1] not in ed_dict.keys()):
                    ed_in.append(ed)
                    ed_dict.update({ed[1]:('in',ed)})
            elif(ed[1][0]==None):#output edges
                ed_out.append(ed)
            else:#middle edges
                if(ed[1] not in ed_dict.keys()):
                    ed_middle.append(ed)
                    ed_dict.update({ed[1]:('middle',ed)})
                elif(ed_dict[ed[1]][1]!=ed):
                    raise ValueError(f"Conflicting edge data {ed_dict[ed[1]][1]} and {ed}.")
        return ed_in,ed_middle,ed_out

    def _compute_domain(self,ed_in):
        """Computes domain of the whole operator by extracting and combining the domains from the input edge data

        Parameters
        ----------
        ed_in : list of tuple 
            List of input edge data

        Returns
        -------
        vecsps.VectorSpace
            domain of the operator
        """        
        space_dict={}
        for ed in ed_in:
            total_domain=ed[1][0].domain[ed[1][1]] if isinstance(ed[1][0].domain,vecsps.DirectSum) else ed[1][0].domain
            for j,index in enumerate(ed[0][1]):
                domain=total_domain[j] if isinstance(total_domain,vecsps.DirectSum) else total_domain
                if(index not in space_dict.keys()):
                    space_dict.update({index:domain})
                elif(space_dict[index]!=domain):
                    raise ValueError(f"Input {index} used for different spaces")
        return vecsps.DirectSum(*(space_dict[j] for j in sorted(space_dict.keys()))) if len(space_dict.keys())>1 else space_dict[space_dict.keys()[0]]

    def _calc_exec_order(self):
        """Computes execution order using a topological sort.

        Raises
        ------
        ValueError
            If the graph has cycles or start and end are not connected.

        Returns
        -------
        list
            List of ordered operators
        """        
        in_sets={op:self.node_dict[op].get_in_nodes() for op in self.node_dict.keys()}
        out_sets={op:self.node_dict[op].get_out_nodes() for op in self.node_dict.keys()}
        current_ops={self.input_op}
        op_order=[]
        while(current_ops!={self.output_op}):
            if(current_ops==set()):
                raise ValueError('Given graph has cycles or is not connected.')
            current_op=current_ops.pop()
            op_order.append(current_op)
            potential_next=out_sets[current_op]
            for next_node in potential_next:
                in_sets[next_node.op].remove(self.node_dict[current_op])
                if(in_sets[next_node.op]==set()):
                    current_ops.add(next_node.op)
        op_order.append(self.output_op)
        return op_order
    
    def _eval(self, x, differentiate=False):
        data_dict={self.node_dict[self.input_op]:x}
        for i in range(1,len(self.operators)):
            current_node=self.node_dict[self.operators[i]]
            x_input=current_node.combine_input(data_dict)
            if(current_node.op.linear):
                y=current_node.op(x_input)
                # y=current_node.op._eval(x_input)
            else:
                if(differentiate):
                    y,current_node.deriv=current_node.op.linearize(x_input)
                else:
                    y=current_node.op(x_input)
                # y=current_node.op._eval(x_input,differentiate=differentiate)
            data_dict.update({current_node:y})
        return data_dict[self.node_dict[self.output_op]]
    
    def _derivative(self, x):
        data_dict={self.node_dict[self.input_op]:x}
        for i in range(1,len(self.operators)):
            current_node=self.node_dict[self.operators[i]]
            x_input=current_node.combine_input(data_dict)
            if(current_node.op.linear):
                y=current_node.op(x_input)
            else:
                y=current_node.deriv(x_input)
                # y=current_node.op._derivative(x_input)
            data_dict.update({current_node:y})
        return data_dict[self.node_dict[self.output_op]]
    
    def _adjoint(self, y):
        data_dict={self.node_dict[self.output_op]:y}
        for i in range(1,len(self.operators)):
            current_node=self.node_dict[self.operators[len(self.operators)-1-i]]
            y_input=current_node.combine_output(data_dict)
            if(current_node.op.linear):
                x=current_node.op.adjoint(y_input)
            else:
                x=current_node.deriv.adjoint(y_input)
            data_dict.update({current_node:x})
        return data_dict[self.node_dict[self.input_op]]



