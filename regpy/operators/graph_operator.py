from regpy.operators import Operator,PartOfOperator
from regpy import vecsps

class OperatorAdress:

    def get_adress_from_data(data,is_input_adress):
        if(data is None):
            return OperatorAdressNull(is_input_adress)
        if(isinstance(data,OperatorAdress)):
            assert data.is_input_adress==is_input_adress
            return data
        if(isinstance(data,OperatorNode)):
            return OperatorAdressAll(data,is_input_adress)
        if(isinstance(data,tuple)):
            assert isinstance(data[0],OperatorNode)
            assert len(tuple)==2
            if(isinstance(data[1]),int):
                return OperatorAdressIndex(data[0],is_input_adress,data[1])
            if(isinstance(data[1],list)):
                return OperatorAdressIndexList(data[0],is_input_adress,data[1])
        raise ValueError(f"{data} cannot be converted to OperatorAdress")

    def __init__(self,node,is_input_adress,adress_data=None):
        self.node=node
        assert isinstance(is_input_adress,bool)
        self.is_input_adress=is_input_adress#decides if adress is adress on input or output of operator
        self.adress_data=adress_data
        if(node is not None):
            self.relevant_dom=node.op.domain if is_input_adress else node.op.codomain

    def __str__(self):
        in_out="input" if self.is_input_adress else "output"
        return f"{self.node.op}:{in_out}:{self.adress_data}"


    def get_part_forward(self,x):
        raise NotImplementedError
    
    def get_part_backward(self,y,split=False):
        raise NotImplementedError

class OperatorAdressNull(OperatorAdress):

    def __init__(self,is_input_adress=False):
        super().__init__(None,is_input_adress)
    
    def __str__(self):
        return f"NullAdress"

    def get_part_forward(self,x):
        raise ValueError("Null adress cannot divide data.")
    
    def get_part_backward(self,y,split=False):
        raise ValueError("Null adress cannot divide data.")

class OperatorAdressAll(OperatorAdress):

    def __init__(self,node,is_input_adress):
        super().__init__(node,is_input_adress,'all')

    def get_part_forward(self, x):
        if(not self.is_input_adress):
            assert x in self.node.codomain
            return x
        else:
            if(isinstance(x,tuple)):
                return self.relevant_dom.join(*x)
            else:
                return x

    def get_part_backward(self,y,split=False):
        if(self.is_input_adress):
            if(split):
                assert isinstance(self.relevant_dom,vecsps.DirectSum)
                return self.relevant_dom.split(y)
            else:
                return y
        else:
            if(isinstance(y,tuple)):
                return [('all',self.relevant_dom.join(*y))]
            else:
                return [('all',y)]


class OperatorAdressIndex(OperatorAdress):

    def __init__(self,node,is_input_adress,index):
        super().__init__(node,is_input_adress,adress_data=index)
        assert isinstance(self.relevant_dom,vecsps.DirectSum)
        assert index<len(self.relevant_dom.summands) and index>=0
        self.relevant_subdom=self.relevant_dom.summands[self.adress_data]
        
        
    def get_part_forward(self,x):
        if(not self.is_input_adress):
            assert x in self.node.codomain
            return self.relevant_dom.split(x)[self.adress_data]
        else:
            if(isinstance(x,tuple)):
                return self.relevant_subdom.join(*x)
            else:
                return x
            
    def get_part_backward(self,y,split=False):
        if(self.is_input_adress):
            if(split):
                assert isinstance(self.relevant_subdom,vecsps.DirectSum)
                return self.relevant_subdom.split(y)
            else:
                return y
        else:
            if(isinstance(y,tuple)):
                assert isinstance(self.relevant_subdom,vecsps.DirectSum)
                return [(self.adress_data,self.relevant_subdom.join(*y))]
            else:
                return [(self.adress_data,y)]
    
class OperatorAdressIndexList(OperatorAdress):

    def __init__(self, node, index_list):
        is_input_adress=False#setting multiple inputs at once is forbidden
        self.relevant_dom=node.op.domain if is_input_adress else node.op.codomain
        assert isinstance(self.relevant_dom,vecsps.DirectSum)
        assert len(index_list)>1 #Use index instead of list of length one
        assert all(i<len(self.relevant_dom.summands) and i>=0 for i in index_list)
        super().__init__(node, is_input_adress, index_list)

    def get_part_forward(self,x):
        x_split=self.relevant_dom.split(x)
        return tuple([x_split[i] for i in self.adress_data])
    
    def get_part_backward(self, y, split=False):
        assert isinstance(y,tuple)
        return list(zip(self.adress_data,y))


class OperatorNode:

    def get_node_from_operator(op):
        if(isinstance(op,PartOfOperator)):
            node=OperatorNode(op.base_op)
            out_adress=None
            if(isinstance(op.index,int)):
                out_adress=OperatorAdressIndex(node,False,op.index)
            elif(isinstance(op.index,list)):
                if(len(op.index)==1):
                    out_adress=OperatorAdressIndex(node,False,op.index)
                else:
                    out_adress=OperatorAdressIndexList(node,False,op.index)
            node.output_edges.append(Edge(out_adress,OperatorAdressNull(is_input_adress=True)))
            return node
        if(isinstance(op,Operator)):
            node=OperatorNode(op)
            node.output_edges.append(Edge(OperatorAdressAll(node,False),OperatorAdressNull(is_input_adress=True)))

    def __init__(self,op):
        self.op=op
        self.N_in=len(self.op.domain.summands) if isinstance(self.op.domain,vecsps.DirectSum) else 1
        self.N_out=len(self.op.codomain.summands) if isinstance(self.op.domain,vecsps.DirectSum) else 1
        self.input_edges=Edge(OperatorAdressNull(),OperatorAdressAll(self,True))#can also be list
        self.output_edges=[]

    def get_free_inputs(self):
        free_inputs=set()
        if(isinstance(self.input_edges,list)):
            for i, input_edge in enumerate(self.input_edges):
                if(isinstance(input_edge.start_adress,OperatorAdressNull)):
                    free_inputs.add(i)
            if(len(free_inputs)==self.N_in):
                free_inputs.add('all')
            return free_inputs
        else:
            if(isinstance(self.input_edges.start_adress,OperatorAdressNull)):
                return set(['all']+list(range(self.N)))
            
    def combine_input(self,data_dict):
        if(isinstance(self.input_edges),Edge):
            return self.input_edges.pass_forward(data_dict[self.input_edges.start_adress.node])
        else:
            parts=[]
            for edge in self.input_edges:
                parts.append(edge.pass_forward(data_dict[edge.start_adress.node]))
            return self.op.domain.join(*parts)
    
    def combine_output(self,data_dict):
        if(not isinstance(self.op.codomain,vecsps.DirectSum)):
            res=None
            for edge in self.output_edges:
                data=edge.pass_backward(data_dict[edge.end_adress.node])[0][1]
                if(res is None):
                    res=data
                else:
                    res+=data
            return res
        parts_all=None
        parts_list=[None for _ in range(self.N_out)]
        all_none=True
        for edge in self.output_edges:
            data_parts=edge.pass_backward(data_dict[edge.end_adress.node])
            for part in data_parts:
                if(part[0]=='all'):
                    if(parts_all is None):
                        parts_all=part[1]
                    else:
                        parts_all+=part[1]
                else:
                    all_none=False
                    if(parts_list[part[0]] is None):
                        parts_list[part[0]]=part[1]
                    else:
                        parts_list[part[0]]+=part[1]
        if(all_none):
            if(parts_all is None):
                return self.codomain.zeros()
            else:
                return parts_all
        else:
            res=self.codomain.join(*parts_list)
            if(parts_all is not None):
                res+=parts_all
            return res
    
    def connect(start_adress,end_adress,override_inputs=False):
        new_edge=Edge(start_adress,end_adress)
        #check if input is free
        if(not override_inputs):
            free_inputs=end_adress.node.get_free_inputs()
            if(end_adress.adress_data not in free_inputs):
                raise ValueError("Cannot override input {end_adress.adress_data}")
        if(end_adress.adress_data=='all'):
            end_adress.node.input_edges=new_edge
        else:
            end_adress.node.input_edges[end_adress.adress_data]=new_edge
        start_adress.node.output_edges.append(new_edge)



class Edge:

    def __init__(self,start_adress,end_adress):
        assert start_adress.is_input_adress==False
        assert end_adress.is_input_adress
        self.start_adress=start_adress
        self.end_adress=end_adress

    def pass_forward(self,x):
        return self.end_adress.get_part_forward(self.start_adress.get_part_forward(x))

    def pass_backward(self,y):
        if(not isinstance(self.start_adress,OperatorAdressIndexList)):
            return self.start_adress.get_part_backward(self.end_adress.get_part_backward(y))
        else:
            return self.start_adress.get_part_backward(self.end_adress.get_part_backward(y,split=True))


class OperatorGraph(Operator):

    def __init__(self, operators,edges):
        self.node_dict={op:OperatorNode.get_node_from_operator(op) for op in operators}
        linear=all(op.linear for op in  self.node_dict.keys())
        super().__init__(None, None, linear)


from regpy.operators import PtwMultiplication

dom=vecsps.UniformGridFcts(2,4)
A=PtwMultiplication(dom,2)
B=PtwMultiplication(dom,3)
C=PtwMultiplication(dom,4)

og=OperatorGraph([A,B,C],None)
print(og)