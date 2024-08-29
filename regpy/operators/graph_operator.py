from regpy.operators import Operator,PartOfOperator,Identity
from regpy import vecsps
import itertools



class OperatorNode:

    def __init__(self,op):
        self.op=op
        self.N_in=len(self.op.domain.summands) if isinstance(self.op.domain,vecsps.DirectSum) else 1
        self.N_out=len(self.op.codomain.summands) if isinstance(self.op.domain,vecsps.DirectSum) else 1
        self.input_edges=[None]*self.N_in
        self.output_edges=[]

    def __str__(self):
        return str(self.op)

    def get_free_inputs(self):
        return set([i for i in range(self.N_in) if self.input_edges[i] is None])
    
    def get_in_nodes(self):
        return set([edge.start_node for edge in self.input_edges if edge is not None and edge.start_node is not None])
    
    def get_out_nodes(self):
        return set([edge.end_node for edge in self.output_edges if edge is not None and edge.end_node is not None])
    
    def combine_input(self,data_dict):
        assert all(edge is not None for edge in self.input_edges)
        if(self.N_in==1):
            edge=self.input_edges[0]
            return edge.pass_forward(data_dict[edge.start_node])
        else:
            return self.op.domain.join(*[edge.pass_forward(data_dict[edge.start_node]) for edge in self.input_edges])
        
    def combine_output(self,data_dict):
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

    def __init__(self,start_node,end_node,start_list,end_index):
        assert isinstance(start_node,OperatorNode) or start_node is None
        assert isinstance(end_node,OperatorNode) or end_node is None
        assert isinstance(start_list,list)
        assert isinstance(end_index,int)
        self.start_node=start_node
        self.end_node=end_node
        self.start_list=start_list
        self.end_index=end_index
        if(self.start_node is not None):
            assert all(i>=0 and i<self.start_node.N_out for i in self.start_list)
            self.start_node.output_edges.append(self)
        if(self.end_node is not None):
            assert end_index<self.end_node.N_in
            assert self.end_node.input_edges[self.end_index] is None
            self.end_node.input_edges[self.end_index]=self
            if(self.end_node.N_in==1):
                self.end_space=self.end_node.op.domain
            else:
                self.end_space=self.end_node.op.domain.summands[self.end_index]

    def construct_start_space(self):
        assert self.start_node is not None
        if(self.start_node.N_out==1):
            return self.start_node.op.codomain
        if(len(self.start_list)==1):
            return self.start_node.op.codomain.summands[self.start_list[0]]
        return vecsps.DirectSum(*[self.start_node.codomain.summands[i] for i in self.start_list])

    def remove(self):
        if(self.start_node is not None):
            self.start_node.output_edges.remove(self)
        if(self.end_node is not None):
            self.end_node.input_edges[self.end_index]=None
        self.start_node=None
        self.end_node=None

    def __str__(self):
        return f"{self.start_node}{self.start_list}-->[{self.end_index}]{self.end_node}"

    def pass_forward(self,x):
        assert self.end_node is not None and self.start_node is not None
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
        assert isinstance(index,int)
        if index == 0:
            return self.start_node.op,self.start_list
        elif index == 1:
            return self.end_node.op,self.end_index
        else:
            raise IndexError("Only index 0 for start,1 for end allowed.")
        

class OperatorGraph(Operator):

    def __init__(self, operators,edges,calc_exec_order=True):
        self.node_dict={op:OperatorNode(op) for op in operators}
        self.edges=[]
        linear=all(op.linear for op in  self.node_dict.keys())
        ed_in,ed_middle,ed_out=OperatorGraph._clean_edge_data(edges)
        self.N_in=len(ed_in)
        self.N_out=len(ed_out)
        domains=[]
        codomains=[]
        for edge in ed_in:
            op_end,end_index=edge[1]
            new_edge=Edge(None,self.node_dict[op_end],[0],end_index)
            self.edges.append(new_edge)
            domains.append(new_edge.end_space)
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
        domain=vecsps.DirectSum(*domains) if len(domains)>1 else domains[0]
        codomain=vecsps.DirectSum(*codomains) if len(codomains)>1 else codomains[0]
        self.input_op=Identity(domain,copy=False)
        self.output_op=Identity(codomain,copy=False)
        self.node_dict.update({self.input_op:OperatorNode(self.input_op),self.output_op:OperatorNode(self.output_op)})
        for i in range(len(ed_in)):
            edge=self.edges[i]
            edge.start_node=self.node_dict[self.input_op]
            edge.start_list=[i]
            self.node_dict[self.input_op].output_edges.append(edge)
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

    def _clean_edge_data(edge_data):
        ed_dict={}
        ed_in=[]
        ed_middle=[]
        ed_out=[]
        for ed in edge_data:
            if(ed[0][0]==None):
                if(ed[1] not in ed_dict.keys()):
                    ed_in.append(ed)
                    ed_dict.update({ed[1]:('in',ed)})
            elif(ed[1][0]==None):
                ed_out.append(ed)
            else:
                if(ed[1] not in ed_dict.keys()):
                    ed_middle.append(ed)
                    ed_dict.update({ed[1]:('middle',ed)})
                elif(ed_dict[ed[1]][0]=='in'):
                    ed_middle.append(ed)
                    ed_in.remove(ed_dict[ed[1]][1])
                    ed_dict[ed[1]]=('middle',ed)
                elif(ed_dict[ed[1]][1]!=ed):
                    raise ValueError(f"Conflicting edge data {ed_dict[ed[1]][1]} and {ed}.")
        return ed_in,ed_middle,ed_out

    def _calc_exec_order(self):
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
                y=current_node.op._eval(x_input)
            else:
                y=current_node.op._eval(x_input,differentiate=differentiate)
            data_dict.update({current_node:y})
        return data_dict[self.node_dict[self.output_op]]
    
    def _derivative(self, x):
        data_dict={self.node_dict[self.input_op]:x}
        for i in range(1,len(self.operators)):
            current_node=self.node_dict[self.operators[i]]
            x_input=current_node.combine_input(data_dict)
            if(current_node.op.linear):
                y=current_node.op._eval(x_input)
            else:
                y=current_node.op._derivative(x_input)
            data_dict.update({current_node:y})
        return data_dict[self.node_dict[self.output_op]]
    
    def _adjoint(self, y):
        data_dict={self.node_dict[self.output_op]:y}
        for i in range(1,len(self.operators)):
            current_node=self.node_dict[self.operators[len(self.operators)-1-i]]
            y_input=current_node.combine_output(data_dict)
            x=current_node.op._adjoint(y_input)
            data_dict.update({current_node:x})
        return data_dict[self.node_dict[self.input_op]]


def merge_operators(*op_eds):#input format is tuple with elements of form (ops,edges,N_in,N_out) edges are sorted
    ops=set()
    edge_data=[]
    for op_ed in op_eds:
        ops|=set(op_ed[0])
        edges=op_ed[1]
        N_out=op_ed[3]
        if(N_out==1):
            edge_data+=edges
        else:
            #combine multiple output edges into one single edge for each operator
            out_edges=edges[len(edges)-N_out:]
            edge_data+=edges[:len(edges)-N_out]
            output_edge=((out_edges[0][0][0],itertools.chain.from_iterable(edge[0][1] for edge in out_edges)),(None,0))
            edge_data.append(output_edge)
    return OperatorGraph(list(ops),edge_data)

def concatenate_operators(op_eds_start,op_eds_end):#input format is (ops,edges,N_in,N_out) edges are sorted
    assert(op_eds_start[3]==op_eds_end[2])
    offset_output=len(op_eds_start[1])-op_eds_start[3]
    offset_input=op_eds_end[2]
    start_output_eds=op_eds_start[1][offset_output:]
    end_input_eds=op_eds_end[1][:offset_input]
    bridge_eds=[]
    for start_ed,end_ed in zip(start_output_eds,end_input_eds):
        bridge_eds.append((start_ed[0],end_ed[1]))
    return OperatorGraph(list(set(op_eds_start[0]+op_eds_end[0])),op_eds_start[1][:offset_output]+bridge_eds+op_eds_end[1][offset_input:])

def get_operators_and_edges(op):#output format is (ops,edges,N_in,N_out) edges are sorted
    assert isinstance(op,Operator)
    if(isinstance(op,OperatorGraph)):
        ops=op.operators[1:len(op.operators)-1]
        edge_data=[]
        for i,edge in enumerate(op.edges):
            ed=[edge[0],edge[1]]
            if(i<op.N_in):
                ed[0]=(None,[0])
            if(i>=len(op.edges)-op.N_out):
                ed[1]=(None,0)
            edge_data.append(tuple(ed))
        return ops,edge_data,op.N_in,op.N_out
    if(isinstance(op,PartOfOperator)):
        ops=[op.base_op]
        indices=[op.index] if isinstance(op.index,int) else op.index
        N_in=1 if not isinstance(op.domain,vecsps.DirectSum) else len(op.domain.summands)
        N_out=len(indices)
        edge_data=[((None,[0]),(op.base_op,i)) for i in range(N_in)]
        edge_data+=[((op.base_op,[indices[i]]),(None,0)) for i in range(N_out)]
        return ops,edge_data,N_in,N_out
    if(isinstance(op,Operator)):
        ops=[op]
        N_in=1 if not isinstance(op.domain,vecsps.DirectSum) else len(op.domain.summands)
        N_out=1 if not isinstance(op.codomain,vecsps.DirectSum) else len(op.codomain.summands)
        edge_data=[((None,[0]),(op,i)) for i in range(N_in)]
        edge_data+=[((op,[i]),(None,0)) for i in range(N_out)]
        return ops,edge_data,N_in,N_out



