from regpy.operators import Operator,PartOfOperator,Identity
from regpy import vecsps



class OperatorNode:

    def __init__(self,op):
        self.op=op
        self.N_in=len(self.op.domain.summands) if isinstance(self.op.domain,vecsps.DirectSum) else 1
        self.N_out=len(self.op.codomain.summands) if isinstance(self.op.domain,vecsps.DirectSum) else 1
        self.input_edges=[None for _ in range(self.N_in)]
        self.output_edges=[]

    def __str__(self):
        return str(self.op)

    def get_free_inputs(self):
        free_inputs=set()
        for i in range(self.N_in):
            if(self.input_edges[i] is None):
                free_inputs.add(i)
        return free_inputs
    
    def combine_input(self,data_dict):
        assert all(edge is not None for edge in self.input_edges)
        if(self.N_in==1):
            edge=self.input_edges[0]
            return edge.pass_forward(data_dict[edge.start_node])
        else:
            inputs=[edge.pass_forward(data_dict[edge.start_node]) for edge in self.input_edges]
            return self.op.domain.join(*inputs)
        
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
            return self.data_list[0]
        for i,d in enumerate(data_list):
            if(d is None):
                data_list[i]=self.op.codomain.summands[i].zeros()
        return self.op.codomain.join(*data_list)

class Edge:

    def __init__(self,start_node,end_node,start_list,end_index,overwrite=False):
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
            assert overwrite or self.end_node.input_edges[self.end_index] is None
            if(overwrite and self.end_node.input_edges[self.end_index] is not None):
                self.end_node.input_edges[self.end_index].remove()
            self.end_node.input_edges[self.end_index]=self
        if(self.end_node is not None):
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
        x_split=self.start_node.codomain.split(x)
        if(len(self.start_list)==1):
            return x_split[self.start_list[0]]
        else:
            return self.end_space.join(*[x_split[i] for i in self.start_list])
            

    def pass_backward(self,y):
        if(self.end_node.N_in>1):
            y=self.end_node.op.domain.split(y)[self.end_index]
        x_vals=[None for _ in range(self.start_node.N_out)]
        if(len(self.start_list)==1):
            x_vals[self.start_list[0]]=y
            return x_vals
        y_split=self.end_space.split(y)
        for i,index in enumerate(self.start_list):
            if(x_vals[index] is None):
                x_vals[index]=y_split[i]
            else:
                x_vals[index]+=y_split[i]
        return x_vals
        

class OperatorGraph(Operator):

    def __init__(self, operators,edges):
        self.node_dict={op:OperatorNode(op) for op in operators}
        self.edges=[]
        linear=all(op.linear for op in  self.node_dict.keys())
        start_edges=[]
        domains=[]
        end_edges=[]
        codomains=[]
        for edge in edges:
            op_start,start_list=edge[0]
            op_end,end_index=edge[1]
            start_node=self.node_dict[op_start] if op_start is not None else None
            end_node=self.node_dict[op_end] if op_end is not None else None
            new_edge=Edge(start_node,end_node,start_list,end_index)
            self.edges.append(new_edge)
            if(op_start is None):
                start_edges.append(new_edge)
                domains.append(new_edge.end_space)
            if(op_end is None):
                end_edges.append(new_edge)
                codomains.append(new_edge.construct_start_space())
        domain=vecsps.DirectSum(*domains) if len(domains)>1 else domains[0]
        codomain=vecsps.DirectSum(*codomains) if len(codomains)>1 else codomains[0]
        self.input_op=Identity(domain,copy=False)
        self.output_op=Identity(codomain,copy=False)
        self.node_dict.update({self.input_op:OperatorNode(self.input_op),self.output_op:OperatorNode(self.output_op)})
        for i,edge in enumerate(start_edges):
            edge.start_node=self.node_dict[self.input_op]
            edge.start_list=[i]
            self.node_dict[self.input_op].output_edges.append(edge)
        for i,edge in enumerate(end_edges):
            edge.end_node=self.node_dict[self.output_op]
            edge.end_index=i
            self.node_dict[self.output_op].input_edges[i]=edge
        super().__init__(self.input_op.domain, self.output_op.codomain, linear)


from regpy.operators import PtwMultiplication


dom=vecsps.UniformGridFcts(2,4)
A=PtwMultiplication(dom,2)
B=PtwMultiplication(dom,3)
C=PtwMultiplication(dom,4)

og=OperatorGraph([A,B,C],[((None,[0]),(A,0)),((A,[0]),(B,0)),((B,[0]),(C,0)),((C,[0]),(None,0))])
for v in og.node_dict.values():
    print(v)
    print(v.input_edges)
print(og)
print(og.domain)
print(og.codomain)