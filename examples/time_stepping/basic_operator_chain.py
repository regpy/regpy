from regpy.operators.graph_operator import OperatorGraph

def time_dependent_operator_chain(time_step_ops,output_intermediate_solutions=False,separate_parameter_inputs=False):
    """Connects operators to form a chain that can be used for time stepping.

    Parameters:
        time_step_ops (list of Operator): List of operators ordered by time step. 
        output_intermediate_solutions (bool, optional): Controls if intermediate solutions are outputted. Defaults to False.
        separate_parameter_inputs (bool, optional): Controls if each operator gets a seperate additional input. Otherwise all get the same input. Defaults to False.

    Returns:
        OperatorGraph: Operator that has the connections depicted below. 
        False, True (Default)
        In0--->Op0--->Op1--->...--->OpN-1--->OpN--->Out0
                /      /             /        /  
               /      /             /        /
             In1     In2          InN       InN+1
        ################################################
        True, True
        In0--->Op0--->Op1--->...--->OpN-1--->OpN--->OutN
               / \    / \           / \      /  
              / Out0 /  Out1       / OutN-1 /
            In1     In2          InN       InN+1
        ################################################
        False, False
        In0--->Op0--->Op1--->...--->OpN-1--->OpN--->Out0
               /      /             /        /  
              /      /             /        /
        In1__/______/_____________/________/
        ################################################
        True, False
        In0--->Op0--->Op1--->...--->OpN-1--->OpN--->OutN
               / \    / \           / \      /  
              / Out0 /  Out1       / OutN-1 /
        In1__/______/_____________/________/
        """
    edges=[((None,[0]),(time_step_ops[0],0))]#connect start value
    edges+=[((time_step_ops[i],[0]),(time_step_ops[i+1],0)) for i in range(len(time_step_ops)-1)]#passing of solution through time
    if(separate_parameter_inputs):
        edges+=[((None,[j+1]),(time_step_op,1)) for j,time_step_op in enumerate(time_step_ops)]#connecting parameters to operators
    else:
        edges+=[((None,[1]),(time_step_op,1)) for j,time_step_op in enumerate(time_step_ops)]#connecting parameter to operators
    if(output_intermediate_solutions):
        edges+=[((time_step_ops[i],[0]),(None,0)) for i in range(len(time_step_ops)-1)]#get output from each operator except the last
    edges.append(((time_step_ops[-1],[0]),(None,0)))#get output from final operator
    return OperatorGraph(time_step_ops,edges)



#Example where time steps are replaced by simple product operators
from regpy.operators import Product
from regpy.vecsps import UniformGridFcts
from regpy.util.operator_tests import test_operator

N=9
doms=[UniformGridFcts(2,3) for _ in range(N)]
time_step_ops=[Product(doms[0]+doms[i]) for i in range(1,N)]

chain=time_dependent_operator_chain(time_step_ops,output_intermediate_solutions=False,separate_parameter_inputs=True)
test_operator(chain)
print('Operator passed tests successfully!')
print('Zeros in domain:')
print(chain.domain.zeros())
print('Result for constant 2 input:')
print(chain(chain.domain.ones()*2))