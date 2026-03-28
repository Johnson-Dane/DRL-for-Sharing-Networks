import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Composite
from torchrl.envs import (
    EnvBase,
    Transform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs


import numpy as np
from torch.distributions import Multinomial, Exponential
from pypoman.duality import compute_polytope_vertices

#Function which speeds up network parameter creation for test networks
def param_creator(network: str, #valid options are '2LLN', '3LLN', and 'C3LN'
                  traffic_rho: float, #traffic parameter in (0,1) measuring what proportion of resource capacity is needed to process jobs at the exact rate they arrive.  I used traffic_rho=.8 (Note: rho is used in traffic_rho to be consistent with the notation in the HGI paper referenced in the documentation, but IT IS DIFFERENT from rho=alpha/beta used in the DRL algorithm)
                  h: torch.tensor, #linear holding cost vector.  I used 1 for all nonlocal jobs, 1 for local jobs in the "uniform" example and 4 for local jobs in the "increased local" example
                  cost_bnd: torch.tensor, #Used to bound the state space for training.  I used 150 for the "uniform" example and 400 for the "increased local" example.  Once a state is reached whose holding cost is at least the cost_bnd the reward is set to -cost_bnd/kappa (as if the trajectory incurred a constant holding cost of cost_bnd indefinitely), the queue length is fixed, and the trajecotry is terminated by setting ('next','terminated')=True
                  device):
    if network=='2LLN':
        td_params = TensorDict(
            {
                "alpha":torch.tensor([.5*traffic_rho,.5*traffic_rho,.5*traffic_rho]),
                "beta":torch.tensor([1.,1.,1.]),
                "K":torch.tensor([[1.,0.,1.],[0.,1.,1.]]),
                "C":torch.tensor([1.,1.]),
                "init_queue":torch.zeros(3),
                "kappa":torch.tensor([.01]),
            },
            [],
            device=device,
        )
    elif network=='3LLN':
        td_params = TensorDict(
            {
                "alpha":torch.tensor([.5*traffic_rho,.5*traffic_rho,.5*traffic_rho,.5*traffic_rho]),
                "beta":torch.tensor([1.,1.,1.,1.]),
                "K":torch.tensor([[1.,0.,0.,1.],[0.,1.,0.,1.],[0.,0.,1.,1.]]),
                "C":torch.tensor([1.,1.,1.]),
                "init_queue":torch.zeros(4),
                "kappa":torch.tensor([.01]),
            },
            [],
            device=device,
        )
    elif network=='C3LN':
        td_params = TensorDict(
            {
                "alpha":torch.tensor([.4*traffic_rho,.2*traffic_rho,.5*traffic_rho,.3*traffic_rho,.2*traffic_rho,.3*traffic_rho]),
                "beta":torch.tensor([1.,1.,1.,1.,1.,1.]),
                "K":torch.tensor([[1.,0.,0.,1.,0.,1.],[0.,1.,0.,1.,1.,1.],[0.,0.,1.,0.,1.,1.]]),
                "C":torch.tensor([1.,1.,1.]),
                "init_queue":torch.zeros(6),
                "kappa":torch.tensor([.01]),
                'term_time':torch.tensor([1000.]),
            },
            [],
            device=device,
        )
    else:
        raise ValueError(f'{network} not supported by this function')

    td_params['h']=h
    td_params['cost_bnd']=cost_bnd
    return td_params


#Function which returns the neighboring queue lengths
def neighbor_fcn(transitions:torch.Tensor, #tensor which lists the possible jumps.  Sent as input to avoid having to recreate it everytime this function is used
                 queue:torch.Tensor, #current queue length
                 device='cpu'):
    mask=torch.ones(tuple(list(queue.shape[:-1]) +[2*queue.shape[-1]]), dtype=torch.bool,device=device)
    mask[...,queue.shape[-1]:2*queue.shape[-1]]=queue>0   

    neighbor_queues=torch.broadcast_to(torch.unsqueeze(queue,-2),list(mask.shape)+[queue.shape[-1]]).clone()
    neighbor_queues=neighbor_queues+mask.unsqueeze(-1)*transitions

    return neighbor_queues




#Function which take the incidence matrix K and capacity constraint vector C and returns the vertices of the action space
#I assume K is a matrix of 1s and 0s, C is a positive vector with the same number of rows as K, and that these correspdong to a bounded action space
def find_vertices_for_action_space(K: torch.Tensor, C: torch.Tensor, device="cpu"):
    K.to("cpu")
    C.to("cpu")
    #I add the nonnegativity constraint
    A=torch.zeros(K.shape[0]+K.shape[1],K.shape[1],device="cpu")
    A[:K.shape[0],:]=K
    b=torch.zeros(C.shape[0]+K.shape[1],device="cpu")
    b[:C.shape[0]]=C

    for i in range(K.shape[1]):
        A[K.shape[0]+i,i]=-1
    
    A_np=A.numpy()
    b_np=b.numpy()

    #find the vertices
    v_np=compute_polytope_vertices(A_np,b_np)

    #convert the tensor on the appropriate device and return
    return torch.tensor(np.array(v_np).T,dtype=torch.float32,device=device)


#Sharing network environment used for training (as opposed to evaluating performance where cost_bnd isn't used)
def _make_spec(self):    
    self.observation_spec = Composite(
        #nonnegative queue lengths
        queue=Bounded(
            low=0,
            high=torch.inf,
            shape=tuple(list(self.batch_size)+list(self.params["init_queue"].shape)),
            dtype=torch.float32,
        ),
        #tracks the run time (not the number of steps)
        run_time=Bounded(
            low=0,
            high=torch.inf,
            shape=tuple(list(self.batch_size)+[1]),
            dtype=torch.float32,
        ),
        #nonnegative holding cost.
        hold_cost=Bounded(
            low=0,
            high=torch.inf,
            shape=tuple(list(self.batch_size)+[1]),
            dtype=torch.float32,
        ),
        shape=self.batch_size,
    )

    self.state_spec = self.observation_spec.clone()

    #This is a necessary but not sufficient condition for admissible actions to satisfy.  
    #The real condition, given by the linear capacity constraint inequalities, in checked 
    #(and necessary adjustments are made) during the step
    self.action_spec = Bounded(
        low=0,
        high=torch.amax(self.params["C"]),
        shape=tuple(list(self.batch_size)+[self.params["beta"].shape[-1]]),
        dtype=torch.float32,
    )
    #reward is the negative cost
    self.reward_spec = Bounded(
        low=-torch.inf,
        high=0,
        shape=tuple(list(self.batch_size)+[1]),
        dtype=torch.float32,
    )

def _step(self, tensordict):
    #pass parameter values
    alpha = self.params["alpha"]
    beta = self.params["beta"]
    K = self.params["K"]
    C = self.params["C"]
    kappa=self.params["kappa"]
    cost_bnd=self.params["cost_bnd"]
    #pass state and action values
    queue=tensordict["queue"]
    hold_cost = tensordict["hold_cost"]
    run_time=tensordict["run_time"]
    action=tensordict["action"]

    #Determine if the current state exceeds the cost_bnd, and if so, give the corresponding reward (as if the trajectory incurred a constant holding cost of cost_bnd indefinitely) and terminate the process
    n_terminated=n_done=hold_cost>=cost_bnd
    n_reward=n_terminated*(-cost_bnd/kappa)

    #Adjust the action if it doesn't satify the capacity contraint (which often happens during a random rollout)
    adjust_denom=torch.maximum(torch.amax(torch.linalg.vecdot(action.unsqueeze(-2),K)/C,dim=-1,keepdim=True),torch.ones(hold_cost.shape,device=self.device))
    action=action/adjust_denom

    #Create a tensor of jump rates
    jump_rate=torch.cat((alpha.expand(action.shape),beta*action*(queue>0)),dim=-1)

    #Determine the time until the next jump
    wait_time_dist=Exponential(torch.sum(jump_rate,dim=-1,keepdim=True))
    wait_time=wait_time_dist.sample(tensordict.batch_size)

    #Increase the total run time unless the process has been terminated
    n_run_time=run_time+(~n_terminated)*wait_time

    #Compute the reward (discount cost between jumps) if the process hasn't been terminated
    n_reward+=(~n_terminated)*(1-torch.exp(-kappa*wait_time))*(-hold_cost/kappa)

    #Compute the change in queue length  
    move_dist=Multinomial(1,jump_rate)
    move=move_dist.sample(tensordict.batch_size)

    #Alter the queue length if the process hasn't been terminated
    n_queue=queue+(~n_terminated)*move[...,:queue.shape[-1]]
    n_queue=n_queue-(~n_terminated)*move[...,queue.shape[-1]:]
    #Compute the holding cost for the new queue length
    n_hold_cost=self.hold_cost_fcn(n_queue)

    #produce output tensordict
    out = TensorDict(
        {
            "queue":n_queue,
            "hold_cost":n_hold_cost,
            "run_time":n_run_time,
            "done":n_done,
            "terminated":n_terminated,
            "reward":n_reward,
        },
        tensordict.shape,
        device=self.device,
    )
    return out

#If paramater values haven't been provided this returns parameter valeus for a 2LLN with traffic_rho=.8
def set_params(self):    
    td_params = TensorDict(
        {
            "alpha":torch.tensor([.4,.4,.4]), #arrival rates
            "beta":torch.tensor([1.0,1.0,1.0]), #mean job sizes are 1/\beta
            "K":torch.tensor([[1.,0.,1.],[0.,1.,1.]]), #incidence matrix
            "C":torch.tensor([1.0,1.0]), #capacity constraints
            "V":torch.tensor([[1.,0.,0.,1.,0.],[0.,0.,1.,1.,0.],[0.,0.,0.,0.,1.]]), #vertices of action space.  isn't required to use the environment (but is used by the DRL algoirthm).  can be computed with 'find_vertices_for_action_space' function
            "init_queue":torch.zeros(3,dtype=torch.float32), #initial queue length
            "cost_bnd":torch.tensor([150.]), #cost bound which determines how large the holding cost must be for the trajectory to be terminated. when this happens it receives a final reward as if it incurred a holding cost equal to cost_bnd indefinitely
            "kappa":torch.tensor([.1]), #continuous time discount rate used in the infinite horizon discounted cost
        },
        [],
        device=self.device,
    )
    return td_params

def _reset(self, tensordict: TensorDict):

    terminated=done=torch.tensor([0],dtype=torch.bool,device=self.device)

    out = TensorDict(
        {
            "done": done,
            "terminated": terminated,
        },
        [],
        device=self.device,
    )
    out["run_time"]=torch.tensor([0.])
    out["queue"] = self.params["init_queue"].clone()
    out["hold_cost"]=self.hold_cost_fcn(out["queue"])

    return out
    
    
def _set_seed(self, seed: int | None) -> None:
    rng = torch.manual_seed(seed)
    self.rng = rng

class SharingNetworkTrainingEnv(EnvBase):
    batch_locked = False

    def __init__(self, td_params=None, #see the 'set_params' for details about the network parameters
                 hold_cost_fcn=None, #should be a nonnegative, monotonically nondecreasing function of the queue length which assigned positive cost to any nonzero queue length
                 seed=None, 
                 device="cpu"):
        # if td_params is empty use the default parameter values from set_params()
        if td_params is None:
            td_params = self.set_params()
        # if no holding cost function is provided use a linear holding cost where all job types have a holding cost of 1
        if hold_cost_fcn is None:
            def sum_fcn(queue:torch.Tensor):
                return torch.sum(queue,dim=-1, keepdim=True)
            hold_cost_fcn=sum_fcn


        super().__init__(device=device, batch_size=[])
        self.params=td_params
        self.hold_cost_fcn=hold_cost_fcn
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helper functions
    set_params = set_params
    _make_spec = _make_spec

    # Mandatory methods
    _reset = _reset
    _step = _step
    _set_seed = _set_seed


#Transformation used to convert queue length to its direction and magnitude
#The motivation is to improve the value network learning process
class DirNormTransform(Transform):
    def _apply_transform(self, queue: torch.Tensor) -> torch.Tensor:
        norm=torch.sum(queue,dim=-1,keepdim=True)
        mul=torch.minimum(torch.ones(norm.shape),1/norm)
        dir = queue*mul
        dir_norm=torch.cat((dir,norm),dim=-1)
        return dir_norm

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        new_shape_list=list(observation_spec.shape)
        new_shape_list[-1]+=1
        return Bounded(
            low=0,
            high=torch.inf,
            shape=tuple(new_shape_list),
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )


