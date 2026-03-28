import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import TensorDictModule
from torchrl.objectives import LossModule
import types
from torchrl.objectives.value.utils import roll_by_gather, _custom_conv1d
from torchrl.objectives.utils import distance_loss

#Class used to define the CTD(lambda) loss for sharing networks


#Computes the 1-step conditional target values and jumprates (not valid if 'next','terminated'=True)
def one_step_targ_val(self, tensordict: TensorDict):
    #Create a copy containing only the relevant information
    td_copy=tensordict.select('queue','hold_cost','neighbor_dir_norms','action').clone()

    #Find the neighbor values
    td_copy['dir_norm']=td_copy['neighbor_dir_norms']
    td_copy=self.value_tdm(td_copy)
    td_copy['neighbor_values']=td_copy['value']
    

    #Compute the jumprates
    td_copy=self.jumprate_tdm(td_copy)
    
    #Compute the target value
    targ_val= -td_copy['hold_cost']+torch.linalg.vecdot(td_copy['neighbor_values'].squeeze(-1),td_copy['jumprates'],dim=-1).unsqueeze(-1)
    targ_val=targ_val/(self.td_params["kappa"]+torch.sum(td_copy['jumprates'],dim=-1,keepdim=True))

    return targ_val, td_copy['jumprates']

#Computes the CT(0) target values more (efficiently than using targ_val_CTDlambda_fcn with lmbda=0)
#Uses the regular value network parameters (as opposed to target parameters) to determine the action (like double DQN)
#tensordict should already be flattened, so batch_size.ndim=1, and the first dimension is the time dimension
def targ_val_CTD0_fcn(self,tensordict: TensorDict)-> torch.Tensor:
    #Create a copy only containing the relevant values
    td_copy=tensordict.select('queue','dir_norm','hold_cost','neighbor_dir_norms').clone()
    #Get the action (opt_vert_act_fcn should not track parameter gradients)
    with self.value_tdm_params.to_module(self.value_tdm):
        td_copy=self.opt_vert_act_fcn(self.td_params,td_copy,self.value_tdm,self.jumprate_tdm)

    #Get the (1 step) target value of the current state using the target value_net parameters, don't track parameter gradient
    #Not valid if ()'next','terminated')=True
    with torch.no_grad(), self.target_value_tdm_params.to_module(self.value_tdm):
        targ_val,_=self.one_step_targ_val(td_copy)
    targ_val=targ_val.squeeze(-1)

    #Use the terminal next reward as the target value for terminated trajectories (in the environment it is adjusted appropriately)
    targ_val[tensordict["next","terminated"].squeeze(-1)]=tensordict["next","reward"][tensordict["next","terminated"].squeeze(-1)].squeeze(-1)

    return targ_val

#Computes the CT(0) target values more (efficiently than using targ_val_CTDlambda_targ_act_fcn with lmbda=0)
#Uses the target value network parameters to determine the action (like DQN)
#tensordict should already be flattened, so batch_size.ndim=1, and the first dimension is the time dimension
def targ_val_CTD0_targ_act_fcn(self,tensordict: TensorDict,)-> torch.Tensor:
    #Create a copy only containing the relevant values
    td_copy=tensordict.select('queue','dir_norm','hold_cost','neighbor_dir_norms').clone()
    #Get the action using the target value parameters (opt_vert_act_fcn should not track parameter gradients)
    with self.target_value_tdm_params.to_module(self.value_tdm):
        td_copy=self.opt_vert_act_fcn(self.td_params,td_copy,self.value_tdm,self.jumprate_tdm)

    #Get the (1 step) target value of the current state using the target value_net parameters, don't track parameter gradient
    #Not valid if ('next','terminated')=True
    with torch.no_grad(), self.target_value_tdm_params.to_module(self.value_tdm):
        targ_val,_=self.one_step_targ_val(td_copy)
    targ_val=targ_val.squeeze(-1)

    #Use the terminal next reward as the target value for terminated trajectories (in the environment it is adjusted appropriately)
    targ_val[tensordict["next","terminated"].squeeze(-1)]=tensordict["next","reward"][tensordict["next","terminated"].squeeze(-1)].squeeze(-1)

    return targ_val

#Computes the CT(lambda) target values
#Uses the regular value network parameters (as opposed to target parameters) to determine the action (like double DQN)
#tensordict should already be flattened, so batch_size.ndim=1, and the first dimension is the time dimension
def targ_val_CTDlambda_fcn(self, tensordict: TensorDict)-> torch.Tensor:
    #Create a copy only containing the relevant values
    td_copy=tensordict.select('queue','dir_norm','hold_cost','neighbor_dir_norms').clone()

    #Get the action (opt_vert_act_fcn should not track parameter gradients)
    with self.value_tdm_params.to_module(self.value_tdm):
        td_copy=self.opt_vert_act_fcn(self.td_params,td_copy,self.value_tdm,self.jumprate_tdm)

    #Get the (1 step) target value of the current state using the target value_net parameters, don't track parameter gradient
    #Not valid if ('next','terminated')=True
    with torch.no_grad(), self.target_value_tdm_params.to_module(self.value_tdm):
        targ_val,jumprate=self.one_step_targ_val(td_copy)
    targ_val=targ_val.squeeze(-1)

    #Use the terminal next reward as the target value for terminated trajectories (in the environment it is adjusted appropriately)
    targ_val[tensordict["next","terminated"].squeeze(-1)]=tensordict["next","reward"][tensordict["next","terminated"].squeeze(-1)].squeeze(-1)

    #Get the value estimate for the current state from the value_net using the target parameters, don't track parameter gradient
    with torch.no_grad(), self.target_value_tdm_params.to_module(self.value_tdm):
        td_copy=self.value_tdm(td_copy)
    
    #Compute zeta (see documentation for definition and explanation)
    zeta= targ_val-td_copy['value'].squeeze(-1)
    zeta=zeta[1:]
    zeta=zeta.unsqueeze(0)
    #Compute the queue length change
    change=tensordict["next","queue"]-tensordict["queue"]
    #Compute xi (see documentation for definition and explanation) based on the queue length change and action jumprates
    xi=torch.zeros(tensordict.batch_size).unsqueeze(-1)
    xi+=torch.sum((change>0)*jumprate[...,:change.shape[-1]],dim=-1,keepdim=True)
    xi+=torch.sum((change<0)*jumprate[...,change.shape[-1]:],dim=-1,keepdim=True)
    xi=xi*(~tensordict["next","done"])


    #Compute the CTD(lambda) target value
    T=targ_val.shape[0]
    temp=xi[:-1]
    temp=temp.expand(T-1,T-1)
    xis=roll_by_gather(temp, 0, -torch.arange(T-1))
    mat=xis*self.lmbda
    mat=mat.flip(-1).triu(diagonal=0).flip(-1)
    mat_cp=mat.cumprod(dim=-1).unsqueeze(0)

    targ_adj=_custom_conv1d(zeta.unsqueeze(0),mat_cp.unsqueeze(-1)).squeeze()

    targ_val[:-1]=targ_val[:-1]+targ_adj
    return targ_val

#Computes the CT(0) target values more (efficiently than using targ_val_CTDlambda_targ_act_fcn with lmbda=0)
#Uses the target value network parameters to determine the action (like DQN)
#tensordict should already be flattened, so batch_size.ndim=1, and the first dimension is the time dimension
def targ_val_CTDlambda_targ_act_fcn(self, tensordict: TensorDict)-> torch.Tensor:
    #Create a copy only containing the relevant values
    td_copy=tensordict.select('queue','dir_norm','hold_cost','neighbor_dir_norms').clone()

    #Get the action using the target value parameters (opt_vert_act_fcn should not track parameter gradients)
    with self.target_value_tdm_params.to_module(self.value_tdm):
        td_copy=self.opt_vert_act_fcn(self.td_params,td_copy,self.value_tdm,self.jumprate_tdm)

    #Get the (1 step) target value of the current state using the target value_net parameters, don't track parameter gradient
    #Not valid if ('next','terminated')=True
    with torch.no_grad(), self.target_value_tdm_params.to_module(self.value_tdm):
        targ_val,jumprate=self.one_step_targ_val(td_copy)
    targ_val=targ_val.squeeze(-1)

    #Use the terminal next reward as the target value for terminated trajectories (in the environment it is adjusted appropriately)
    targ_val[tensordict["next","terminated"].squeeze(-1)]=tensordict["next","reward"][tensordict["next","terminated"].squeeze(-1)].squeeze(-1)

    #Get the value estimate for the current state from the value_net using the target parameters, don't track parameter gradient
    with torch.no_grad(), self.target_value_tdm_params.to_module(self.value_tdm):
        td_copy=self.value_tdm(td_copy)
    
    #Compute zeta (see documentation for definition and explanation)
    zeta= targ_val-td_copy['value'].squeeze(-1)
    zeta=zeta[1:]
    zeta=zeta.unsqueeze(0)
    #Compute the queue length change
    change=tensordict["next","queue"]-tensordict["queue"]
    #Compute xi (see documentation for definition and explanation) based on the queue length change and action jumprates
    xi=torch.zeros(tensordict.batch_size).unsqueeze(-1)
    xi+=torch.sum((change>0)*jumprate[...,:change.shape[-1]],dim=-1,keepdim=True)
    xi+=torch.sum((change<0)*jumprate[...,change.shape[-1]:],dim=-1,keepdim=True)
    xi=xi*(~tensordict["next","done"])


    #Compute the CTD(lambda) target value
    T=targ_val.shape[0]
    temp=xi[:-1]
    temp=temp.expand(T-1,T-1)
    xis=roll_by_gather(temp, 0, -torch.arange(T-1))
    mat=xis*self.lmbda
    mat=mat.flip(-1).triu(diagonal=0).flip(-1)
    mat_cp=mat.cumprod(dim=-1).unsqueeze(0)

    targ_adj=_custom_conv1d(zeta.unsqueeze(0),mat_cp.unsqueeze(-1)).squeeze()

    targ_val[:-1]=targ_val[:-1]+targ_adj
    return targ_val



def _init(
    self,
    jumprate_tdm: TensorDictModule, #used to provide jumprates based on actoin
    value_tdm: TensorDictModule, #used to provide state value estimates, this is converted to a functional and target parameters are created
    opt_vert_act_fcn, #uses jumprate_tdm and value_tdm to find the optimal vertex action
    lmbda,
    td_params, 
    loss_type, #options are 'CTD0' and 'CTDlambda'
    targ_act, #indicates whether target value parameters are used to determine the optimal vertex action.  True is like DQN and False is like double DQN.  If target value parameters are just copies of the regular value parameters this has no impact
) -> None:
    super(type(self), self).__init__()
    self.jumprate_tdm=jumprate_tdm
    self.convert_to_functional(
        value_tdm,
        "value_tdm",
        create_target_params=True,
    )
    self.opt_vert_act_fcn=opt_vert_act_fcn
    self.lmbda=lmbda
    self.td_params=td_params

    #Assign the correct loss function.  Note that using 'CTD0' is more efficient than using 'CTDlambda' with lmbda=0
    if loss_type=="CTD0":
        if targ_act==True:
            self.targ_val_fcn=types.MethodType(targ_val_CTD0_targ_act_fcn, self)
        else:
            self.targ_val_fcn=types.MethodType(targ_val_CTD0_fcn, self)
    elif loss_type=="CTDlambda":
        if targ_act==True:
            self.targ_val_fcn=types.MethodType(targ_val_CTDlambda_targ_act_fcn, self)
        else:
            self.targ_val_fcn=types.MethodType(targ_val_CTDlambda_fcn, self)
    else:
        raise ValueError(f'{loss_type} not supported by this class')

#Returns a tensordict containing loss under key 'value_loss'
#Adds td_error under key 'td_error' to the input tensordict which can be used for priority sampling
#Input tensordict must contain keys 'queue','dir_norm','hold_cost','neighbor_dir_norms'
def _forward(self, tensordict: TensorDictBase) -> TensorDict:
    #Use the chosen target value function to compute the target values
    targ_val=self.targ_val_fcn(tensordict)

    #Use value_tdm to compute the predicted value
    td_copy=tensordict.select('dir_norm').clone()
    with self.value_tdm_params.to_module(self.value_tdm):
        td_copy=self.value_tdm(td_copy)
    pred_val=td_copy["value"].squeeze(-1)

    #Save the td_error to the tensordict (could be used for priority sampling)
    tensordict['td_error']=((targ_val-pred_val)**2).detach()
    #Compute and return the value loss
    value_loss = distance_loss(pred_val, targ_val, loss_function="l2").mean()

    return TensorDict(
        {
            'value_loss':value_loss,
        },
        [],
    )

class Cond_Value_Loss(LossModule):
    value_tdm: TensorDictModule
    value_tdm_params: TensorDictParams
    target_value_tdm_params: TensorDictParams
    __init__ = _init
    forward = _forward
    one_step_targ_val=one_step_targ_val