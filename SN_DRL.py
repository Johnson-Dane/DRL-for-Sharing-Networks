def SN_DRL_fcn():
    import torch
    import tqdm
    import json
    import multiprocessing
    from collections import defaultdict
    from tensordict import TensorDict
    from torch import nn
    from tensordict.nn import TensorDictModule
    from torchrl.collectors import Collector
    import tempfile
    from SN_Train_Env import (
        SharingNetworkTrainingEnv, 
        find_vertices_for_action_space, 
        DirNormTransform,
        neighbor_fcn,
        param_creator,
    )
    from torchrl.envs import (
        TransformedEnv,
        ParallelEnv,
    )
    from torchrl.data.replay_buffers import (
        LazyMemmapStorage,
        RandomSampler,
        TensorDictReplayBuffer,
        SliceSampler,
    )
    from pathlib import Path
    from torch.distributions import Multinomial
    from torchrl.envs.utils import ExplorationType
    from torch import optim
    from torchrl.objectives.utils import SoftUpdate
    from functools import partial
    from CondValueLoss import Cond_Value_Loss

    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    torch.set_default_device(device)

    #torch.autograd.set_detect_anomaly(True)
    num_workers=5  # number of parallel environments
    frames_per_worker_per_batch=1_000 #number of frames per parallel environment per batch
    init_rand_batches=1 #number of initial batches using random vertex actions
    num_cells= 64  # number of cells in each layer
    policy_batches= 200 #number of batches using the epsilon-greedy policy
    epsilon_reduct_prop=1. #the proportion of policy batches before epsilon in the epsilon-greedy policy is reduced to its minimum value 
    epsilon_max=1. #Starting (and maximum) epsilon value for the epsilon-greedy policy
    epsilon_min=.1 #Final (and minimum ) epsilon value for the epsilon-greedy policy
    eval_interval=None #How many batches between policy evaluation (0/False/None skips this)
    buffer_prop=.2 #determines the size of the replay buffer as a proportion of the number of frames generated under the epsilon-greedy policy
    frame_update_per_frame_coll=200 #number of frames used in value function training per frame collected (used to determine the number of training minibatches per batch from environment)
    value_param_lr=1e-3 #value network learning rate
    value_param_wd=1e-2 #weight decay used in Adam optimizer for value network
    value_grad_bnd=10. #value network gradient norm bound used in parameter updates
    rollout_max_steps=1_000 #maximum number of steps used in policy evaluation rollout (note that too small a value will lead to trajectory cost underestimation)
    frames_per_batch = num_workers*frames_per_worker_per_batch #total frames collected per batch
    total_frames = policy_batches*frames_per_batch #total number of frames collected from the epsilon-greedy policy
    epsilon_min_frames=epsilon_reduct_prop*total_frames #number of frames under the epsilon-greedy policy until epsilon reaches its final (minimum) value
    slice_len=10 #maximum length of trajectories sampled from the replay buffer
    num_slices=100 #number of trajectories sampled from the replay buffer
    minibatch_size=slice_len*num_slices #number of frames in minibatches used for value network training
    updates_per_batch=-((frame_update_per_frame_coll * frames_per_batch) // -(minibatch_size)) #number of minibatch training updates for value network between batches from environment
    buffer_size=total_frames//(1/buffer_prop) #size of replay buffer
    lmbda=1. #lambda parameter used in the CTD(lambda) value estimates 
    loss_type="CTDlambda" #Options are 'CTDlambda' and 'CTD0' (choosing 'CTD0' rather than 'CTDlambda' and setting lmbda=0 uses more efficient value loss code and removes trajectory sampling from replay buffer)
    soft_up_eps=0. #soft update parameter for target value parameters (choosing 0 results in target parameters that are copies of original parameters that are removed from the computation graph)
    targ_act=False #Determines if the greedy action choice in the value loss estimate uses the target value parameters (False is Double DQN approach while True is DQN approach).  There is no impact if target parameters are copies of original parameters (if soft_up_eps=0).

    #Locations to save parameters
    value_param_save_path='value_params.pt'
    jump_param_save_path='jump_params.pt'

    #Locations to save training performance info
    loss_logs_save_path='loss_logs.json'
    perform_logs_save_path='perform_logs.json'

    #Create environment parameter tensordict.  Can use param_creator function for '2LLN','3LLN', and 'C3LN' networks
    td_params=param_creator('2LLN', .8, torch.tensor([1.,1.,1.]), torch.tensor([150.]), device)


    #Add the vertices of the action space to the environment parameter tensordict
    td_params["V"]=find_vertices_for_action_space(td_params["K"],td_params["C"],device=device)

    #Define the holding cost function for the environment
    def hold_cost_fcn(queue):
        return torch.linalg.vecdot(queue,td_params["h"]).unsqueeze(-1)

    #def SN_DRL_fcn():

    #Transform the environment state to be the direction and magnitude of the queue length
    #The motivation is to improve the value network learning process
    def make_transformed_env(env):
        n_env = TransformedEnv(
            env, 
            DirNormTransform(in_keys=["queue"], out_keys=["dir_norm"]),
        )
        return n_env

    #Function which creates an environment instance
    dcEnvCreator=partial(SharingNetworkTrainingEnv,td_params,hold_cost_fcn,None,device)

    #Transform which is used when using the value network to compute values of neighboring queue lengths
    neighbor_transform=DirNormTransform(in_keys=["neighbor_queues"], out_keys=["neighbor_dir_norms"])



    #Create parallel environments and apply the transform
    par_dc_env = ParallelEnv(
        num_workers=num_workers,
        create_env_fn=dcEnvCreator,
        create_env_kwargs=None,
    )
    t_par_dc_env=make_transformed_env(par_dc_env)

    #create matrix of possible transitions (jumps) which is later used in computing the value loss
    transitions=torch.zeros(td_params["init_queue"].shape[-1]*2,td_params["init_queue"].shape[-1],device=device)
    for i in range(td_params["init_queue"].shape[-1]):
        transitions[i,i]=1
    for i in range(td_params["init_queue"].shape[-1]):
        transitions[td_params["init_queue"].shape[-1]+i,i]=-1

    #Create a function which identifies neighboring queue lengths
    loc_neighbor_fcn=partial(neighbor_fcn,transitions)

    #Class which stores rate estimates and returns the jumprates given queue length and action (factoring in which queues are empty)
    class JumpRateModule(torch.nn.Module):
        def __init__(self, td_params:torch.Tensor):
            super(JumpRateModule, self).__init__()
            self.kappa=td_params["kappa"]
            self.alpha = torch.nn.Parameter(torch.ones(td_params["alpha"].shape,dtype=torch.float32,device=device))
            self.beta = torch.nn.Parameter(torch.ones(td_params["beta"].shape,dtype=torch.float32,device=device))
            self.alpha_tot_time=torch.nn.Parameter(torch.ones(td_params["alpha"].shape,dtype=torch.float32,device=device))
            self.beta_tot_time=torch.nn.Parameter(torch.ones(td_params["alpha"].shape,dtype=torch.float32,device=device))

        def forward(self, queue:torch.Tensor, action:torch.Tensor):
            jumprate=torch.cat((self.alpha.broadcast_to(action.shape),self.beta*action*(queue>0)),dim=-1)
            return jumprate


    #Create instance of jumprate module
    jumprate_module=JumpRateModule(td_params)

    #Create a tensordict module of the jumprate module
    jumprate_tensordictmodule=TensorDictModule(jumprate_module,in_keys=['queue','action'],out_keys=['jumprates'])


    #create the value function network (recall that the input comes form the DirNormTransform is one longer than the queue length)
    value_net = nn.Sequential(
        nn.Linear(td_params["init_queue"].shape[-1]+1,num_cells, device=device),
        nn.Tanh(),
    #    nn.Dropout(p=dropout_prob),
        nn.Linear(num_cells,num_cells, device=device),
        nn.Tanh(),
    #    nn.Dropout(p=dropout_prob),
        nn.Linear(num_cells,num_cells, device=device),
        nn.Tanh(),
    #    nn.Dropout(p=dropout_prob),
        nn.Linear(num_cells, 1, device=device),
    )

    #initialize the rows of the parameter matrices in the value network to be orthogonal
    with torch.no_grad():
        for param in value_net.parameters():
            if param.ndim==2:
                _=nn.init.orthogonal_(param)

    #Create a tensordict module of the value network 
    value_tensordictmodule=TensorDictModule(value_net,in_keys=['dir_norm'],out_keys=['value'])

    #Function which collects the relevant information from an environment batch needed to update the 
    #rate parameter estimates
    def jump_param_est_info(tensordict):
        #select relevant info from tensordict contain environment batch
        td_clone = tensordict.select("queue","action","run_time",("next","queue"),("next","run_time"),("next","terminated")).clone()
        #only use transitions which don't end in a terminal state
        #Recall that the transition which ends in a terminal state doesn't involve a
        #jump, it just provides a final "reward" which assumes incurring a holding
        #cost of cost_bnd indefinitely into the future
        td_clone=td_clone[~td_clone["next","terminated"].squeeze(-1)]

        #Compute the wait time between jumps
        td_clone["wait_time"]=td_clone["next","run_time"]-td_clone["run_time"]

        #For beta estimates provide a sum of wait times multiplied by flowrate assigned to that job type
        beta_time=torch.sum(td_clone["wait_time"]*td_clone["action"]*(td_clone["queue"]>0),dim=list(range(td_clone["action"].ndim-1)))
        #For alpha estimates provide a sum of wait times
        alpha_time=torch.sum(td_clone["wait_time"],dim=list(range(td_clone["action"].ndim-1)))

        #Compute the transition (the type of jump which occurred)
        change=td_clone["next","queue"]-td_clone["queue"]
        inc=torch.zeros(change.shape)
        dec=torch.zeros(change.shape)
        inc_mask=(torch.sum(change,dim=-1,keepdim=True)>0)
        inc=inc+change*inc_mask
        dec=dec-(change*(~inc_mask))

        #Expand the alpha time to a vector corresponding to job types
        alpha_time=alpha_time.expand(change.shape[-1])

        #Sum the transitions (jump types) for rate estimates
        alpha_occ=torch.sum(inc,dim=list(range(inc.ndim-1)))
        beta_occ=torch.sum(dec,dim=list(range(dec.ndim-1)))

        return alpha_occ, alpha_time, beta_occ, beta_time

    #Function used to update the rate parameters in the jumprate module and the rho estimate (for random walk actions) in the actor exploration module
    def jump_param_update(td_params, alpha_occ, alpha_time, beta_occ, beta_time, jumprate_module, actor_exploration_module):
        with torch.no_grad():
            #Creates alpha estimates based on observations in the most recent batch
            alpha_est=torch.zeros(jumprate_module.alpha.shape)
            alpha_est[alpha_time>0]=alpha_occ[alpha_time>0]/alpha_time[alpha_time>0]
            #Creates new overall alpha estimate using new observations and previous history
            n_alpha_tot_time=jumprate_module.alpha_tot_time+alpha_time
            n_alpha=(jumprate_module.alpha*jumprate_module.alpha_tot_time+alpha_est*alpha_time)/n_alpha_tot_time

            #Creates beta estimates based on observations in the most recent batch
            beta_est=torch.zeros(jumprate_module.beta.shape)
            beta_est[beta_time>0]=beta_occ[beta_time>0]/beta_time[beta_time>0]
            #Creates new overall beta estimate using new observations and previous history
            n_beta_tot_time=jumprate_module.beta_tot_time+beta_time
            n_beta=(jumprate_module.beta*jumprate_module.beta_tot_time+beta_est*beta_time)/n_beta_tot_time

            #Computes the norm between previous estimates and estimates based on most recent batch to be returned for performance tracking
            alpha_diff=torch.linalg.vector_norm(n_alpha-jumprate_module.alpha)
            beta_diff=torch.linalg.vector_norm(n_beta-jumprate_module.beta)

            #Updates overall estimates in modules while ensuring that an action based on rho satisfies the network capacity constraints        
            jumprate_module.alpha.copy_(n_alpha)
            jumprate_module.beta.copy_(n_beta)
            jumprate_module.alpha_tot_time.copy_(n_alpha_tot_time)
            jumprate_module.beta_tot_time.copy_(n_beta_tot_time)
            rho=jumprate_module.alpha/jumprate_module.beta
            denom=max((torch.linalg.vecdot(td_params["K"],rho)/td_params["C"]).max().item(),1)
            actor_exploration_module.rho.copy_(rho/denom)
        return alpha_diff, beta_diff

    #Function which takes the value and jumprate tensordict modules and returns the optimal 
    #action (key='action') among all vertex actions and the corresponding value estimate of 
    #that action (key='max_vert_act_val')
    #Output is written to the input tensordict
    #Input tensordict which must contain the keys: 'queue', 'neighbor_dir_norms', and 'hold_cost'
    def opt_vert_act_fcn(td_params,tensordict,value_tdm,jumprate_tdm):
        #Collects the values for neighboring queue lengths
        td_copy_neighbor=tensordict.select('neighbor_dir_norms').clone()
        td_copy_neighbor['dir_norm']=td_copy_neighbor['neighbor_dir_norms']
        value_tdm.requires_grad_(False)
        td_copy_neighbor=value_tdm(td_copy_neighbor)
        value_tdm.requires_grad_(True)
        td_copy_neighbor['neighbor_values']=td_copy_neighbor['value']

        #Creates the inputs to jumprate_tdm in order to find the jumprate to all neighbors using vertex actions
        td_copy_neighbor['queue']=tensordict['queue'].unsqueeze(-2).expand(tuple(list(tensordict.batch_size)+list(td_params["V"].transpose(-2,-1).shape)))
        td_copy_neighbor['action']=td_params["V"].transpose(-2,-1).expand(tuple(list(tensordict.batch_size)+list(td_params["V"].transpose(-2,-1).shape)))

        #Computes the jumprates associated with vertex actions
        jumprate_tdm.requires_grad_(False)
        td_copy_neighbor=jumprate_tdm(td_copy_neighbor)
        jumprate_tdm.requires_grad_(True)
        td_copy_neighbor['vert_act_jumprates']=td_copy_neighbor['jumprates']

        #Computes the optimal vertex action.  Returns the corresponding value and action
        vert_act_vals= -tensordict['hold_cost']+torch.linalg.vecdot(td_copy_neighbor['neighbor_values'].squeeze(-1).unsqueeze(-2),td_copy_neighbor['vert_act_jumprates'],dim=-1)
        vert_act_vals=vert_act_vals/(td_params["kappa"]+torch.sum(td_copy_neighbor['vert_act_jumprates'],dim=-1))
        max_vert_act_val, max_vert_act=torch.max(vert_act_vals, dim=-1)
        tensordict['max_vert_act_val']=max_vert_act_val
        tensordict['action']=td_params["V"].transpose(-2,-1)[max_vert_act]
        return tensordict

    #Class for creating epsilon-greedy policy
    class actorExplorationModule(torch.nn.Module):
        def __init__(self, 
                    td_params:torch.Tensor, 
                    rho_prob:torch.Tensor, #initial probability of random walk action (epsilon)
                    neighbor_fcn, #used to compute neighboring queue lengths to find the optimal vertex action
                    neighbor_trans, #transform which allows the value network to approximate the value of neighboring queue lengths
                    value_tdm, #value module used to approximate value of neighboring queue lengths
                    jumprate_tdm, #jumprate module used to find jumprates associated with vertex actions
                    opt_vert_act_fcn, #function which computes the optimal vertex action
                    device):
            super(actorExplorationModule, self).__init__()
            rho=torch.ones(td_params["init_queue"].shape[-1]) #initial estimate of rho (alpha/beta)
            denom=max((torch.linalg.vecdot(td_params["K"],rho)/td_params["C"]).max().item(),1)
            self.rho=torch.nn.Parameter(rho/denom) #adjust initial rho estimate so that a rho action satisfies network capacity constraints
            self.rho_prob=torch.nn.Parameter(rho_prob)
            self.neighbor_fcn=neighbor_fcn
            self.neighbor_trans=neighbor_trans
            self.device=device
            self.td_params=td_params
            self.value_tdm=value_tdm
            self.jumprate_tdm=jumprate_tdm
            self.opt_vert_act_fcn=opt_vert_act_fcn

        #determines the probability (epsilon) of using a rho action (random walk) instead of approximate optimal action
        #this allows us to decrease epsilon (exploration) as simulation progresses
        #epsilon decreases linearly from max_prob to min_prob as current frames increases to complete frames
        def rho_prob_adj(self, max_prob,min_prob,current_frames,complete_frames):
            new_rho=torch.tensor([max_prob+(min_prob-max_prob)*(min(current_frames,complete_frames)/complete_frames)])
            with torch.no_grad():
                self.rho_prob.copy_(new_rho)

        #returns an action given a queue length and holding cost
        def forward(self, queue, hold_cost):
            #find the optimal vertex action using the opt_vert_act_fcn
            neighbor_queues = self.neighbor_fcn(queue,self.device)
            temp_td=TensorDict(
                {
                    'queue':queue,
                    'hold_cost':hold_cost,
                    'neighbor_queues':neighbor_queues,
                },
                batch_size=queue.shape[:-1],
                device=self.device
            )
            temp_td=self.neighbor_trans(temp_td)
            neighbor_dir_norms=temp_td['neighbor_dir_norms']
            temp_td=self.opt_vert_act_fcn(self.td_params,temp_td,self.value_tdm,self.jumprate_tdm)
            action= temp_td['action']

            #replace the optimal action with a rho action with probability self.rho_prob.  
            #the actor bool variable indicates if the optimal action was used
            with torch.no_grad():
                replace=torch.rand(tuple(list(queue.shape[:-1])+[1]))<=self.rho_prob
                n_action=self.rho.clone().expand(queue.shape)*replace+action*(~replace)
                actor=~replace
            #also return neigh_dir_norms which is used again later when computing the value loss
            return n_action, neighbor_dir_norms, actor

    #Create an instance of the epsilon-greedy policy
    actor_exploration_module=actorExplorationModule(td_params,torch.tensor([epsilon_max]),loc_neighbor_fcn, neighbor_transform, value_tensordictmodule, jumprate_tensordictmodule, opt_vert_act_fcn, device)
    #Create a tensordict module with the epsilon-greedy policy
    actor_exploration_policy = TensorDictModule(
        actor_exploration_module,in_keys=["queue","hold_cost"],out_keys=["action", "neighbor_dir_norms","actor"])

        

    #Class which creates the random vertex action policy
    class randomActionModule(torch.nn.Module):
        #Although neighboring queue lengths aren't needed for this policy, we provide that
        #information here so it can be used later when computing the value loss
        def __init__(self, td_params:torch.Tensor, neighbor_fcn,  neighbor_trans, device):
            super(randomActionModule, self).__init__()
            self.rand_vert_act_dist=Multinomial(1,torch.ones(td_params["V"].shape[-1])) #distribution used to select a random vertex action
            self.neighbor_fcn=neighbor_fcn
            self.neighbor_trans=neighbor_trans
            self.device=device

        def forward(self, queue:torch.Tensor):
            #compute neighbor_dir_norms
            neighbor_queues = self.neighbor_fcn(queue,device)
            temp_td=TensorDict(
                {
                    'neighbor_queues':neighbor_queues,
                },
                batch_size=queue.shape[:-1],
                device=self.device,
            )
            temp_td=self.neighbor_trans(temp_td)
            neighbor_dir_norms=temp_td['neighbor_dir_norms']
            #compute a random vertex action
            weights = self.rand_vert_act_dist.sample(queue.shape[:-1])
            action=torch.linalg.vecdot(weights.unsqueeze(-2),td_params["V"])
            #set actor bool variable to False to indicate that the approximate optimal action was not used
            actor=torch.zeros(queue.shape[:-1],dtype=torch.bool).unsqueeze(-1)        
            return action, neighbor_dir_norms, actor

    #Create instance of the random vertex action policy
    rand_act_module=randomActionModule(td_params, loc_neighbor_fcn, neighbor_transform, device)

    #Create a tensordict module for the random vertex action policy
    random_action_policy=TensorDictModule(
        rand_act_module, in_keys=["queue"], out_keys=["action","neighbor_dir_norms","actor"]
    )

    #Create the loss module
    loss_module=Cond_Value_Loss(jumprate_tensordictmodule, value_tensordictmodule, opt_vert_act_fcn,lmbda,td_params,loss_type, targ_act)

    #Create the collector for the batches of random vertex actions
    init_rand_collector = Collector(
        create_env_fn=t_par_dc_env,
        policy=random_action_policy,
        frames_per_batch=frames_per_batch,
        total_frames=init_rand_batches*frames_per_batch,
        reset_at_each_iter=False,
        split_trajs=False,
        device=device,
        exploration_type=ExplorationType.RANDOM,
        trust_policy=True,
    )

    #Create the collector for the batches using the epsilon-greedy policy
    collector = Collector(
        create_env_fn=t_par_dc_env,
        policy=actor_exploration_policy,
        total_frames=total_frames,
        frames_per_batch=frames_per_batch,
        reset_at_each_iter=False,
        split_trajs=False,
        device=device,
        exploration_type=ExplorationType.RANDOM,
        trust_policy=True,
    )

    #Create the replay buffer
    tmpdir = tempfile.TemporaryDirectory()
    buffer_scratch_dir = tmpdir.name

    #If using 'CTD0' value loss don't use slice sample to sample trajectories
    if loss_type=="CTD0":
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                max_size=buffer_size,
                scratch_dir=buffer_scratch_dir,
            ),
            sampler=RandomSampler(),
            batch_size=minibatch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                max_size=buffer_size,
                scratch_dir=buffer_scratch_dir,
            ),
            sampler=SliceSampler(
                slice_len=slice_len,
                strict_length=False,
            ),
            batch_size=minibatch_size,
        )

    #Create the value network optimizer
    optimizer_value = optim.Adam(
        loss_module.value_tdm_params.values(True, True), lr=value_param_lr, weight_decay=value_param_wd #used 1e-2 to get decent performance previously
    )
    #Create the value target parameter updater
    if soft_up_eps<.5:
        target_net_updater = SoftUpdate(loss_module, tau=1-soft_up_eps)
    else:
        target_net_updater = SoftUpdate(loss_module, eps=soft_up_eps)

    #Track training information
    loss_logs = defaultdict(list)

    #Generate a small number of batches using random vertex actions to have better estimates (particularly for rates)
    #prior to using optimal action estimates
    for i, tensordict in enumerate(init_rand_collector):
        #Indicate the end of consecutive entries from the same trajectory for the replay buffer
        tensordict["next","done"][:,-1]=True

        #Adjust shape for the replay buffer
        tensordict=tensordict.reshape(-1)

        #Update rate parameters using MLE
        with torch.no_grad():
            alpha_occ, alpha_time, beta_occ, beta_time = jump_param_est_info(tensordict)
            _, _=jump_param_update(td_params=td_params, alpha_occ=alpha_occ, alpha_time=alpha_time, beta_occ=beta_occ, beta_time=beta_time, jumprate_module=jumprate_module,actor_exploration_module=actor_exploration_module)

        #Add frames to the replay buffer
        _=replay_buffer.extend(tensordict.cpu())

        #Sample minibatches to train value network
        for _ in range(updates_per_batch):
            sampled_tensordict = replay_buffer.sample().to(device)
            # Compute loss
            loss_td=loss_module(sampled_tensordict)
            value_loss=loss_td['value_loss']
            #Record the MSE value loss 
            loss_logs["value_loss"].append(value_loss.detach().item())
            value_loss.backward()
            #Clip the value gradient
            value_grad = torch.nn.utils.clip_grad_norm_(
                loss_module.value_tdm_params.values(True, True), value_grad_bnd
            )
            #Record the magnitude of the value gradient
            loss_logs["value_grad"].append(value_grad.item())
            #Update value parameters
            optimizer_value.step()
            optimizer_value.zero_grad()
            #Update value target parameters
            target_net_updater.step()

            #Record the rate parameter error (note that it is constant over the value training minibatch)
            loss_logs["rate_error"].append(torch.linalg.vector_norm(torch.cat((td_params["alpha"]-jumprate_module.alpha,td_params["beta"]-jumprate_module.beta))).item())
    init_rand_collector.shutdown()
    del init_rand_collector


    #If evaluating the policy during training track performance
    if eval_interval:
        perform_logs = defaultdict(list)

    #Collected frames is used to update pbar and update epsilon in the epsilon-greedy policy using actor_exploration_module.rho_prob_adj
    collected_frames = 0
    pbar = tqdm.tqdm(total=total_frames)

    for i, tensordict in enumerate(collector):
        #Update collected frames for pbar and adjusting the epsilon-greedy probability (rho probability)
        current_frames = tensordict.numel()
        collected_frames += current_frames
        #Indicate the end of consecutive entries from the same trajectory for the replay buffer
        tensordict["next","done"][:,-1]=True
        #Adjust shape for replay buffer
        tensordict=tensordict.reshape(-1)
        
        #Update rate parameters using MLE
        with torch.no_grad():
            alpha_occ, alpha_time, beta_occ, beta_time = jump_param_est_info(tensordict)
            _, _=jump_param_update(td_params=td_params, alpha_occ=alpha_occ, alpha_time=alpha_time, beta_occ=beta_occ, beta_time=beta_time, jumprate_module=jumprate_module,actor_exploration_module=actor_exploration_module)
        #Add frames to the replay buffer
        _=replay_buffer.extend(tensordict.cpu())
        #Sample minibatches to train value network
        for j in range(updates_per_batch):
            sampled_tensordict = replay_buffer.sample().to(device)
            
            #Compute loss
            loss_td=loss_module(sampled_tensordict)
            value_loss=loss_td['value_loss']
            #Record the MSE value loss
            loss_logs["value_loss"].append(value_loss.detach().item())
            value_loss.backward()
            #Clip the value gradient
            value_grad = torch.nn.utils.clip_grad_norm_(
                loss_module.value_tdm_params.values(True, True), value_grad_bnd
            )
            #Record the magnitued of the value gradient
            loss_logs["value_grad"].append(value_grad.item())
            #Update value parameters
            optimizer_value.step()
            optimizer_value.zero_grad()
            #Update target value parameters
            target_net_updater.step()

            #Record the rate parameter error (note that it is constant over the value training minibatch)
            loss_logs["rate_error"].append(torch.linalg.vector_norm(torch.cat((td_params["alpha"]-jumprate_module.alpha,td_params["beta"]-jumprate_module.beta))).item())
        
        #Test the policy every eval_interval batches (set eval_interval to 0/None/False to skip this step)
        #Keep in mind that the sample size of this estimate is the number of parallel environments, so it is likely
        #to have a high variance.  It is meant to give a quick but inaccurate impression of performance during training
        #and isn't a substitute for a thorough performance estimate after training is complete
        if eval_interval:
            if i % eval_interval==0:      

                #Set the probability of rho actions (random walks) to 0 (only use the policy)
                actor_exploration_module.rho_prob_adj(0.,0.,1,1)

                #Collect a batch of length rollout_max_steps (note that a short length will underestimate the total trajectory cost)
                eval_rollout=t_par_dc_env.rollout(rollout_max_steps,actor_exploration_policy)
                #Compute the cost from transitions which don't result in termination (don't exceed the cost_bnd) for each trajectory
                discMult=torch.exp(-td_params['kappa']*eval_rollout['run_time'].squeeze(-1))*(~eval_rollout['next','terminated'].squeeze(-1))
                cost=torch.sum(discMult*eval_rollout['next','reward'].squeeze(-1),dim=-1)
                #Add costs from final terminated state (these states are repeated in the rollout trajecotry and we only want to add their contribution once)
                cost+=eval_rollout['next','terminated'].squeeze(-1)[...,-1]*torch.exp(-td_params['kappa']*eval_rollout['run_time'].squeeze(-1))[...,-1]*eval_rollout['next','reward'].squeeze(-1)[...,-1]
                #Compute the mean overall all sample trajectories (one for each parallel environment)
                mean_cost=cost.mean().item()
                #Record mean cost and add it to the info provided by the pbar update
                perform_logs["mean_cost"].append(mean_cost)
                val_comp_str=f"mean_cost: {mean_cost}"

        #Update the rho action probability (the epsilon used in the epsilon-greedy policy) 
        actor_exploration_module.rho_prob_adj(epsilon_max,epsilon_min,collected_frames,epsilon_min_frames)

        #If periodically testing policy performance update pbar with this info
        if eval_interval:
            pbar.set_description(val_comp_str)
        pbar.update(tensordict.numel())

    collector.shutdown()
    del collector

    #Save parameter values
    Path(value_param_save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(value_net.state_dict(), value_param_save_path)
    Path(jump_param_save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(jumprate_module.state_dict(), jump_param_save_path)

    #Save training logs
    Path(loss_logs_save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(loss_logs_save_path, 'w') as json_file:
        json.dump(loss_logs, json_file)
    if eval_interval:
        Path(perform_logs_save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(perform_logs_save_path, 'w') as json_file:
            json.dump(perform_logs, json_file)

if __name__ == "__main__":
    SN_DRL_fcn()