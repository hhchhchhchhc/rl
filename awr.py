import datetime
import json
import os
import random
from collections import deque
from enum import Enum, auto

import gym
import numpy as np
import torch
from gym import Space
from tensorboardX import SummaryWriter
from torch import nn, distributions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def t(v, dtype=None, device=device, requires_grad=False):
    """Shortcut for tensor instantiation with device."""
    return torch.tensor(v, device=device, dtype=dtype, requires_grad=requires_grad)


def mc_values(rewards, hyper_ps):
    """
    Gives a list of MC estimates for a given list of samples from an RL environment.
    The MC estimator is used for this computation.

    :param rewards: The rewards that were obtained while exploring the RL environment.
    :param hyper_ps: The hyper-parameters to be used.

    :return: The MC estimates.
    """

    mcs = np.zeros(shape=(len(rewards),))
    discount_factor = dict_with_default(hyper_ps, 'discount_factor', .9)

    for i, reward in enumerate(rewards):
        ret = reward
        gamma = 1.

        for j in range(i + 1, len(rewards)):
            gamma *= discount_factor
            ret += gamma * rewards[j]

        mcs[i] = ret

    return mcs


def td_values(replay_buffers, state_values, hyper_ps):
    """
    Gives a list of TD estimates for a given list of samples from an RL environment.
    The TD(λ) estimator is used for this computation.

    :param replay_buffers: The replay buffers filled by exploring the RL environment.
    Includes: states, rewards, "final state?"s.
    :param state_values: The currently estimated state values.
    :param hyper_ps: The hyper-parameters to be used.

    :return: The TD estimates.
    """

    states, rewards, dones = replay_buffers
    sample_count = len(states)
    tds = np.zeros(shape=(sample_count,))

    discount_factor = dict_with_default(hyper_ps, 'discount_factor', .9)
    alpha = dict_with_default(hyper_ps, 'alpha', .95)
    lam = dict_with_default(hyper_ps, 'lambda', .95)

    val = 0.
    for i in range(sample_count - 1, -1, -1):
        state_value = state_values[i]
        next_value = 0. if dones[i] else state_values[i + 1]

        error = rewards[i] + discount_factor * next_value - state_value
        val = alpha * error + discount_factor * lam * (1 - dones[i]) * val

        tds[i] = val + state_value

    return tds

def critic_inputs(trajectories, next_states=False):
    """
    Extracts the relevant inputs for the V-critic from the given list of trajectories.

    :param trajectories: The trajectories from which the information should be taken.
    :param next_states: Extract the next-state entries from the samples instead of the current states.

    :return: The extracted information in the form of a batched tensor.
    """

    return torch.cat([(tr.next_state if next_states else tr.state).flatten().unsqueeze(0) for tr in trajectories]).to(device)


def nan_in_model(model):
    """
    Checks if the given model holds any parameters that contain NaN values.

    :param model: The model to be checked for NaN entries.

    :return: Whether the model contain NaN parameters.
    """

    for p in model.parameters():
        p_nan = torch.isnan(p.data).flatten().tolist()
        if any(p_nan):
            return True

    return False

def dict_with_default(dict, key, default):
    """
    Returns the value contained in the given dictionary for the given key, if it exists.
    Otherwise, returns the given default value.

    :param dict: The dictionary from which the value should be read.
    :param key: The key to look for in the dictionary.
    :param default: The fallback value, in case the dictionary doesn't contain the desired key.

    :return: The value read from the dictionary, if it exists. The default value otherwise.
    """

    if key in dict:
        return dict[key]
    else:
        return default


def xavier_init(m):
    """
    Xavier normal initialisation for layer m.

    :param m: The layer to have its weight and bias initialised.
    """

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def kaiming_init(m):
    """
    Kaiming normal initialisation for layer m.

    :param m: The layer to have its weight and bias initialised.
    """

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def obs_to_state(observation):
    """
    Converts a given observation into a state tensor.
    Necessary as a catch-all for MuJoCo environments.

    :param observation: The observation received from the environment.

    :return: The state tensor.
    """

    if type(observation) is dict:
        state = state_from_mujoco(observation)
    else:
        state = observation

    return t(state).float()


def state_from_mujoco(observation):
    """
    Converts the observation parts returned by a MuJoCo environment into a single vector of values.

    :param observation: The observation containing the relevant parts.

    :return: A single vector containing all the observation information.
    """

    ag = observation['achieved_goal']
    dg = observation['desired_goal']
    obs = observation['observation']

    return np.concatenate([ag, dg, obs])


class DebugType(Enum):
    NONE = auto()  # no debug output
    EVAL = auto()  # debug output only in evaluation, i. e. when computing the average return 
    FULL = auto()  # full debug output

class Model(nn.Module):

    init_func = None

    def evaluate(self, x):
        self.eval()
        with torch.no_grad():
            y = self.forward(x)
        self.train()

        return y

class Critic(Model):

    name = "critic"

    def __init__(self):
        super(Critic, self).__init__()

        self.fc = None
        self.optimiser = None
        self.criterion = torch.nn.MSELoss()

    def set_params(self, hyper_ps):
        hidden_size = hyper_ps['c_hidden_size']
        hidden_layers = hyper_ps['c_hidden_layers']

        fcs = [
            nn.Linear(hyper_ps['state_dim'], hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        ]
        for _ in range(hidden_layers):
            fcs.append(nn.Linear(hidden_size, hidden_size))
            fcs.append(nn.ReLU())
            fcs.append(nn.BatchNorm1d(hidden_size))
        fcs.append(nn.Linear(hidden_size, 1))

        self.fc = nn.Sequential(*fcs)

        self.optimiser = torch.optim.SGD(
            lr=hyper_ps['c_learning_rate'],
            momentum=hyper_ps['c_momentum'],
            params=self.parameters()
        )

    def forward(self, state):
        return self.fc(state)

    def backward(self, out, target):
        loss = self.criterion(out, target.float())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss

class Actor(Model):

    name = "actor"

    def __init__(self):
        super(Actor, self).__init__()

        self.fc_base = None
        self.fc_mean = None
        self.fc_logsd = None
        self.optimiser = None

    def set_params(self, hyper_ps):
        hidden_size = hyper_ps['a_hidden_size']
        hidden_layers = hyper_ps['a_hidden_layers']
        action_dim = hyper_ps['action_dim']

        fcs = [nn.Linear(hyper_ps['state_dim'], hidden_size), nn.ReLU()]
        for _ in range(hidden_layers):
            fcs.append(nn.Linear(hidden_size, hidden_size))
            fcs.append(nn.ReLU())

        self.fc_base = nn.Sequential(*fcs)
        self.fc_mean = nn.Linear(hidden_size, action_dim)
        self.fc_logsd = nn.Linear(hidden_size, action_dim)

        self.optimiser = torch.optim.SGD(
            lr=hyper_ps['a_learning_rate'],
            momentum=hyper_ps['a_momentum'],
            params=self.parameters()
        )

    def forward(self, state):
        x = self.fc_base(state)
        mean = self.fc_mean(x)

        # 将动作均值mean传入softmax层,得到归一化的概率分布
        action_probs = nn.functional.softmax(mean, dim=-1)

        # 根据概率分布采样离散动作
        dist = distributions.Categorical(action_probs)
        action = dist.sample()

        return dist, action

    def backward(self, loss):      
        self.optimiser.zero_grad()

        # 使用dist.log_prob计算action的对数概率
        log_pis = dist.log_prob(t(actions[indices]))
        losses = -log_pis * advantage_weights
        losses = losses / batch_size  # normalise wrt the batch size

        loss.backward()
        self.optimiser.step()

class AWRAgent:

    name = "awr"

    @staticmethod
    def train(models, environment, hyper_ps, debug_type, writer):
        assert len(models) == 2, "AWR needs exactly two models to function properly."
        actor, critic = models

        # replay buffer
        sample_mod = dict_with_default(hyper_ps, 'sample_mod', 10)
        max_buffer_size = hyper_ps['replay_size']
        states = deque(maxlen=max_buffer_size)
        actions = deque(maxlen=max_buffer_size)
        rewards = deque(maxlen=max_buffer_size)
        dones = deque(maxlen=max_buffer_size)
        replay_fill_threshold = dict_with_default(hyper_ps, 'replay_fill_threshold', 0.)
        random_exploration = dict_with_default(hyper_ps, 'random_exploration', False)

        # learning time setup
        max_epoch_count = hyper_ps['max_epochs']
        epoch = 0
        pre_training_epochs = 0
        max_pre_epochs = 150

        # algorithm specifics
        beta = hyper_ps['beta']
        critic_steps_start = hyper_ps['critic_steps_start']
        critic_steps_end = hyper_ps['critic_steps_end']
        actor_steps_start = hyper_ps['actor_steps_start']
        actor_steps_end = hyper_ps['actor_steps_end']
        batch_size = hyper_ps['batch_size']
        max_advantage_weight = hyper_ps['max_advantage_weight']
        min_log_pi = hyper_ps['min_log_pi']

        # debug helper field
        debug_full = debug_type == DebugType.FULL

        # critic pre-training
        critic_threshold = hyper_ps['critic_threshold']
        critic_suffices_required = hyper_ps['critic_suffices_required']
        critic_suffices_count = 0
        critic_suffices = False

        # evaluation
        validation_epoch_mod = dict_with_default(hyper_ps, 'validation_epoch_mod', 30)
        test_iterations = hyper_ps['test_iterations']

        AWRAgent.compute_validation_return(
            actor,
            environment,
            hyper_ps,
            debug_type,
            test_iterations,
            epoch,
            writer
        )

        while epoch < max_epoch_count + pre_training_epochs:
            print(f"\nEpoch: {epoch}")

            # set actor and critic update steps
            epoch_percentage = ((epoch - pre_training_epochs) / max_epoch_count)
            critic_steps = critic_steps_start + int((critic_steps_end - critic_steps_start) * epoch_percentage)
            actor_steps = actor_steps_start + int((actor_steps_end - actor_steps_start) * epoch_percentage)

            # sampling from env
            for _ in range(sample_mod):
                AWRAgent.sample_from_env(
                    actor,
                    environment,
                    debug_full,
                    exploration=random_exploration and len(states) < replay_fill_threshold * max_buffer_size,
                    replay_buffers=(states, actions, rewards, dones)
                )

            if len(states) < replay_fill_threshold * max_buffer_size:
                continue

            dq_states = states
            states = np.array(states)
            dq_actions = actions
            actions = np.array(actions)
            dq_rewards = rewards
            rewards = np.array(rewards)

            # training the critic
            avg_loss = 0.
            state_values = np.array(critic.evaluate(t(states)).squeeze(1).cpu())
            tds = td_values((states, rewards, dones), state_values, hyper_ps)
            for _ in range(critic_steps):
                indices = random.sample(range(len(states)), batch_size)
                ins = t(states[indices])
                tars = t(tds[indices])

                outs = critic(ins)
                loss = critic.backward(outs.squeeze(1), tars)
                avg_loss += loss
            avg_loss /= critic_steps
            print(f"average critic loss: {avg_loss}")

            if nan_in_model(critic):
                print("NaN values in critic\nstopping training")
                break

            writer.add_scalar('critic_loss', avg_loss, epoch)

            if avg_loss <= critic_threshold:
                critic_suffices_count += 1
            else:
                critic_suffices_count = 0

            if critic_suffices_count >= critic_suffices_required:
                critic_suffices = True

            if critic_suffices:
                # training the actor
                avg_loss = 0.
                state_values = np.array(critic.evaluate(t(states)).squeeze(1).cpu())
                returns = td_values((states, rewards, dones), state_values, hyper_ps)
                advantages = returns - state_values
                for _ in range(actor_steps):
                    indices = random.sample(range(len(states)), batch_size)

                    advantage_weights = np.exp(advantages[indices] / beta)
                    advantage_weights = t(np.minimum(advantage_weights, max_advantage_weight))

                    normal, _ = actor(t(states[indices]))
                    log_pis = normal.log_prob(t(actions[indices]))
                    log_pis = torch.sum(log_pis, dim=1)
                    log_pis = torch.clamp(log_pis, min=min_log_pi)

                    losses = -log_pis * advantage_weights
                    losses = losses / batch_size  # normalise wrt the batch size
                    actor.backward(losses)

                    mean_loss = torch.sum(losses)
                    avg_loss += mean_loss
                avg_loss /= actor_steps
                print(f"average actor loss: {avg_loss}")

                if nan_in_model(actor):
                    print("NaN values in actor\nstopping training")
                    break

                writer.add_scalar('actor_loss', avg_loss, epoch)
            else:
                pre_training_epochs += 1
                if pre_training_epochs > max_pre_epochs:
                    print("critic couldn't be trained in appropriate time\nstopping training")
                    break

            epoch += 1

            if critic_suffices and epoch % validation_epoch_mod == 0:
                AWRAgent.compute_validation_return(
                    actor,
                    environment,
                    hyper_ps,
                    debug_type,
                    test_iterations,
                    epoch,
                    writer
                )

            states = dq_states
            actions = dq_actions
            rewards = dq_rewards

        return actor, critic

    @staticmethod
    def compute_validation_return(actor, env, hyper_ps, debug_type, iterations, epoch, writer):
        print("computing average return")
        sample_return = AWRAgent.validation_return(actor, env, hyper_ps, debug_type, iterations)
        writer.add_scalar('return', sample_return, epoch)
        print(f"return: {sample_return}")

    @staticmethod
    def validation_return(actor, env, hyper_ps, debug_type, iterations):
        sample_return = 0.
        for _ in range(iterations):
            s, a, r, d = [], [], [], []
            AWRAgent.sample_from_env(
                actor,
                env,
                debug_type != DebugType.NONE,
                exploration=False,
                replay_buffers=(s, a, r, d)
            )
            mcs = mc_values(r, hyper_ps)
            sample_return += np.mean(mcs)

        sample_return /= iterations
        return sample_return

    @staticmethod
    def sample_from_env(actor_model, env, debug, exploration, replay_buffers):
        states, actions, rewards, dones = replay_buffers
        obs = env.reset()
        state = obs_to_state(obs[0])
        done = False

        if debug:
            env.render()

        while not done:
            if exploration:
                action = env.action_space.sample()
            else:
                dist, action = actor_model.evaluate(state)
                action = action.cpu().item()  # 转为int类型
            res = env.step(action)

            reward = res[1]
            done = res[2]
            states.append(np.array(state.cpu()))
            actions.append(action)  # 存储int类型的action
            rewards.append(reward)
            dones.append(done)

            state = obs_to_state(res[0])

            if debug:
                env.render()

    @staticmethod
    def test(models, environment, hyper_ps, debug_type):
        actor, _ = models
        return AWRAgent.validation_return(actor, environment, hyper_ps, debug_type, hyper_ps['test_iterations'])


class Training:

    @staticmethod
    def train(models, agent, environment, hyper_ps, save=True, debug_type=DebugType.NONE):
        """
        Executes a full training and testing cycle and stores the results.

        :param models: The models to train.
        :param agent: The agent to train the models.
        :param environment: The environment to be trained in.
        :param hyper_ps: All hyper-parameters required for the given models and the agent.
        :param save: Triggers saving of parameters, models and results.
        :param debug_type: Specifies the amount of debug information to be printed during training and testing.
        """

        # setting the random seed
        if 'seed' in hyper_ps:
            torch.manual_seed(hyper_ps['seed'])

        # creating the directory for this test
        # this assumes that your working directory is awr/src/
        dir_path = "../trained_models/"
        dir_path += f"[{environment.unwrapped.spec.id}]_[{agent.name}]_["
        for m in models:
            dir_path += f"({type(m).name})_"
        dir_path += "]_"
        now = datetime.datetime.now()
        dir_path += now.strftime("%d.%m.%Y,%H:%M:%S.%f")
        dir_path += "/"

        os.makedirs(dir_path)

        # creating the tensorboardX writer
        writer = SummaryWriter(dir_path + 'tensorboard/')

        if 'test_iterations' not in hyper_ps:
            hyper_ps['test_iterations'] = 100
        test_iterations = hyper_ps['test_iterations']

        # extend the hyper-parameters to include environment information
        if not issubclass(type(environment.observation_space), Space):  # MuJoCo Robotics environment
            desired_goal_dims = environment.observation_space.spaces['desired_goal'].shape[0]
            achieved_goal_dims = environment.observation_space.spaces['achieved_goal'].shape[0]
            observation_dims = environment.observation_space.spaces['observation'].shape[0]
            hyper_ps['state_dim'] = desired_goal_dims + achieved_goal_dims + observation_dims
        else:
            hyper_ps['state_dim'] = environment.observation_space.shape[0]
        # 判断 action space 类型
        if isinstance(environment.action_space, gym.spaces.Discrete):
            hyper_ps['action_dim'] = environment.action_space.n
        else:
            hyper_ps['action_dim'] = environment.action_space.shape[0]

        # passing the hyper-parameters to the models
        for m in models:
            m.set_params(hyper_ps)
            init_func = type(m).init_func
            m.apply(xavier_init if init_func is None else init_func)

        # converting the models to the current device
        models = [m.to(device) for m in models]

        # training and testing
        print("training")
        models = agent.train(models, environment, hyper_ps, debug_type, writer)

        nans = [nan_in_model(m) for m in models]
        if any(nans):
            print("NaN values in some models\nskipping testing")
            return

        print("testing")
        rewards = []
        for _ in range(test_iterations):  # expected reward estimated with average
            rew = agent.test(models, environment, hyper_ps, debug_type)
            rewards.append(rew)
        reward = np.mean(np.array(rewards))

        if not save:
            print(f"Average reward: {reward}")
        else:
            # saving the hyper-parameters
            parameter_text = json.dumps(hyper_ps)
            parameter_file = open(dir_path + "hyper-parameters.json", "w")
            parameter_file.write(parameter_text)
            parameter_file.close()

            # saving the models
            for model in models:
                params = model.state_dict()
                file_path = type(model).name + ".model"

                torch.save(params, dir_path + file_path)

            # saving the results
            result_text = "Results\n===\n"
            result_text += f"Testing was conducted {test_iterations} times to obtain an estimate of the expected return.\n\n\n"
            result_text += f"\nAverage return\n---\n{reward}"

            results_file = open(dir_path + "results.md", "w")
            results_file.write(result_text)
            results_file.close()

# setting the hyper-parameters
hyper_ps = {
    'replay_size': 50000,
    'max_epochs': 150,
    'sample_mod': 10,
    'beta': 2.5,
    'max_advantage_weight': 50.,
    'min_log_pi': -50.,
    'discount_factor': .9,
    'alpha': 0.95, 
    'c_hidden_size': 150,
    'c_hidden_layers': 3,
    'a_hidden_size': 200,
    'a_hidden_layers': 5,
    'c_learning_rate': 1e-4,
    'c_momentum': .9,
    'a_learning_rate': 5e-5,
    'a_momentum': .9,
    'critic_threshold': 17.5,
    'critic_suffices_required': 1,
    'critic_steps_start': 200, 
    'critic_steps_end': 200,
    'actor_steps_start': 1000,
    'actor_steps_end': 1000,
    'batch_size': 256,
    'seed': 123456,
    'replay_fill_threshold': 1.,
    'random_exploration': True,
    'test_iterations': 30,
    'validation_epoch_mod': 3,
}

# configuring the environment 
environment = gym.make('CartPole-v1')

# setting up the training components
agent = AWRAgent
actor = Actor()  
critic = Critic()

# training and testing
Training.train(
    (actor, critic),
    agent, 
    environment,
    hyper_ps,
    save=True,
    debug_type=DebugType.NONE
)
