from agentlace.data.data_store import DataStoreBase
from threading import Lock
from typing import Any, List, NamedTuple
import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch as th
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class BernoulliMaskReplayBufferSamples(NamedTuple):
    """
    Necessary for the creation of samples based on the ReplayBuffer,
    including the bernoulli mask.

    Note:
        Due to using UTD-ratio in CHEQ, this might have an additional dimension
        depicting the updates specified in the config-file.
    """
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    masks: th.Tensor

    def __getitem__(self, index):
        """
        Returns the batch of a specified slice.
        Helpful for the training of EnsembleSAC.
        """
        
        item = (
            self.observations[index],
            self.actions[index],
            self.next_observations[index],
            self.dones[index],
            self.rewards[index],
            self.masks[index]
        )

        return BernoulliMaskReplayBufferSamples(*item)
    
    @property
    def get_shape(self):
        """
        This returns the shape of the sample.
        """
        shape = list(self.observations.shape)
        shape.insert(0, 6) # since we have observations, actions, next...
        return shape

class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        #super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        
        # get obs and action space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]
        self.action_dim = get_action_dim(action_space)

        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer (in the form of *args).
        """
        # Do a for-loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        Sample random batch from the buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, BernoulliMaskReplayBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        assert isinstance(array, np.ndarray), f"Assumed to get an np.ndarray as input, but got {type(array)}."

        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward
    

class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray
    bernoulli_mask: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        ensemble_size: Optional[int] = None,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size=buffer_size, 
            observation_space=observation_space, 
            action_space=action_space,
            device=device, 
            n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.ensemble_size = ensemble_size
        if self.ensemble_size:
            self.bernoulli_mask = np.zeros((self.buffer_size, self.n_envs, ensemble_size, ), dtype=np.bool_)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    @property
    def get_buffer_richness(self) -> float:
        """
        Computes the percentage of how much the buffer is filled.

        Returns:
            float: Current percentage of used capacity
        """
        # maybe this has to be adapted if more than one env is used
        return self.pos / self.buffer_size

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        mask: Optional[np.ndarray] = None
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        # if we use Bernoulli-mask
        if self.ensemble_size:
            self.bernoulli_mask[self.pos] = np.array(mask)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> Union[ReplayBufferSamples, BernoulliMaskReplayBufferSamples]:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> Union[ReplayBufferSamples, BernoulliMaskReplayBufferSamples]:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        temp = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )

        # either add mask to output or not, depending on usage
        if self.ensemble_size:
            data = temp + (self.bernoulli_mask[batch_inds, env_indices, :],)
            return BernoulliMaskReplayBufferSamples(*tuple(map(self.to_torch, data)))
        else:
            data = temp
            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):

    def __init__(
            self, 
            capacity: int, 
            observation_space: spaces.Space, 
            action_space: spaces.Space, 
            device: Union[th.device, str] = "auto", 
            ensemble_size: Optional[int] = None, 
            kappa: Optional[float] = None, 
            n_envs: int = 1, 
            optimize_memory_usage: bool = False,  # still a problem here with True 
            handle_timeout_termination: bool = False
    ):
        
        # created by us (SERL) to manage the data
        ReplayBuffer.__init__(
            self, 
            buffer_size=capacity, 
            observation_space=observation_space, 
            action_space=action_space,
            device=device, 
            n_envs=n_envs, 
            ensemble_size=ensemble_size,
            optimize_memory_usage=optimize_memory_usage, 
            handle_timeout_termination=handle_timeout_termination
        )
        # created from agentlace to manage the buffers state
        DataStoreBase.__init__(self, capacity=capacity)

        self._lock = Lock()

        # initialize for Bernoulli masking
        self.ensemble_size = ensemble_size
        self.kappa = kappa

    def latest_data_id(self):
        return self.pos

    def get_latest_data(self, from_id: int):
        raise NotImplementedError

    def __len__(self):
        return self.size()

    def insert(self, data: Dict) -> None:
        """
        Insert new data to the Buffer. Could be multi-dimensional.
        
        Note:
            The input is the same, independent on using Bernoulli or not.
            However, depending on if we use a Bernoulli mask the dimensions added to the buffer differ.
        
        Args:
            data (Dict): Data to be added to the buffer.
        """
        if not self.kappa:
            with self._lock:
                super(ReplayBufferDataStore, self).add(obs=data.get("observations"),
                                                       next_obs=data.get("next_observations"),
                                                       action=data.get("actions"),
                                                       reward=data.get("rewards"),
                                                       done=data.get("dones"),
                                                       infos=data.get("infos"))
        else:
            assert (0 <= self.kappa <= 1) and (isinstance(self.kappa, float)), \
                "Please make sure that your bernoulli mask coefficient is a float between 0 and 1."
            assert (0 < self.ensemble_size) and (isinstance(self.ensemble_size, int)), \
                "Please make sure that your ensemble size is a positive integer."

            with self._lock:
                super(ReplayBufferDataStore, self).add(obs=data.get("observations"),
                                                       next_obs=data.get("next_observations"),
                                                       action=data.get("actions"),
                                                       reward=data.get("rewards"),
                                                       done=data.get("dones"),
                                                       infos=data.get("infos"),
                                                       mask=self._generate_bernoulli_mask())
                
    def sample(self, batch_size: Union[int, List[int]]) -> Union[ReplayBufferSamples, BernoulliMaskReplayBufferSamples]:
        """
        Get a random sample from the ReplayBuffer.
        Depending on the initialization of the ReplayBuffer, 
        the sample will include the Bernoulli-masking or not.

        Args:
            batch_size (int, List): The number of samples in the requested batch.
                
        Note:
            Using a list such as [2, 20] you can specify to include multiple batches.
            Here we would have 2 batches of size 20. The first dimension of the outcome
            depicts the number of batches.
            This is relevant for the update-to-data ratio in CHEQ!

        Returns:
            Union[ReplayBufferSamples, BernoulliMaskReplayBufferSamples]: NamedTuple
        """
        with self._lock:
            if isinstance(batch_size, int):
                # handle normal batch creation
                return super(ReplayBufferDataStore, self).sample(batch_size=batch_size)
            if isinstance(batch_size, list):
                # handle batch creation with bernoulli mask and utd ratio
                all_observations = []
                all_actions = []
                all_next_observations = []
                all_dones = []
                all_rewards = []
                all_masks = []

                for _ in range(batch_size[0]):  # utd ratio
                    batch = super(ReplayBufferDataStore, self).sample(batch_size=batch_size[1])
                    all_observations.append(batch.observations.cpu())
                    all_actions.append(batch.actions.cpu())
                    all_next_observations.append(batch.next_observations.cpu())
                    all_dones.append(batch.dones.cpu())
                    all_rewards.append(batch.rewards.cpu())
                    all_masks.append(batch.masks.cpu())
                
                new_observations = np.array([observations for observations in all_observations])
                new_actions = np.array([actions for actions in all_actions])
                new_next_observations = np.array([next_observations for next_observations in all_next_observations])
                new_dones = np.array([dones for dones in all_dones])
                new_rewards = np.array([rewards for rewards in all_rewards])
                new_masks = np.array([masks for masks in all_masks])

                return BernoulliMaskReplayBufferSamples(
                    *tuple(map(self.to_torch, 
                               (new_observations,
                                new_actions,
                                new_next_observations,
                                new_dones,
                                new_rewards,
                                new_masks)
                            )
                        )
                )
            else:
                raise ValueError(f"You should input an integer or a list as the batch_size for the sample but inserted a {type(batch_size)}!")
        
    def __getstate__(self) -> Dict[str, Any]:
        """
        Remove lock from object state to allow pickling of the replay buffer.       
        """
        state = self.__dict__.copy()
        del state['_lock']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = Lock()

    def _generate_bernoulli_mask(self) -> np.ndarray:
        """
        Generate a Bernoulli mask for the current input.
        
        Returns:
            np.ndarray: Array of Bernoulli masks for the current data.
        """
        bernoulli_mask = np.random.binomial(n=1, p=self.kappa, size=self.ensemble_size)
        return bernoulli_mask


def inject_weight_into_state(state: Union[np.ndarray,th.Tensor], weight: Any):
    """
    Concatenate the state vector with the weight.
    Further, allows for multi-dimensional state vectors.

    Returns:
        torch.tensor: Concatenated state-weight-vector.
    """
    if isinstance(state, np.ndarray):
        if len(state.shape) == 1:
            weight_array = np.array([weight], dtype=state.dtype)
            array = np.append(state, weight_array)
        else:
            weight_array = np.array([[weight]], dtype=state.dtype)
            array = np.append(state, weight_array, axis=1)

        array = th.tensor(array)

    elif isinstance(state, th.Tensor):
        if len(state.shape) == 1:
            weight_tensor = th.tensor([weight], dtype=state.dtype)
            array = th.cat((state, weight_tensor), dim=0)
        else:
            weight_tensor = th.tensor([[weight]], dtype=state.dtype)
            array = th.cat((state, weight_tensor), dim=1)

    return array

def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")
