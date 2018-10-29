import numpy as np
from baselines.common.vec_env import VecEnv
import ray

from nips.round2_env import OBSERVATION_SPACE


class TaskPool(object):
    """Helper class for tracking the status of many in-flight actor tasks."""

    def __init__(self, timeout=1):
        self._tasks = {}
        self.timeout = timeout

    def add(self, worker, obj_id):
        self._tasks[obj_id] = worker

    def completed(self):
        pending = list(self._tasks)
        if pending:
            ready, _ = ray.wait(pending, num_returns=len(pending), timeout=self.timeout)
            if not ready:
                return []
            for obj_id in ready:
                yield (self._tasks.pop(obj_id), obj_id)

    @property
    def count(self):
        return len(self._tasks)


class Actor(object):

    def __init__(self, aid, env_fn):
        self.aid = aid
        self.env = env_fn()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        if done:
            ob = self.env.reset()
        return ob, reward, done, info

    def reset(self):
        return self.env.reset()

    def get_spaces(self):
        return self.env.observation_space, self.env.action_space

    def get_id(self):
        return self.aid


class RemoteVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.task_pool = TaskPool(timeout=10)

        nenvs = len(env_fns)

        self.actors = []
        self.actor_to_i = {}
        remote_actor = ray.remote(Actor)
        for i in range(nenvs):
            actor = remote_actor.remote(i, env_fns[i])
            self.actors.append(actor)
            self.actor_to_i[actor] = i

        observation_space, action_space = ray.get(self.actors[0].get_spaces.remote())
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        self.results = [([0] * OBSERVATION_SPACE, 0, False, {"bad": True})] * self.num_envs

    def step_async(self, actions):
        for actor, action in zip(self.actors, actions):
            # print(action, any(action))
            if any(action):
                # print(self.actor_to_i[actor], action)
                self.task_pool.add(actor, actor.step.remote(action))
        self.waiting = True

    def step_wait(self):
        done_ids = set([])

        count = 0
        while count * 2 < self.task_pool.count:
            for actor, obj in self.task_pool.completed():
                _i = self.actor_to_i[actor]
                done_ids.add(_i)
                self.results[_i] = ray.get(obj)
                count += 1
        # print(count, self.task_pool.count, done_ids)
        self.waiting = False
        obs, rews, dones, infos = zip(*self.results)
        for i in range(self.num_envs):
            infos[i]["bad"] = i not in done_ids
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obj_ids = [actor.reset.remote() for actor in self.actors]
        results = ray.get(ray.wait(obj_ids, num_returns=self.num_envs)[0])
        # TODO: should update self.results, but it's ok because this function will be invoked only at first
        return np.stack(results)

    def close(self):
        if self.closed:
            return
        self.closed = True
