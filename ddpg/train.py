import time
import multiprocessing
import threading
from threading import Lock
import numpy as np
from evaluator import Evaluator
from my_env import MyEnv


def sample(agent, writer, logger, episode, lock, args):
    env = MyEnv(accuracy=args.accuracy, action_repeat=args.action_repeat, reward_type=args.reward_type)
    max_episode_length = args.max_episode_length // args.action_repeat

    episode_steps = 0
    episode_reward = 0.
    observation = env.reset()
    agent.reset(observation)

    done = False
    while not done and episode_steps < max_episode_length:
        # obtain an action
        if episode < args.warmup and args.resume is None:
            action = agent.random_action()
        else:
            lock.acquire()
            action = agent.select_action(observation)
            lock.release()
        
        episode_steps += 1
        observation, reward, done = env.step(action)
        episode_reward += reward

        lock.acquire()
        agent.observe(reward, observation, done)
        lock.release()
        episode_reward += reward
    
    env.close()
    lock.acquire()
    writer.add_scalar('sample/score', episode_reward, episode)
    writer.add_scalar('sample/stpes', episode_steps, episode)
    logger.debug(' [sample] episode #{}: score = {}, steps = {}'.format(episode, episode_reward, episode_steps))
    lock.release()


def train(agent, writer, logger, args):
    lock = Lock()
    log = iteration = episode = 0
    num_workers = args.num_workers
    checkpoint_num = 0 if args.resume is None else args.resume_num
    
    evaluator = Evaluator(args)

    logger.debug('----------------- random sampling -----------------')
    if args.resume is None:
        while episode < args.warmup:
            threads = []
            for _ in range(num_workers):
                thread = threading.Thread(target=sample, args=(agent, writer, logger, episode, lock, args,))
                thread.setDaemon(True)
                threads.append(thread)
                episode += 1

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            writer.add_scalar('sample/replay buffer size', agent.memory.size(), episode)
            logger.debug('[internal] replay buffer size: {}'.format(agent.memory.size()))
    logger.debug('------------------ warmup finish ------------------')

    while iteration < args.nb_iterations:
        threads = []
        for _ in range(num_workers):
            thread = threading.Thread(target=sample, args=(agent, writer, logger, episode, lock, args,))
            thread.setDaemon(True)
            threads.append(thread)
            episode += 1

        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()

        writer.add_scalar('sample/replay buffer size', agent.memory.size(), episode)
        logger.debug(' [internal] replay buffer size: {}'.format(agent.memory.size()))

        for _ in range(args.nb_train_steps):
            iteration += 1
            Q, critic_loss, critic_output = agent.update_policy()
            log += 1
            writer.add_scalar('train/Q', Q, log)
            writer.add_scalar('train/critic loss', critic_loss, log)
            writer.add_scalar('train/critic output', critic_output, log)
        
        # validation
        validation_rewards, validation_steps = evaluator(agent.select_action, logger)
        writer.add_scalar('validation/scores', np.mean(validation_rewards), iteration)
        writer.add_scalar('validation/steps', np.mean(validation_steps), iteration)
        logger.info(' [validation] scores = {}, steps = {}'.format(np.mean(validation_rewards), np.mean(validation_steps)))

        # checkpoint
        if iteration % args.checkpoint_interval == 0:
            checkpoint_num += 1
            logger.debug(' [checkpoint] #{} at {}'.format(checkpoint_num, args.output))
            agent.save_model(args.output, checkpoint_num)
