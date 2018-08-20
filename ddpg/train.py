import time
import threading
import numpy as np
from evaluator import Evaluator
from my_env import MyEnv

# sample function, for multi thread
def sample(agent, logger, episode, lock, args):
    env = MyEnv(accuracy=args.accuracy, action_repeat=args.action_repeat, reward_type=args.reward_type)
    max_episode_length = args.max_episode_length // args.action_repeat

    episode_steps = 0
    episode_reward = 0.
    observation = env.reset()
    agent.reset(observation)

    done = False
    while not done and episode_steps < max_episode_length:
        # obtain an action
        lock.acquire()
        if episode < args.warmup_episodes and args.resume is False:
            action = agent.random_action()
        else:
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
    logger.debug('[sample] episode #{}: reward = {}, length = {}'.format(episode, episode_reward, episode_steps))
    lock.release()


def train(agent, writer, logger, args):
    num_workers = args.num_workers
    lock = threading.Lock()

    log = iteration = episode = 0
    checkpoint_num = 0 if args.resume is False else args.resume_num
    
    evaluator = Evaluator(args)

    # random sampling phase
    if args.resume is False:
        while episode < args.warmup_episodes:
            threads = []
            for _ in range(num_workers):
                thread = threading.Thread(target=sample, args=(agent, logger, episode, lock, args,))
                thread.setDaemon(True)
                threads.append(thread)
                episode += 1
                if episode >= args.warmup_episodes:
                    break
            
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            writer.add_scalar('sample/replay_buffer_size', agent.memory.size(), episode)
            logger.debug('[sample] replay buffer size: {}'.format(agent.memory.size()))

    logger.info('warmup phase finish, replay buffer size: {}'.format(agent.memory.size()))

    # training phase
    while iteration < args.nb_iterations:
        logger.debug('<----- iterations #{} ----->'.format(iteration))
        threads = []
        for _ in range(num_workers):
            thread = threading.Thread(target=sample, args=(agent, logger, episode, lock, args,))
            thread.setDaemon(True)
            threads.append(thread)
            episode += 1

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        writer.add_scalar('sample/replay_buffer_size', agent.memory.size(), episode)
        logger.debug('[sample] replay buffer size: {}'.format(agent.memory.size()))

        for _ in range(args.nb_train_epochs):
            Q, critic_loss, critic_output = agent.update_policy()
            log += 1
            writer.add_scalar('train/Q', Q, log)
            writer.add_scalar('train/critic_loss', critic_loss, log)
            writer.add_scalar('train/critic_output', critic_output, log)
            logger.debug('[training] Q value: {}'.format(Q))
            logger.debug('[training] critic loss: {}'.format(critic_loss))
        
        # validation
        if iteration > 0 and iteration % args.validation_interval == 0:
            validation_rewards, validation_steps = evaluator(agent.select_action, logger)
            writer.add_scalar('validation/score', np.mean(validation_rewards), iteration)
            writer.add_scalar('validation/steps', np.mean(validation_steps), iteration)
            logger.info('[validation] iteration #{}: scores = {}, steps = {}'.format(
                iteration, np.mean(validation_rewards), np.mean(validation_steps)
            ))

        # checkpoint
        if iteration > 0 and iteration % args.checkpoint_interval == 0:
            checkpoint_num += 1
            agent.save_model(args.output, checkpoint_num)
            logger.info('[checkpoint] iteration #{}: checkpoint-{}'.format(iteration, checkpoint_num))
        
        iteration += 1
