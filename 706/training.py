import time
import numpy as np
from observation_process import obs_process

def train(agent, env, evaluator, writer, args):
    apply_noise = args.apply_noise
    action_repeat = args.action_repeat
    max_episode_length = args.max_episode_length // action_repeat
    save_times = 0 if args.resume is None else args.resume_num

    log = step = episode = episode_steps = 0
    episode_reward = 0.
    obs = observation = None

    time_stamp = time.time()
    while step <= args.iterations:
        if observation is None:
            # get unprojected observation from environment
            obs = env.reset(project=False)
            observation = obs_process(obs)
            agent.reset(observation)

        # random or select action
        if step <= args.warmup and args.resume is None:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, apply_noise=apply_noise)

        # action repeat
        total_reward = 0.
        for _ in range(action_repeat):
            obs, reward, done, _ = env.step(action, project=False)
            total_reward += reward
            if done:
                break
        observation = obs_process(obs)
        reward = total_reward
        
        agent.observe(reward, observation, done)

        step += 1
        episode_steps += 1
        episode_reward += reward

        if done or (episode_steps >= max_episode_length and max_episode_length):
            # log
            episode_time = time.time() - time_stamp
            time_stamp = time.time()
            print('episode #{}: reward={}, steps={}, time={:.2f}'.format(
                    episode, episode_reward, episode_steps, episode_time
            ))
            writer.add_scalar('train/reward', episode_reward, episode)

            if step > args.warmup:
                # train
                for _ in range(args.nb_train_steps):
                    log += 1
                    Q, critic_loss = agent.update_policy()
                    writer.add_scalar('train/Q', Q, log)
                    writer.add_scalar('train/critic loss', critic_loss, log)
                    
                # validation
                if episode > 0 and episode % args.validation_interval == 0:
                    validation_reward = evaluator(env, agent.select_action)
                    print('[validation] episode #{}, reward={}'.format(episode, np.mean(validation_reward)))
                    writer.add_scalar('validation/reward', np.mean(validation_reward), episode)

                # checkpoint
                if episode > 0 and episode % args.save_interval == 0:
                    save_times += 1
                    print('[save model] #{} in {}'.format(save_times, args.output))
                    agent.save_model(args.output, save_times)
            else:
                print('warmup stage...')

            observation = None
            episode_steps = 0
            episode_reward = 0
            episode += 1
