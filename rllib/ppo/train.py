def train(agent, evaluator, logger, writer, args):
    while True:
        train_result = agent.train()

        # logger
        logger.info('iteration No.{}:'.format(train_result.training_iteration))
        logger.info('time this iteration: {}'.format(train_result.time_this_iter_s))
        logger.debug('# timesteps = {}'.format(train_result.timesteps_this_iter))
        logger.debug('# total timesteps = {}'.format(train_result.timesteps_total))
        logger.debug('# episodes = {}'.format(train_result.episodes_total))
        logger.debug('episode mean length: {}'.format(train_result.episode_len_mean))
        logger.debug('episode reward:')
        logger.debug('  [mean] {}'.format(train_result.episode_reward_mean))
        logger.debug('  [max] {}'.format(train_result.episode_reward_max))
        logger.debug('  [min] {}'.format(train_result.episode_reward_min))
        logger.debug('--------------------------------')
        # summary
        writer.add_scalar('train/mean_reward', train_result.episode_reward_mean, train_result.training_iteration)
        writer.add_scalar('train/mean_steps', train_result.episode_len_mean, train_result.training_iteration)
        writer.add_scalar('train/time', train_result.time_this_iter_s, train_result.training_iteration)

        # validation
        if train_result.training_iteration % args.validation_interval == 0:
            validation_reward, validation_steps = evaluator(agent, False)
            logger.info('# validation at iteration No.{}: reward = {}, episode length = {}'.format(
                train_result.training_iteration, validation_reward, validation_steps
            ))
            # record validation score/steps in private tensorboard
            writer.add_scalar('validation/reward', validation_reward, train_result.training_iteration)
            writer.add_scalar('validation/steps', validation_steps, train_result.training_iteration)

        # checkpoint
        if train_result.training_iteration % args.checkpoint_interval == 0:
            save_result = agent.save(args.checkpoint_dir)
            logger.info('# checkpoint at iteration No.{}, save at {}'.format(train_result.training_iteration, save_result))
        
        if train_result.training_iteration >= args.iterations:
            break
