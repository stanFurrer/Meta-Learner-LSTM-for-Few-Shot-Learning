import argparse
import os
import random

import h5py
import numpy as np
import tensorflow as tf

import source

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode', choices=['train', 'test', "resume"], default="train")

# Hyper-parameters
FLAGS.add_argument('--n-shot', type=int, default=5,
                   help="How many examples per class for training (k, n_support)")
FLAGS.add_argument('--n-eval', type=int, default=15,
                   help="How many examples per class for evaluation (n_query)")
FLAGS.add_argument('--n-class', type=int, default=5,
                   help="How many classes (N, n_way)")
# FLAGS.add_argument('--input-size', type=int, default=4,
#                    help="Input size for the first LSTM")
FLAGS.add_argument('--hidden-size', type=int, default=20,
                   help="Hidden size for the first LSTM")
FLAGS.add_argument('--lr', type=float, default=1e-3,
                   help="Learning rate")
FLAGS.add_argument('--episode', type=int, default=50000,
                   help="Episodes to train")
FLAGS.add_argument('--episode-val', type=int, default=100,
                   help="Episodes to eval")
FLAGS.add_argument('--episode-test', type=int, default=600,
                   help="Episodes to test")
FLAGS.add_argument('--epoch', type=int, default=8,
                   help="Epoch to train for an episode")
FLAGS.add_argument('--batch-size', type=int, default=25,
                   help="Batch size when training an episode")
FLAGS.add_argument('--image-size', type=int, default=84,
                   help="Resize image to this size")
FLAGS.add_argument('--grad-clip', type=float, default=0.25,
                   help="Clip gradients larger than this number")
FLAGS.add_argument('--bn-momentum', type=float, default=0.95,
                   help="Momentum parameter in BatchNorm2d")
FLAGS.add_argument('--bn-eps', type=float, default=1e-3,
                   help="Eps parameter in BatchNorm2d")

FLAGS.add_argument('--meta-learner', choices=['lstm', 'full-lstm', 'gru', 'corr-lstm', 'lstm-wo-init', 'sgd'], default="lstm")
FLAGS.add_argument('--meta-dropout', type=float, default=0,
                   help="Dropout applied to meta-learner cI (learner initializer)")

# Paths
FLAGS.add_argument('--data', default='CUB',
                   help="Name of dataset")
FLAGS.add_argument('--data-root', type=str, default="data/CUB",
                   help="Location of data")
FLAGS.add_argument('--resume', action='store_true')
FLAGS.add_argument('--save', type=str, default='logs',
                   help="Location to logs and ckpts")

# Others
FLAGS.add_argument('--cpu', action='store_true',
                   help="Set this to use CPU, default use CUDA")
FLAGS.add_argument('--visible-gpu', type=str, default="0")
FLAGS.add_argument('--log-freq', type=int, default=50,
                   help="Logging frequency")
FLAGS.add_argument('--val-freq', type=int, default=500,
                   help="Validation frequency")
FLAGS.add_argument('--seed', type=int,
                   help="Random seed")
FLAGS.add_argument('--ID', type=int,
                   help="ID of the run")


@tf.function
def train(learner: source.Learner, meta_learner: source.MetaLearner, train_input, train_target, epoch, batch_size, meta_mode, dropout):
    """trains the model over a data set for given number of epochs"""

    if meta_mode == "train":
        cI = tf.nn.dropout(meta_learner.meta_lstm.cI, dropout)
    else:
        cI = meta_learner.meta_lstm.cI

    hs = [[None, cI]]

    for j in range(epoch):
        for i in range(0, len(train_input), batch_size):
            x = train_input[i:i + batch_size]
            y = train_target[i:i + batch_size]

            # call the learner and get the loss/grad
            learner.set_flat_trainable_params(cI)  #  no meta learning back-prop
            loss, gradients = learner.call_with_grad(x, y)

            # preprocess grad & loss
            flat_gradient_list = []
            for grad_element in gradients:
                flat_gradient_list.append(tf.reshape(grad_element, shape=[-1, 1]))
            flat_gradient = tf.concat(flat_gradient_list, 0)
            loss_rs = tf.reshape(loss, shape=[-1, 1])
            grad_prep = source.preprocess_grad_loss(flat_gradient)  # [n_learner_params, 2]
            loss_prep = source.preprocess_grad_loss(loss_rs)  # [1, 2]

            # call the meta-learner to get the updated learner parameters
            meta_learner_input = [loss_prep, grad_prep, flat_gradient, hs[-1]]
            cI, h = meta_learner(meta_learner_input)
            hs.append(h)

    return cI


@tf.function
def episode_routine(learner, learner_mock, meta_learner, shot_input, shot_target, test_input, test_target, opt, epoch, batch_size, grad_clip, dropout):

    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as meta_gradient_tape:
        meta_gradient_tape.watch(meta_learner.trainable_variables)
        cI = train(learner, meta_learner, shot_input, shot_target, epoch, batch_size, "train", dropout)
        meta_gradient_tape.watch(cI)
        predictions = learner_mock.set_flat_params_and_predict(cI, learner.get_batch_stats(), test_input)
        loss = tf.reduce_mean(learner_mock.loss_object(test_target, predictions))
        metrics = [tf.reduce_mean(metric(test_target, predictions)) for metric in learner_mock.train_metrics]

    nabla = meta_gradient_tape.gradient(loss, meta_learner.trainable_variables)
    nabla, _ = tf.clip_by_global_norm(nabla, grad_clip)

    opt.apply_gradients(zip(nabla, meta_learner.trainable_variables))

    return loss, metrics


@tf.function
def meta_test(learner, learner_mock, meta_learner, shot_input, shot_target, test_input, test_target, epoch, batch_size, dropout):

    cI = train(learner, meta_learner, shot_input, shot_target, epoch, batch_size, "test", dropout)
    predictions = learner_mock.set_flat_params_and_predict(cI, learner.get_batch_stats(), test_input)
    loss = tf.reduce_mean(learner_mock.loss_object(test_target, predictions))
    metrics = [tf.reduce_mean(metric(test_target, predictions)) for metric in learner_mock.train_metrics]

    return loss, metrics


def main():
    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

    if args.seed is None:
        if args.mode == "test" or args.mode == "resume":
            args.seed = int(input("No seed is given. Specify the seed of the checkpoint: "))
            args.ID = int(input("No ID is given. Specify the ID of the checkpoint: "))
        else:
            args.seed = random.randint(0, 1e3)

    # set all seeds equal to global seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    # suspend some info from TF
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # use float32
    tf.keras.backend.set_floatx('float32')

    # set visible GPUs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    # set memory growth (only for non-XLA GPUs)
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # create logger
    logger = source.GOATLogger(args)
    args.logger = logger

    # ONLY FOR DEBUGGING
    # tf.config.experimental_run_functions_eagerly(True)
    # tf.autograph.set_verbosity(2)

    # Set up learner, meta-learner
    learner = source.Learner(args.bn_eps, args.bn_momentum, args.n_class, args.lr, args.image_size)
    learner_mock = source.Learner(args.bn_eps, args.bn_momentum, args.n_class, args.lr, args.image_size)
    meta_learner = source.MetaLearner(args.hidden_size, learner, args)

    if args.mode == 'test':
        logger.resume_ckpt(meta_learner)

        #   random seed for data drawing
        seed = random.randint(0, 1e3)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        test_loader = source.prepare_data_imagenet(args)
        logger.loginfo("Meta learner has been loaded")
        for test_episode_x, test_episode_y in test_loader:
            test_shot_input, test_test_input = tf.split(test_episode_x, [args.n_class * args.n_shot, args.n_class * args.n_eval], axis=0)
            test_shot_target, test_test_target = tf.split(test_episode_y, [args.n_class * args.n_shot, args.n_class * args.n_eval], axis=0)
            test_loss, test_metrics = meta_test(learner, learner_mock, meta_learner, test_shot_input, test_shot_target, test_test_input,
                                                test_test_target, args.epoch, args.batch_size, args.meta_dropout)
            logger.batch_info(loss=test_loss.numpy(), acc=100 * test_metrics[0].numpy(), phase='eval')
        logger.batch_info(totaleps=args.episode_test, phase='testdone')
        return

    train_hist = []
    test_hist = []

    if args.mode == "resume":
        logger.resume_ckpt(meta_learner)
        with h5py.File(logger.get_save_dir() + "/hist.hdf5", "r") as h5f:
            train_hist = list(h5f["train_hist"])
            test_hist = list(h5f["test_hist"])

    # Get data loaders
    train_loader, val_loader = source.prepare_data_imagenet(args)

    """==========   Testing the efficiency of data pipeline     ==========="""
    # import time
    # default_timeit_steps = 500
    #
    # def timeit(ds, steps=default_timeit_steps):
    #     start = time.time()
    #     it = iter(ds)
    #     for i in range(steps):
    #         _ = next(it)
    #         if i % 10 == 0:
    #             print('.', end='')
    #     print()
    #     end = time.time()
    #     duration = end - start
    #     print("{} batches: {} s".format(steps, duration))
    #     print("{:0.5f} Images/s".format(args.batch_size * steps / duration))
    #
    # timeit(train_loader)
    """====================================================================="""

    logger.loginfo("Start training")

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

    print("Building episode routine")

    for eps, (episode_x, episode_y) in enumerate(train_loader):

        shot_input, test_input = tf.split(episode_x, [args.n_class * args.n_shot, args.n_class * args.n_eval], axis=0)
        shot_target, test_target = tf.split(episode_y, [args.n_class * args.n_shot, args.n_class * args.n_eval], axis=0)

        loss, metrics = episode_routine(learner, learner_mock, meta_learner, shot_input, shot_target, test_input, test_target,
                                        opt, args.epoch, args.batch_size, args.grad_clip, args.meta_dropout)

        hist = logger.batch_info(eps=eps, totaleps=args.episode, loss=loss.numpy(), acc=100 * metrics[0].numpy(), phase='train')
        train_hist.append(hist)

        # Meta-validation and checkpoint saving
        if eps % args.val_freq == 0 and eps != 0:
            logger.save_ckpt(meta_learner)
            batch_stats = learner.get_batch_stats()

            for val_episode_x, val_episode_y in val_loader:
                val_shot_input, val_test_input = tf.split(val_episode_x, [args.n_class * args.n_shot, args.n_class * args.n_eval], axis=0)
                val_shot_target, val_test_target = tf.split(val_episode_y, [args.n_class * args.n_shot, args.n_class * args.n_eval], axis=0)

                val_loss, val_metrics = meta_test(learner, learner_mock, meta_learner, val_shot_input, val_shot_target, val_test_input,
                                                  val_test_target, args.epoch, args.batch_size, args.meta_dropout)
                logger.batch_info(loss=val_loss.numpy(), acc=100 * val_metrics[0].numpy(), phase='eval')

            hist = logger.batch_info(eps=eps, totaleps=args.episode_val, phase='evaldone')
            test_hist.append(hist)

            learner.set_batch_stats(batch_stats)

    with h5py.File(logger.get_save_dir()+"/hist.hdf5", "w") as h5f:
        h5f.create_dataset("train_hist", data=np.asarray(train_hist))
        h5f.create_dataset("test_hist", data=np.asarray(test_hist))

    logger.loginfo("Done")


if __name__ == '__main__':
    main()
