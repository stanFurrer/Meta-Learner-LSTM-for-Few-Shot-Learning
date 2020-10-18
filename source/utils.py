import logging
import os

import numpy as np
import tensorflow as tf


class GOATLogger:

    def __init__(self, args):
        args.save = "/" + args.save + '/log-{}'.format(args.seed)

        self.mode = args.mode
        self.save_root = os.getcwd() + args.save
        self.log_freq = args.log_freq
        self.best_acc = 0.0

        if self.mode == 'train' or self.mode == 'resume':
            if args.ID is None:
                idx = 0
                while os.path.exists(self.save_root+"-"+str(idx)):
                    idx += 1
                self.save_root = self.save_root+"-"+str(idx)
                os.mkdir(self.save_root)
            else:
                idx = args.ID
                self.save_root = self.save_root+"-"+str(idx)

            filename = os.path.join(self.save_root, 'console.log')
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s.%(msecs)03d - %(message)s',
                                datefmt='%b-%d %H:%M:%S',
                                filename=filename,
                                filemode='a')
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter('%(message)s'))
            logging.getLogger('').addHandler(console)

            logging.info("Logger at {}".format(filename))
        else:
            self.save_root = self.save_root+"-"+str(args.ID)

            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s.%(msecs)03d - %(message)s',
                                datefmt='%b-%d %H:%M:%S')

        logging.info("Seed: {}".format(args.seed))
        self.reset_stats()

    def reset_stats(self):
        if self.mode == 'train' or self.mode == 'resume':
            self.stats = {'train': {'loss': [], 'acc': []},
                          'eval': {'loss': [], 'acc': []}}
        else:
            self.stats = {'eval': {'loss': [], 'acc': []}}

    def batch_info(self, **kwargs):
        if kwargs['phase'] == 'train':
            self.stats['train']['loss'].append(kwargs['loss'])
            self.stats['train']['acc'].append(kwargs['acc'])

            if kwargs['eps'] % self.log_freq == 0 and kwargs['eps'] != 0:
                loss_mean = np.mean(self.stats['train']['loss'])
                acc_mean = np.mean(self.stats['train']['acc'])
                # self.draw_stats()
                self.loginfo("[{:5d}/{:5d}] loss: {:6.4f} ({:6.4f}), acc: {:6.3f}% ({:6.3f}%)".format( \
                    kwargs['eps'], kwargs['totaleps'], kwargs['loss'], loss_mean, kwargs['acc'], acc_mean))
            return kwargs['loss'], kwargs['acc']

        elif kwargs['phase'] == 'eval':
            self.stats['eval']['loss'].append(kwargs['loss'])
            self.stats['eval']['acc'].append(kwargs['acc'])

        elif kwargs['phase'] == 'evaldone':
            loss_mean = np.mean(self.stats['eval']['loss'])
            loss_std = np.std(self.stats['eval']['loss'])
            acc_mean = np.mean(self.stats['eval']['acc'])
            acc_std = np.std(self.stats['eval']['acc'])
            acc_conf = 1.96 * acc_std / np.sqrt(len(self.stats['eval']['acc']))
            self.loginfo("[{:5d}] Eval ({:3d} episode) - loss: {:6.4f} +- {:6.4f}, acc: {:6.3f} +- {:5.3f}%".format(
                kwargs['eps'], kwargs['totaleps'], loss_mean, loss_std, acc_mean, acc_conf))

            if acc_mean > self.best_acc:
                self.best_acc = acc_mean
                self.loginfo("* Best accuracy so far *\n")

            self.reset_stats()
            return loss_mean, loss_std, acc_mean, acc_std

        elif kwargs['phase'] == 'testdone':
            loss_mean = np.mean(self.stats['eval']['loss'])
            loss_std = np.std(self.stats['eval']['loss'])
            acc_mean = np.mean(self.stats['eval']['acc'])
            acc_std = np.std(self.stats['eval']['acc'])
            acc_conf = 1.96 * acc_std / np.sqrt(len(self.stats['eval']['acc']))
            self.loginfo("Test ({:3d} episode) - loss: {:6.4f} +- {:6.4f}, acc: {:6.3f} +- {:5.3f}%".format(
                kwargs['totaleps'], loss_mean, loss_std, acc_mean, acc_conf))
        else:
            raise ValueError("phase {} not supported".format(kwargs['phase']))

    def save_ckpt(self, meta_learner):
        if not os.path.exists(os.path.join(self.save_root, 'ckpts')):
            os.mkdir(os.path.join(self.save_root, 'ckpts'))
        meta_learner.save_weights(os.path.join(self.save_root, 'ckpts', 'meta-learner'))

    def resume_ckpt(self, meta_learner):
        meta_learner.load_weights(os.path.join(self.save_root, 'ckpts', 'meta-learner'))

    def logdebug(self, strout):
        logging.debug(strout)

    def loginfo(self, strout):
        logging.info(strout)

    def get_save_dir(self):
        return self.save_root


@tf.function
def preprocess_grad_loss(x):
    p = 10.0
    absolute = tf.math.abs(x)
    indicator = tf.math.abs(x) >= tf.exp(-p)
    # preprocess 1
    entry_1 = tf.where(indicator, tf.math.log(absolute + 1e-8) / p, -1.0)

    # preprocess 2
    entry_2 = tf.where(indicator, tf.sign(x), tf.exp(p) * x)

    return tf.concat([entry_1, entry_2], -1)
