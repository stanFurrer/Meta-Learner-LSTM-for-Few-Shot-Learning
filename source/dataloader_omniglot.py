import os

import tensorflow as tf
from tqdm import tqdm
import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
AUTO_TUNE = tf.data.experimental.AUTOTUNE


@tf.function
def normalization_processor(image: tf.Tensor) -> tf.Tensor:
    return (image - mean) / std


@tf.function
def resize(image, size):
    return tf.image.resize(image, [size, size], name="resize")


@tf.function
def load_and_resize(image_path, image_size, phase):
    image_file = tf.io.read_file(image_path)
    # convert the compressed string to a 3D uint8 tensor
    image = tf.io.decode_png(image_file, channels=1)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)
    return resize(image, image_size)


def get_loader_and_processor(phase, image_size):
    def loader(path, label):
        return load_and_resize(path, image_size, phase), label

    def processor(batch, labels):
        return normalization_processor(batch), labels

    return loader, processor


class EpisodicSampler(tf.data.Dataset):

    @staticmethod
    def generate_sample_indices(data, n_cls, n_shot, n_eval):
        """ Draw random batches """
        classes = np.random.choice(len(data), n_cls, replace=False)
        shot_indices = []
        eval_indices = []
        for cls in classes:
            idx = np.random.choice(len(data[cls]), n_shot + n_eval, replace=False)
            shot_indices.append(idx[:n_shot])
            eval_indices.append(idx[n_shot:])
        return classes, shot_indices, eval_indices

    @staticmethod
    def _generator(images_dir, phase, n_eps, n_cls, n_shot, n_eval):

        stage_dir = os.path.join(images_dir, phase)
        labels = os.listdir(stage_dir)

        class_dirs = [os.path.join(stage_dir, label) for label in labels]

        data = [[] for _ in class_dirs]

        # Read the input data out of the source files
        for i, cd in enumerate(class_dirs):
            for image_file in os.listdir(cd):
                data[i].append(os.path.join(cd, image_file))

        class_labels = tf.eye(n_cls)

        for _ in range(n_eps):
            cls, shot_samples, eval_samples = EpisodicSampler.generate_sample_indices(data, n_cls, n_shot, n_eval)
            # shot samples
            for c_idx, c in enumerate(cls):
                for s in shot_samples[c_idx]:
                    yield data[c][s], class_labels[c_idx]
            # eval samples
            for c_idx, c in enumerate(cls):
                for s in eval_samples[c_idx]:
                    yield data[c][s], class_labels[c_idx]

    def __new__(cls, **kwargs):
        args = kwargs["image_path"], kwargs["phase"], kwargs["n_eps"], kwargs["n_cls"], kwargs["n_shot"], kwargs["n_eval"]
        loader, processor = get_loader_and_processor(kwargs["phase"], kwargs["image_size"])

        sampler = tf.data.Dataset.from_generator(cls._generator, output_types=(tf.string, tf.int32), output_shapes=([], [kwargs["n_cls"]]), args=args)
        sample_loader = sampler.map(loader, num_parallel_calls=AUTO_TUNE).map(processor, num_parallel_calls=AUTO_TUNE)
        batch_loader = sample_loader.batch(kwargs["batch_size"])
        return batch_loader.prefetch(AUTO_TUNE)


def prepare_data(args):
    batch_size = args.n_class * (args.n_shot + args.n_eval)

    kwargs = {
        "image_path": args.data_root,
        "n_cls": args.n_class,
        "n_shot": args.n_shot,
        "n_eval": args.n_eval,
        "batch_size": batch_size,
        "image_size": args.image_size
    }

    if args.mode == "train" or args.mode == "resume":
        train_loader = EpisodicSampler(**kwargs, phase="train", n_eps=args.episode)
        val_loader = EpisodicSampler(**kwargs, phase="val", n_eps=args.episode_val)
        return train_loader, val_loader
    else:
        return EpisodicSampler(**kwargs, phase="test", n_eps=args.episode_test)
