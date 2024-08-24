import tensorflow as tf
from collections.abc import Mapping
from ml_collections import ConfigDict


def random_resized_crop(image, scale, ratio, seed):
    if len(tf.shape(image)) == 3:
        image = tf.expand_dims(image, axis=0)
    batch_size = tf.shape(image)[0]
    
    log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
    height = tf.shape(image)[1]
    width = tf.shape(image)[2]

    random_scales = tf.random.stateless_uniform((batch_size,), seed, scale[0], scale[1])
    random_ratios = tf.exp(
        tf.random.stateless_uniform((batch_size,), seed, log_ratio[0], log_ratio[1])
    )

    new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
    new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
    height_offsets = tf.random.stateless_uniform(
        (batch_size,), seed, 0, 1 - new_heights
    )
    width_offsets = tf.random.stateless_uniform((batch_size,), seed, 0, 1 - new_widths)

    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (height, width)
    )

    return tf.squeeze(image)


AUGMENT_OPS = {
    "random_resized_crop": random_resized_crop,
    "random_brightness": tf.image.stateless_random_brightness,
    "random_contrast": tf.image.stateless_random_contrast,
    "random_saturation": tf.image.stateless_random_saturation,
    "random_hue": tf.image.stateless_random_hue,
    "random_flip_left_right": tf.image.stateless_random_flip_left_right,
}


def augment(image, seed, **augment_kwargs):
    image = tf.cast(image, tf.float32) / 255  
    for op in augment_kwargs["augment_order"]:
        if op in augment_kwargs:
            if isinstance(augment_kwargs[op], Mapping) or isinstance(
                augment_kwargs[op], ConfigDict
            ):
                image = AUGMENT_OPS[op](image, seed=seed, **augment_kwargs[op])
            else:
                image = AUGMENT_OPS[op](image, seed=seed, *augment_kwargs[op])
        else:
            image = AUGMENT_OPS[op](image, seed=seed)
        image = tf.clip_by_value(image, 0, 1)
    image = tf.cast(image * 255, tf.uint8)
    return image


def augment_batch(images, seed, **augment_kwargs):
    batch_size = tf.shape(images)[0]
    sub_seeds = [seed]
    for _ in range(batch_size):
        sub_seeds.append(
            tf.random.stateless_uniform(
                [2],
                seed=sub_seeds[-1],
                minval=None,
                maxval=None,
                dtype=tf.int32,
            )
        )
    images = tf.cast(images, tf.float32) / 255  
    for op in augment_kwargs["augment_order"]:
        if op in augment_kwargs:
            if isinstance(augment_kwargs[op], Mapping) or isinstance(
                augment_kwargs[op], ConfigDict
            ):
                
                assert op == "random_resized_crop"
                images = AUGMENT_OPS[op](images, seed=seed, **augment_kwargs[op])
            else:
                images_list = []
                for i in range(batch_size):
                    images_list.append(
                        AUGMENT_OPS[op](
                            images[i], seed=sub_seeds[i], *augment_kwargs[op]
                        )
                    )
                images = tf.stack(images_list)
        else:
            images_list = []
            for i in range(batch_size):
                images_list.append(AUGMENT_OPS[op](images[i], seed=sub_seeds[i]))
            images = tf.stack(images_list)
        images = tf.clip_by_value(images, 0, 1)
    images = tf.cast(images * 255, tf.uint8)
    return images
