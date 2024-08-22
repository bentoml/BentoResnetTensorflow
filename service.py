import typing as t
import numpy as np
from PIL.Image import Image

import bentoml


BENTOML_MODEL_TAG = "resnet-v2-tensorflow"

@bentoml.service(
    name="bento-resnet-v2-tensorflow",
    traffic={
        "timeout": 300,
        "concurrency": 8,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-tesla-t4",
    },
)
class Resnet:

    bento_model_ref = bentoml.models.get(BENTOML_MODEL_TAG)

    def __init__(self) -> None:

        import tensorflow as tf
        import tensorflow_hub as hub

        self.device = "cuda" if tf.test.is_gpu_available() else "cpu"
        # we can also use `hub.load` instead of `tf.saved_model.load`
        self.model = tf.saved_model.load(self.bento_model_ref.path_of("model"))

    @bentoml.api
    async def classify(self, image: Image) -> np.ndarray:
        '''
        Classify input image to label
        '''

        import tensorflow as tf

        image = image.resize((640, 640))

        image_tensor = tf.keras.utils.img_to_array(image, dtype="uint8")
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        output = self.model(image_tensor)
        cls_tensor = output["detection_classes"]
        return cls_tensor.numpy().astype(np.int64)
