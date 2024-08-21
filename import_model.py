import bentoml
import shutil

MODEL_ID = "tensorflow/faster-rcnn-inception-resnet-v2/tensorFlow2/640x640"
BENTO_MODEL_TAG = "resnet-v2-tensorflow"

def import_model(model_id, bento_model_tag):
    import kagglehub
    src_path = kagglehub.model_download(MODEL_ID)
    with bentoml.models.create(bento_model_tag) as bento_model_ref:
        shutil.copytree(src_path, bento_model_ref.path_of("model"))


if __name__ == "__main__":
    import_model(MODEL_ID, BENTO_MODEL_TAG)
