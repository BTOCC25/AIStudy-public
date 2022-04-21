import bentoml
from bentoml.io import NumpyNdarray, JSON

iris_cls_runner = bentoml.sklearn.load_runner("iris_cls:latest")

svc = bentoml.Service("iris_cls_service", runners="iris_cls_runner")

@svc.api(input=NumpyNdarray(), output=JSON())
def classifier(input_data):
    result = iris_cls_runner.run(input_data)
    return result