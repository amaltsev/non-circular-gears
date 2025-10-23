from typing import Tuple, List, Union
import yaml
import os


class Model:
    def __init__(self, name, sample_num, center_point, tooth_height, tooth_num, k=1, smooth=0, oversampling = 32):
        self.name = name
        self.oversampling = oversampling
        self.sample_num = sample_num or (tooth_num * oversampling)
        self.center_point = center_point
        self.tooth_height = tooth_height
        self.tooth_num = tooth_num
        self.k = k
        self.smooth = smooth


def load_model_from_file(data: dict) -> Model:
    data['center_point'] = tuple(data['center_point'])
    return Model(**data)


def load_models(filename: str) -> List[Model]:
    with open(filename) as file:
        return [load_model_from_file(data) for data in yaml.safe_load(file)]


our_models = load_models(os.path.join(os.path.dirname(__file__), 'models.yaml'))


def find_model_by_name(model_name: str) -> Union[Model, None]:
    if '/' in model_name:
        folder, name = model_name.split('/')
        return retrieve_model_from_folder(folder, name)
    else:
        for model in our_models:
            if model.name == model_name:
                return model
    return None


def retrieve_model_from_folder(folder_name, model_name):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../silhouette/'))
    models = retrieve_models_from_folder(os.path.join(base_dir, folder_name))
    for model in models:
        if model.name.endswith(model_name):
            return model
    return None


def generate_model_pool(model_names: Tuple[str]):
    model_pool = []
    for available_model in our_models:
        if available_model.name in model_names:
            model_pool.append(available_model)
    assert len(model_pool) == len(model_names)
    return model_pool


def retrieve_models_from_folder(folder_name):
    ### print(f'=== retrieve_models_from_folder({folder_name})')

    assert os.path.isdir(folder_name)

    return [Model(
        name=f'({os.path.basename(folder_name)}){filename[:-4]}',
        sample_num=0,
        center_point=(0, 0),
        tooth_num=32,
        tooth_height=0.04,
        k=1,
        smooth=0
    ) for filename in os.listdir(folder_name) if '.txt' in filename]


if __name__ == '__main__':
    print(our_models)