import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
import torch

from sklearn.pipeline import Pipeline

def hydra_helpers(func):
    """
    Hydra 설정을 사용할 때 필요한 추가 Resolver(함수)들을
    사전에 등록해주는 데코레이터 함수.
    """

    def inner(*args, **kwargs):
        # 여기서 실제로 resolver를 등록한다.

        # OmegaConf에 커스텀 함수(Resolver)를 등록
        # - 문자열을 파이썬의 실제 함수로 매핑시켜주는 기능
        # - 추가 계산(길이, 사칙연산 등)을 설정 파일 내에서 동적으로 가능하게 함
        OmegaConf.register_new_resolver("len", lambda x: len(x), replace=True)
        OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
        OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace=True)
        OmegaConf.register_new_resolver("mul", lambda x, y: x * y, replace=True)
        OmegaConf.register_new_resolver("rdiv", lambda x, y: x / y, replace=True)

        # 문자열로부터 torch dtype을 얻기 위한 매핑 딕셔너리
        STR2TORCHDTYPE = {
            'float32': torch.float32,
            'float64': torch.float64,
            'double': torch.double,
        }
        # "torchdtype"라는 키워드를 hydra config에서 사용하면, 문자열("float32" 등)을 실제 PyTorch dtype으로 변환해줌
        OmegaConf.register_new_resolver(
            "torchdtype",
            lambda x: STR2TORCHDTYPE[x],
            replace=True
        )

        # 데코레이터에 인자로 전달된 함수가 있으면, 해당 함수 실행
        if func is not None:
            func(*args, **kwargs)

    return inner


def make_sklearn_pipeline(steps_config) -> Pipeline:
    """
    Hydra 설정 안에 정의된 여러 단계(step)들을 순서대로
    'scikit-learn Pipeline' 형태로 만들어주는 함수.
    예를 들어, 전처리 단계 + 분류기를 연결한 Pipeline을
    손쉽게 빌드하도록 도와준다.
    """

    steps = []
    # steps_config: 예) [{"scaler": {알고리즘}}, {"clf": {알고리즘}}] 형태를 가정
    for step_config in steps_config:
        # step_config는 DictConfig 형태이거나, 일반 dict일 수도 있음

        # step_config에서 key(name), value(transform) 한 쌍 추출
        step_name, step_transform = next(iter(step_config.items()))

        # step_transform이 DictConfig면 hydra.utils.instantiate로 인스턴스화
        if isinstance(step_transform, DictConfig):
            pipeline_step = (
                step_name,
                hydra.utils.instantiate(step_transform, _convert_='partial')
            )
        else:
            # DictConfig가 아니면, 이미 인스턴스화 되어 있다고 보고 그대로 사용
            pipeline_step = (step_name, step_transform)

        # 각 단계(이름, 변환기/모델)를 리스트에 쌓는다.
        steps.append(pipeline_step)

    # scikit-learn의 Pipeline 객체를 생성해 반환
    return Pipeline(steps)
