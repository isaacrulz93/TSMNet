import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    기본 신경망 모델(BaseModel):
    - PyTorch nn.Module을 상속받는다.
    - 공통적으로 필요한 속성(nclasses, nchannels, nsamples 등)과,
      학습에 사용될 기본 loss function(CrossEntropyLoss)을 정의한다.
    """

    def __init__(self, nclasses=None, nchannels=None, nsamples=None, nbands=None, device=None, input_shape=None):
        super().__init__()
        self.device_ = device  # 사용할 디바이스
        self.lossfn = torch.nn.CrossEntropyLoss()  # 기본 손실 함수
        self.nclasses_ = nclasses  # 클래스 개수
        self.nchannels_ = nchannels  # 채널 개수
        self.nsamples_ = nsamples  # 샘플(시간축) 개수
        self.nbands_ = nbands  # (필터뱅크) 밴드 개수
        self.input_shape_ = input_shape  # (batch, *shape) 형태 중 shape 부분

    # AUXILIARY METHODS
    def calculate_classification_accuracy(self, Y, Y_lat):
        """
        (Y_lat -> 예측 로짓 or 확률)와 실제 라벨(Y)을 비교해,
        정확도(accuracy) 계산.

        반환:
         - acc: 정확도 (float)
         - P_hat: softmax 확률 분포
        """
        Y_hat = Y_lat.argmax(1)  # 가장 높은 점수를 가진 클래스로 예측
        acc = Y_hat.eq(Y).float().mean().item()  # 예측 vs 실제 라벨 비교 -> 평균
        P_hat = torch.softmax(Y_lat, dim=1)  # 확률 분포
        return acc, P_hat

    def calculate_objective(self, model_pred, y_true, model_inp=None):
        """
        모델의 최적화 목적함수를 계산.
        일반적으로 cross-entropy loss를 사용.

        - model_pred: 모델 출력(로짓)
        - y_true: 실제 라벨
        """
        if isinstance(model_pred, (list, tuple)):
            # 모델이 여러 출력을 내는 경우(예: (logits, latent_space, ...)) 중 첫 번째를 사용
            y_class_hat = model_pred[0]
        else:
            y_class_hat = model_pred

        loss = self.lossfn(y_class_hat, y_true.to(y_class_hat.device))
        return loss

    def get_hyperparameters(self):
        """
        하이퍼파라미터(채널 수, 클래스 수, 샘플 수, 밴드 수 등)를 딕셔너리로 반환.
        """
        return dict(
            nchannels=self.nchannels_,
            nclasses=self.nclasses_,
            nsamples=self.nsamples_,
            nbands=self.nbands_
        )


class CPUModel:
    """
    CPU 전용 모델임을 나타내는 '빈' 클래스.
    - 특별한 메서드는 없고, 'CPU 전용' 이라는 태그/인터페이스 역할을 한다.
    - 이를 상속하면 'GPU 연산'이 아닌 'CPU 연산'만 처리하도록
      별도의 로직을 넣을 수도 있다.
    """
    pass


class FineTuneableModel:
    """
    '파인튜닝 기능'을 제공하는 모델임을 나타내는 믹스인 클래스.
    - finetune 메서드를 추상적으로 정의만 해두고,
      실제 구현은 이를 상속하는 모델에서 구체화한다.
    """

    def finetune(self, x, y, d):
        raise NotImplementedError()


class DomainAdaptBaseModel(BaseModel):
    """
    '도메인 적응' 기능을 가진 모델의 베이스 클래스.
    - BaseModel을 상속받아, 도메인 목록(domains_)을 추가로 관리한다.
    """

    def __init__(self, domains=[], **kwargs):
        super().__init__(**kwargs)  # BaseModel의 __init__ 호출
        self.domains_ = domains  # 추가적으로, '어떤 도메인들이 있는지' 저장


class DomainAdaptFineTuneableModel(DomainAdaptBaseModel):
    """
    도메인 적응 + 파인튜닝 기능을 동시에 지원하는 모델(추상 클래스).
    - DomainAdaptBaseModel과 FineTuneableModel(믹스인)을 모두 상속받을 수도 있음.
    - domainadapt_finetune 메서드는 아직 구현이 없고 NotImplementedError 발생.
      자식 클래스에서 실제 로직을 오버라이드해야 한다.
    """

    def domainadapt_finetune(self, x, y, d, target_domains):
        """
        도메인 적응을 위한 파인튜닝 로직.
        - 실제 구현은 자식 클래스가 담당.
        - x: 입력 데이터
        - y: 라벨
        - d: 도메인 정보
        - target_domains: 적응을 수행하고자 하는 타겟 도메인 목록
        """
        raise NotImplementedError()


class DomainAdaptJointTrainableModel(DomainAdaptBaseModel):
    """
    도메인 적응(타겟 도메인) 데이터를 소스 데이터와 함께 joint training할 수 있는 모델.
    - calculate_objective를 오버라이드 하여,
      라벨이 -1인(마스킹된) 샘플은 학습(손실 계산)에서 제외.
    """

    def calculate_objective(self, model_pred, y_true, model_inp=None):
        # -1 라벨은 "마스킹"이므로 무시
        keep = y_true != -1  # True인 인덱스만 손실 계산

        if isinstance(model_pred, (list, tuple)):
            y_class_hat = model_pred[0]
        else:
            y_class_hat = model_pred

        # 마스킹되지 않은 부분만 부모 클래스(BaseModel)의 calculate_objective 호출
        return super().calculate_objective(y_class_hat[keep], y_true[keep], None)


class PatternInterpretableModel:
    """
    '패턴 가중치'나 '공간 패턴'을 해석할 수 있는 모델임을 나타내는 믹스인 클래스.
    - compute_patterns 메서드를 구체 구현 없이 선언만 해둔다.
    - 실제로 EEGNet 등에서 '컴포넌트(Spatial Filter)'에 대응하는 '패턴'을
      어떻게 계산할지 오버라이드하여 구현할 수 있다.
    """

    def compute_patterns(self, x, y, d):
        raise NotImplementedError()