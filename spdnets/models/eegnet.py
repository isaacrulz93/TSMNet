import torch
from .base import BaseModel
from .dann import DANNBase
import spdnets.modules as modules


class EEGNetv4(BaseModel):
    """
    'EEGNet v4' 모델 구현 (Lawhern et al. 2018 등을 참조).
    - 가상의 시간 축, 채널 축에 대해 소규모 CNN+DepthwiseConv를 사용하여
      뇌파(EEG) 신호 특성을 추출.
    - BaseModel을 상속하며, CrossEntropyLoss 기반의 분류가 기본.
    """

    def __init__(self, is_within=False, srate=128, f1=8, d=2, **kwargs):
        """
        EEGNetv4 초기화.

        매개변수:
        - is_within: within-subject 실험인지(드롭아웃 등 하이퍼파라미터 변경용).
        - srate: 샘플링 레이트(초당 샘플 수).
        - f1: 첫 번째 Conv 필터 개수.
        - d: depth multiplier (DepthwiseConv 후 확장 배수).
        - kwargs: BaseModel에 필요한 기타 인자들 (nclasses, nchannels, nsamples 등).
        """
        super().__init__(**kwargs)
        self.is_within_ = is_within
        self.srate_ = srate    # EEG 샘플링 레이트
        self.f1_ = f1          # 첫 번째 필터 개수
        self.d_ = d            # depth multiplier
        self.f2_ = self.f1_ * self.d_  # depthwise conv 뒤 확장된 필터 개수

        momentum = 0.01

        # kernel_length: 시간축 Conv2D 필터 길이, 대략 srate/2
        kernel_length = int(self.srate_ // 2)

        # Convolution + pooling을 거친 후, 시간 축 샘플 수가 얼마나 줄어드는지 계산
        nlatsamples_time = self.nsamples_ // 32
        # (예: pooling 4, pooling 8 적용 시 총 32배 축소된다고 가정)

        temp2_kernel_length = int(self.srate_ // 2 // 4)
        # 두 번째 Temporal Convolution의 커널 길이 (추가로 나눈 값)

        # within-subject이면 드롭아웃율 0.5, 아니면 0.25
        if self.is_within_:
            drop_prob = 0.5
        else:
            drop_prob = 0.25

        # BatchNorm 객체를 별도로 초기화 (momentum=0.01)
        bntemp = torch.nn.BatchNorm2d(self.f1_, momentum=momentum, affine=True, eps=1e-3)
        bnspat = torch.nn.BatchNorm2d(self.f1_ * self.d_, momentum=momentum, affine=True, eps=1e-3)

        # CNN(Feature Extractor) 구성
        self.cnn = torch.nn.Sequential(
            # (batch, 1, nchs, nsamples) 형태로 입력 들어온다고 가정
            torch.nn.Conv2d(1, self.f1_, (1, kernel_length), bias=False, padding='same'),
            bntemp,
            # DepthwiseConv
            modules.Conv2dWithNormConstraint(
                self.f1_, self.f1_ * self.d_, (self.nchannels_, 1),
                max_norm=1, stride=1, bias=False, groups=self.f1_, padding=(0, 0)
            ),
            bnspat,
            torch.nn.ELU(),
            # 첫 번째 AvgPool -> 시간축 4배 축소
            torch.nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            torch.nn.Dropout(p=drop_prob),

            # 두 번째 Temporal Convolution (depthwise)
            torch.nn.Conv2d(
                self.f1_ * self.d_, self.f1_ * self.d_,
                (1, temp2_kernel_length),
                stride=1, bias=False, groups=self.f1_ * self.d_, padding='same'
            ),
            # PointwiseConv (f1_*d_ -> f2_)
            torch.nn.Conv2d(
                self.f1_ * self.d_, self.f2_, (1, 1),
                stride=1, bias=False, padding=(0, 0)
            ),
            torch.nn.BatchNorm2d(self.f2_, momentum=momentum, affine=True, eps=1e-3),
            torch.nn.ELU(),
            # 두 번째 AvgPool -> 시간축 추가 8배 축소 => 총 32배
            torch.nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            torch.nn.Dropout(p=drop_prob),
        ).to(self.device_)

        # 분류기(classifier) 계층
        self.classifier = torch.nn.Sequential(
            # (batch, f2_, nchs', time')를 Flatten
            torch.nn.Flatten(start_dim=1),
            # LinearWithNormConstraint: 가중치 정규화 제한 (max_norm=0.25)
            modules.LinearWithNormConstraint(
                self.f2_ * nlatsamples_time, self.nclasses_, max_norm=0.25
            )
        ).to(self.device_)

    def get_hyperparameters(self):
        """
        상위(BaseModel)의 하이퍼파라미터 외에,
        EEGNetv4 고유의 하이퍼파라미터도 딕셔너리에 추가.
        """
        kwargs = super().get_hyperparameters()
        kwargs['nsamples'] = self.nsamples_
        kwargs['is_within_subject'] = self.is_within_
        kwargs['srate'] = self.srate_
        kwargs['f1'] = self.f1_
        kwargs['d'] = self.d_
        return kwargs

    def forward(self, x, d):
        """
        순전파:
        - x: (batch, nchannels, nsamples) 형태의 입력 EEG (이미 배치 차원 존재)
        - d: 도메인 정보(지금은 사용 안 함, 인터페이스 일치 목적)

        반환:
        - y: (batch, nclasses) 로짓
        - l: (batch, f2_, 1, time') 2차원 CNN feature (마지막 pooling 출력)
        """
        # (batch, 1, nchannels, nsamples) 형태로 차원 추가
        l = self.cnn(x[:, None, ...])
        y = self.classifier(l)
        return y, l


class DANNEEGNet(DANNBase, EEGNetv4):
    """
    Domain Adversarial Neural Network(DANN) + EEGNet 조합:
    - Ozdenizci et al. 2020 (IEEE Access) 등에서 사용된 방식.
    - EEGNetv4로 feature를 추출하고, DANNBase(Adversary)를 통해
      도메인 분류 손실(gradient reversal)을 추가하는 구조.
    """

    def __init__(self, daloss_scaling=0.03, dann_mode='ganin2016', **kwargs):
        """
        DANNEEGNet 초기화:
        - daloss_scaling: 도메인 적대적 손실 스케일링 (기본 0.03)
        - dann_mode: 'ganin2016' (기본값) or 'ganin2015'

        **kwargs 안에 nclasses, nchannels, nsamples, domains 등
        BaseModel, DANNBase, EEGNetv4가 필요로 하는 인자들 포함.
        """
        kwargs['daloss_scaling'] = daloss_scaling
        kwargs['dann_mode'] = dann_mode
        # DANNBase.__init__ -> (domains 정렬 등) + EEGNetv4.__init__ -> (CNN, classifier 등)
        super().__init__(**kwargs)

    def _ndim_latent(self):
        """
        DANNBase에서 '_ndim_latent()'가 필요했음.
        - 여기서는 classifier의 마지막 Linear 차원을 그대로 사용.
        - self.classifier[-1]은 LinearWithNormConstraint.
        - weight.shape[-1] -> in_features 차원.
        """
        return self.classifier[-1].weight.shape[-1]

    def forward(self, x, d):
        """
        순전파:
        - EEGNetv4.forward로 분류(y, l) 얻고,
        - DANNBase.forward(l, d)로 도메인 분류 로짓(y_domain)을 얻어
        - (y, y_domain) 튜플 반환.

        이때 y: (batch, nclasses),
            y_domain: (batch, n_domain)
        """
        # EEGNetv4의 forward
        y, l = EEGNetv4.forward(self, x, d)
        # DANNBase의 forward(=도메인 분류기)
        y_domain = DANNBase.forward(self, l, d)
        return y, y_domain
