from typing import Union
import torch
from .base import BaseModel, DomainAdaptFineTuneableModel
from .dann import DANNBase
import spdnets.batchnorm as bn
import spdnets.modules as modules


class ShallowConvNet(BaseModel):
    def __init__(self, spatial_filters=40, temporal_filters=40, pdrop=0.5, **kwargs):
        super().__init__(**kwargs)
        self.spatial_filters_ = spatial_filters
        self.temporal_filters_ = temporal_filters

        temp_cnn_kernel = 25
        temp_pool_kernel = 75
        temp_pool_stride = 15

        # 1) temp_cnn_kernel=25 => conv 커널 길이(시간축)
        #    temporal_filters=40 => 시간 방향으로 필터(=채널 수)
        # 2) spatial_filters=40 => 공간 방향(conv over channels)
        # -> EEG에서 통상 "시공간분리(temporal conv 먼저, spatial conv 그 뒤)" 구조를 가정

        # (예) 입력이 (batch, nchannels, nsamples)
        #  -> 첫 conv2d에서 (batch, temporal_filters, nchannels, (nsamples - kernel+1))로 바뀔 것

        # temp_cnn_kernel=25를 사용하면,
        # (nsamples - 1*(25-1) - 1)/1 + 1 = nsamples-24
        ntempconvout = int((self.nsamples_ - 1 * (temp_cnn_kernel - 1) - 1) / 1 + 1)
        # 아래 AvgPooling (커널=75, stride=15) 통과 후 출력 크기
        navgpoolout = int((ntempconvout - temp_pool_kernel) / temp_pool_stride + 1)

        self.bn = torch.nn.BatchNorm2d(self.spatial_filters_)
        drop = torch.nn.Dropout(p=pdrop)

        # CNN 부분(Temporal Conv + Spatial Conv)
        self.cnn = torch.nn.Sequential(
            # 첫 번째 Conv2d (batch, 1, nchannels, nsamples) -> (batch, temporal_filters, nchannels, ntempconvout)
            torch.nn.Conv2d(1, self.temporal_filters_, (1, temp_cnn_kernel)),

            # 두 번째 Conv2d (spatial filter):
            # (batch, temporal_filters, nchannels, ~) -> (batch, spatial_filters, 1, ~)
            # 커널이 (nchannels_,1)이므로 채널 축에 대해 완전히 펼쳐서 학습
            torch.nn.Conv2d(self.temporal_filters_, self.spatial_filters_, (self.nchannels_, 1)),
        ).to(self.device_)

        # Pool + Activation + Dropout
        self.pool = torch.nn.Sequential(
            modules.MySquare(),
            # MySquare: 보통 ShallowConvNet에서 ReLU 대신 Squared nonlinearity를 사용 가능
            torch.nn.AvgPool2d(kernel_size=(1, temp_pool_kernel), stride=(1, temp_pool_stride)),
            modules.MyLog(),
            drop,
            torch.nn.Flatten(start_dim=1),
        ).to(self.device_)

        # 최종 분류기
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.spatial_filters_ * navgpoolout, self.nclasses_),
        ).to(self.device_)

    def forward(self, x, d):
        # x shape: (batch, nchannels, nsamples)
        # d: 도메인 인덱스 (여기서는 사용 안 함, 인터페이스 일치 목적)
        #  => (batch,1,nchannels,nsamples)로 reshape
        l = self.cnn(x.to(self.device_)[:, None, ...])
        # 공간 필터 출력에 배치 정규화
        l = self.bn(l)
        l = self.pool(l)
        # (batch, spatial_filters * navgpoolout)
        y = self.classifier(l)
        return y, l


class DANNShallowConvNet(DANNBase, ShallowConvNet):
    """
    DANN(도메인 적대적) + ShallowConvNet
    """

    def __init__(self, daloss_scaling=0.05, dann_mode='ganin2016', **kwargs):
        # DANNBase에서 요구하는 인자 (daloss_scaling, dann_mode) 추가
        kwargs['daloss_scaling'] = daloss_scaling
        kwargs['dann_mode'] = dann_mode
        super().__init__(**kwargs)

    def _ndim_latent(self):
        # 도메인 분류기(adversary)에 필요한 latent 차원
        # classifier[-1] => Linear, weight.shape[-1] = in_features
        return self.classifier[-1].weight.shape[-1]

    def forward(self, x, d):
        # ShallowConvNet.forward -> (y, l)
        y, l = ShallowConvNet.forward(self, x, d)
        # DANNBase.forward -> (배치, #domains) 도메인 분류 로짓
        y_domain = DANNBase.forward(self, l, d)
        return y, y_domain


class ShConvNetDSBN(ShallowConvNet, DomainAdaptFineTuneableModel):
    def __init__(self,
                 bnorm_dispersion: Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.VECTOR,
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(bnorm_dispersion, str):
            bnorm_dispersion = bn.BatchNormDispersion[bnorm_dispersion]

        self.bn = bn.AdaMomDomainBatchNorm(
            (1, self.spatial_filters_, 1, 1),
            batchdim=[0,2,3], # 배치 차원을 (0,2,3)으로 보고, channel dim=1
            domains=self.domains_,
            dispersion=bnorm_dispersion,
            eta=1., eta_test=.1
        ).to(self.device_)

    def forward(self, x, d):
        l = self.cnn(x.to(self.device_)[:,None,...])
        # 여기서 bn에 (특징, 도메인)d 를 함께 넘긴다 => 도메인별 BN 파라미터 사용
        l = self.bn(l, d.to(device=self.device_))
        l = self.pool(l)
        y = self.classifier(l)
        return y, l

    def domainadapt_finetune(self, x, y, d, target_domains):
        """
        DomainAdaptFineTuneableModel에 정의된
        'domainadapt_finetune(x, y, d, target_domains)'를 구현.
        여기서는 domain별 BatchNorm 통계(평균/분산)만 갱신(refit)한다.
        """
        # set_test_stats_mode(REFIT) -> 도메인별로 moving avg/var를 다시 추정
        self.bn.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        for du in d.unique():
            # 해당 도메인 샘플만 forward 하여 통계 갱신
            self.forward(x[d == du], d[d == du])

        # 다시 BUFFER 모드로 변경(즉, 통계 버퍼 사용)
        self.bn.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
