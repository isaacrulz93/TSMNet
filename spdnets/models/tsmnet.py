import math
from typing import Optional, Union
import torch

import spdnets.modules as modules
import spdnets.batchnorm as bn
from .base import DomainAdaptFineTuneableModel, FineTuneableModel, PatternInterpretableModel


class TSMNet(DomainAdaptFineTuneableModel, FineTuneableModel, PatternInterpretableModel):
    def __init__(self, temporal_filters, spatial_filters=40,
                 subspacedims=20,
                 temp_cnn_kernel=25,
                 bnorm: Optional[str] = 'spdbn',
                 bnorm_dispersion: Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 **kwargs):
        """
        TSMNet: 공분산 풀링(CovPooling)을 사용하고,
        SPD(대칭 양의정부 행렬) 기반 배치 정규화 등을 지원하는 네트워크.

        주요 단계:
        1. CNN을 통해 시간/채널 축 컨볼루션
        2. CovariancePool()로 특성을 공분산 형태로 요약
        3. SPD-BN(또는 Domain-Specific SPD-BN 등) 적용
        4. LogEig 등을 통해 SPD -> 벡터 특성
        5. 최종 Linear 분류

        믹스인(Mixin):
        - DomainAdaptFineTuneableModel: domainadapt_finetune() 구현
        - FineTuneableModel: finetune() 구현
        - PatternInterpretableModel: compute_patterns() (아직 pass)
        """
        super().__init__(**kwargs)

        # 하이퍼파라미터 설정
        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.subspacedimes = subspacedims
        self.bnorm_ = bnorm
        self.spd_device_ = torch.device('cpu')  # SPD 연산을 CPU에서 한다고 가정

        # 배치 정규화 분산 파라미터(bnorm_dispersion)를 enum 형태로 변환
        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion

        # SPD 행렬 크기: subspacedims x subspacedims
        # 로그공간에서 벡터 크기: tsdim = subspacedims*(subspacedims+1)/2
        tsdim = int(subspacedims * (subspacedims + 1) / 2)

        # CNN: (batch, 1, nchannels, nsamples) -> (batch, spatial_filters_, ?) -> Flatten(시간축)
        # temp_cnn_kernel=25, padding='same' => 시간 축 길이 동일 유지
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(
                1, self.temporal_filters_,
                kernel_size=(1, temp_cnn_kernel),
                padding='same', padding_mode='reflect'
            ),
            torch.nn.Conv2d(
                self.temporal_filters_, self.spatial_filters_,
                (self.nchannels_, 1)
            ),
            torch.nn.Flatten(start_dim=2),
        ).to(self.device_)

        # 공분산 풀링(채널 간 공분산 구하기)
        self.cov_pooling = torch.nn.Sequential(
            modules.CovariancePool(),
        )

        # SPD BN 설정 (spdbn, brooks, tsbn, spddsbn, tsdsbn 등 옵션)
        if self.bnorm_ == 'spdbn':
            self.spdbnorm = bn.AdaMomSPDBatchNorm(
                (1, subspacedims, subspacedims),
                batchdim=0,
                dispersion=self.bnorm_dispersion_,
                learn_mean=False, learn_std=True,
                eta=1., eta_test=.1,
                dtype=torch.double,
                device=self.spd_device_
            )
        elif self.bnorm_ == 'brooks':
            self.spdbnorm = modules.BatchNormSPDBrooks(
                (1, subspacedims, subspacedims),
                batchdim=0,
                dtype=torch.double,
                device=self.spd_device_
            )
        elif self.bnorm_ == 'tsbn':
            # SPD를 로그벡터로 바꾼 뒤 batchnorm
            self.tsbnorm = bn.AdaMomBatchNorm(
                (1, tsdim),
                batchdim=0,
                dispersion=self.bnorm_dispersion_,
                eta=1., eta_test=.1,
                dtype=torch.double,
                device=self.spd_device_
            ).to(self.device_)
        elif self.bnorm_ == 'spddsbn':
            self.spddsbnorm = bn.AdaMomDomainSPDBatchNorm(
                (1, subspacedims, subspacedims),
                batchdim=0,
                domains=self.domains_,
                learn_mean=False, learn_std=True,
                dispersion=self.bnorm_dispersion_,
                eta=1., eta_test=.1,
                dtype=torch.double,
                device=self.spd_device_
            )
        elif self.bnorm_ == 'tsdsbn':
            self.tsdsbnorm = bn.AdaMomDomainBatchNorm(
                (1, tsdim),
                batchdim=0,
                domains=self.domains_,
                dispersion=self.bnorm_dispersion_,
                eta=1., eta_test=.1,
                dtype=torch.double
            ).to(self.device_)
        elif self.bnorm_ is not None:
            raise NotImplementedError('requested undefined batch normalization method.')

        # spdnet: BiMap -> ReEig (subspace projection & regularization)
        self.spdnet = torch.nn.Sequential(
            modules.BiMap(
                (1, self.spatial_filters_, subspacedims),
                dtype=torch.double, device=self.spd_device_
            ),
            modules.ReEig(threshold=1e-4),
        )

        # 로그-고윳값(LogEig) + Flatten
        self.logeig = torch.nn.Sequential(
            modules.LogEig(subspacedims),
            torch.nn.Flatten(start_dim=1),
        )

        # 최종 분류기 (tsdim => nclasses)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(tsdim, self.nclasses_).double(),
        ).to(self.spd_device_)

    def to(self, device: Optional[Union[int, torch.device]] = None, dtype: Optional[Union[int, torch.dtype]] = None,
           non_blocking: bool = False):
        """
        SPD 부분은 CPU에서 돌리되, CNN 부분은 GPU에 올릴 수도 있으므로
        device_와 spd_device_를 분리해둔 것 같음.
        """
        if device is not None:
            self.device_ = device
            self.cnn.to(self.device_)
        return super().to(device=None, dtype=dtype, non_blocking=non_blocking)

    def forward(self, x, d, return_latent=True, return_prebn=False, return_postbn=False):
        """
        순전파 과정:
        1. CNN -> (batch, spatial_filters_, time) 형태
        2. CovariancePool() -> (batch, spatial_filters_, spatial_filters_) SPD 행렬
        3. SPDNet(BiMap, ReEig) -> subspacedims x subspacedims
        4. (옵션) SPD-BN or Domain SPD-BN
        5. LogEig -> 벡터화
        6. (옵션) TS-BN or Domain TS-BN
        7. Linear classifier
        return_latent, return_prebn, return_postbn 플래그로 중간 값을 튜플로 반환.
        """
        out = ()
        # 1. CNN (GPU)
        h = self.cnn(x.to(device=self.device_)[:, None, ...])
        # 2. Cov pooling -> SPD (CPU, double)
        C = self.cov_pooling(h).to(device=self.spd_device_, dtype=torch.double)
        # 3. SPDNet
        l = self.spdnet(C)
        out += (l,) if return_prebn else ()
        # 4. (도메인)SPD-BN
        l = self.spdbnorm(l) if hasattr(self, 'spdbnorm') else l
        l = self.spddsbnorm(l, d.to(device=self.spd_device_)) if hasattr(self, 'spddsbnorm') else l
        out += (l,) if return_postbn else ()
        # 5. LogEig -> 벡터
        l = self.logeig(l)
        # 6. (도메인)TS-BN
        l = self.tsbnorm(l) if hasattr(self, 'tsbnorm') else l
        l = self.tsdsbnorm(l, d) if hasattr(self, 'tsdsbnorm') else l
        out += (l,) if return_latent else ()
        # 7. 분류기
        y = self.classifier(l)
        # 여러 플래그에 따라 반환 튜플 구성
        out = y if len(out) == 0 else (y, *out[::-1])
        return out

    def domainadapt_finetune(self, x, y, d, target_domains):
        """
        target domain 데이터로 BN 통계만 재추정하는 로직.
        """
        if hasattr(self, 'spddsbnorm'):
            self.spddsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
        if hasattr(self, 'tsdsbnorm'):
            self.tsdsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            for du in d.unique():
                self.forward(x[d == du], d[d == du], return_latent=False)

        if hasattr(self, 'spddsbnorm'):
            self.spddsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
        if hasattr(self, 'tsdsbnorm'):
            self.tsdsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)

    def finetune(self, x, y, d):
        """
        소스 데이터 등으로 BN 통계를 다시 잡고자 할 때.
        """
        if hasattr(self, 'spdbnorm'):
            self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
        if hasattr(self, 'tsbnorm'):
            self.tsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            self.forward(x, d, return_latent=False)

        if hasattr(self, 'spdbnorm'):
            self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
        if hasattr(self, 'tsbnorm'):
            self.tsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)

    def compute_patterns(self, x, y, d):
        """
        PatternInterpretableModel 인터페이스
        - 아직 구현되지 않음(pass)
        """
        pass


class CNNNet(DomainAdaptFineTuneableModel, FineTuneableModel):
    def __init__(self, temporal_filters, spatial_filters=40,
                 temp_cnn_kernel=25,
                 bnorm: Optional[str] = 'bn',
                 bnorm_dispersion: Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 **kwargs):
        """
        CNNNet: Covariance pooling + diagonal만 사용 (diagonal = var?)
        - DomainAdaptFineTuneableModel, FineTuneableModel 믹스인
        - (도메인)BatchNorm를 사용할 수도 있음
        """
        super().__init__(**kwargs)

        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.bnorm_ = bnorm

        # BN 분산 파라미터 설정
        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion

        # (batch,1,nchannels,nsamples) -> temporal_filters, spatial_filters
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(
                1, self.temporal_filters_,
                kernel_size=(1, temp_cnn_kernel),
                padding='same', padding_mode='reflect'
            ),
            torch.nn.Conv2d(
                self.temporal_filters_, self.spatial_filters_,
                (self.nchannels_, 1)
            ),
            torch.nn.Flatten(start_dim=2),
        ).to(self.device_)

        # Covariance pooling -> SPD
        self.cov_pooling = torch.nn.Sequential(
            modules.CovariancePool(),
        )

        # bnorm='bn' => 일반 AdaMomBatchNorm(특징 크기: spatial_filters)
        # bnorm='dsbn' => 도메인별 BN
        if self.bnorm_ == 'bn':
            self.bnorm = bn.AdaMomBatchNorm(
                (1, self.spatial_filters_), batchdim=0,
                dispersion=self.bnorm_dispersion_,
                eta=1., eta_test=.1
            ).to(self.device_)
        elif self.bnorm_ == 'dsbn':
            self.dsbnorm = bn.AdaMomDomainBatchNorm(
                (1, self.spatial_filters_), batchdim=0,
                domains=self.domains_,
                dispersion=self.bnorm_dispersion_,
                eta=1., eta_test=.1
            ).to(self.device_)
        elif self.bnorm_ is not None:
            raise NotImplementedError('requested undefined batch normalization method.')

        # MyLog + Flatten
        self.logarithm = torch.nn.Sequential(
            modules.MyLog(),
            torch.nn.Flatten(start_dim=1),
        )

        # 최종 분류기
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.spatial_filters_, self.nclasses_),
        ).to(self.device_)

    def forward(self, x, d, return_latent=True):
        """
        순전파:
        1) cnn -> (batch, spatial_filters, time)
        2) cov_pooling -> (batch, spatial_filters, spatial_filters)
        3) l = torch.diagonal(C) -> (batch, spatial_filters)
        4) MyLog -> log( . ), Flatten
        5) (도메인)BN
        6) Linear
        반환: (y, [l]) or y 단독
        """
        out = ()
        # (batch,1,nchannels,nsamples)
        h = self.cnn(x.to(device=self.device_)[:, None, ...])
        # Covariance pool -> SPD
        C = self.cov_pooling(h)
        # 대각원소만 뽑음 => (batch, spatial_filters_) (채널별 분산 정도를 보는 듯)
        l = torch.diagonal(C, dim1=-2, dim2=-1)
        # log 변환
        l = self.logarithm(l)
        # (도메인)BN
        l = self.bnorm(l) if hasattr(self, 'bnorm') else l
        l = self.dsbnorm(l, d) if hasattr(self, 'dsbnorm') else l
        out += (l,) if return_latent else ()
        y = self.classifier(l)
        out = y if len(out) == 0 else (y, *out[::-1])
        return out

    def domainadapt_finetune(self, x, y, d, target_domains):
        """
        (도메인)DSBN을 사용하는 경우, target domain 데이터로 BN 통계 refit
        """
        if hasattr(self, 'dsbnorm'):
            self.dsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            for du in d.unique():
                self.forward(x[d == du], d[d == du], return_latent=False)

        if hasattr(self, 'dsbnorm'):
            self.dsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)

    def finetune(self, x, y, d):
        """
        일반 BN을 사용하는 경우 refit
        """
        if hasattr(self, 'bnorm'):
            self.bnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            self.forward(x, d, return_latent=False)

        if hasattr(self, 'bnorm'):
            self.bnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
