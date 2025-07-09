############################
# 배치 정규화(BN) 확장 파트 #
############################

from builtins import NotImplementedError
from enum import Enum
from typing import Tuple
import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.types import Number

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from .manifolds import SymmetricPositiveDefinite
from . import functionals
from skorch.callbacks import Callback
from skorch import NeuralNet


############################
# 섹션 1) Momentum 스케줄러 #
############################

class DummyScheduler(Callback):
    """
    아무것도 하지 않는 스케줄러(콜백).
    skorch Callback과 호환성 유지 목적으로 존재.
    """
    pass


class ConstantMomentumBatchNormScheduler(Callback):
    """
    상수 모멘텀(eta, eta_test)을 BN 모듈들에 적용하는 스케줄러.
    skorch의 콜백을 상속받아,
    - on_train_begin 단계에서 BN 모듈을 찾아 eta/eta_test를 설정.
    """

    def __init__(self, eta, eta_test) -> None:
        self.eta0 = eta  # 학습 시 eta
        self.eta0_test = eta_test  # 추론 시 eta_test
        self.bn_modules_ = []

    def initialize(self):
        # 스케줄러 초기화 시점
        super().initialize()
        self.eta_ = self.eta0  # 실제 적용 중인 학습 모멘텀
        self.eta_test_ = self.eta0_test  # 실제 적용 중인 추론 모멘텀
        self.bn_modules_ = []
        return self

    def on_train_begin(self, net: NeuralNet, **kwargs):
        # 학습 시작 시, 모델 안의 BN 모듈들을 찾아 eta, eta_test를 설정
        model = net.module_
        if model is not None:
            # model.modules()에서 SchedulableBatchNorm 또는 SchedulableDomainBatchNorm인 모듈만 골라 담음
            self.bn_modules_ = [
                m for m in model.modules()
                if isinstance(m, SchedulableBatchNorm) or isinstance(m, SchedulableDomainBatchNorm)
            ]
        else:
            self.bn_modules_ = []

        # 찾아낸 BN 모듈에 eta_, eta_test_를 일괄 설정
        for m in self.bn_modules_:
            m.set_eta(eta=self.eta_, eta_test=self.eta_test_)

    def __repr__(self) -> str:
        return f'ConstantMomentumBatchNormScheduler - eta={self.eta_:.3f}, eta_test={self.eta_test_:.3f}'


class MomentumBatchNormScheduler(ConstantMomentumBatchNormScheduler):
    """
    Epoch가 지날수록 모멘텀을 동적으로 조정하는 스케줄러.
    bs, bs0, tau0 등으로 eta를 점차 바꿈.
    """

    def __init__(self, epochs: Number, bs: Number = 32, bs0: Number = 64, tau0: Number = 0.9) -> None:
        # bs <= bs0 가정
        super().__init__(1. - tau0, 1. - tau0 ** (bs / bs0))
        self.epochs = epochs
        self.rho = (bs / bs0) ** (1 / self.epochs)
        self.tau0 = tau0
        self.bs = bs
        self.bs0 = bs0

    def initialize(self):
        super().initialize()
        self.epoch_ = 1
        return self

    def __repr__(self) -> str:
        return f'MomentumBatchNormScheduler - eta={self.eta_:.3f}, eta_tst={self.eta_test_:.3f}'

    def on_epoch_begin(self, net, **kwargs):
        """
        에폭 시작 시, eta_, eta_test_ 값을 재계산하여
        BN 모듈에 적용.
        """
        # eta_를 rho^(...) 기반으로 업데이트
        self.eta_ = 1. - (self.rho ** (
                    self.epochs * max(self.epochs - self.epoch_, 0) / (self.epochs - 1)) - self.rho ** self.epochs)

        for m in self.bn_modules_:
            m.set_eta(eta=self.eta_)

        w = max(self.epochs - self.epoch_, 0) / (self.epochs - 1)
        tau_test = self.tau0 ** (self.bs / self.bs0 * (1 - w) + w * 1)
        self.eta_test_ = 1 - tau_test

        for m in self.bn_modules_:
            m.set_eta(eta_test=1. - self.eta_test_)

        self.epoch_ += 1


##########################################
# 섹션 2) BatchNormTestStatsMode와 스케줄 #
##########################################

class BatchNormTestStatsMode(Enum):
    """
    추론 시점에 BN 통계를 어떻게 사용할지 지정:
    - BUFFER: (기존 방식) running_mean/var 버퍼 사용
    - REFIT:  실제 입력 X로부터 다시 추정(re-fit)
    - ADAPT:  온라인으로 적응? (여기선 NotImplemented)
    """
    BUFFER = 'buffer'
    REFIT = 'refit'
    ADAPT = 'adapt'


class BatchNormTestStatsModeScheduler(Callback):
    """
    스케줄러: train(학습) 모드와 predict(추론) 모드가 바뀔 때,
    fit_mode/predict_mode 중 어느 것을 쓸지 지정.
    """

    def __init__(self,
                 fit_mode: BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER,
                 predict_mode: BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER) -> None:
        self.fit_mode = fit_mode
        self.predict_mode = predict_mode

    def on_train_begin(self, net: NeuralNet, **kwargs):
        model = net.module_
        for m in model.modules():
            if isinstance(m, BatchNormTestStatsInterface):
                # 학습 중 모드로 설정
                m.set_test_stats_mode(self.fit_mode)

    def on_train_end(self, net: NeuralNet, **kwargs):
        model = net.module_
        for m in model.modules():
            if isinstance(m, BatchNormTestStatsInterface):
                # 학습 끝나면 추론 모드로 전환
                m.set_test_stats_mode(self.predict_mode)


class BatchNormDispersion(Enum):
    """
    BN에서 '분산'을 어떻게 취급할지:
    - NONE: 전혀 안 씀(=평균만 보정?)
    - SCALAR: 하나의 값으로 전체를 스케일링
    - VECTOR: 각 채널별(또는 각 차원별)로 스케일링
    """
    NONE = 'mean'
    SCALAR = 'scalar'
    VECTOR = 'vector'


class BatchNormTestStatsInterface:
    """
    Test Stats 모드(BUFFER, REFIT 등)를 바꾸는 API를 제공.
    """

    def set_test_stats_mode(self, mode: BatchNormTestStatsMode):
        pass


##################################################
# 섹션 3) BaseBatchNorm, DomainBatchNorm의 베이스 #
##################################################

class BaseBatchNorm(nn.Module, BatchNormTestStatsInterface):
    """
    BN의 기본 클래스.
    - eta, eta_test (학습, 추론 모멘텀)
    - test_stats_mode (BUFFER, REFIT 등)
    """

    def __init__(self, eta=1.0, eta_test=0.1, test_stats_mode: BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER):
        super().__init__()
        self.eta = eta
        self.eta_test = eta_test
        self.test_stats_mode = test_stats_mode

    def set_test_stats_mode(self, mode: BatchNormTestStatsMode):
        self.test_stats_mode = mode


class SchedulableBatchNorm(BaseBatchNorm):
    """
    모멘텀(eta, eta_test)을 외부(스케줄러)에서 주기적으로 바꿀 수 있게 하는 믹스인 클래스.
    """

    def set_eta(self, eta=None, eta_test=None):
        if eta is not None:
            self.eta = eta
        if eta_test is not None:
            self.eta_test = eta_test


class BaseDomainBatchNorm(nn.Module, BatchNormTestStatsInterface):
    """
    Domain별로 BN 객체를 따로 관리하기 위한 베이스 클래스.
    - batchnorm: ModuleDict 형태로, key='dom {도메인ID}' => 각각의 BN.
    - forward 시, 도메인별로 X를 나눠서 BN 처리 후 다시 합침.
    """

    def __init__(self):
        super().__init__()
        self.batchnorm = torch.nn.ModuleDict()

    def set_test_stats_mode(self, mode: BatchNormTestStatsMode):
        # 내부에 있는 BN이 BatchNormTestStatsInterface를 상속하면 거기에 모드 설정
        for bn in self.batchnorm.values():
            if isinstance(bn, BatchNormTestStatsInterface):
                bn.set_test_stats_mode(mode)

    def add_domain_(self, layer: BaseBatchNorm, domain: Tensor):
        """
        특정 도메인(domain.item())에 해당하는 BN 레이어를 ModuleDict에 등록.
        """
        self.batchnorm[f'dom {domain.item()}'] = layer

    def get_domain_obj(self, domain: Tensor):
        """
        도메인 domain.item()에 해당하는 BN 레이어를 반환.
        """
        return self.batchnorm[f'dom {domain.item()}']

    @torch.no_grad()
    def initrunningstats(self, X, domain):
        """
        러닝 스탯 초기화 시, 해당 도메인 BN 객체에 위임.
        """
        self.batchnorm[f'dom {domain.item()}'].initrunningstats(X)

    def forward_domain_(self, X, domain):
        """
        특정 도메인 데이터 X를 그 도메인의 BN 레이어에 통과.
        """
        res = self.batchnorm[f'dom {domain.item()}'](X)
        return res

    def forward(self, X, d):
        """
        실제 forward: d(unique domains)별로 X를 쪼개서 forward_domain_로 처리 후, 순서 맞게 합침.
        """
        du = d.unique()
        X_normalized = torch.empty_like(X)

        # domain별로 X[d==domain]을 BN에 통과
        res = [
            (self.forward_domain_(X[d == domain], domain), torch.nonzero(d == domain))
            for domain in du
        ]
        # 도메인별 결과를 하나로 concat
        X_out, ixs = zip(*res)
        X_out, ixs = torch.cat(X_out), torch.cat(ixs).flatten()
        X_normalized[ixs] = X_out

        return X_normalized


class SchedulableDomainBatchNorm(BaseDomainBatchNorm, SchedulableBatchNorm):
    """
    도메인별 BN(BaseDomainBatchNorm) + 모멘텀 스케줄러(SchedulableBatchNorm)
    => 각 도메인 BN에도 eta, eta_test 동시 적용 가능.
    """

    def set_eta(self, eta=None, eta_test=None):
        # self.batchnorm에 들어있는 모든 BN 객체가 SchedulableBatchNorm이면 set_eta 반복
        for bn in self.batchnorm.values():
            if isinstance(bn, SchedulableBatchNorm):
                bn.set_eta(eta, eta_test)


#########################################################
# 섹션 4) Euclidean BN (일반 텐서) 구현 (BatchNormImpl) #
#########################################################

class BatchNormImpl(BaseBatchNorm):
    """
    실제 유클리드 공간에서의 BN을 구현한 핵심 클래스.
    - running_mean, running_var, (learnable) mean, std
    - dispersion=NONE,SCALAR,VECTOR
    - training 중이면 batch_mean/batch_var로 running 통계 업데이트
    - test_stats_mode(BUFFER, REFIT) 등에 따라 추론시 통계 처리
    """

    def __init__(
            self,
            shape: Tuple[int, ...] or torch.Size,
            batchdim: int,
            eta=0.1,
            eta_test=0.1,
            dispersion: BatchNormDispersion = BatchNormDispersion.NONE,
            learn_mean: bool = True,
            learn_std: bool = True,
            mean=None,
            std=None,
            eps=1e-5,
            **kwargs
    ):
        super().__init__(eta=eta, eta_test=eta_test)
        self.dispersion = dispersion
        self.batchdim = batchdim
        self.eps = eps

        # 러닝 평균, 러닝 분산
        init_mean = torch.zeros(shape, **kwargs)
        self.register_buffer('running_mean', init_mean)
        self.register_buffer('running_mean_test', init_mean.clone())

        # mean (학습 가능 or 고정)
        if mean is not None:
            self.mean = mean
        else:
            if learn_mean:
                self.mean = nn.parameter.Parameter(init_mean.clone())
            else:
                self.mean = init_mean.clone()

        # std(스케일)도 유사하게 설정
        if std is not None:
            self.std = std
            self.register_buffer('running_var', torch.ones(std.shape, **kwargs))
            self.register_buffer('running_var_test', torch.ones(std.shape, **kwargs))
        elif self.dispersion == BatchNormDispersion.SCALAR:
            var_init = torch.ones((*shape[:-1], 1), **kwargs)
            self.register_buffer('running_var', var_init)
            self.register_buffer('running_var_test', var_init.clone())
            if learn_std:
                self.std = nn.parameter.Parameter(var_init.clone().sqrt())
            else:
                self.std = var_init.clone().sqrt()
        elif self.dispersion == BatchNormDispersion.VECTOR:
            var_init = torch.ones(shape, **kwargs)
            self.register_buffer('running_var', var_init)
            self.register_buffer('running_var_test', var_init.clone())
            if learn_std:
                self.std = nn.parameter.Parameter(var_init.clone().sqrt())
            else:
                self.std = var_init.clone().sqrt()

    @torch.no_grad()
    def initrunningstats(self, X):
        """
        REFIT 모드 등에 필요:
        입력 X로부터 running_mean, running_var를 새로 초기화.
        """
        self.running_mean = X.mean(dim=self.batchdim, keepdim=True).clone()
        self.running_mean_test = self.running_mean.clone()

        if self.dispersion == BatchNormDispersion.SCALAR:
            self.running_var = (X - self.running_mean).square().mean(keepdim=True)
            self.running_var_test = self.running_var.clone()
        elif self.dispersion == BatchNormDispersion.VECTOR:
            self.running_var = (X - self.running_mean).square().mean(dim=self.batchdim, keepdim=True)
            self.running_var_test = self.running_var.clone()

    def forward(self, X):
        if self.training:
            # 학습 모드 => 배치 통계(batch_mean, batch_var)를 계산
            batch_mean = X.mean(dim=self.batchdim, keepdim=True)
            rm = (1. - self.eta) * self.running_mean + self.eta * batch_mean

            if self.dispersion is not BatchNormDispersion.NONE:
                if self.dispersion == BatchNormDispersion.SCALAR:
                    batch_var = (X - batch_mean).square().mean(keepdim=True)
                elif self.dispersion == BatchNormDispersion.VECTOR:
                    batch_var = (X - batch_mean).square().mean(dim=self.batchdim, keepdim=True)

                rv = (1. - self.eta) * self.running_var + self.eta * batch_var

        else:
            # 추론 모드 => test_stats_mode에 따라
            if self.test_stats_mode == BatchNormTestStatsMode.BUFFER:
                pass  # running_mean, running_var_test 그대로 사용
            elif self.test_stats_mode == BatchNormTestStatsMode.REFIT:
                self.initrunningstats(X)
            elif self.test_stats_mode == BatchNormTestStatsMode.ADAPT:
                raise NotImplementedError()

            rm = self.running_mean_test
            if self.dispersion is not BatchNormDispersion.NONE:
                rv = self.running_var_test

        # (X - rm) / sqrt(rv+eps) * self.std + self.mean
        if self.dispersion is not BatchNormDispersion.NONE:
            Xn = (X - rm) / (rv + self.eps).sqrt() * self.std + self.mean
        else:
            # dispersion=NONE => 분산 없이 평균만 보정
            Xn = X - rm + self.mean

        if self.training:
            # 러닝 통계 업데이트
            with torch.no_grad():
                self.running_mean = rm.clone()
                self.running_mean_test = (1. - self.eta_test) * self.running_mean_test + self.eta_test * batch_mean
                if self.dispersion is not BatchNormDispersion.NONE:
                    self.running_var = rv.clone()
                    self.running_var_test = (1. - self.eta_test) * self.running_var_test + self.eta_test * batch_var

        return Xn


class BatchNorm(BatchNormImpl):
    """
    Ioffe & Szegedy (2015) 형태의 표준 BN. (VECTOR dispersion)
    eta_test 대신 eta 하나만 사용.
    """

    def __init__(self, shape: Tuple[int, ...] or torch.Size, batchdim: int, eta=0.1, **kwargs):
        if 'dispersion' not in kwargs.keys():
            kwargs['dispersion'] = BatchNormDispersion.VECTOR
        if 'eta_test' in kwargs.keys():
            raise RuntimeError('이 subclass에서는 eta_test 무시')
        super().__init__(shape=shape, batchdim=batchdim, eta=1.0, eta_test=eta, **kwargs)


class BatchReNorm(BatchNormImpl):
    """
    Ioffe (2017) ReNorm 기법 (VECTOR dispersion).
    """

    def __init__(self, shape: Tuple[int, ...] or torch.Size, batchdim: int, eta=0.1, **kwargs):
        if 'dispersion' not in kwargs.keys():
            kwargs['dispersion'] = BatchNormDispersion.VECTOR
        if 'eta_test' in kwargs.keys():
            raise RuntimeError('이 parameter는 무시')
        super().__init__(shape=shape, batchdim=batchdim, eta=eta, eta_test=eta, **kwargs)


class AdaMomBatchNorm(BatchNormImpl, SchedulableBatchNorm):
    """
    Yong et al. (2020) ECCV: Adaptive Momentum BatchNorm.
    - momemtum(eta, eta_test)을 스케줄러로 조정 가능.
    """

    def __init__(self, shape: Tuple[int, ...] or torch.Size, batchdim: int, eta=1.0, eta_test=0.1, **kwargs):
        super().__init__(shape=shape, batchdim=batchdim, eta=eta, eta_test=eta_test, **kwargs)


###############################
# 섹션 5) DomainBatchNorm 구현 #
###############################

class DomainBatchNormImpl(BaseDomainBatchNorm):
    """
    Domain별로 BatchNormImpl을 따로 유지.
    - domain_bn_cls: 어떤 BN 구현을 써야 할지 지정 (BatchNormImpl 등).
    - add_domain_()로 실제 도메인별 BN 인스턴스 생성.
    """

    def __init__(
            self, shape: Tuple[int, ...] or torch.Size,
            batchdim: int,
            learn_mean: bool = True,
            learn_std: bool = True,
            dispersion: BatchNormDispersion = BatchNormDispersion.NONE,
            test_stats_mode: BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER,
            eta=1.,
            eta_test=0.1,
            domains: list = [],
            **kwargs
    ):
        super().__init__()
        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std

        # 도메인별이지만, scale/bias를 전부 공유하는 경우 init?
        init_mean = torch.zeros(shape, **kwargs)
        if self.learn_mean:
            self.mean = nn.parameter.Parameter(init_mean)
        else:
            self.mean = init_mean

        if self.dispersion is BatchNormDispersion.SCALAR:
            init_var = torch.ones((*shape[:-2], 1), **kwargs)
        elif self.dispersion == BatchNormDispersion.VECTOR:
            init_var = torch.ones(shape, **kwargs)
        else:
            init_var = None

        if self.learn_std:
            self.std = nn.parameter.Parameter(init_var.clone()) if init_var is not None else None
        else:
            self.std = init_var

        cls = type(self).domain_bn_cls
        for domain in domains:
            # domains 리스트에 있는 각 도메인마다 BN 인스턴스 생성
            self.add_domain_(
                cls(
                    shape=shape,
                    batchdim=batchdim,
                    learn_mean=learn_mean,
                    learn_std=learn_std,
                    dispersion=dispersion,
                    mean=self.mean,
                    std=self.std,
                    eta=eta,
                    eta_test=eta_test,
                    **kwargs
                ),
                domain
            )

        self.set_test_stats_mode(test_stats_mode)


class DomainBatchNorm(DomainBatchNormImpl):
    """
    Chang et al. (2019, CVPR): Domain-specific BN (Euclidean).
    - domain_bn_cls = BatchNormImpl
    """
    domain_bn_cls = BatchNormImpl


class AdaMomDomainBatchNorm(DomainBatchNormImpl, SchedulableDomainBatchNorm):
    """
    도메인별 BN + Adaptive Momentum BN.
    """
    domain_bn_cls = AdaMomBatchNorm


###############################################
# 섹션 6) SPD(리만공간)용 BN (SPDBatchNormImpl) #
###############################################

class SPDBatchNormImpl(BaseBatchNorm):
    """
    SPD 행렬에 대한 배치 정규화.
    - shape의 마지막 2차원은 SPD 크기([..., n, n]).
    - logm, expm, 병렬이동(parallel transport) 등으로 mean, var 처리.
    """

    def __init__(
            self,
            shape: Tuple[int, ...] or torch.Size,
            batchdim: int,
            eta=1.,
            eta_test=0.1,
            karcher_steps: int = 1,
            learn_mean=True,
            learn_std=True,
            dispersion: BatchNormDispersion = BatchNormDispersion.SCALAR,
            eps=1e-5,
            mean=None,
            std=None,
            **kwargs
    ):
        super().__init__(eta, eta_test)
        # shape[-1] == shape[-2] => SPD 행렬
        assert (shape[-1] == shape[-2])

        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std
        self.batchdim = batchdim
        self.karcher_steps = karcher_steps
        self.eps = eps

        init_mean = torch.diag_embed(torch.ones(shape[:-1], **kwargs))
        init_var = torch.ones((*shape[:-2], 1), **kwargs)

        # running_mean, running_var도 SPD(ManifoldTensor)로 저장
        self.register_buffer('running_mean', ManifoldTensor(init_mean, manifold=SymmetricPositiveDefinite()))
        self.register_buffer('running_var', init_var)
        self.register_buffer('running_mean_test',
                             ManifoldTensor(init_mean.clone(), manifold=SymmetricPositiveDefinite()))
        self.register_buffer('running_var_test', init_var.clone())

        if mean is not None:
            self.mean = mean
        else:
            if self.learn_mean:
                # SPD manifold 파라미터
                self.mean = ManifoldParameter(init_mean.clone(), manifold=SymmetricPositiveDefinite())
            else:
                self.mean = ManifoldTensor(init_mean.clone(), manifold=SymmetricPositiveDefinite())

        if self.dispersion is not BatchNormDispersion.NONE:
            if std is not None:
                self.std = std
            else:
                if self.learn_std:
                    self.std = nn.parameter.Parameter(init_var.clone())
                else:
                    self.std = init_var.clone()

    @torch.no_grad()
    def initrunningstats(self, X):
        """
        X(배치 SPD 행렬)에 대해 karcher flow(spd_mean_kracher_flow)로 평균을 구함.
        dispersion이 SCALAR이면 var도 구함.
        """
        self.running_mean.data, geom_dist = functionals.spd_mean_kracher_flow(X, dim=self.batchdim, return_dist=True)
        self.running_mean_test.data = self.running_mean.data.clone()

        if self.dispersion is BatchNormDispersion.SCALAR:
            self.running_var = \
            geom_dist.square().mean(dim=self.batchdim, keepdim=True).clamp(min=functionals.EPS[X.dtype])[..., None]
            self.running_var_test = self.running_var.clone()

    def forward(self, X):
        manifold = self.running_mean.manifold
        if self.training:
            # 배치 평균: Karcher mean(=Riemannian barycenter)
            batch_mean = X.mean(dim=self.batchdim, keepdim=True)
            for _ in range(self.karcher_steps):
                bm_sq, bm_invsq = functionals.sym_invsqrtm2.apply(batch_mean.detach())
                XT = functionals.sym_logm.apply(bm_invsq @ X @ bm_invsq) #batch mean to identity then matrix logarithm
                GT = XT.mean(dim=self.batchdim, keepdim=True)            #geodesic mean
                batch_mean = bm_sq @ functionals.sym_expm.apply(GT) @ bm_sq #back to batch mean

            # running mean 업데이트
            rm = functionals.spd_2point_interpolation(self.running_mean, batch_mean, self.eta)
            # running var 업데이트
            if self.dispersion is BatchNormDispersion.SCALAR:
                # 배치 분산(batch_var) 계산
                # XT = 중심이 평균으로 이동된 Batch의 X의 logm, GT = batch mean을 통해 I 근처로 이동한 Running Mean의 logm
                GT = functionals.sym_logm.apply(bm_invsq @ rm @ bm_invsq) #move running mean around to Identity by batch mean then matrix logarithm
                batch_var = torch.norm(XT - GT, p='fro', dim=(-2, -1), keepdim=True).square().mean(dim=self.batchdim, keepdim=True).squeeze(-1)
                rv = (1. - self.eta) * self.running_var + self.eta * batch_var

        else:
            # 추론 모드
            if self.test_stats_mode == BatchNormTestStatsMode.BUFFER:
                pass
            elif self.test_stats_mode == BatchNormTestStatsMode.REFIT:
                self.initrunningstats(X)
            elif self.test_stats_mode == BatchNormTestStatsMode.ADAPT:
                raise NotImplementedError()

            rm = self.running_mean_test
            if self.dispersion is BatchNormDispersion.SCALAR:
                rv = self.running_var_test
#지금 running mean, running variance는 일단 업데이트를 했어?

#X를 running mean으로 Identity로 옮기고 스케일링하고, Identity에서 그걸 다시 bias으로 옮겨.
        # (X -> rm) 사이의 scaling
        if self.dispersion is BatchNormDispersion.SCALAR:
            Xn = manifold.transp_identity_rescale_transp(X, rm, self.std / (rv + self.eps).sqrt(), self.mean)
        else:
            Xn = manifold.transp_via_identity(X, rm, self.mean)

        if self.training:
            # 러닝 통계 업데이트
            with torch.no_grad():
                self.running_mean.data = rm.clone()
                self.running_mean_test.data = functionals.spd_2point_interpolation(self.running_mean_test, batch_mean,
                                                                                   self.eta_test)
                if self.dispersion is not BatchNormDispersion.NONE:
                    self.running_var = rv.clone()
                    GT_test = functionals.sym_logm.apply(bm_invsq @ self.running_mean_test @ bm_invsq)
                    batch_var_test = torch.norm(XT - GT_test, p='fro', dim=(-2, -1), keepdim=True).square().mean(
                        dim=self.batchdim, keepdim=True).squeeze(-1)
                    self.running_var_test = (1. - self.eta_test) * self.running_var_test + self.eta_test * batch_var_test

        return Xn


class SPDBatchNorm(SPDBatchNormImpl):
    """
    SPD BN: dispersion=SCALAR가 default (Kobler et al. 2022)
    eta_test는 별도 활용
    """

    def __init__(self, shape: Tuple[int, ...] or torch.Size, batchdim: int, eta=0.1, **kwargs):
        if 'dispersion' not in kwargs.keys():
            kwargs['dispersion'] = BatchNormDispersion.SCALAR
        if 'eta_test' in kwargs.keys():
            raise RuntimeError('이 서브클래스에서 eta_test 무시')
        super().__init__(shape=shape, batchdim=batchdim, eta=1.0, eta_test=eta, **kwargs)


class SPDBatchReNorm(SPDBatchNormImpl):
    """
    SPD 상에서 ReNorm
    """

    def __init__(self, shape: Tuple[int, ...] or torch.Size, batchdim: int, eta=0.1, **kwargs):
        if 'dispersion' not in kwargs.keys():
            kwargs['dispersion'] = BatchNormDispersion.SCALAR
        if 'eta_test' in kwargs.keys():
            raise RuntimeError('무시됨')
        super().__init__(shape=shape, batchdim=batchdim, eta=eta, eta_test=eta, **kwargs)


class AdaMomSPDBatchNorm(SPDBatchNormImpl, SchedulableBatchNorm):
    """
    SPD 상에서 모멘텀 스케줄이 가능한 BN
    """

    def __init__(self, shape: Tuple[int, ...] or torch.Size, batchdim: int, eta=1.0, eta_test=0.1, **kwargs):
        super().__init__(shape=shape, batchdim=batchdim, eta=eta, eta_test=eta_test, **kwargs)


############################################################
# 섹션 7) Domain + SPD BN 구현 (DomainSPDBatchNormImpl 등) #
############################################################

class DomainSPDBatchNormImpl(BaseDomainBatchNorm):
    """
    SPD를 도메인별로 나누어 BN.
    domain_bn_cls: 실제 SPD BN 구현 클래스를 지정 (SPDBatchNormImpl 등).
    """

    domain_bn_cls = None  # 서브클래스에서 오버라이드

    def __init__(
            self,
            shape: Tuple[int, ...] or torch.Size,
            batchdim: int,
            learn_mean: bool = True,
            learn_std: bool = True,
            dispersion: BatchNormDispersion = BatchNormDispersion.NONE,
            test_stats_mode: BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER,
            eta=1.,
            eta_test=0.1,
            domains: Tensor = Tensor([]),
            **kwargs
    ):
        super().__init__()
        assert (shape[-1] == shape[-2])

        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std

        # 공통 mean, std
        init_mean = torch.diag_embed(torch.ones(shape[:-1], **kwargs))
        if self.learn_mean:
            self.mean = ManifoldParameter(init_mean, manifold=SymmetricPositiveDefinite())
        else:
            self.mean = ManifoldTensor(init_mean, manifold=SymmetricPositiveDefinite())

        if self.dispersion is BatchNormDispersion.SCALAR:
            init_var = torch.ones((*shape[:-2], 1), **kwargs)
            if self.learn_std:
                self.std = nn.parameter.Parameter(init_var.clone())
            else:
                self.std = init_var.clone()
        else:
            self.std = None

        cls = type(self).domain_bn_cls
        for domain in domains:
            self.add_domain_(
                cls(
                    shape=shape,
                    batchdim=batchdim,
                    learn_mean=learn_mean,
                    learn_std=learn_std,
                    dispersion=dispersion,
                    mean=self.mean,
                    std=self.std,
                    eta=eta,
                    eta_test=eta_test,
                    **kwargs
                ),
                domain
            )

        self.set_test_stats_mode(test_stats_mode)


class DomainSPDBatchNorm(DomainSPDBatchNormImpl):
    """
    SPD 도메인별 BN + momentum(고정)
    """
    domain_bn_cls = SPDBatchNormImpl


class AdaMomDomainSPDBatchNorm(SchedulableDomainBatchNorm, DomainSPDBatchNormImpl):
    """
    SPD 도메인별 BN + adaptive momentum
    """
    domain_bn_cls = AdaMomSPDBatchNorm
