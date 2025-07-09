import math
import torch
from typing import Callable, Tuple
from typing import Any
from torch.autograd import Function, gradcheck
from torch.functional import Tensor
from torch.types import Number
# Geoopt에서 일부 로직 참고함 (안정화 기법, 로브너 미분 등)

# 각 데이터타입(float32, float64)에 따라 다른 epsilon을 설정
EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


def ensure_sym(A: Tensor) -> Tensor:
    """
    마지막 두 차원을 대칭(symmetric) 행렬로 만들어주는 함수
    Parameters
    ----------
    A : torch.Tensor
        (B, n, n) 등, 마지막 두 차원을 행렬로 보는 텐서
    -------
    return : torch.Tensor
        A와 A^T의 평균을 취해 강제로 대칭화한 결과
    """
    return 0.5 * (A + A.transpose(-1,-2)) # A와 A^T를 더한 뒤 절반을 취하면 대칭행렬이 됨.


def broadcast_dims(A: torch.Size, B: torch.Size, raise_error:bool=True) -> Tuple:
    """
    A와 B의 shape을 비교해, 브로드캐스팅이 필요한 차원을 찾아 반환
    Parameters
    ----------
    A : torch.Size
        첫 번째 텐서의 shape
    B : torch.Size
        두 번째 텐서의 shape
    raise_error : bool (=True)
        브로드캐스팅 불가능 시 에러를 낼지 여부
    -------
    Returns : Tuple
        브로드캐스팅 차원 인덱스
    """
    # 만약 A와 B의 차원 수가 다르면 예외 발생
    if raise_error:
        if len(A) != len(B):
            raise ValueError('The number of dimensions must be equal!')

    # 두 shape를 합쳐서 텐서로 만듦
    tdim = torch.tensor((A, B), dtype=torch.int32)

    # 서로 다른 차원 인덱스 찾기
    bdims = tuple(torch.where(tdim[0].ne(tdim[1]))[0].tolist())

    # 서로 다른 차원 중 하나가 1이어야만 브로드캐스트 가능
    if raise_error:
        if not tdim[:,bdims].eq(1).any(dim=0).all():
            raise ValueError('Broadcast not possible! One of the dimensions must be 1.')

    return bdims


def sum_bcastdims(A: Tensor, shape_out: torch.Size) -> Tensor:
    """
    A를 shape_out으로 브로드캐스트 맞춘 뒤,
    브로드캐스트된 차원에 대해 sum을 취해 shape_out에 맞게 변환
    Parameters
    ----------
    A : torch.Tensor
    shape_out : torch.Size
        원하는 최종 shape
    -------
    Returns : torch.Tensor
        브로드캐스트 차원들이 합산된 결과
    """
    bdims = broadcast_dims(A.shape, shape_out)

    if len(bdims) == 0:
        return A
    else:
        # bdims 차원을 합(keepdim=True)으로 처리
        return A.sum(dim=bdims, keepdim=True)


def randn_sym(shape, **kwargs):
    # shape 크기의 텐서를 랜덤 생성한 뒤 대칭 형태로 만들어 반환
    ndim = shape[-1]             # 행렬의 크기 (마지막 차원)
    X = torch.randn(shape, **kwargs)   # shape에 맞춰 랜덤 생성
    ixs = torch.tril_indices(ndim, ndim, offset=-1)
    # 하삼각부(tril) 인덱스 가져오기

    X[...,ixs[0],ixs[1]] /= math.sqrt(2)
    # 하삼각부 값을 sqrt(2)로 나눠서 정규화
    X[...,ixs[1],ixs[0]] = X[...,ixs[0],ixs[1]]
    # 하삼각부를 위삼각부에 대칭 복사

    return X


def spd_2point_interpolation(A : Tensor, B : Tensor, t : Number) -> Tensor:
    """
    SPD 행렬 A, B 사이의 2점 보간(geodesic interpolation) 함수
    A^(1/2), A^(-1/2) 등을 이용해 A에서 B까지 t만큼 이동한 지점 계산
    """
    # 먼저 A의 sqrt(A), invsqrt(A)를 구함
    rm_sq, rm_invsq = sym_invsqrtm2.apply(A)
    # (A^-1/2) * B * (A^-1/2)를 power로 들어올려(^(t)) 다시 (A^1/2)로 감싸기
    return rm_sq @ sym_powm.apply(rm_invsq @ B @ rm_invsq, torch.tensor(t)) @ rm_sq


class reverse_gradient(Function):
    """
    gradient 반전 기능을 제공하는 간단한 Function
    scaling 파라미터로 반전된 gradient의 세기 조절 가능
    """
    @staticmethod
    def forward(ctx, x, scaling = 1.0):
        ctx.scaling = scaling
        return x.view_as(x)    # forward는 입력을 그대로 반환

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.scaling
        # gradient의 부호를 반전하고 scaling 곱
        return grad_output, None


class sym_modeig:
    """
    고유분해 후, 고유값에 임의의 elementwise 함수(fun)를 적용하는 공통 클래스
    Brooks et al. (2019) Riemannian Batch Normalization 아이디어 기반
    """

    @staticmethod
    def forward(M : Tensor,
                fun : Callable[[Tensor], Tensor],
                fun_param : Tensor = None,
                ensure_symmetric : bool = False,
                ensure_psd : bool = False) -> Tensor:
        """
        대칭행렬 M의 고유값을 fun을 통해 수정한 뒤 다시 재구성
        M : (배치, n, n) 형태
        fun : 고유값에 적용할 원소별 함수
        ensure_symmetric : True면 M을 대칭화
        ensure_psd : True면 음수 고유값을 clamp하여 SPD로 유지
        """
        if ensure_symmetric:
            M = ensure_sym(M)  # 대칭 보정

        # 고유분해
        s, U = torch.linalg.eigh(M)
        if ensure_psd:
            # 고유값이 0보다 작으면 EPS까지 clamp
            s = s.clamp(min=EPS[s.dtype])

        # fun을 통해 고유값 수정
        smod = fun(s, fun_param)

        # U diag(smod) U^T로 재구성
        X = U @ torch.diag_embed(smod) @ U.transpose(-1,-2)
        return X, s, smod, U

    @staticmethod
    def backward(dX : Tensor,
                 s : Tensor,
                 smod : Tensor,
                 U : Tensor,
                 fun_der : Callable[[Tensor], Tensor],
                 fun_der_param : Tensor = None) -> Tensor:
        """
        Loewner 미분 기법으로 dX -> dM 역전파 계산
        dX : forward 결과 X에 대한 gradient
        s, smod : 원본/수정된 고유값
        U : 고유벡터
        fun_der : f'(고유값)에 해당하는 함수
        """
        # Lowener(Loewner) 행렬 계산
        L_den = s[...,None] - s[...,None].transpose(-1,-2)  # 분모 (σ_i - σ_j)
        is_eq = L_den.abs() < EPS[s.dtype]
        L_den[is_eq] = 1.0

        # 분자 (smod_i - smod_j)
        L_num_ne = smod[...,None] - smod[...,None].transpose(-1,-2)
        L_num_ne[is_eq] = 0

        # 대각(i=j)인 경우: f'(σ_i)의 평균 => 0.5*(f'(σ_i)+f'(σ_j)) (i=j 이므로 둘이 동일)
        sder = fun_der(s, fun_der_param)
        L_num_eq = 0.5 * (sder[...,None] + sder[...,None].transpose(-1,-2))
        L_num_eq[~is_eq] = 0

        # 최종 L = (L_num_ne + L_num_eq)/L_den
        L = (L_num_ne + L_num_eq) / L_den

        # ensure_sym(dX)로 대칭화 후, U^T dX U
        dX_sym = ensure_sym(dX)
        tmp = U.transpose(-1,-2) @ dX_sym @ U

        # L * tmp => 원소별 곱
        dM = U @ (L * tmp) @ U.transpose(-1,-2)
        return dM


class sym_reeig(Function):
    """
    ReEig : 고유값을 threshold 이하로는 threshold로, 이상은 그대로 (ReLU 유사)
    """

    @staticmethod
    def value(s : Tensor, threshold : Tensor) -> Tensor:
        # s를 threshold 이상으로 clamp
        return s.clamp(min=threshold.item())

    @staticmethod
    def derivative(s : Tensor, threshold : Tensor) -> Tensor:
        # s>threshold 이면 1, 아니면 0
        return (s>threshold.item()).type(s.dtype)

    @staticmethod
    def forward(ctx: Any, M: Tensor, threshold : Tensor, ensure_symmetric : bool = False) -> Tensor:
        # 고유값에 value 함수를 적용
        X, s, smod, U = sym_modeig.forward(M, sym_reeig.value, threshold,
                                           ensure_symmetric=ensure_symmetric)
        # 역전파 때 쓸 변수 저장
        ctx.save_for_backward(s, smod, U, threshold)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, threshold = ctx.saved_tensors
        # derivative 함수를 이용한 Loewner 미분
        dM = sym_modeig.backward(dX, s, smod, U, sym_reeig.derivative, threshold)
        # threshold(스칼라)에 대한 grad는 여기서 None
        return dM, None, None

    @staticmethod
    def tests():
        """
        sym_reeig이 제대로 동작하는지 테스트 (gradcheck 등)
        """
        ndim = 2
        nb = 1
        # 임의의 SPD 비슷한 행렬 만들기
        A = torch.randn((1,ndim,ndim), dtype=torch.double)
        U, s, _ = torch.linalg.svd(A)

        threshold = torch.tensor([1e-3], dtype=torch.double)

        # (1) 모든 고유값이 threshold보다 큰 경우
        s = threshold * 1e1 + torch.rand((nb,ndim), dtype=torch.double) * threshold
        M = U @ torch.diag_embed(s) @ U.transpose(-1,-2)
        assert (sym_reeig.apply(M, threshold, False).allclose(M))
        M.requires_grad_(True)
        assert(gradcheck(sym_reeig.apply, (M, threshold, True)))

        # (2) 일부 고유값이 threshold보다 작은 경우
        s = torch.rand((nb,ndim), dtype=torch.double) * threshold
        s[::2] += threshold
        M = U @ torch.diag_embed(s) @ U.transpose(-1,-2)
        assert (~sym_reeig.apply(M, threshold, False).allclose(M))
        M.requires_grad_(True)
        assert(gradcheck(sym_reeig.apply, (M, threshold, True)))

        # (3) 모든 고유값이 같을 경우
        s = torch.ones((nb,ndim), dtype=torch.double)
        M = U @ torch.diag_embed(s) @ U.transpose(-1,-2)
        assert (sym_reeig.apply(M, threshold, True).allclose(M))
        M.requires_grad_(True)
        assert(gradcheck(sym_reeig.apply, (M, threshold, True)))

class sym_abseig(Function):
    """
    고유값의 절댓값을 취하는 함수
    음수 고유값도 양수화해서 다루는 경우에 쓰임
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.abs()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        # d/ds(|s|) = sign(s)
        return s.sign()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_abseig.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        # 고유값 미분 => sign(s)
        return sym_modeig.backward(dX, s, smod, U, sym_abseig.derivative), None


class sym_logm(Function):
    """
    SPD 행렬에 대한 matrix-log 함수
    고유값이 양수라고 가정하므로, s<=EPS인 경우 clamp 후 log
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).log()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        sder = s.reciprocal()  # 1/s
        # clamp된 부분은 0으로 처리
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_logm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_logm.derivative), None


class sym_expm(Function):
    """
    대칭행렬에 대한 matrix-exp 함수
    고유값 e^s 형태로 변환
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_expm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_expm.derivative), None


class sym_powm(Function):
    """
    대칭행렬에 대한 power 연산: M^exponent
    고유값 s^exponent 꼴로 변환
    """
    @staticmethod
    def value(s : Tensor, exponent : Tensor) -> Tensor:
        return s.pow(exponent=exponent)

    @staticmethod
    def derivative(s : Tensor, exponent : Tensor) -> Tensor:
        return exponent * s.pow(exponent=exponent-1.)

    @staticmethod
    def forward(ctx: Any, M: Tensor, exponent : Tensor, ensure_symmetric : bool = False) -> Tensor:
        # exponent도 텐서 형태일 수 있음
        X, s, smod, U = sym_modeig.forward(M, sym_powm.value, exponent, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U, exponent)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, exponent = ctx.saved_tensors
        # 먼저 M에 대한 grad
        dM = sym_modeig.backward(dX, s, smod, U, sym_powm.derivative, exponent)

        # exponent에 대한 grad도 구함
        #  d/d(exp) [s^exp] = s^exp * log(s)
        #  Chain rule 고려: dX에 대한 부분
        dXs = (U.transpose(-1,-2) @ ensure_sym(dX) @ U).diagonal(dim1=-1,dim2=-2)
        # smod = s^exponent
        dexp = dXs * smod * s.log()

        return dM, dexp, None


class sym_sqrtm(Function):
    """
    SPD 행렬의 sqrt : 고유값을 sqrt(s)로 변환
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).sqrt()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        # 0.5 * 1/sqrt(s)
        sder = 0.5 * s.rsqrt()
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_sqrtm.derivative), None


class sym_invsqrtm(Function):
    """
    SPD 행렬의 inverse sqrt : 고유값 1/sqrt(s)로 변환
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).rsqrt()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        # d/ds(1/sqrt(s)) = -0.5 s^(-3/2)
        sder = -0.5 * s.pow(-1.5)
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invsqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invsqrtm.derivative), None

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invsqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invsqrtm.derivative), None


class sym_invsqrtm2(Function):
    """
    SPD 행렬에 대해 sqrt, invsqrt를 동시에 반환하는 함수
    """
    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        # 먼저 sqrt(M)을 구함
        Xsq, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)

        # 이어서 같은 s에 대해 inverse sqrt도 구함
        smod2 = sym_invsqrtm.value(s)
        Xinvsq = U @ torch.diag_embed(smod2) @ U.transpose(-1,-2)

        # 필요한 값 저장
        ctx.save_for_backward(s, smod, smod2, U)
        # Xsq : sqrt(M), Xinvsq : invsqrt(M)
        return Xsq, Xinvsq

    @staticmethod
    def backward(ctx: Any, dXsq: Tensor, dXinvsq: Tensor):
        s, smod, smod2, U = ctx.saved_tensors
        # sqrt(M)에 대한 grad
        dMsq = sym_modeig.backward(dXsq, s, smod, U, sym_sqrtm.derivative)
        # invsqrt(M)에 대한 grad
        dMinvsq = sym_modeig.backward(dXinvsq, s, smod2, U, sym_invsqrtm.derivative)

        # 합쳐서 반환
        return dMsq + dMinvsq, None


class sym_invm(Function):
    """
    SPD 행렬의 inverse : 고유값 1/s 로 변환
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).reciprocal()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        # d/ds(1/s) = -1/s^2
        sder = -1. * s.pow(-2)
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invm.derivative), None


def spd_mean_kracher_flow(X : Tensor,
                          G0 : Tensor = None,
                          maxiter : int = 50,
                          dim = 0,
                          weights = None,
                          return_dist = False,
                          return_XT = False) -> Tensor:
    """
    SPD 행렬들(X)의 Riemannian 평균을 Karcher flow 방식으로 구함
    X : (..., n, n) 형태 (dim=0이면 batch 차원)
    G0 : 초기값(초기 추정 평균)
    maxiter : 최대 반복 횟수
    dim : 평균 낼 차원(기본 0)
    weights : 가중치 (weighted mean)
    return_dist : True면 각 step의 dist 반환
    return_XT : True면 과정 중의 XT(로그값)도 반환
    """
    # 만약 averaging할 데이터가 1개뿐이면 그냥 반환
    if X.shape[dim] == 1:
        if return_dist:
            return X, torch.tensor([0.0], dtype=X.dtype, device=X.device)
        else:
            return X

    # weights가 없으면 균등 분포
    if weights is None:
        n = X.shape[dim]
        weights = torch.ones((*X.shape[:-2], 1, 1), dtype=X.dtype, device=X.device)
        weights /= n

    # 초깃값 설정
    if G0 is None:
        G = (X * weights).sum(dim=dim, keepdim=True)
    else:
        G = G0.clone()

    nu = 1.  # step size
    dist = tau = crit = torch.finfo(X.dtype).max
    i = 0

    while (crit > EPS[X.dtype]) and (i < maxiter) and (nu > EPS[X.dtype]):
        i += 1

        # G의 sqrt, invsqrt
        Gsq, Ginvsq = sym_invsqrtm2.apply(G)
        # (Ginvsq * X * Ginvsq)의 로그
        XT = sym_logm.apply(Ginvsq @ X @ Ginvsq)
        # XT 가중 평균
        GT = (XT * weights).sum(dim=dim, keepdim=True)
        # 지수로 돌아오면서 nu 배
        G = Gsq @ sym_expm.apply(nu * GT) @ Gsq

        if return_dist:
            # XT와 GT의 차이(norm) 관찰
            dist = torch.norm(XT - GT, p='fro', dim=(-2,-1))
        # 수렴 판정 위한 norm
        crit = torch.norm(GT, p='fro', dim=(-2,-1)).max()
        h = nu * crit
        # 하강폭이 이전보다 작아지면 nu를 줄여서 스텝 크기 감소
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    if return_dist:
        return G, dist
    if return_XT:
        return G, XT
    return G