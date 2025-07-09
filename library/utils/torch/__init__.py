from importlib.metadata import metadata
from geoopt.manifolds.sphere import Sphere
from omegaconf.dictconfig import DictConfig
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset, TensorDataset
from library.utils.moabb import CachedParadigm
from spdnets.manifolds import SymmetricPositiveDefinite
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Stiefel
import numpy as np
from copy import deepcopy
import time
import torch
from typing import Iterator, Sequence, Tuple
from sklearn.model_selection import StratifiedKFold
from datasetio.eeg.moabb import CachableDatase
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings


class BufferDataset(torch.utils.data.Dataset):
    """
    간단히 'items'라는 리스트(또는 튜플)를 Dataset으로 감싸는 클래스.
    items[0], items[1], ... 각각이 텐서일 때,
    __getitem__에서 해당 index의 데이터를 모두 묶어 반환할 수 있음.
    """

    def __init__(self, items) -> None:
        super().__init__()
        self.items = items  # items: list나 튜플 형태로 여러 텐서를 갖고 있을 수 있음

    def __len__(self):
        return self.items[0].shape[0]  # 첫 번째 텐서의 첫 차원 크기를 Dataset 길이로 간주

    def __getitem__(self, index):
        # index 위치의 값을 items 배열의 각 텐서에서 추출하여 리스트로 반환
        return [item[index] for item in self.items]


class DomainIndex(int):
    '''
    도메인 전체를 한 번에 가져오고 싶을 때, index로 전달하기 위해 사용하는 클래스.
    예: dataset[DomainIndex(도메인번호)] -> 해당 도메인 전체 데이터 반환
    '''
    pass


class DomainDataset(torch.utils.data.Dataset):
    """
    도메인 정보(domains), 라벨(labels), 메타데이터(metadata), mask_indices 등을 포함하는
    PyTorch Dataset 추상 클래스.
    - get_feature(index), get_features(indices) 메서드를 구현해서
      실제 '특징' 텐서를 어떻게 가져올지 정의해야 함.
    - training/eval 모드에 따라, 마스킹된 라벨(-1)을 쓸지 정상 라벨을 쓸지 결정.
    """

    def __init__(self,
                 labels: torch.LongTensor,
                 domains: torch.LongTensor,
                 metadata: pd.DataFrame,
                 training: bool = True,
                 dtype: torch.dtype = torch.double,
                 mask_indices: Sequence[int] = None):

        self.dtype = dtype  # 텐서 자료형(기본 float64=double)
        self._training = training  # 현재 Dataset이 학습 모드인가? (True/False)
        self._metadata = metadata
        self._mask_indices = mask_indices
        assert (len(metadata) == len(labels))
        assert (len(metadata) == len(domains))
        self._metadata = metadata
        self._domains = domains
        self._labels = labels

    @property
    def features(self) -> torch.Tensor:
        """
        Dataset 전체에 대한 특징 텐서를 반환.
        기본적으로 get_features(range(len(self)))로 구현.
        """
        return self.get_features(range(len(self))).to(dtype=self.dtype)

    @property
    def metadata(self) -> pd.DataFrame:
        return self._metadata

    @property
    def domains(self) -> torch.Tensor:
        return self._domains

    @property
    def labels(self) -> torch.Tensor:
        """
        라벨 텐서.
        만약 training 모드이고, _mask_indices가 설정되어 있다면,
        해당 인덱스의 라벨을 -1로 바꾸어 반환한다 (라벨 마스킹).
        """
        labels = self._labels.clone()
        if self._mask_indices is not None and self.training:
            labels[self._mask_indices] = -1
        return labels

    @property
    def training(self) -> bool:
        return self._training

    @property
    def shape(self):
        # 하위 클래스에서 구현해야 함
        raise NotImplementedError()

    @property
    def ndim(self):
        # 하위 클래스에서 구현
        raise NotImplementedError()

    def train(self):
        """ 학습 모드로 전환 -> 라벨 마스킹이 활성화될 수 있음 """
        self._training = True

    def eval(self):
        """ 평가 모드로 전환 -> 라벨 마스킹이 해제됨 """
        self._training = False

    def set_masked_labels(self, indices):
        """ indices에 해당하는 라벨들을 마스킹(-1)처리하도록 설정 """
        self._mask_indices = indices

    def get_feature(self, index: int) -> torch.Tensor:
        """
        인덱스 하나에 해당하는 특징 텐서를 가져오는 메서드 (하위 클래스에서 구현).
        """
        raise NotImplementedError()

    def get_features(self, indices) -> torch.Tensor:
        """
        인덱스 여러 개에 대한 특징 텐서를 동시에 가져오는 메서드 (하위 클래스에서 구현).
        """
        raise NotImplementedError()

    def copy(self, deep=False):
        """ 이 Dataset 객체를 복제(clone)하는 메서드 (하위 클래스에서 구현). """
        raise NotImplementedError()

    def __len__(self):
        return len(self.metadata)  # metadata 길이 == 전체 샘플 수

    def __getitem__(self, index):
        """
        PyTorch Dataset 필수 구현 함수.
        index가 DomainIndex이면 해당 도메인 전체를 리턴,
        그렇지 않으면 하나의 샘플만 리턴.
        """
        if isinstance(index, DomainIndex):
            # 도메인 인덱스가 넘어오면, 해당 도메인에 속한 모든 샘플을 한꺼번에 반환
            indices = np.flatnonzero(self.domains.numpy() == index)
            features = self.get_features(indices)
            return [
                dict(x=features.to(dtype=self.dtype), d=self.domains[indices]),
                self.labels[indices]
            ]
        else:
            # 일반 정수 인덱스면, 하나의 샘플(특징 + 도메인 + 라벨)을 반환
            feature = self.get_feature(index)
            return [
                dict(x=feature.to(dtype=self.dtype), d=self.domains[index]),
                self.labels[index]
            ]


class CachedDomainDataset(DomainDataset):
    """
    실제 텐서(전체 features)를 _features로 보유하고 있는 Dataset.
    - get_feature, get_features가 _features[index]로 바로 접근할 수 있어 빠름.
    - copy(deep=True) 시에는 features를 clone()해서 새로 할당.
    """

    def __init__(self, features, **kwargs) -> None:
        super().__init__(**kwargs)
        assert (len(self) == len(features))
        self._features = features  # 전체 샘플에 대한 특징 텐서

    @property
    def shape(self):
        return self._features.shape

    @property
    def ndim(self):
        return self._features.ndim

    def get_feature(self, index: int) -> torch.Tensor:
        # _features에서 index에 해당하는 샘플 하나를 가져온다
        return self._features[index]

    def get_features(self, indices) -> torch.Tensor:
        # _features에서 indices 리스트에 해당하는 샘플들을 한번에 가져온다
        return self._features[indices]

    def set_features(self, features) -> torch.Tensor:
        """
        외부에서 features를 교체하고 싶을 때 사용.
        np.ndarray => torch.Tensor로 변환,
        torch.Tensor => 그대로 저장
        """
        if isinstance(features, np.ndarray):
            self._features = torch.from_numpy(features)
        elif isinstance(features, torch.Tensor):
            self._features = features
        else:
            raise ValueError()

    def copy(self, deep=False):
        """
        이 Dataset 객체를 복제.
        deep=True이면 features, labels, domains를 모두 clone()하여 새 텐서로 복제.
        """
        features = self._features.clone() if deep else self._features
        labels = self._labels.clone() if deep else self._labels
        domains = self._domains.clone() if deep else self._domains

        obj = CachedDomainDataset(
            features,
            labels=labels,
            domains=domains,
            metadata=self._metadata.copy(deep=deep),
            training=self.training,
            dtype=self.dtype,
            mask_indices=self._mask_indices
        )
        return obj


class CombinedDomainDataset(DomainDataset, torch.utils.data.ConcatDataset):
    """
    여러 개의 features(각각 다른 subject/세션 등에 해당)를
    하나의 Dataset처럼 합쳐서 관리하기 위해
    PyTorch의 ConcatDataset과 DomainDataset을 다중 상속.
    """

    def __init__(self, features: Sequence[torch.Tensor], **kwargs):
        # features는 리스트 형태로, 각각 torch.Tensor (ex. subject별 data)
        torch.utils.data.ConcatDataset.__init__(self, features)
        DomainDataset.__init__(self, **kwargs)

    @classmethod
    def from_moabb(cls,
                   paradigm: CachedParadigm,
                   ds: CachableDatase,
                   subjects: list = None,
                   domain_expression="session + subject * 1000",
                   sessions: DictConfig = None,
                   **kwargs):
        """
        MOABB 기반 데이터셋(ds)와 CachedParadigm(전처리, epoching 등),
        원하는 subjects 목록을 받아와서,
        subject별로 데이터를 로드한 뒤 합치는 메서드.
        domain_expression 파라미터에 따라 각 샘플의 도메인 인덱스를 계산.
        예: session + subject*1000
        (kwargs는 상위 클래스 DomainDataset의 __init__에 전달.)
        """
        if subjects is None:
            subjects = ds.subject_list

        features = []
        metadata = []
        labels = []
        with warnings.catch_warnings():
            # UserWarning 같은 것은 무시
            warnings.simplefilter("ignore", UserWarning)

            for ix, subject in enumerate(subjects):
                x, l, md = paradigm.get_data(ds, [subject], False)

                # sessions 설정이 있다면, 예: 특정 세션만 골라 쓰도록 제한
                if sessions is not None:
                    unique_sessions = md.session.unique()
                    if 'order' in sessions and sessions['order'] == 'last':
                        unique_sessions = unique_sessions[::-1]
                    msk = md.session.isin(unique_sessions[0: sessions.get('n', len(unique_sessions))])
                    x = x[msk]
                    l = l[msk]
                    md = md[msk]

                # torch 텐서로 변환
                features += [torch.from_numpy(x)]
                # metadata에는 어떤 subject(subset index)인지 기록
                md['setindex'] = ix
                metadata += [md]
                labels += [l]

        # 모든 subject의 metadata를 수직으로 연결
        metadata = pd.concat(metadata, ignore_index=True)
        # 라벨(문자열 등)을 숫자로 변환
        labels = torch.from_numpy(
            LabelEncoder().fit_transform(np.concatenate(labels))
        ).to(dtype=torch.long)

        # domain_expression을 eval하여(예: session + subject*1000)
        # 각 샘플의 도메인 값을 만든다
        domains = torch.from_numpy(metadata.eval(domain_expression).to_numpy(dtype=np.int64))

        # CombinedDomainDataset 객체 생성
        return CombinedDomainDataset(
            features=features,
            labels=labels,
            domains=domains,
            metadata=metadata,
            **kwargs
        )

    @property
    def shape(self):
        """
        대략 (총 샘플 수, 채널, 시간...) 형태.
        - 첫 번째 ConcatDataset의 dataset[0].shape에서 가져옴
        - 그러나 0번째 하나만으로는 전체 길이가 아님
        - 실제 shape[0]은 ConcatDataset 전체 길이로 대체
        """
        shape = list(self.datasets[0].shape)
        shape[0] = len(self)
        return tuple(shape)

    @property
    def ndim(self):
        return self.datasets[0].ndim

    def get_feature(self, index: int) -> torch.Tensor:
        """
        ConcatDataset.__getitem__ 사용.
        index 하나에 해당하는 샘플을 반환.
        """
        return torch.utils.data.ConcatDataset.__getitem__(self, index)

    def get_features(self, indices) -> torch.Tensor:
        """
        여러 인덱스를 한 번에 가져오려 할 때,
        우선 metadata로부터 어떤 sub-dataset(=어떤 subject setindex)에 속하는지 확인.
        setindex(한 subject) 내에서만 일괄 접근 가능하도록 한정.
        (여러 subject가 섞인 인덱스 요청은 ValueError)
        """
        setix = self.metadata.loc[indices, 'setindex'].unique()
        if len(setix) > 1:
            raise ValueError('Domain data has to be contained in a single subset!')

        setix = setix[0]
        # 만약 setix != 0이면, ConcatDataset에서는 인덱스가
        # cumulative_sizes[setix-1]만큼 offset이 있음
        if setix == 0:
            subindices = indices
        else:
            subindices = indices - self.cumulative_sizes[setix - 1]

        return self.datasets[setix][subindices]

    def cache(self) -> CachedDomainDataset:
        """
        전체 features를 하나의 텐서로 모아 CachedDomainDataset 형태로 반환.
        - ConcatDataset에 들어있는 텐서들을 torch.cat으로 이어붙임
        """
        features = torch.cat([ds.to(dtype=self.dtype) for ds in self.datasets])
        obj = CachedDomainDataset(
            features,
            labels=self.labels,
            domains=self._domains,
            metadata=self.metadata,
            training=self.training,
            dtype=self.dtype,
            mask_indices=self._mask_indices
        )
        return obj

    def copy(self, deep=False):
        """
        deepcopy or shallow copy 수행.
        features 목록에 대해 clone()을 할지 여부 결정.
        """
        features = [dataset.clone() if deep else dataset for dataset in self.datasets]
        obj = CombinedDomainDataset(
            features,
            labels=self.labels,
            domains=self._domains,
            metadata=self.metadata,
            training=self.training,
            dtype=self.dtype,
            mask_indices=self._mask_indices
        )
        return obj


class StratifyableDataset(torch.utils.data.Dataset):
    """
    (특정) 라벨 또는 분류 기준에 대해 Stratify(계층화)할 수 있도록
    stratvar(계층화 변수)을 추가로 보관하는 래퍼 Dataset.
    - sampler가 stratvar를 이용해 균형 잡힌 배치를 구성할 수 있도록
      추가 정보를 제공.
    """

    def __init__(self, dataset, stratvar) -> None:
        super().__init__()
        self.dataset = dataset
        self.stratvar = stratvar
        assert (self.stratvar.shape[0] == len(dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class BalancedDomainDataLoader(DataLoader):
    """
    여러 도메인이 섞여 있는 데이터셋에서,
    배치마다 '도메인별 일정 샘플 수'가 들어오도록 샘플링하는 DataLoader.
    - domains_per_batch만큼 도메인을 골라,
      각 도메인에서 batch_size/domains_per_batch개의 샘플을 추출.
    - BalancedDomainSampler로 구현.
    """

    def __init__(self, dataset=None, batch_size=1, domains_per_batch=1, shuffle=True, replacement=False, **kwargs):
        # dataset이 DomainDataset이거나, 그 Subset이어야 함
        if isinstance(dataset, Subset) and isinstance(dataset.dataset, DomainDataset):
            domains = dataset.dataset.domains[dataset.indices]
        elif isinstance(dataset, DomainDataset):
            domains = dataset.domains
        else:
            raise NotImplementedError()

        # sampler를 BalancedDomainSampler로 정의
        sampler = BalancedDomainSampler(
            domains,
            int(batch_size / domains_per_batch),
            shuffle=shuffle,
            replacement=replacement
        )
        super().__init__(dataset=dataset, sampler=sampler, batch_size=batch_size, **kwargs)


class StratifiedDataLoader(DataLoader):
    """
    라벨(또는 특정 stratvar)에 대해 균형 잡힌 배치를 뽑아오는 DataLoader.
    - StratifiedSampler를 이용해 샘플링.
    """

    def __init__(self, dataset=None, batch_size=1, shuffle=True, **kwargs):
        # dataset이 StratifyableDataset이거나, 그 Subset이어야 함
        if isinstance(dataset, Subset) and isinstance(dataset.dataset, StratifyableDataset):
            stratvar = dataset.dataset.stratvar[dataset.indices]
        elif isinstance(dataset, StratifyableDataset):
            stratvar = dataset.stratvar
        else:
            raise NotImplementedError()

        sampler = StratifiedSampler(stratvar=stratvar, batch_size=batch_size, shuffle=shuffle)
        super().__init__(dataset=dataset, sampler=sampler, batch_size=batch_size, **kwargs)


class StratifiedDomainDataLoader(DataLoader):
    """
    라벨(또는 stratvar)뿐만 아니라,
    도메인별로 균형을 맞추면서 배치를 구성하는 DataLoader.
    - domains_per_batch개 도메인을 뽑아,
      각 도메인에서 samples_per_domain 개 샘플을 stratify하게 뽑음.
    - StratifiedDomainSampler를 이용.
    """

    def __init__(self, dataset=None, batch_size=1, domains_per_batch=1, shuffle=True, **kwargs):

        if isinstance(dataset, Subset) and isinstance(dataset.dataset, Subset) and isinstance(dataset.dataset.dataset, (
        DomainDataset, CachedDomainDataset)):
            # Subset(Subset(Dataset)) 구조
            domains = dataset.dataset.dataset.domains[dataset.dataset.indices][dataset.indices]
            labels = dataset.dataset.dataset.domains[dataset.dataset.indices][
                dataset.indices]  # ??? 여기서 labels에 domains를 할당한 것으로 보임 (오타 가능성)
        elif isinstance(dataset, Subset) and isinstance(dataset.dataset, (DomainDataset, CachedDomainDataset)):
            # Subset(Dataset) 구조
            domains = dataset.dataset.domains[dataset.indices]
            labels = dataset.dataset.domains[dataset.indices]  # ??? 동일 (labels 대신 domains를 사용)
        elif isinstance(dataset, (DomainDataset, CachedDomainDataset)):
            # 직접 Dataset
            domains = dataset.domains
            labels = dataset.labels
        else:
            raise NotImplementedError()

        # StratifiedDomainSampler 초기화
        sampler = StratifiedDomainSampler(
            domains, labels,
            int(batch_size / domains_per_batch),
            domains_per_batch,
            shuffle=shuffle
        )

        super().__init__(dataset=dataset, sampler=sampler, batch_size=batch_size, **kwargs)


def sample_without_replacement(domainlist, shuffle=True):
    """
    BalancedDomainSampler에서 replacement=False인 경우,
    도메인을 중복 없이 모두 한 번씩 샘플링하기 위한 헬퍼 함수.
    (도메인별 카운트가 남아있는 동안 루프)
    """
    du, counts = domainlist.unique(return_counts=True)
    dl = []
    while counts.sum() > 0:
        mask = counts > 0
        if shuffle:
            ixs = torch.randperm(du[mask].shape[0])
        else:
            ixs = range(du[mask].shape[0])
        counts[mask] -= 1
        dl.append(du[mask][ixs])
    return torch.cat(dl, dim=0)


class BalancedDomainSampler(torch.utils.data.Sampler[int]):
    """
    BalancedDomainDataLoader에서 사용.
    도메인별로 동일한 개수(samples_per_domain)만큼 샘플을 뽑도록 하는 샘플러.
    - self.domainlist: 도메인 번호를 나열
    - 도메인마다 (samples_per_domain) 배치 크기로 묶어서 반환
    """

    def __init__(self, domains, samples_per_domain: int, shuffle=False, replacement=True) -> None:
        super().__init__(domains)
        self.samples_per_domain = samples_per_domain
        self.shuffle = shuffle
        self.replacement = replacement

        du, didxs, counts = domains.unique(return_inverse=True, return_counts=True)
        du = du.tolist()  # 고유 도메인들
        didxs = didxs.tolist()  # 전체 샘플이 어떤 도메인에 속하는지
        counts = counts.tolist()

        # 각 도메인에 대해, samples_per_domain단위로 몇 번의 배치를 뽑을 수 있는지
        self.domainlist = torch.cat(
            [
                domain * torch.ones((counts[ix] // self.samples_per_domain), dtype=torch.long)
                for ix, domain in enumerate(du)
            ]
        )

        # 도메인 -> 샘플 인덱스 목록
        self.domaindict = {}
        for domix, domid in enumerate(du):
            self.domaindict[domid] = torch.LongTensor(
                [idx for idx, dom in enumerate(didxs) if dom == domix]
            )

    def __iter__(self) -> Iterator[int]:
        # 도메인 순서를 섞거나 유지
        if self.shuffle:
            if self.replacement:
                # 도메인 목록 랜덤 셔플
                permidxs = torch.randperm(self.domainlist.shape[0])
                domainlist = self.domainlist[permidxs]
            else:
                # sample_without_replacement를 사용해 순서 섞어서 모든 도메인 소진
                domainlist = sample_without_replacement(self.domainlist, shuffle=True)
        else:
            if self.replacement:
                domainlist = self.domainlist
            else:
                domainlist = sample_without_replacement(self.domainlist, shuffle=False)

        # 각 도메인별로 샘플 인덱스를 batch 단위로 생성
        generators = {}
        for domain in self.domaindict.keys():
            if self.shuffle:
                permidxs = torch.randperm(self.domaindict[domain].shape[0])
            else:
                permidxs = range(self.domaindict[domain].shape[0])
            # BatchSampler를 사용해 samples_per_domain 단위로 묶음
            generators[domain] = iter(
                torch.utils.data.BatchSampler(
                    self.domaindict[domain][permidxs].tolist(),
                    batch_size=self.samples_per_domain, drop_last=True
                )
            )

        # domainlist를 순회하며, 해당 도메인에서 next(batch)를 가져옴
        for item in domainlist.tolist():
            batch = next(generators[item])
            # batch에 포함된 인덱스들을 yield
            yield from batch

    def __len__(self) -> int:
        # domainlist에 기록된 배치 수 * 한 배치당 samples_per_domain
        return len(self.domainlist) * self.samples_per_domain


class StratifiedDomainSampler():
    """
    도메인별로, 라벨 stratify를 적용하면서 샘플링하는 샘플러.
    - domains, stratvar(=labels)로부터,
      domains_per_batch개 도메인을 뽑아,
      그 각 도메인에서 samples_per_domain개를 StratifiedSampler로 뽑아서 yield.
    """

    def __init__(self, domains, stratvar, samples_per_domain, domains_per_batch, shuffle=True) -> None:
        self.samples_per_domain = samples_per_domain
        self.domains_per_batch = domains_per_batch
        self.shuffle = shuffle
        self.stratvar = stratvar

        du, didxs, counts = domains.unique(return_inverse=True, return_counts=True)
        du = du.tolist()
        didxs = didxs.tolist()

        # 도메인별로 몇 번 배치를 만들 수 있는지
        self.domaincounts = torch.LongTensor((counts / self.samples_per_domain).tolist())

        # 각 도메인에 속하는 샘플 인덱스
        self.domaindict = {}
        for domix, _ in enumerate(du):
            self.domaindict[domix] = torch.LongTensor(
                [idx for idx, dom in enumerate(didxs) if dom == domix]
            )

    def __iter__(self) -> Iterator[int]:
        domaincounts = self.domaincounts.clone()

        # 도메인별 StratifiedSampler 생성
        generators = {}
        for domain in self.domaindict.keys():
            if self.shuffle:
                permidxs = torch.randperm(self.domaindict[domain].shape[0])
            else:
                permidxs = range(self.domaindict[domain].shape[0])

            # StratifiedSampler: 라벨이 골고루 나오도록 순서를 정해줌
            # self.stratvar[self.domaindict[domain]] -> 해당 도메인에 속한 샘플들의 라벨
            generators[domain] = iter(
                StratifiedSampler(
                    self.stratvar[self.domaindict[domain]],
                    batch_size=self.samples_per_domain,
                    shuffle=self.shuffle
                )
            )

        # 아직 뽑아야 할 도메인(도메인 카운트가 남아있는) 동안 반복
        while domaincounts.sum() > 0:
            assert (domaincounts >= 0).all()
            # 샘플링할 수 있는(카운트 >0) 도메인들의 인덱스
            candidates = torch.nonzero(domaincounts, as_tuple=False).flatten()

            if candidates.shape[0] < self.domains_per_batch:
                # 남아있는 도메인 수가 domains_per_batch보다 적으면 종료
                break

            # 도메인 순서를 셔플
            permidxs = torch.randperm(candidates.shape[0])
            candidates = candidates[permidxs]

            # 이번 배치에서 사용할 도메인들
            batchdomains = candidates[:self.domains_per_batch]

            # 각 도메인에서 samples_per_domain개를 stratified하게 뽑아 batch로 yield
            for item in batchdomains.tolist():
                within_domain_idxs = [next(generators[item]) for _ in range(self.samples_per_domain)]
                batch = self.domaindict[item][within_domain_idxs]
                domaincounts[item] = domaincounts[item] - 1
                yield from batch

        # 마지막에 아무것도 없으면 그냥 끝
        yield from []

    def __len__(self) -> int:
        # 전체 도메인 배치 횟수 * 한 배치당 샘플 수
        return self.domaincounts.sum() * self.samples_per_domain


class StratifiedSampler(torch.utils.data.Sampler[int]):
    """
    라벨(Stratvar)을 기준으로 계층적(Stratified)으로 샘플링.
    - skorch.BatchSampler처럼
      StratifiedKFold를 통해 라벨 별 비율이 유지되도록 인덱스 순서를 생성.
    """

    def __init__(self, stratvar, batch_size, shuffle=True):
        self.n_splits = max(int(stratvar.shape[0] / batch_size), 2)
        self.stratvar = stratvar
        self.shuffle = shuffle

    def gen_sample_array(self):
        # stratvar에 대해 StratifiedKFold 수행 -> 데이터를 n_splits로 나눠
        # 차례대로 테스트 인덱스를 이어붙여 하나의 순서 배열로 만든다
        s = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle)
        indices = [test for _, test in s.split(self.stratvar, self.stratvar)]
        # test 인덱스들을 순차적으로 연결(hstack)
        return np.hstack(indices)

    def __iter__(self):
        # 위에서 만든 인덱스 순서를 yield
        return iter(self.gen_sample_array())

    def __len__(self):
        # 전체 데이터 수
        return len(self.stratvar)
