import logging
import re
import json
import os
import hashlib
import numpy as np
import pandas as pd
import mne

from sklearn.base import BaseEstimator
from moabb.paradigms.base import BaseParadigm
from moabb.paradigms.motor_imagery import FilterBankMotorImagery, MotorImagery
from mne import get_config, set_config
from mne.datasets.utils import _get_path
from mne.io import read_info, write_info
from skorch import dataset

# 로깅 객체 생성 (로그 찍을 때 사용)
log = logging.getLogger(__name__)

class CachedParadigm(BaseParadigm):
    """
    MOABB의 BaseParadigm을 상속하여,
    - 전처리/필터링/에포칭 결과를 디스크에 캐싱해두고
    - 동일한 설정일 경우(같은 dataset, 같은 필터, 같은 파라다임 등)
      이미 계산된 결과를 재사용하는 클래스
    """

    def _get_string_rep(self, obj):
        """
        주어진 객체(obj)가 BaseEstimator(주로 sklearn 변환기나 분류기)라면,
        get_params()를 repr로 만든다.
        그렇지 않으면 그냥 repr(obj)를 사용.

        이 때, 0x(주소) 부분은 모두 0x__ 로 바꿔주고,
        줄바꿈 문자는 없앤다.
        """
        if issubclass(type(obj), BaseEstimator):
            str_repr = repr(obj.get_params())
        else:
            str_repr = repr(obj)
        str_no_addresses = re.sub("0x[a-z0-9]*", "0x__", str_repr)
        return str_no_addresses.replace("\n", "")

    def _get_rep(self, dataset):
        """
        해당 dataset과 (self)Paradigm 객체를 함께 문자열로 표현(repr).
        이 문자열을 해시(sha1)하면 캐시 디렉토리를 구분짓는 기준이 됨.
        """
        return self._get_string_rep(dataset) + '\n' + self._get_string_rep(self)

    def _get_cache_dir(self, rep):
        """
        해시된 문자열(rep)을 바탕으로
        캐시를 저장할 디렉토리를 결정한다.
        (기본적으로는 ~user/mne_data/preprocessed/<sha1digest>)
        """
        # MNEDATASET_TMP_DIR이 설정되지 않았다면,
        # 사용자 홈 디렉토리 아래 mne_data를 기본값으로 설정
        if get_config("MNEDATASET_TMP_DIR") is None:
            set_config("MNEDATASET_TMP_DIR", os.path.join(os.path.expanduser("~"), "mne_data"))

        # MNE가 사용하는 임시 디렉토리 경로
        base_dir = _get_path(None, "MNEDATASET_TMP_DIR", "preprocessed")

        # rep(문자열)을 sha1 해시
        digest = hashlib.sha1(rep.encode("utf8")).hexdigest()

        cache_dir = os.path.join(
            base_dir,
            "preprocessed",
            digest
        )
        return cache_dir

    def process_raw(self, raw, dataset, return_epochs=False):
        """
        BaseParadigm의 핵심 전처리 로직.
        - 이벤트 추출
        - 필터링(필요하다면, bandpass)
        - 에포칭(Epochs)
        - (옵션) 리샘플링
        - 데이터 배열(X), 라벨(labels), 메타데이터 반환

        여기서는 디스크 캐시를 만들지는 않고,
        실제 연산을 수행하는 로직만 담당.
        """
        # 이벤트 id (사용할 이벤트 종류)
        event_id = self.used_events(dataset)

        # 이벤트 검출.
        # 일반적이면 mne.find_events(raw) 이용,
        # stim 채널이 없으면 annotations에서 이벤트 읽기
        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if len(stim_channels) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

        # EEG 채널만 선택
        if self.channels is None:
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
        else:
            picks = mne.pick_types(raw.info, stim=False, include=self.channels)

        # 쓰고자 하는 이벤트만 필터링
        try:
            events = mne.pick_events(events, include=list(event_id.values()))
        except RuntimeError:
            # 해당 이벤트가 없으면 None 반환
            return

        # 시간 구간 설정 (MOABB dataset에서 interval 가져오고, tmin/tmax 더하기)
        tmin = self.tmin + dataset.interval[0]
        if self.tmax is None:
            tmax = dataset.interval[1]
        else:
            tmax = self.tmax + dataset.interval[0]

        X = []
        # self.filters: 예) 여러 개의 (fmin, fmax) 밴드패스 튜플을 담고 있음
        for bandpass in self.filters:
            fmin, fmax = bandpass
            # 밴드패스가 (None, None)이면 원본 raw 그대로 사용
            if fmin is None and fmax is None:
                raw_f = raw
            else:
                # copy 후 필터 적용
                raw_f = raw.copy().filter(fmin, fmax, method='iir', picks=picks, verbose=False)

            # Epochs 생성
            epochs = mne.Epochs(raw_f, events, event_id=event_id,
                                tmin=tmin, tmax=tmax, proj=False,
                                baseline=None, preload=True,
                                verbose=False, picks=picks,
                                event_repeated='drop',
                                on_missing='ignore')

            # (옵션) 리샘플링
            if self.resample is not None:
                epochs = epochs.resample(self.resample)

            # return_epochs=False => 실제 numpy 데이터만 반환
            if return_epochs:
                X.append(epochs)
            else:
                X.append(dataset.unit_factor * epochs.get_data())

        # 이벤트 id <-> class name 간 역매핑
        inv_events = {k: v for v, k in event_id.items()}
        # 레이블 벡터 생성
        labels = np.array([inv_events[e] for e in epochs.events[:, -1]])

        # 필터가 여러 개면 (n_epochs, n_ch, n_samples, n_bands) 형태가 됨
        if len(self.filters) == 1:
            X = X[0]
        else:
            X = np.array(X).transpose((1, 2, 3, 0))

        # 메타데이터(여기서는 빈 DataFrame, index는 레이블 개수만큼)
        metadata = pd.DataFrame(index=range(len(labels)))
        return X, labels, metadata

    def get_data(self, dataset, subjects=None, return_epochs=False):
        """
        BaseParadigm의 get_data를 오버라이드.
        - 캐시 디렉토리를 확인하고,
        - (없으면) super().get_data(...)를 호출해 계산 후 저장
        - (있으면) 계산된 파일(.npy, .csv)을 로드
        - 최종적으로 X, labels, metadata를 반환
        """

        # 현재 cache 방식은 return_epochs=False만 지원
        if return_epochs:
            raise ValueError("Only return_epochs=False is supported.")

        # 데이터셋 + 파라다임 표현 문자열
        rep = self._get_rep(dataset)
        # 해당 표현(rep)에 대한 캐시 디렉토리 경로
        cache_dir = self._get_cache_dir(rep)
        os.makedirs(cache_dir, exist_ok=True)

        # X, labels, metadata 초기화
        X = [] if return_epochs else np.array([])
        labels = []
        metadata = pd.Series([])

        if subjects is None:
            subjects = dataset.subject_list

        # 캐시를 표현할 repr.json 파일이 없으면 만들어둔다
        if not os.path.isfile(os.path.join(cache_dir, 'repr.json')):
            with open(os.path.join(cache_dir, 'repr.json'), 'w+') as f:
                f.write(self._get_rep(dataset))

        # subject 별로 반복
        for subject in subjects:
            # 캐시에 해당 subject의 데이터가 없다면(super().get_data로 실제 계산)
            if not os.path.isfile(os.path.join(cache_dir, f'{subject}.npy')):
                x, lbs, meta = super().get_data(dataset, [subject], return_epochs)
                # numpy array로 저장
                np.save(os.path.join(cache_dir, f'{subject}.npy'), x)
                # csv로 label과 기타 메타정보 저장
                meta['label'] = lbs
                meta.to_csv(os.path.join(cache_dir, f'{subject}.csv'), index=False)
                log.info(f'saved cached data in directory {cache_dir}')
            else:
                log.info(f'loading cached data from directory {cache_dir}')

            # load from cache
            x = np.load(os.path.join(cache_dir, f'{subject}.npy'), mmap_mode='r')
            meta = pd.read_csv(os.path.join(cache_dir, f'{subject}.csv'))
            lbs = meta['label'].tolist()

            # return_epochs=False면 X는 np.array를 가로로 누적
            if return_epochs:
                X.append(x)
            else:
                # X가 비어있다면(첫 subject) -> 그냥 대입
                # 이후부터는 np.append로 연결
                X = np.append(X, x, axis=0) if len(X) else x

            labels = np.append(labels, lbs, axis=0)
            metadata = pd.concat([metadata, meta], ignore_index=True)

        return X, labels, metadata

    def get_info(self, dataset):
        """
        MNE Epochs에서 info(채널 수, 샘플링 레이트, 채널명 등)를 추출해
        <cache_dir>/raw-info.fif로 저장하고 재사용.
        """
        rep = self._get_rep(dataset)
        cache_dir = self._get_cache_dir(rep)
        os.makedirs(cache_dir, exist_ok=True)

        info_file = os.path.join(cache_dir, f'raw-info.fif')

        if not os.path.isfile(info_file):
            # 없는 경우 -> 첫 번째 subject 데이터를 로드하여 info를 가져옴
            x, _, _ = super().get_data(dataset, [dataset.subject_list[0]], True)
            info = x.info
            # fif 파일로 저장
            write_info(info_file, info)
            log.info(f'saved cached info in directory {cache_dir}')
        else:
            # 이미 있으면 로드
            log.info(f'loading cached info from directory {cache_dir}')
            info = read_info(info_file)
        return info

    def __repr__(self) -> str:
        """
        json.dumps 형태로 객체 내부 dict를 문자열화.
        """
        return json.dumps({self.__class__.__name__: self.__dict__})


class CachedMotorImagery(CachedParadigm, MotorImagery):
    """
    MotorImagery + CachedParadigm 다중 상속
    (BaseMotorImagery 파라다임을 그대로 쓰면서 캐시 기능 추가)
    """

    def __init__(self, **kwargs):
        # events 항목(움직임 손, 발 등)의 길이를 기준으로 n_classes 설정
        n_classes = len(kwargs['events'])
        super().__init__(n_classes=n_classes, **kwargs)


class CachedFilterBankMotorImagery(CachedParadigm, FilterBankMotorImagery):
    """
    FilterBankMotorImagery + CachedParadigm 다중 상속
    (여러 밴드패스를 사용하는 motor imagery 파라다임 + 캐시 기능)
    """

    def __init__(self, **kwargs):
        n_classes = len(kwargs['events'])
        super().__init__(n_classes=n_classes, **kwargs)
