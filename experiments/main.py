# %%
import os  # 운영 체제와 상호 작용(파일/디렉토리 경로 처리 등)에 사용
from time import time  # 시간 측정을 위해 사용 (학습에 걸린 시간 계산)
from hydra.core.hydra_config import HydraConfig  # Hydra 설정 객체에 직접 접근하기 위함
import pandas as pd  # 데이터프레임으로 결과를 정리하고 저장하기 위해 사용
from skorch.callbacks.scoring import EpochScoring  # Skorch 학습 중 에폭별로 지정된 스코어를 계산하기 위한 콜백
from skorch.dataset import ValidSplit  # Skorch에서 학습 데이터의 일부분을 검증용으로 분할하는 기능
from skorch.callbacks import Checkpoint  # 학습 중 모델 파라미터의 체크포인트(최적 스코어 시점)를 저장하기 위한 콜백

import logging  # 로그 기록을 위한 기본 파이썬 모듈
import hydra  # Hydra: 설정 파일로부터 인스턴스를 생성하고 관리할 때 사용
import torch  # PyTorch: 딥러닝 프레임워크
import numpy as np  # 수치 계산용 라이브러리
from omegaconf import DictConfig, OmegaConf, open_dict  # Hydra 설정 관리(딕트 형태)와 파일 입출력 관련

import moabb  # MOABB: EEG(뇌파) 관련 벤치마크 라이브러리 (실험 데이터셋과 프로토콜 제공)
from sklearn.metrics import get_scorer, make_scorer  # 분류/회귀용 스코어 함수를 가져오기 위해 사용
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold  # 데이터 분할(교차검증 등) 전략 제공
from library.utils.moabb import CachedParadigm  # MOABB 기반 커스텀 전처리/파라다임을 위한 클래스
from spdnets.models import DomainAdaptBaseModel, DomainAdaptJointTrainableModel, EEGNetv4  # TSMNet에서 사용하는 모델들
from spdnets.models import CPUModel  # CPU에서만 동작하는 간단한 모델 정의
# from library.utils.moabb.evaluation import extract_adapt_idxs
import warnings  # 경고 메시지를 제어하기 위해 사용
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning  # 학습 실패 및 수렴 관련 경고
from library.utils.torch import BalancedDomainDataLoader, CombinedDomainDataset, DomainIndex, StratifiedDomainDataLoader
    # TSMNet 및 도메인 적응 관련 데이터셋/로더 클래스
from spdnets.models.base import DomainAdaptFineTuneableModel, FineTuneableModel  # 파인튜닝 기능을 가진 모델들

from spdnets.utils.skorch import DomainAdaptNeuralNetClassifier  # Skorch를 상속받아 도메인 적응 기능을 넣은 분류기
from library.utils.hydra import hydra_helpers  # Hydra 설정에 필요한 헬퍼 함수들

# 로깅 객체 생성 (로그를 찍을 때 사용)
log = logging.getLogger(__name__)

# 재현 가능성을 위해 RNG(난수 발생기) 시드를 설정
rng_seed = 42

# 특정 종류의 경고를 무시 (학습 과정에서 발생할 수 있는 FitFailedWarning, ConvergenceWarning 등)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Hydra 설정을 초기화하는 데코레이터
@hydra_helpers  # 커스텀 헬퍼
@hydra.main(config_path='conf/', config_name='mims-nnets.yaml')  # config_path와 config_name을 통해 설정 파일 지정
def main(cfg: DictConfig):
    """
    Hydra로부터 설정(cfg)을 받아,
    MOABB와 skorch, PyTorch 등을 사용하여 EEG 도메인 적응 모델을 학습 및 평가하는 메인 함수.
    """
    moabb.set_log_level("info")  # MOABB의 로그 레벨 설정(정보 레벨만 표시)

    # __file__ 위치 기준으로 프로젝트 루트 디렉토리를 절대경로로 변환
    rootdir = hydra.utils.to_absolute_path(os.path.dirname(__file__))

    # GPU 사용 가능 여부 파악, Hydra가 제공하는 job num을 이용해 GPU 아이디 결정
    if torch.cuda.is_available():
        gpuid = f"cuda:{HydraConfig.get().job.get('num', 0) % torch.cuda.device_count()}"
        log.debug(f"GPU ID: {gpuid}")
        device = torch.device(gpuid)
    else:
        device = torch.device('cpu')
    cpu = torch.device('cpu')

    # Hydra 설정 중 nnet 내부에 ft_pipeline, prep_pipeline 항목이 없으면 None으로 초기화
    with open_dict(cfg):
        if 'ft_pipeline' not in cfg.nnet:
            cfg.nnet.ft_pipeline = None
        if 'prep_pipeline' not in cfg.nnet:
            cfg.nnet.prep_pipeline = None

    # 데이터셋 인스턴스화 (cfg.dataset.type에 정의된 클래스를 가져와 생성)
    dataset = hydra.utils.instantiate(cfg.dataset.type, _convert_='partial')

    # 전처리 관련 설정 불러오기
    ppreprocessing_dict = hydra.utils.instantiate(cfg.preprocessing, _convert_='partial')
    # 설정 파일 구조상, 한 번의 호출에는 하나의 파라다임만 있어야 하므로 가정
    assert(len(ppreprocessing_dict) == 1)
    # 전처리 파이프라인 이름과 paradigm(파라다임) 객체를 하나 뽑아옴
    prep_name, paradigm = next(iter(ppreprocessing_dict.items()))

    # 결과 저장 디렉토리 설정 (evaluation.strategy와 prep_name을 조합)
    res_dir = os.path.join(cfg.evaluation.strategy, prep_name)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # 실험 결과를 저장할 pandas DataFrame 초기화
    results = pd.DataFrame(
        columns=['dataset', 'subject', 'session', 'method', 'score_trn', 'score_tst',
                 'time', 'n_test', 'classes'])
    resix = 0  # 결과 DataFrame에 행을 추가할 때 사용할 인덱스

    # 각 컬럼의 데이터 타입 미리 지정
    results['score_trn'] = results['score_trn'].astype(np.double)
    results['score_tst'] = results['score_tst'].astype(np.double)
    results['time'] = results['time'].astype(np.double)
    results['n_test'] = results['n_test'].astype(int)
    results['classes'] = results['classes'].astype(int)

    # 학습 과정의 중간(에폭별) 로그를 저장할 리스트
    results_fit = []

    # cfg.score에 해당하는 스코어 함수를 가져옴 (예: accuracy_score 등)
    scorefun = get_scorer(cfg.score)._score_func

    # -1로 마스킹된 타겟(라벨)은 계산에서 제외하는 스코어 함수를 정의
    def masked_scorefun(y_true, y_pred, **kwargs):
        masked = y_true == -1  # 라벨이 -1인 경우를 마스킹
        if np.all(masked):  # 전부 마스킹이면 점수를 낼 수 없음
            log.warning('Nothing to score because all target values are masked (value = -1).')
            return np.nan
        return scorefun(y_true[~masked], y_pred[~masked], **kwargs)

    # 위에서 만든 함수를 make_scorer로 감싸 최종 Scorer 생성
    scorer = make_scorer(masked_scorefun)

    # 도메인 적응 정보(dadapt)를 설정파일로부터 가져옴
    dadapt = cfg.evaluation.adapt

    # Epoch 단위로 train/validation 스코어를 계산하여 로그로 남기는 콜백
    bacc_val_logger = EpochScoring(scoring=scorer, lower_is_better=False, on_train=False, name='score_val')
    bacc_trn_logger = EpochScoring(scoring=scorer, lower_is_better=False, on_train=True, name='score_trn')

    # 평가 전략(evaluation.strategy)에 따라 세션 단위, 혹은 서브젝트 단위로 나누어 학습
    # 예: inter-session => 세션별로 나누어 train/test, inter-subject => 피험자별로 나누어 train/test
    if 'inter-session' in cfg.evaluation.strategy:
        subset_iter = iter([[s] for s in dataset.subject_list])  # 세션 기준 분할
        groupvarname = 'session'
    elif 'inter-subject' in cfg.evaluation.strategy:
        subset_iter = iter([None])  # 서브젝트 기준 분할
        groupvarname = 'subject'
    else:
        raise NotImplementedError()

    # session 정보를 설정 파일에서 가져올 수 있다면 가져옴 (없으면 None)
    selected_sessions = cfg.dataset.get("sessions", None)

    # subset_iter를 순회하면서 실험 진행
    for subset in subset_iter:

        # groupvarname='session'이면 domain_expression에 session을, 'subject'면 session + subject * 1000 형태로 지정
        if groupvarname == 'session':
            domain_expression = "session"
        elif groupvarname == 'subject':
            domain_expression = "session + subject * 1000"

        # MOABB 데이터셋에서 도메인(index)까지 합쳐서 데이터셋을 생성 (CombinedDomainDataset)
        ds = CombinedDomainDataset.from_moabb(
            paradigm, dataset, subjects=subset, domain_expression=domain_expression,
            dtype=cfg.nnet.inputtype, sessions=selected_sessions
        )

        # prep_pipeline이 있으면 전체 데이터를 메모리에 캐싱 (전처리 수행을 위해 필요)
        if cfg.nnet.prep_pipeline is not None:
            ds = ds.cache()  # 전체 로드를 위해 캐싱

        # session, subject, group 등의 정보를 NumPy array로 추출
        sessions = ds.metadata.session.astype(np.int64).values
        subjects = ds.metadata.subject.astype(np.int64).values

        # groupvarname에 따른 그룹 벡터를 생성 (예: 세션별 or 서브젝트별)
        g = ds.metadata[groupvarname].astype(np.int64).values
        groups = np.unique(g)  # 그룹(세션 혹은 서브젝트) 고유 값들

        # 도메인 인덱스 목록
        domains = ds.domains.unique()

        # 클래스 개수
        n_classes = len(ds.labels.unique())

        # 그룹 개수가 2개 미만이면(1개 이하), 교차검증 불가
        if len(groups) < 2:
            log.warning(f"Insufficient number (n={len(groups)}) of groups ({groupvarname}) in the (sub-)dataset to run leave 1 group out CV!")
            continue

        # 모델에 필요한 파라미터를 담을 딕셔너리 생성
        mdl_kwargs = dict(nclasses=n_classes)

        # 데이터셋 차원 정보를 모델 파라미터에 할당
        mdl_kwargs['nchannels'] = ds.shape[1]
        mdl_kwargs['nsamples'] = ds.shape[2]
        mdl_kwargs['nbands'] = ds.shape[3] if ds.ndim == 4 else 1
        mdl_kwargs['input_shape'] = (1,) + ds.shape[1:]

        # Hydra 설정 파일에서 모델 관련 정보를 꺼냄
        mdl_dict = OmegaConf.to_container(cfg.nnet.model, resolve=True)
        # _target_에 정의된 클래스를 가져옴 (예: EEGNetv4 등)
        mdl_class = hydra.utils.get_class(mdl_dict.pop('_target_'))

        # DomainAdaptBaseModel의 하위 클래스면 domains 파라미터를 함께 넣어줌
        if issubclass(mdl_class, DomainAdaptBaseModel):
            mdl_kwargs['domains'] = domains

        # EEGNetv4 클래스를 사용하는 경우, srate(샘플링 레이트) 설정이 필요
        if issubclass(mdl_class, EEGNetv4):
            if isinstance(paradigm, CachedParadigm):
                info = paradigm.get_info(dataset)
                mdl_kwargs['srate'] = int(info['sfreq'])
            else:
                raise NotImplementedError()

        # FineTuneableModel의 하위 클래스이고, CombinedDomainDataset이면 캐싱되어야 함
        if issubclass(mdl_class, FineTuneableModel) and isinstance(ds, CombinedDomainDataset):
            ds = ds.cache()  # 전체 로드

        # 모델 파라미터를 최종적으로 dict 형태로 업데이트
        mdl_kwargs = {**mdl_kwargs, **mdl_dict}

        # Optimizer 관련 설정 불러오기
        optim_kwargs = OmegaConf.to_container(cfg.nnet.optimizer, resolve=True)
        optim_class = hydra.utils.get_class(optim_kwargs.pop('_target_'))

        # 모델 메타데이터(어떤 모델, 어떤 옵티마 등)를 저장하기 위한 dict
        metaddata = {
            'model_class': mdl_class,
            'model_kwargs': mdl_kwargs,
            'optim_class': optim_class,
            'optim_kwargs': optim_kwargs
        }

        # 모델 메타데이터 및 Hydra 설정을 저장해놓을 디렉토리
        mdl_metadata_dir = os.path.join(res_dir, 'metadata')
        if not os.path.exists(mdl_metadata_dir):
            os.makedirs(mdl_metadata_dir)

        # torch 형태로 메타데이터 저장
        torch.save(metaddata, f=os.path.join(mdl_metadata_dir, f'meta-{cfg.nnet.name}.pth'))
        # yaml 형태로 Hydra 전체 설정 파일도 저장
        with open(os.path.join(mdl_metadata_dir, f'config-{cfg.nnet.name}.yaml'), 'w+') as f:
            f.writelines(OmegaConf.to_yaml(cfg))

        # CPU 전용 모델일 경우, device를 CPU로 강제 고정
        if issubclass(mdl_class, CPUModel):
            device = cpu

        # 최종 모델 파라미터에 device 항목 추가
        mdl_kwargs['device'] = device

        # leave x group out 방식 (test_size 비율만큼 그룹을 테스트로)
        n_test_groups = int(np.clip(np.round(len(groups) * cfg.fit.test_size), 1, None))
        log.info(f"Performing leave {n_test_groups} (={cfg.fit.test_size*100:.0f}%) {groupvarname}(s) out CV")

        # 그룹 단위로 KFold 분할
        cv = GroupKFold(n_splits=int(len(groups) / n_test_groups))

        # labels를 모두 노출한 상태 (eval)로 변경
        ds.eval()

        # train/test 그룹을 분할
        for train, test in cv.split(ds.labels, ds.labels, g):

            # target 도메인만 추출
            target_domains = ds.domains[test].unique().numpy()
            # 도메인마다 다른 시드값으로 실험 가능하도록
            torch.manual_seed(rng_seed + target_domains[0])

            # 모델에 적용할 전처리, 파인튜닝 파이프라인 가져오기
            prep_pipeline = hydra.utils.instantiate(cfg.nnet.prep_pipeline, _convert_='partial')
            ft_pipeline = hydra.utils.instantiate(cfg.nnet.ft_pipeline, _convert_='partial')

            # 도메인 적응이 설정된 경우(예: sda, uda 등)
            if dadapt is not None and dadapt.name != 'no':
                # JointTrainableModel이면 target domain 데이터를 학습에 포함
                if issubclass(mdl_class, DomainAdaptJointTrainableModel):
                    # stratvar = ds.labels + ds.domains * n_classes
                    # adapt_domain = extract_adapt_idxs(dadapt.nadapt_domain, test, stratvar)
                    # 위 라인은 주석처리되어 있으므로, 실제로는 아래와 같은 방식의 대체가 필요
                    adapt_domain = test
                else:
                    # 그 외 모델은 target domain 학습 불필요하다고 가정
                    adapt_domain = np.array([], dtype=np.int64)
                    log.info("Model does not require adaptation. Using original training data.")

                # train_source_doms는 소스(원천) 도메인 학습 인덱스 (적응 전)
                train_source_doms = train
                # 실제 train 인덱스에는 테스트 도메인의 적응 데이터도 추가
                train = np.concatenate((train, adapt_domain))

                # uda(unsupervised domain adaptation)이면 target domain 라벨을 마스킹
                if dadapt.name == 'uda':
                    ds.set_masked_labels(adapt_domain)
                # sda(supervised domain adaptation)이면 test셋에서 적응에 사용된 부분 제외
                elif dadapt.name == 'sda':
                    test = np.setdiff1d(test, adapt_domain)

                if len(test) == 0:
                    raise ValueError('No data left in the test set!')
            else:
                train_source_doms = train

            # test 그룹 정보를 다시 정리하여 subject, session별로 구분
            test_groups = np.unique(g[test])
            test_group_list = []
            for test_group in test_groups:
                test_dict = {}
                subject = np.unique(subjects[g == test_group])
                assert(len(subject) == 1)  # 한 그룹에는 한 피험자만 들어있어야 함
                test_dict['subject'] = subject[0]

                if groupvarname == 'subject':
                    test_dict['session'] = -1  # subject 기준이면 session 구분 의미 x
                else:
                    session = np.unique(sessions[g == test_group])
                    assert(len(session) == 1)
                    test_dict['session'] = session[0]

                test_dict['idxs'] = np.intersect1d(test, np.nonzero(g == test_group))
                test_group_list.append(test_dict)

            # 학습 및 추론 시간 측정 시작
            t_start = time()

            ## ------------ 전처리 파이프라인 적용 -------------
            dsprep = ds.copy(deep=False)  # (features, labels, domains) 구조만 복사, 실제 데이터는 공유
            dsprep.train()  # 라벨을 다시 마스크 (fit 전처리 시, 라벨 누출 방지)

            if prep_pipeline is not None:
                # train 인덱스 부분으로 전처리 파이프라인을 학습
                prep_pipeline.fit(dsprep.features[train].numpy(), dsprep.labels[train])
                # 전처리된 결과를 dsprep에 업데이트
                dsprep.set_features(prep_pipeline.transform(dsprep.features))

            # 검증셋의 batch size 계산
            batch_size_valid = (cfg.fit.validation_size if isinstance(cfg.fit.validation_size, int)
                                else int(np.ceil(cfg.fit.validation_size * len(train))))

            # 전처리 파이프라인을 거친 뒤 라벨을 공개 (eval) 상태로 바꿈
            dsprep.eval()

            # train의 도메인+클래스 조합을 stratvar로 사용 (stratify 위해)
            stratvar = dsprep.labels[train] + dsprep.domains[train] * n_classes

            # validation split 설정 (train 내부에서 일부를 valid로)
            valid_cv = ValidSplit(
                iter(StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=cfg.fit.validation_size,
                    random_state=rng_seed + target_domains[0]
                ).split(stratvar, stratvar))
            )

            # skorch(Net) 초기화 시 전달해야 하는 파라미터를 구성
            netkwargs = {'module__' + k: v for k, v in mdl_kwargs.items()}
            netkwargs = {**netkwargs, **{'optimizer__' + k: v for k, v in optim_kwargs.items()}}

            # stratified 옵션이 켜져 있으면 StratifiedDomainDataLoader를 사용
            if cfg.fit.stratified:
                n_train_domains = len(dsprep.domains[train].unique())
                domains_per_batch = min(cfg.fit.domains_per_batch, n_train_domains)
                batch_size_train = int(max(np.round(cfg.fit.batch_size_train / domains_per_batch), 2) * domains_per_batch)

                netkwargs['iterator_train'] = StratifiedDomainDataLoader
                netkwargs['iterator_train__domains_per_batch'] = domains_per_batch
                netkwargs['iterator_train__shuffle'] = True
                netkwargs['iterator_train__batch_size'] = batch_size_train
            else:
                # 그렇지 않으면 BalancedDomainDataLoader 사용
                netkwargs['iterator_train'] = BalancedDomainDataLoader
                netkwargs['iterator_train__domains_per_batch'] = cfg.fit.domains_per_batch
                netkwargs['iterator_train__drop_last'] = True
                netkwargs['iterator_train__replacement'] = False
                netkwargs['iterator_train__batch_size'] = cfg.fit.batch_size_train

            netkwargs['iterator_valid__batch_size'] = batch_size_valid
            netkwargs['max_epochs'] = cfg.fit.epochs
            # skorch의 print_log 콜백 로그에 prefix를 달아서 보기 쉽게
            netkwargs['callbacks__print_log__prefix'] = f'{dataset.code} {n_classes}cl | {test_groups} | {cfg.nnet.name} :'

            # 모델 파일 임시 저장 경로
            mdl_path_tmp = os.path.join(res_dir, 'models', 'tmp', f'{test_groups}_{cfg.nnet.name}.pth')
            if not os.path.exists(os.path.split(mdl_path_tmp)[0]):
                os.makedirs(os.path.split(mdl_path_tmp)[0])

            # 체크포인트 콜백: 검증 손실(valid_loss)이 가장 좋을 때 모델 파라미터를 저장, 학습 끝나면 로드
            checkpoint = Checkpoint(
                f_params=mdl_path_tmp,
                f_criterion=None,
                f_optimizer=None,
                f_history=None,
                monitor='valid_loss_best',
                load_best=True
            )

            # 학습률 스케줄러
            scheduler = hydra.utils.instantiate(cfg.nnet.scheduler, _convert_='partial')

            # DomainAdaptNeuralNetClassifier: skorch.NeuralNetClassifier를 상속 + 도메인 적응 기능 포함
            net = DomainAdaptNeuralNetClassifier(
                mdl_class,
                train_split=valid_cv,
                callbacks=[bacc_trn_logger, bacc_val_logger, scheduler, checkpoint],
                optimizer=optim_class,
                verbose=0,
                device=device,
                **netkwargs
            )

            # 학습 데이터를 Subset으로 만들기 (train 인덱스만 골라냄)
            dsprep.train()  # 라벨 마스킹
            dstrn = torch.utils.data.Subset(dsprep, train)

            # 실제 학습 진행 (fit)
            net.fit(dstrn, None)

            # 에폭별 결과(history) DataFrame 추출, 일부 열은 제거
            res = pd.DataFrame(net.history)
            res = res.drop(res.filter(regex='.*batches|_best|_count').columns, axis=1)
            res = res.drop(res.filter(regex='event.*').columns, axis=1)
            # 컬럼명 정리
            res = res.rename(columns=dict(train_loss="loss_trn", valid_loss="loss_val", dur="time"))
            res['domains'] = str(test_groups)
            res['method'] = cfg.nnet.name
            res['dataset'] = dataset.code
            results_fit.append(res)

            # 모델을 최적 epoch로부터 로딩 완료된 상태
            # uda(비지도)라면, DomainAdaptFineTuneableModel의 domainadapt_finetune을 수행
            if cfg.evaluation.adapt.name == "uda":
                if isinstance(net.module_, DomainAdaptFineTuneableModel):
                    dsprep.train()  # 타겟 도메인 라벨 마스킹
                    for du in dsprep.domains.unique():
                        domain_data = dsprep[DomainIndex(du.item())]
                        net.module_.domainadapt_finetune(x=domain_data[0]['x'], y=domain_data[1], d=domain_data[0]['d'], target_domains=target_domains)
            elif cfg.evaluation.adapt.name == "no":
                # 도메인 적응이 전혀 없더라도, FineTuneableModel이면 기본 finetune 시도
                if isinstance(net.module_, FineTuneableModel):
                    dsprep.train()
                    net.module_.finetune(x=dsprep.features[train], y=dsprep.labels[train], d=dsprep.domains[train])

            # 학습에 걸린 시간 계산
            duration = time() - t_start

            # 최종 모델(.pth)을 주어진 경로에 저장 (각 테스트 그룹별로)
            for test_group in test_group_list:
                mdl_path = os.path.join(res_dir, 'models', f'{test_group["subject"]}', f'{test_group["session"]}', f'{cfg.nnet.name}.pth')
                if not os.path.exists(os.path.split(mdl_path)[0]):
                    os.makedirs(os.path.split(mdl_path)[0])
                net.save_params(f_params=mdl_path)

            ## ------------------- 평가 -------------------
            dsprep.eval()  # 테스트 시점이므로 라벨 노출

            # 예측값 배열과 latent space(특징 벡터) 배열 준비
            y_hat = np.empty(dsprep.labels.shape)
            # latent space 하나를 먼저 forward 해 보고 차원을 확인
            _, l0 = net.forward(dsprep[DomainIndex(dsprep.domains[0])][0])
            l = np.empty((len(dsprep),) + l0.shape[1:])

            # 각 도메인마다 순차적으로 예측
            for du in dsprep.domains.unique():
                ixs = np.flatnonzero(dsprep.domains == du)
                domain_data = dsprep[DomainIndex(du)]

                # 모델 순전파 (출력 로짓과 latent)
                y_hat_domain, l_domain, *_ = net.forward(domain_data[0])
                # 예측 라벨은 argmax로 뽑고, latent는 CPU로 옮겨 numpy 변환
                y_hat_domain = y_hat_domain.numpy().argmax(axis=1)
                l_domain = l_domain.to(device=cpu).numpy()

                y_hat[ixs] = y_hat_domain
                l[ixs] = l_domain

            # 소스(원천) 도메인에 대해서만 학습 스코어 계산
            score_trn = scorefun(dsprep.labels[train_source_doms], y_hat[train_source_doms])

            # test 그룹별로 테스트 스코어 계산
            for test_group in test_group_list:
                score_tst = scorefun(dsprep.labels[test_group["idxs"]], y_hat[test_group["idxs"]])

                # 결과 DataFrame에 기록
                res = pd.DataFrame({
                    'dataset': dataset.code,
                    'subject': test_group["subject"],
                    'session': test_group["session"],
                    'method': cfg.nnet.name,
                    'score_trn': score_trn,
                    'score_tst': score_tst,
                    'time': duration,
                    'n_test': len(test),
                    'classes': n_classes
                }, index=[resix])

                results = results.append(res)
                resix += 1

                # 로그에 출력
                r = res.iloc[0, :]
                log.info(f'{r.dataset} {r.classes}cl | {r.subject} | {r.session} | {r.method} :       trn={r.score_trn:.2f} tst={r.score_tst:.2f}')

            ## ------------------- 파인튜닝 (FT Pipeline) -------------------
            if ft_pipeline is not None:
                # latent space를 기반으로 추가적인 학습(FT: Fine Tuning)
                dsprep.train()  # 라벨 마스킹 (fit 시점)
                ft_pipeline.fit(l[train], dsprep.labels[train])

                # 예측
                y_hat_ft = ft_pipeline.predict(l)

                # 트레이닝 스코어
                dsprep.eval()
                ft_score_trn = scorefun(dsprep.labels[train_source_doms], y_hat_ft[train_source_doms])

                # 테스트 스코어
                for test_group in test_group_list:
                    ft_score_tst = scorefun(dsprep.labels[test_group["idxs"]], y_hat_ft[test_group["idxs"]])

                    res = pd.DataFrame({
                        'dataset': dataset.code,
                        'subject': test_group["subject"],
                        'session': test_group["session"],
                        'method': f'{cfg.nnet.name}+FT',
                        'score_trn': ft_score_trn,
                        'score_tst': ft_score_tst,
                        'time': duration,
                        'n_test': len(test),
                        'classes': n_classes
                    }, index=[resix])

                    results = results.append(res)
                    resix += 1

                    r = res.iloc[0, :]
                    log.info(f'{r.dataset} {r.classes}cl | {r.subject} | {r.session} | {r.method} :    trn={r.score_trn:.2f} tst={r.score_tst:.2f}')

    # 모든 fold의 epoch별 학습 로그를 취합하여 CSV로 저장
    if len(results_fit):
        results_fit = pd.concat(results_fit)
        results_fit['preprocessing'] = prep_name
        results_fit['evaluation'] = cfg.evaluation.strategy
        results_fit['adaptation'] = cfg.evaluation.adapt.name

        for method in results_fit['method'].unique():
            method_res = results[results['method'] == method]
            results_fit.to_csv(os.path.join(res_dir, f'nnfitscores_{method}.csv'), index=False)

    # 최종 test 스코어들을 CSV로 저장
    if len(results) > 0:
        results['preprocessing'] = prep_name
        results['evaluation'] = cfg.evaluation.strategy
        results['adaptation'] = cfg.evaluation.adapt.name

        for method in results['method'].unique():
            method_res = results[results['method'] == method]
            method_res.to_csv(os.path.join(res_dir, f'scores_{method}.csv'), index=False)

        # 그룹화하여 평균/표준편차를 표시
        print(results.groupby('method').agg(['mean', 'std']))

# 메인 함수 실행
if __name__ == '__main__':
    main()
