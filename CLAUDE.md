# DuBLoNet

## 프로젝트 목적

DuBLoNet 음성 향상(speech enhancement) 모델에 대한 학회 논문 작성을 위한 학습 및 실험 프로젝트.
- **모델**: Backbone — 주파수 도메인 기반 causal speech enhancement 모델
- **핵심 연구**: 비대칭 convolution의 padding ratio를 조절하여 algorithmic latency를 제어하고, state buffer + lookahead buffer 기반 streaming 추론 구현
- **데이터셋**: VoiceBank-DEMAND 16kHz (HuggingFace: `JacobLinCool/VoiceBank-DEMAND-16k`)
- **평가 지표**: PESQ, STOI, CSIG, CBAK, COVL, segSNR

## 코드베이스 구성

```
dublonet/
├── conf/
│   └── config.yaml              # Hydra 학습 설정 (모델, 데이터셋, 손실함수, 최적화)
├── src/
│   ├── train.py                 # 학습 엔트리포인트 (Hydra)
│   ├── evaluate.py              # 평가 스크립트 (PESQ/STOI 등 메트릭 산출)
│   ├── enhance.py               # 음성 향상 및 wav 저장 (streaming 옵션 포함)
│   ├── solver.py                # 학습 루프 (MetricGAN discriminator 포함)
│   ├── data.py                  # VoiceBankDataset, 세그먼트 샘플링
│   ├── stft.py                  # mag-phase STFT/iSTFT 유틸리티
│   ├── compute_metrics.py       # PESQ, STOI 등 메트릭 계산
│   ├── pcs400.py                # Perceptual Contrast Stretching
│   ├── receptive_field.py       # 수용장(receptive field) 계산기
│   ├── utils.py                 # 모델 로드, 체크포인트, 로깅 유틸리티
│   ├── models/
│   │   ├── backbone.py         # Backbone 모델 정의 (CausalConv, DenseEncoder, TSBlock 등)
│   │   ├── discriminator.py     # MetricGAN Discriminator
│   │   └── streaming/           # 스트리밍 추론 관련 모듈
│   │       ├── layers/          # StatefulConv, ReshapeFree 등 스트리밍 레이어
│   │       ├── converters/      # 일반 모델 → 스트리밍 모델 변환기 (conv, reshape_free)
│   │       ├── core/            # 상태 관리, 모델 빌더
│   │       ├── wrappers/        # DuBLoNet (STFT overlap-add + lookahead buffering)
│   │       └── onnx/            # ONNX 내보내기용 stateful wrapper
├── results/
│   └── experiments/             # Hydra 실험 출력 (체크포인트, 로그, TensorBoard)
├── paper_works/                 # 논문 작성 관련 문서
└── requirements.txt             # 의존성 목록
```

## 주요 커맨드

### 학습
```bash
python -m src.train                                    # 기본 설정으로 학습
python -m src.train model.norm_type=batch               # BatchNorm으로 학습
python -m src.train model.causal=true model.encoder_padding_ratio=[0.5,0.5]  # padding ratio 조절
python -m src.train hydra.run.dir=./results/experiments/my_exp  # 출력 디렉토리 지정
```

### 평가
```bash
python -m src.evaluate --model_config <exp_dir>/.hydra/config.yaml --chkpt_dir <exp_dir>
```

### 향상 (Enhancement)
```bash
# 기본 (non-streaming)
python -m src.enhance --chkpt_dir <exp_dir>

# Streaming (stateful conv)
python -m src.enhance --chkpt_dir <exp_dir> --use_stateful_conv
```

## 문서 구성 (paper_works/)

| 파일 | 내용 |
|------|------|
| `experiment_design.md` | 실험 설계서: latency sweep, padding ratio 매핑, 학습 config matrix, 논문 Figure/Table 구상 |
| `benchmark_comparison.md` | 벤치마크 모델(RNNoise, GaGNet, SEMamba 등) 대비 latency/성능 비교표 |
| `references.md` | 관련 논문 레퍼런스 목록 |
| `dns3_training_recipes.md` | DNS Challenge 3 학습 레시피 참고 자료 |

## 설정 (conf/config.yaml)

- Hydra 기반 설정 관리, `hydra.run.dir`로 실험별 디렉토리 자동 생성
- 주요 모델 파라미터: `causal`, `encoder_padding_ratio`, `decoder_padding_ratio`, `norm_type`, `sca_kernel_size`
- 실험 결과는 `results/experiments/<날짜시간>/`에 저장 (체크포인트, `.hydra/config.yaml`, TensorBoard 로그)
