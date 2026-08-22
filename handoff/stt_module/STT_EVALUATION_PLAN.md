# 한국어 Streaming STT 성능 평가 및 개선 계획

이 문서는 경진대회 최종 보고서에 사용할 STT 평가 기준, 전문 어휘 표기,
현재 모델 특징과 최종 성능 지표를 정리한다. 2026-08-22 Mac 합성 음성
30문장 결과를 로컬 기준선으로 기록하며, 실제 Windows·Jetson·ReSpeaker에서
측정하지 않은 값은 `TBD`로 표시한다.

## 1. 평가지표

### 정확도

| 지표 | 계산 방법 | 용도 |
|---|---|---|
| CER | `(대체 문자 + 삭제 문자 + 추가 문자) / 정답 문자 수` | 한국어 주 지표. 띄어쓰기 영향이 WER보다 작음 |
| WER | `(대체 단어 + 삭제 단어 + 추가 단어) / 정답 단어 수` | 단어 단위 오류 확인. 띄어쓰기 기준을 고정해야 함 |
| Domain Keyword Accuracy | `정확히 인식된 전문 용어 수 / 전체 전문 용어 수` | Jetson, DOA 등 핵심 용어 평가 |
| Sentence Accuracy | `정답과 완전히 같은 문장 수 / 전체 문장 수` | 자막 전체 문장의 완전 일치율 |
| 빈 출력률 | `빈 Final 결과 수 / 전체 발화 수` | 음성이 인식되지 않는 실패 확인 |

한국어는 조사·어미가 단어에 결합되고 띄어쓰기 표기가 달라질 수 있으므로 **CER을 주 지표, WER을 보조 지표**로 사용한다. 비교 전에는 Unicode NFC 정규화, 연속 공백 정리, 영문 대소문자 통일, 문장부호 처리 기준을 고정한다. 원문 점수와 정규화 후 점수를 모두 보존한다.

### 실시간 성능

| 지표 | 의미 |
|---|---|
| Partial Latency | 중간 자막 추론에 걸린 시간 |
| Final Latency | 발화 종료 후 최종 자막 추론에 걸린 시간 |
| RTF | `음성 처리 시간 / 음성 길이`; 1 미만이면 음성 길이보다 빠름 |
| 모델 로딩 시간 | 최초 모델 로딩과 warm-up에 걸린 시간 |
| 최대 Audio Queue | 처리 지연 중 쌓인 입력 Queue의 최대 길이 |
| Callback overflow | 마이크 입력 callback이 제때 처리되지 못한 횟수 |
| 처리 실패율 | 오류가 발생한 발화 수 / 전체 발화 수 |

### 논문 기준 참고 벤치마크

한국어 자연발화 데이터셋인 **KsponSpeech** 논문은 CER, WER, 띄어쓰기를 정규화한 sWER를 함께 사용했다. 논문의 Large Transformer Baseline은 다음 결과를 보고했다.

| 평가셋 | CER | WER | sWER |
|---|---:|---:|---:|
| Dev | 6.1% | 16.4% | 11.9% |
| Eval-clean | 7.6% | 21.1% | 13.4% |
| Eval-other | 8.5% | 25.5% | 15.4% |

KoSpeech 논문은 KsponSpeech에서 LAS 기반 음향 모델의 CER 10.31%를 보고했다. 이 수치는 **현재 faster-whisper `small` 모델의 예상 점수나 합격 기준이 아니다.** 모델, 데이터 분할, 전처리 및 평가 방식이 다르므로 한국어 ASR 논문에서 사용하는 지표와 결과 표 형식을 참고하는 용도로만 사용한다.

현재 시스템은 같은 평가셋과 정규화 기준으로 직접 CER/WER를 측정한 후에만 논문 결과와 제한적으로 비교해야 한다.

참고 문헌:

- [KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition](https://doi.org/10.3390/app10196936)
- [KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition](https://arxiv.org/abs/2009.03092)
- [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://cdn.openai.com/papers/whisper.pdf)

## 2. 특정 분야 어휘 통일 및 설명

전문 용어는 같은 개념이 한글 음차, 영문, 띄어쓰기 차이로 여러 형태로 출력될 수 있다. 보고서와 자막에서는 아래 표기를 표준으로 사용한다.

| 표준 표기 | 예상 변형 | 설명 |
|---|---|---|
| Jetson Orin Nano | 젯슨 오린 나노, 젯슨 어린 나노 | NVIDIA의 임베디드 AI 컴퓨팅 보드 |
| DOA | 디오에이 | Direction of Arrival, 음원이 들어오는 방향 추정 |
| Beamforming | 빔포밍, 빔 포밍 | 특정 방향의 음성을 강조하고 다른 방향의 신호를 억제하는 기술 |
| MVDR | 엠브이디알 | 왜곡을 최소화하면서 간섭과 잡음을 억제하는 Beamforming 방식 |
| CUDA | 쿠다, 씨유디에이 | NVIDIA GPU 병렬 연산 플랫폼 |
| CTranslate2 | 씨트랜슬레이트투 | faster-whisper가 사용하는 최적화 추론 엔진 |
| faster-whisper | 패스터 위스퍼 | Whisper 모델을 CTranslate2로 실행하는 구현체 |
| Streaming STT | 스트리밍 에스티티 | 음성이 들어오는 동안 Partial/Final 결과를 제공하는 음성 인식 방식 |
| Partial | 파셜, 중간 자막 | 발화 도중 갱신되는 임시 인식 결과 |
| Final | 파이널, 최종 자막 | 발화가 끝난 뒤 확정되는 인식 결과 |

### 통일 방법

1. 위 표준 표기를 `domain_terms.txt`로 관리한다.
2. 전문 용어가 포함된 고정 평가 문장을 별도로 녹음한다.
3. Baseline의 **Domain Keyword Accuracy**를 먼저 측정한다.
4. Final Initial Prompt를 적용해 같은 음성으로 다시 측정한다.
5. 반복적으로 발생하고 의미가 명확한 오류만 제한적으로 후처리한다.

현재 STT 코드는 프로젝트 전문용어를 Final `initial_prompt`로만
제한적으로 제공한다. 실시간 `hotwords`는 잡음에서 BEM·MVDR·CTranslate2
등을 강제로 출력하는 현장 환각이 확인되어 모델에 전달하지 않는다.

```text
이 음성은 Jetson Orin Nano, DOA, MVDR Beamforming, CUDA에 관한 발표입니다.
```

Prompt는 전문 용어 인식 가능성을 높일 수 있지만 정확한 출력을 보장하지 않는다. 단순 문자열 치환 역시 정상 문장을 잘못 바꿀 수 있으므로, 전체 CER와 함께 **전문 용어 정확도 및 오교정 횟수**를 측정해야 한다.

## 3. 현재 모델 특징

아래 내용은 `stt/config.py`, `stt/model.py`, `stt/streaming.py`, `stt/types.py`에서 확인한 현재 설정이다.

| 항목 | 현재 설정 | 특징 |
|---|---|---|
| 모델 | Partial `tiny`, Final `small` | 중간 자막 반응성과 최종 정확도를 분리 |
| 언어 | `ko` | 한국어로 고정 |
| 입력 | mono, 16,000 Hz, NumPy/PCM16 | Beamforming 결과를 직접 연결할 수 있음 |
| 실행 장치 | CUDA 감지 시 GPU, 아니면 CPU | GPU `int8_float16`, CPU `int8` 자동 선택 |
| Partial | 0.25초 간격, 최근 4초, beam 1 | `tiny` 모델로 빠른 갱신 우선 |
| Final | 전체 발화, beam 3, VAD 사용 | `small` 모델과 신뢰도·환각 필터로 확정 |
| 발화 종료 | 시작 1초 잡음 보정, 적응형 RMS, 침묵 0.45초 | 고정 문턱과 배경 잡음을 결합 |
| 모델 관리 | Singleton/Lazy Loading | 모델을 한 번 로드해 재사용 |
| 처리 구조 | 최신 Partial 유지, Final 우선, 추론 직렬화 | Queue 적체와 중복 추론을 줄임 |

현재 방식은 토큰 단위의 진정한 Streaming이 아니라, 최근 음성을 반복 추론하는 **Rolling Buffer 방식**이다. 따라서 Partial은 바뀔 수 있으며 Final에서 전체 발화를 다시 인식한다. 긴 발화는 Final까지 메모리에 유지되고, Jetson GPU 성능과 전문 용어 정확도는 아직 실제 측정 전이다.

Mac CPU에서 macOS 한국어 TTS로 생성한 30문장 기준선은 다음과 같다.
합성 음성 결과는 모델·설정 비교용이며 실제 사용 환경의 최종 성능이 아니다.

| 항목 | 측정값 |
|---|---:|
| 평가 문장/음성 길이 | 30개 / 103.775초 |
| 원시 CER / WER | 13.43% / 18.63% |
| 전문용어 정규화 CER | 11.61% |
| 전문용어 정확도 | 72.73% |
| 문장 완전 일치율 | 46.67% |
| 평균 / P95 Final Latency | 1,421.18 ms / 1,575.89 ms |
| RTF | 0.4108 |
| 빈 결과 / 오류 / 프롬프트 환각 | 0 / 0 / 0 |
| 자동 테스트 | 33개 통과 |

동일 프로세스에서 실행 순서를 번갈아 측정한 설정 A/B 결과는 다음과 같다.
모델 로딩과 공통 warm-up은 비교 지연시간에서 제외했다.

| 설정 | 원시 CER | 정규화 CER | 전문용어 정확도 | 평균 Latency | RTF |
|---|---:|---:|---:|---:|---:|
| 이전: small, beam 5, prompt | 13.16% | 9.29% | 72.73% | 1,489.85 ms | 0.4307 |
| 이전 실험: small, beam 3, prompt+hotwords | 11.26% | 7.84% | 95.45% | 1,462.90 ms | 0.4229 |
| 현재 운영: small, beam 3, Final prompt+VAD+안전 필터 | 13.43% | 11.61% | 72.73% | 1,421.18 ms | 0.4108 |
| 참고: medium, beam 5, prompt | 17.10% | 9.29% | 77.27% | 4,066.47 ms | 1.1756 |

이전 `prompt+hotwords` 설정은 깨끗한 합성 음성에서는 전문용어
수치가 높았지만, 실제 Beamforming 잡음에서 해당 용어를 자막으로 강제하는
현장 실패가 확인됐다. 안전 설정은 이 편향을 제거하여 합성 음성 CER이
2.17%p 높아지고 전문용어 정확도가 22.72%p 낮아졌으나, RMS 0.02 지속
잡음 테스트에서 잘못된 자막을 0건으로 억제했다. 평균 Final 추론시간은
41.72ms(2.85%) 감소했다. Mac CPU Final 500ms 목표에는 아직 미달하며,
실제 사람·마이크·Jetson 성능이 아닌 로컬 합성 음성 기준선이다.

## 4. STT 성능 최종 지표

경진대회 보고서에는 다음 여섯 지표를 대표값으로 사용한다.

1. **한국어 CER**: 전체 자막의 주 정확도 지표
2. **Domain Keyword Accuracy**: 프로젝트 핵심 용어의 정확도
3. **평균 Final Latency**: 발화 종료 후 자막 확정 속도
4. **RTF**: 장치가 실시간 음성을 지속 처리할 수 있는지 판단
5. **Beamforming 전후 CER 차이**: 음향 전처리의 효과
6. **처리 실패율**: 빈 출력과 추론 오류를 포함한 안정성 지표

### 최종 결과 표

| 환경 | 데이터 | CER | WER | 전문 용어 정확도 | Final Latency | RTF | 실패율 |
|---|---|---:|---:|---:|---:|---:|---:|
| Mac CPU | 합성 전체 30문장, 현재 안전 설정 | 13.43% | 18.63% | 72.73% | 1,421.18 ms | 0.4108 | 0% |
| Mac CPU | 합성 전체 30문장, medium(이전 설정) | 17.10% | 17.16% | 77.27% | 4,066.47 ms | 1.1756 | 0% |
| Windows CPU/GPU | Windows TTS 30문장 | TBD | TBD | TBD | TBD | TBD | TBD |
| Jetson GPU | 일반 한국어 | TBD | TBD | N/A | TBD | TBD | TBD |
| Jetson GPU | 전문 분야 | TBD | TBD | TBD | TBD | TBD | TBD |

### Beamforming 효과

| 입력 | CER | 전문 용어 정확도 | Final Latency |
|---|---:|---:|---:|
| Beamforming 전 | TBD | TBD | TBD |
| Beamforming 후 | TBD | TBD | TBD |
| 변화량 | TBD | TBD | TBD |

보고서에서는 평균값만 제시하지 않고 평가 문장 수, 총 음성 길이, 모델 크기, 장치, 정규화 방식과 소음 조건을 함께 기록한다. 동일한 음성과 설정을 사용해야 Mac, Jetson, Prompt 및 Beamforming의 효과를 공정하게 비교할 수 있다.
