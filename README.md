# Embedded Realtime Caption

## 프로젝트 목표

4채널 마이크 입력을 이용하여 음원의 방향을 추정하고, 특정 방향의 음성을 분리한 뒤 한국어 실시간 자막을 생성하는 임베디드 시스템을 개발합니다.

## 전체 파이프라인

```text
4채널 마이크
→ DOA
→ Beamforming
→ Streaming STT
→ 한국어 자막
```

## 브랜치 구조

- `main`: 최종 통합 및 검증 완료 버전
- `develop`: feature 결과 통합 및 테스트
- `feature/byunghyun`: 병현님 담당 기능
- `feature/junhyeong`: 준형님 담당 DOA, Beamforming, Jetson 연동
- `feature/dongmin`: 이동민 담당 Streaming Korean STT

## 담당 브랜치

- 병현님: `feature/byunghyun`
- 준형님: `feature/junhyeong`
- 이동민: `feature/dongmin`

## 병합 흐름

```text
feature/*
→ develop
→ main
```

기능 작업은 담당 feature 브랜치에서 진행하고 Pull Request를 통해 `develop`에 통합합니다. 통합 검증이 끝난 변경만 `main`에 반영합니다.
