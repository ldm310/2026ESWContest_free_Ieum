# Branch Rules

## main

- 최종 검증 완료 버전만 저장
- 직접 push 금지
- `develop`에서 Pull Request로 병합
- force push 금지
- 브랜치 삭제 금지

## develop

- 기능 통합 및 테스트
- feature 브랜치에서 Pull Request로 병합
- 통합 테스트 실패 시 `main`에 병합하지 않음

## feature branches

- 각 담당자 전용 브랜치
- `develop`에서 생성
- 작업 완료 후 `develop`으로 Pull Request
- `main`으로 직접 병합하지 않음

## 담당 브랜치

- 병현: `feature/byunghyun`
- 준형: `feature/junhyeong`
- 이동민: `feature/dongmin`

## 병합 흐름

```text
feature/*
→ develop
→ main
```

## 커밋 금지 대상

- `.venv`
- 음성 파일
- 모델 및 캐시
- 개인정보
- API key와 인증정보
- 생성된 ZIP 파일
