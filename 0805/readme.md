# realtime_doa.py만 보면 됩니다

pipeline.py를 개조하여 실시간성을 부여한 코드입니다. 
--
빔포밍이랑 DOA 알고리즘은 준형님 것을 그대로 사용했으므로 안전합니다. 새로 짠 알고리즘은 없습니다.
마이크 4채널을 기반으로 DOA, 빔포밍한 결과물을 방향과 mono 오디오로 출력하는 코드입니다. 


## 준형님이 채울 곳은 2개입니다. (코드에 'TODO'로 표시해 뒀습니다.)

**TODO(1) — 마이크 장치·채널**
```python
DEVICE = None      # 4채널 마이크의 장치 번호. 아래 명령으로 확인 가능합니다.
CHANNELS = 4
```
장치 번호 확인:
```
py -3.11 -c "import sounddevice as sd; print(sd.query_devices())"
```

**TODO(2) — BEM 테이블·채널 매핑**
```python
BEM_TABLE = "sample_data/bem_table_reduced.h5"
PANEL_INDICES = np.array([0, 1, 2, 3], dtype=np.int64)
```
`PANEL_INDICES`는 **상자에 붙인 마이크 4개의 물리 순서**와 같아야 한다고 합니다.
(네 개의 마이크 자리가 각각 몇 번 채널인지 준형님이 확정해 주세요.)
---
