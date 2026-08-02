cd C:\Users\ybang\OneDrive\Desktop\D455 <br />
py -3.11 week1_viewer.py <br /><br />

mediapipe 1.0.0에서 옛날 방식(mp.solutions)이 없으므로, Tasks API를 사용 <br />
Tasks API : 얼굴 검출 모델 파일을 하나 받아서 쓰는 구조임 <br /><br />

Stage2에 대한 노트 : <br />
az(방위각): 카메라 정면이 0, **오른쪽에 있으면 +, 왼쪽에 있으면 − <br />
얼굴을 화면 왼쪽으로 옮기면 음수로, 오른쪽으로 옮기면 양수로 바뀌어야 정상 <br />
m(거리): 그 사람까지 미터. 앞으로 다가가면 줄고, 물러나면 늘어야 정상 <br /> <br />

#현주소 <br />
지금은 LATERAL_SIGN=1, OFFSET=0 기본값이라 카메라 각도 = 마이크 각도로 가정하고 있음<br />
스피커(또는 손뼉 치는 사람)를 여러 각도에 놓은 뒤<br />
각 위치에서 카메라 각도와 마이크 각도를 동시에 읽어 s 키로 기록함<br />
(지금은 마이크 대신 파란 선 각도를, 나중엔 준형님 값을 기준으로)<br />
5~10개쯤 모아 calib.csv가 생기면, 아래 코드로 두 상수를 뽑음<br />

```python
import numpy as np, csv
rows = list(csv.reader(open("calib.csv")))[1:]
cam = np.array([float(r[0]) for r in rows])
ref = np.array([float(r[1]) for r in rows])
a, b = np.polyfit(cam, ref, 1)
print("LATERAL_SIGN =", round(a, 2), "  (부호가 핵심, 크기는 1 근처여야 정상)")
print("AZIMUTH_OFFSET_DEG =", round(b, 1))
```
<br />
나온 값을 stage3_match.py 맨 위 두 상수에 적으면 됨.<br /><br />

# 추가과제

① 물리적 결정. 카메라를 24×20×16cm 상자 어디에 붙일지 정하고, 마이크 어레이 중심 기준으로 카메라가 얼마나 위/옆에 있는지 자로 재 둬야 함. 발화자가 1.5m보다 멀면 이 오프셋은 무시해도 되지만, 가까우면 필요함.
