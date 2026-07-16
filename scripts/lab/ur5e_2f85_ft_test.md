# UR5e + Robotiq 가상 F/T 센서 예제

## 개요

이 예제는 기존 UR5e와 Robotiq 2F-85 USD를 그대로 사용하면서 두 에셋 사이에 작은 rigid body를 삽입한다. 이 body로 전달되는 joint wrench를 Isaac Lab 텐서로 읽어 가상 6축 F/T 센서처럼 사용한다.

연결 구조는 다음과 같다.

```text
UR5e wrist -> fixed joint -> VirtualFTSensor -> fixed joint -> Robotiq base
```

센서 출력은 `(num_envs, 6)` 형태이며 순서는 `[Fx, Fy, Fz, Tx, Ty, Tz]`이다. 값은 센서 body 좌표계 기준이고, PhysX/Isaac Lab의 장치 텐서에서 직접 얻으므로 RL 관측에 별도 변환 없이 사용할 수 있다.

## 실행

GUI에서 접촉과 화살표를 확인하려면 다음과 같이 실행한다.

```bash
./isaaclab.sh -p scripts/lab/ur5e_2f85_ft_test.py \
  --device cuda:0 --num_envs 4 --contact_twist_deg 5
```

학습 환경과 유사하게 headless로 확인하려면 다음 옵션을 추가한다.

```bash
--headless --steps 600
```

USD 경로는 스크립트 상단의 `UR5E_USD_PATH`, `ROBOTIQ_2F85_USD_PATH`에서 환경에 맞게 설정한다.

## 접촉 검증

기본 검증은 환경별로 `X/Y/Z` 접촉축과 방향을 무작위 선택하고, 접촉축에 수직인 축으로 큐브를 회전시킨다. 이를 통해 force와 torque 성분이 방향에 맞게 변하는지 확인할 수 있다.

- 빨간색 화살표: force
- 파란색 화살표: torque
- `--torque_visualize false`: torque를 숨기고 `Fx/Fy/Fz`를 빨강/초록/파랑으로 분해 표시
- `--disable_contact_test`: 검증용 probe와 큐브 동작 비활성화
- `--disable_ft_visualization`: viewport 화살표 비활성화

검증 큐브는 로봇 링크가 아니라 `FTProbePad`에만 충돌하도록 설정되어 있다. 접촉력 또는 토크가 지정한 한계를 넘으면 큐브가 자동으로 후퇴한다.

## RL 적용

핵심 함수는 `get_virtual_ft_wrench_b()`이다. 반환 텐서를 observation 구성 코드에서 직접 사용하면 된다. 출력 로그와 viewport 시각화에서만 사람이 읽을 수 있는 형식으로 CPU 복사가 발생하며, 센서 텐서 자체에는 영향을 주지 않는다.
