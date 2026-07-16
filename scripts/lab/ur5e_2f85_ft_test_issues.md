# UR5e + Robotiq 가상 F/T 센서 이슈 및 해결 기록

## 목적

기존 Isaac Sim 예제를 Isaac Lab 병렬 환경에서 사용할 수 있도록 변경하면서 발생한 문제와 해결 방법을 정리한다. 최종 목표는 동일한 UR5e 및 Robotiq USD를 사용하고, 가상 F/T 센서 출력을 RL 관측에 바로 사용할 수 있는 텐서로 얻는 것이다.

## 1. Isaac Lab 텐서 출력

### 이슈

기존 예제는 Isaac Sim API 중심으로 작성되어 있었다. 센서 값을 Python 값으로 읽은 뒤 텐서로 변환하는 방식은 RL 병렬 학습 과정에 적합하지 않았다.

### 해결

- UR5e와 Robotiq 사이에 `VirtualFTSensor` rigid body를 삽입했다.
- 연결 구조를 `UR 손목 -> VirtualFTSensor -> Robotiq 베이스`로 구성했다.
- Isaac Lab의 `body_incoming_joint_wrench_b`에서 센서 body의 incoming joint wrench를 직접 읽도록 했다.
- 반환값은 처음부터 시뮬레이션 장치에 있는 `(num_envs, 6)` 크기의 `torch.Tensor`이다.
- 출력 순서는 `[Fx, Fy, Fz, Tx, Ty, Tz]`이며 센서 body 좌표계 기준이다.

CPU 변환은 로그 출력과 viewport 시각화에서만 사용하며 RL 관측 텐서에는 적용하지 않는다.

## 2. 일부 메쉬가 보이지 않거나 바닥에 남는 문제

### 증상

- Robotiq 일부 링크가 손목을 따라가지 않고 바닥에 남아 있는 것처럼 보였다.
- 물리 body 위치는 정상인데 visual mesh만 분리되어 표시되는 경우가 있었다.
- Fabric에서 존재하지 않는 visual prim 경로에 접근했다는 경고가 출력되었다.

### 원인

Robotiq USD의 링크 geometry가 instance proxy로 구성되어 있었다. UR5e와 Robotiq를 하나의 articulation으로 병합한 뒤 Fabric이 rigid body pose를 갱신할 때, 일부 instance proxy visual이 해당 pose를 안정적으로 따라가지 못했다.

### 해결

- 원본 USD 파일과 mesh reference는 변경하지 않았다.
- 각 환경의 Robotiq reference 아래에서 instance root의 `instanceable` 속성을 해제하여 현재 stage에 geometry를 materialize했다.
- UR5e와 Robotiq를 clone 후 수정하는 대신 각 환경의 구체적인 prim 경로에 직접 spawn했다.
- geometry가 아닌 prim에 잘못 적용된 `CollisionAPI`는 기본적으로 제거하도록 했다.

GUI에서 양쪽 finger, knuckle, fingertip이 손목 아래에 정상적으로 연결되어 움직이는 것을 확인했다.

## 3. 그리퍼 조인트가 끊기거나 시작 시 snap되는 문제

### 증상

- Robotiq 일부가 바닥에 놓이거나 시작 직후 잘못된 위치로 이동했다.
- PhysX에서 `joint with disjointed body transforms` 경고가 발생했다.
- 기존 `robot_gripper_joint`의 child body target이 비어 있는 경우가 있었다.

### 원인

- 기존 장착 joint와 새 가상 센서 joint가 동시에 남아 중복 구속이 발생할 수 있었다.
- gripper reference root의 world transform을 환경 원점이 적용된 local transform처럼 기록하면 병렬 환경에서 위치가 어긋났다.
- standalone Robotiq asset의 articulation root가 유지되면 UR5e articulation 내부에 중첩 articulation이 생겼다.
- fixed joint의 local frame이 현재 parent/child body pose와 일치하지 않으면 PhysX가 body를 강제로 맞추면서 snap이 발생했다.

### 해결

- 기존 `robot_gripper_joint`를 비활성화하고 prim도 inactive 상태로 만들었다.
- gripper base를 UR tool frame에 정렬한 뒤 위치 및 회전 오차를 검사했다.
- world pose를 각 환경 parent 기준 local pose로 변환해 reference root에 기록했다.
- Robotiq 아래의 독립 `ArticulationRootAPI`를 제거했다.
- 동일한 world joint frame을 parent와 child 각각의 local frame으로 변환해 fixed joint를 생성했다.

## 4. 검증 큐브가 로봇을 과도하게 밀어 articulation이 붕괴하는 문제

### 증상

초기 접촉 시험에서 수천에서 수십만 N의 힘과 수백 N·m의 토크가 발생했다. 큐브가 손가락이나 다른 로봇 geometry를 직접 밀면서 articulation 전체가 불안정해졌다.

### 해결

- 센서 하류에 별도의 `FTProbePad`를 설치했다.
- 검증 큐브는 `FTProbePad`에만 충돌하고 UR5e 및 다른 Robotiq 링크와는 충돌하지 않도록 filtered pair를 설정했다.
- compliant contact stiffness와 damping을 적용했다.
- minimum-jerk 궤적으로 큐브를 천천히 접근 및 후퇴시켰다.
- 힘 또는 토크가 설정 한계를 넘으면 회전을 풀고 큐브를 후퇴시키는 feedback 안전장치를 추가했다.
- 큐브 회전 시 모서리 때문에 접촉 방향 지지 길이가 늘어나는 양을 보정하여 과도한 침투를 줄였다.
- NaN/Inf 및 안전 범위를 크게 벗어난 wrench를 즉시 감지하도록 했다.

5도 회전 조건의 최종 검증에서는 4개 CUDA 환경 전체 최대값이 약 `52.34 N`, `2.43 N·m`로 유지되었다.

## 5. F/T 값이 합리적인지 확인하기 어려운 문제

### 이슈

센서 텐서가 생성되어도 외력이 없으면 중력과 정적 하중 정도만 관찰된다. 따라서 force와 torque의 방향 및 크기가 올바른지 판단하기 어려웠다.

### 해결

- 환경마다 센서 좌표계 기준 `+X/-X`, `+Y/-Y`, `+Z/-Z` 중 접촉 방향을 선택한다.
- 환경 수가 충분하면 X/Y/Z 축이 가능한 한 균등하게 포함되도록 배정한다.
- 접촉 유지 단계에서 접촉축에 수직인 축을 선택해 큐브를 양방향으로 회전시킨다.
- seed를 지정할 수 있어 동일한 접촉 시험을 재현할 수 있다.
- 로그에 phase, 접촉축, 회전각, 현재 wrench, 누적 최대 force/torque를 함께 출력한다.

## 6. Force/Torque 시각화

### 해결

- 센서 body 좌표계의 force와 torque를 world 좌표계로 회전한 뒤 debug draw로 표시한다.
- force는 빨간색, torque는 파란색 화살표로 구분한다.
- torque 시각화를 끄면 torque 화살표 대신 센서 기준 `Fx/Fy/Fz`를 각각 빨강/초록/파랑으로 표시한다.
- 큰 값에서도 viewport를 가리지 않도록 화살표 길이에 상한을 적용한다.
- torque 화살표는 force 화살표와 겹치지 않도록 센서 위치에서 조금 떨어진 곳에 표시한다.
- headless 모드에서는 debug draw 모듈을 import하지 않는다.

시각화를 위한 CPU 복사는 그리기 직전에만 수행한다. 센서 관측 텐서는 계속 GPU에 유지된다.

## 7. 병렬 환경 일관성

### 이슈

환경 원점을 고려하지 않고 world transform을 적용하면 `env_0`은 정상이어도 다른 환경의 joint와 body가 어긋날 수 있었다.

### 해결

- 각 환경의 reference와 joint를 구체적인 prim 경로에 생성한다.
- 첫 physics step 이후 모든 body 위치에서 환경 원점을 뺀 뒤 `env_0`과 비교한다.
- 허용 오차를 넘으면 body 이름과 최대 위치 오차를 포함한 오류를 발생시킨다.

4개 환경 검증에서 body layout 최대 오차는 약 `2.682e-07 m`였고, F/T 출력은 `cuda:0` 장치의 `(4, 6)` 텐서로 확인되었다.

## 참고할 경고

- `getAttributeCount/getTypes called on non-existent path`와 같은 경고는 원본 USD 내부의 오래된 visual 경로에서 발생할 수 있다. 실제 rigid body와 mesh가 정상적으로 표시되고 동작한다면 가상 F/T joint의 연결 오류와는 구분해서 판단한다.
- `gpu.foundation.plugin ... acquired ... 100 times`는 Isaac Sim 플러그인 측 성능 경고이며 센서 텐서의 값이나 joint 구조가 잘못되었다는 의미는 아니다.
- `enable_external_forces_every_iteration=False` 경고와 함께 속도 노이즈가 관찰되면 PhysX 설정에서 해당 옵션과 velocity iteration 수를 함께 조정하는 것을 검토한다.

## 현재 결과

- 동일한 UR5e 및 Robotiq USD 사용
- Robotiq visual과 rigid body의 정상 연결 확인
- 병렬 환경별 가상 F/T wrench 텐서 출력
- X/Y/Z 접촉과 회전에 의한 force/torque 검증 가능
- force/torque viewport 시각화 지원
- 과도한 접촉과 비정상 wrench에 대한 안전장치 적용
