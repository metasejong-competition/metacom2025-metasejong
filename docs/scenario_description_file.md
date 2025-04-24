# 시나리오 설명 파일

## 파일 구조 설명

### 1. 기본 구조

환경변수로 ENV_METASEJONG_SCENARIO 값에 따라서 {working directory}/scenario/{ENV_METASEJONG_SCENARIO}.yml 파일을 읽어서 임무 환경을 구성한다. 

경진대회 환경 구성에 필요한 고정값들(예: usd 파일 경로)은 기존과 같이 config_xxx.yaml 파일로 정의하고, 시나리오 환경에 따라 지정이 필요한 속성만 시나리오 파일로 정의

```
{project-root}/scenario-data/
├── demo.yml      # 데모 시나리오
├── chungmu.yml      # 충무관 시나리오
├── gunja.yml        # 군자관 시나리오
├── gwanggaeto.yml   # 광개토관 시나리오
└── aicenter.yml       # 대양AI센터 시나리오
```

### 2. YAML 파일 구조
각 시나리오 YAML 파일은 다음 구조를 따릅니다:

```yaml
scenario:
  info: # 시나리오에 대한 설명과 제약
    name: Demo  # yaml 파일 명이 시나리오에 대한 식별자 역할을 하고, 이 이름은 User friendly name
    description: 데모 시나리오  # User friendly description
    time_constraint: 15   # 시나리오 수행에 주어진 시간(분)
  playground: # 시나리오 대상 필드에 대한 USD 파일과 최초 로딩 후 view point 
    usd_file: playground/demo_field.usd  # {python 실행하는 경로. 즉, pwd}/resources/models 로부터의 상대경로
    view_point:
      camera_view_eye: [-113.03272, 64.98621, 30.59225]
      camera_view_target: [-63, 119, 1.02]
  mission:  # 미션과 관련하여 설치된 CCTV를 중심으로 CCTV에 대한 설치 정보, CCTV가 촬영하는 대상 지역(trash를 배치할 대상 영역), 대상 지역에 배치할 trash 종류별 카운트
    {area_name}:  # ROS topic의 namespace로 사용될 값
      camera_info:
        focal_length: 30.0 # 카메라의 촛점거리 (cm)
        horizontal_aperture: 40.0 ##  카메라의 수평 조리개 길이(cm). 수직 조리개 길이는 16:9의 비율을 적용하여 계산됨 
        ground_height: 1.0  # 카메라가 촬영하는 지면의 z 값 
        position: [x, y, z]
        rotation: [rx, ry, rz] # optional. 지정되지 않으면 config_xxx.yaml 파일에 정의된 기본 값 사용
      mission_objects:
        {object_name}: count  # {object_name}_{seq#} 형식으로 prime path, ROS2 topic에 사용 
  robot:
    start_point: [x, y, z]
```

#### 2.1. scenario 개요
```yaml
scenario:
  name: "시나리오 이름"
  description: "시나리오 설명"
  time_constraint: 15  # 시나리오 완료를 위한 제한 시간 (분)
```

시나리오에 대한 이름, 설명, 그리고 완료를 위한 제한 시간을 지정합니다. time_constraint는 분 단위로 지정되며, 해당 시간 내에 시나리오를 완료해야 합니다.

#### 2.2. 시작 화면의 카메라 뷰
```yaml
scenario:
  playground:
    usd_file: playground/demo_area.usd  # {python 실행하는 경로. 즉, pwd}/resources/models 로부터의 상대경로
    view_point:
      camera_view_eye: [-113.03272, 64.98621, 30.59225]
      camera_view_target: [-63, 119, 1.02]
```

다음과 같은 규칙으로 사용될 수 있도록 
```python

  if ENV_METASEJONG_DOCKER == 'YES':
      CONST_WORKING_DIRECTORY = '/simulation_app'
      CONST_METASEJONG_RESOURCES = '/root/Documents/metasejong'
  else:
      CONST_WORKING_DIRECTORY = os.getcwd()
      CONST_METASEJONG_RESOURCES = os.path.join(CONST_WORKING_DIRECTORY, 'resources')

  CONST_METASEJONG_RESOURCES_MODELS = os.path.join(CONST_METASEJONG_RESOURCES, "models")
  CONST_PLAYGROUND_USD_PATH = os.path.join(CONST_METASEJONG_RESOURCES_MODELS, {playground.usd_file})
```

시작 화면의 camera view eye와 view target을 지정 

#### 2.3. 임무 정의
```yaml
scenario:
  mission:
    {area_name}:  # chungmu_1, chungmu_2, gunga_1, ...
      camera_info:
        focal_length: 30.0
        horizontal_aperture: 40.0
        ground_height: 1.0
        position: [x, y, z]
        rotation: [rx, ry, rz]
      mission_objects:
        object_name: count
```
mission: 하나 이상의 임무 영역에 대한 정의를 포함하는 container, 하나의 임무 영역은 설치된 fixed camera를 통해 구성된다. 

{area_name}: 해당 임무 영역에 대한 이름. 예: chungmu_1, chungmu_2, gunja_1, ...

camera_info: 해당 임무 영역에 설치된 fixed camera에 대한 설치 정보. 촛점거리, 위치, 회전 정보를 포함한다.

mission_objects: 하나 이상의 쓰래기 이름과 해당 쓰래기를 몇개 배치할지를 결정하는 정보. 이름과 숫자 쌍으로 기록되며, 지정된 숫자만큼의 쓰래기 개체가 배포된다.

```yaml
      mission_objects:
        master_chef_can: 3
        cracker_box: 3
        mug: 3
```

#### 2.4. 로봇 
```yaml
scenario:
  robot:
    start_point: [x, y, z]
```

로봇이 처음 위치할 좌표를 지정한다.
