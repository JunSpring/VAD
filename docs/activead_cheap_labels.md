# ActiveAD cold-start용 "cheap label" 검토: 날씨·조명 밖에는 무엇이 있나

ActiveAD(Sec. 4.1)는 초기 10 %를 라벨링 없이 얻을 수 있는 정보만으로 고른다. nuScenes에서 그 정보는
`scene.json` description의 `night`/`rain` 단어(날씨·조명), ego 미래 궤적으로 만든 driving command(τc = 4),
scene 평균 속도 세 가지였다(`docs/activead_ed10.md` §3, `tools/active/scene_attrs.py`).
이 문서는 세 질문에 답한다.

1. description의 날씨·조명 **이외의 단어**로 더 적절한 구별이 가능한가?
2. 데이터셋에서 **그대로** 가져다 쓸 수 있는 요소는 무엇인가?
3. **전처리(수식)** 로 만들 수 있는 요소는 무엇인가?

분석 대상은 train 700 scene(28,130 keyframe)이며 val 150 scene은 marginal 확인용으로만 썼다.
스크립트와 scene별 피처 표(87열)는 scratchpad에 있다(부록 A). 모든 수치는 그 출력에서 옮겼다.

## 0. 요약

- **description 단어로는 날씨·조명보다 나은 1단계 기준을 만들기 어렵다.** 큰 주제(peds 439, intersection 238,
  construction 191, parking lot 157 …)는 모두 촬영 **지역(location)과 얽혀 있고**(construction·truck·parking lot은
  Boston, bus·bump는 Queenstown), 주행 행동(속도·정지·선회)은 거의 설명하지 못한다(η² ≤ 0.18).
  유일하게 쓸 만한 것은 `parking lot / intersection / construction / 기타` 네 분류(정지 비율 η² 0.18, 객체 수 η² 0.14).
- **날씨·조명 자체가 사실상 지역 분할이다.** night는 Singapore(Queenstown·Holland Village)에만, rain은 Boston·Holland
  Village에만 있고 One North(148 scene)에는 둘 다 없다. 날씨 × location의 Cramér V = 0.49.
- **`night`·`rain` 단어는 센서로 대체 가능하다.** CAM_FRONT 평균 휘도는 night 단어와 AUC 0.992(700개 중 694개 일치),
  LiDAR 30 m 내 평균 intensity는 Boston에서 rain 단어와 AUC 0.905로 일치한다. 즉 논문의 "cheap label"은 사람이 쓴
  텍스트 없이도 재현된다.
- **그대로 쓸 수 있는 요소**: `scene.description`, `log.location/vehicle/date_captured`, `ego_pose`(20 Hz 위치·자세),
  CAN bus 7채널(그중 converter가 읽는 것은 `pose`, `steeranglefeedback` 둘뿐; `vehicle_monitor`의 방향지시등·브레이크·
  스로틀, `zoe_veh_info`의 가감속, `route`는 미사용), HD map, 원본 카메라·LiDAR 파일. `gt_*`·`attribute`·`visibility`·
  `num_lidar_pts`는 사람 annotation이므로 cold-start에 쓰면 안 된다.
- **전처리로 얻는 요소** 중 주행 행동을 실제로 통제하는 것은 ego kinematics다: 정지 비율(`stop_frac`), 평균 속도,
  heading 변화량(`yaw_total`), 최대 감속. 현재 ED는 속도를 cell 안 정렬에만 쓰므로 정지 scene(47/700)·선회 각도 등은
  통제되지 않는다(η² ≈ 0).
- 다음에 시도할 대안: (a) 1단계를 `location × lighting`(6 cell)으로 바꿔 지역 편중을 명시적으로 제어, (b) 2단계를
  τc command 대신 `yaw_total` 구간으로, (c) 속도 정렬 대신 `stop_bin`을 층으로. cell 표는 §6.

## 1. 데이터와 방법

| 항목 | 값 |
|---|---|
| scene 메타 | `/media/junspring/JunBoom/dataset/nuScenes/Trainval/v1.0-trainval_meta/v1.0-trainval/{scene,log}.json` |
| ego/궤적 | `data/nuscenes/vad_nuscenes_infos_temporal_{train,val}.pkl` (VAD converter 출력; ego 필드만 사용) |
| CAN bus | `<NUSC>/can_bus/scene-XXXX_{pose,steeranglefeedback,vehicle_monitor,zoe_veh_info,zoesensors,ms_imu,route}.json` |
| 센서 원본 | `Trainval/samples/CAM_FRONT/*.jpg`, `samples/LIDAR_TOP/*.pcd.bin` (keyframe만, train 28,130 + val 6,019) |
| 환경 | host `mlops` python(numpy·pandas·scipy·PIL), devkit 불필요. 센서 pass 6 worker로 train 37 s / val 7 s |

검증: 스크립트 자체 열로 논문 Table 4 marginal 재현(DS/DR/NS/NR 491/125/71/13, S/R/L/O 423/132/112/33),
`mean_speed`는 `scene_attrs.per_scene_stats`와 차이 2e-15, scene-0001의 CAN `left_signal` 평균 0.25·`vehicle_speed`
19.3 km/h를 원본 json과 대조.

## 2. Q1. description의 다른 단어로 더 나은 구별이 가능한가

### 2.1 어휘

700 scene의 description은 쉼표로 구분된 구 평균 4.6개, 구 671종·단어 452종이다. 날씨·조명 단어(`rain` 136, `night` 84)
는 전체 구 3,225개 중 7 %에 불과하고 나머지는 객체(parked cars 175, peds 168, bus 76, truck 69, cones 50 …), 장소
(parking lot 142, intersection 관련 98+62+45 …), 행동(turn right 90, turn left 85, wait at intersection 98 …)이다.
"Random scene" 같은 무의미 설명은 1건.

### 2.2 키워드 그룹과 그것이 무엇과 얽혀 있는가

정규식으로 20개 그룹을 만들고(부록 B) 각 그룹의 scene 수, night/rain scene 수, 지역 분포, 다른 속성과의 연관
(Cramér V; 수치형은 그룹 평균)을 봤다. 전체 평균: 속도 4.98 m/s, 정지 비율 0.20, frame당 GT box 34.6개.

| 그룹 | n | night | rain | BOS/ON/QT/HV | V(날씨) | V(지역) | V(command) | 속도 | 정지 | box/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| pedestrian | 439 | 53 | 68 | 233/100/74/32 | 0.14 | 0.19 | 0.07 | 4.66 | 0.21 | 35.6 |
| two_wheeler | 261 | 38 | 41 | 112/73/37/39 | 0.12 | 0.21 | 0.09 | 5.36 | 0.18 | 31.1 |
| intersection | 238 | 17 | 65 | 153/43/21/21 | 0.17 | 0.13 | 0.18 | 3.52 | 0.38 | 36.1 |
| stop_wait | 204 | 24 | 46 | 120/43/21/20 | 0.05 | 0.06 | 0.16 | 3.06 | 0.47 | 34.8 |
| turn_word | 203 | 20 | 40 | 100/62/21/20 | 0.05 | 0.15 | **0.55** | 4.86 | 0.12 | 33.2 |
| construction | 191 | 2 | 42 | 128/48/10/5 | 0.21 | 0.22 | 0.09 | 4.93 | 0.18 | **47.4** |
| truck_trailer | 185 | 5 | 49 | 144/27/8/6 | 0.20 | 0.28 | 0.08 | 4.71 | 0.22 | 36.5 |
| bus | 161 | 39 | 26 | 63/33/46/19 | 0.21 | 0.26 | 0.02 | 5.67 | 0.17 | 29.0 |
| parking_lot | 157 | 2 | 42 | 114/26/17/0 | 0.20 | 0.22 | 0.07 | 4.99 | 0.13 | 42.7 |
| crosswalk | 98 | 8 | 14 | 62/23/9/4 | 0.08 | 0.10 | 0.14 | 2.59 | 0.46 | 40.4 |
| bump_slope_bridge | 65 | 22 | 9 | 25/5/23/12 | 0.24 | 0.25 | 0.08 | 5.99 | 0.07 | 26.8 |
| jaywalker | 59 | 2 | 8 | 30/18/8/3 | 0.11 | 0.08 | 0.07 | 4.52 | 0.18 | 35.9 |
| overtake_lanechange | 56 | 3 | 4 | 31/14/7/4 | 0.13 | 0.04 | 0.09 | 5.42 | 0.11 | 36.2 |
| traffic_density | 56 | 18 | 5 | 20/10/12/14 | 0.21 | 0.18 | 0.08 | 5.45 | 0.19 | 31.1 |
| narrow_residential | 43 | 10 | 2 | 9/7/2/25 | 0.13 | **0.41** | 0.10 | 7.12 | 0.08 | 20.5 |
| nature | 25 | 2 | 0 | 4/8/6/7 | 0.10 | 0.17 | 0.11 | 6.67 | 0.06 | 19.2 |
| wet_reflection_dark | 23 | 13 | 9 | 10/0/7/6 | 0.29 | 0.16 | 0.05 | 5.96 | 0.09 | 20.6 |
| speed_word | 13 | 11 | 4 | 1/0/1/11 | 0.36 | 0.34 | 0.04 | 13.05 | 0.01 | 16.4 |
| animal | 12 | 2 | 3 | 9/1/0/2 | 0.04 | 0.07 | 0.04 | 4.54 | 0.20 | 33.5 |
| roundabout | 9 | 3 | 2 | 5/0/4/0 | 0.09 | 0.12 | 0.12 | 5.19 | 0.06 | 33.0 |

읽는 법:
- **지역 confound.** construction·truck·parking lot은 Boston(128/191, 144/185, 114/157), bus·bump는 Queenstown,
  narrow/residential·speed word는 Holland Village에 몰린다. 이런 단어로 층을 만들면 날씨·조명과 마찬가지로 지역을 다시
  나누는 셈이다.
- **주행 행동과의 연관은 약하다.** command와 V ≥ 0.2인 그룹은 `turn_word`(0.55) 하나뿐이고, 그것은 궤적으로 이미
  더 정확히 얻는다(§2.3). 정지 비율을 올리는 그룹(intersection 0.38, stop_wait 0.47, crosswalk 0.46)은 서로 겹친다
  (intersection∩stop_wait Jaccard 0.44).
- **객체 밀도 proxy.** construction(47.4)·parking lot(42.7)·crosswalk(40.4)는 GT box 수가 많고 narrow/nature/speed는
  적다. 즉 description 단어는 "장면 복잡도"의 공짜 proxy로는 쓸 수 있다(그 자체가 annotation인 box 수를 쓰지 않고).
- 커버리지: 절반 이상의 scene을 덮는 그룹은 pedestrian 하나이고, 나머지는 소수 집합이라 1단계 층으로 쓰면 "기타"
  cell이 거대해진다.

### 2.3 "turn left/right" 단어 vs 궤적 command (τc = 4)

| description | L | O | R | S | 합 |
|---|---|---|---|---|---|
| turn 단어 없음 | 46 | 20 | 53 | 399 | 518 |
| turn right만 | 2 | 7 | 78 | 13 | 100 |
| turn left만 | 63 | 5 | 1 | 11 | 80 |
| 둘 다 | 1 | 1 | 0 | 0 | 2 |

단어가 있으면 궤적 cell과 78~80 % 일치하지만, 궤적상 L/R인 scene의 **40 %(46+53/244)에는 turn 단어가 없다**.
궤적 command가 더 완전하고, 단어는 annotator가 적을 때만 적는 부분 라벨이다.

### 2.4 "overtake / lane change" 단어 vs Overtake cell

O cell 33개 중 overtake·lane change 단어가 있는 scene은 5개, 단어가 있는 56개 중 36개는 S다. 논문의 τc 규칙
"L ≥ 4 ∧ R ≥ 4 → Overtake"는 실제로는 **좌우로 모두 꺾인 scene**(scene-1110 "turn right" yaw 107°, scene-0070 turn
right + turn left 61°)과 **직진 도로에서 비켜 간 scene**(scene-0034 overtake parked truck, yaw 5°)을 섞는다.
O의 yaw_total 분포(p10 3.3°, 중앙값 22.7°, p90 67°)가 이를 보여 준다. 추월만 뽑고 싶다면 `yaw_total < 20°` ∧
`fut_lat_absmax > 1.5 m`(옆으로 비킴) 같은 kinematic 정의가 단어보다 낫다.

### 2.5 결론

description의 다른 단어는 (a) 지역과 얽히고, (b) 주행 행동을 통제하지 못하며, (c) 커버리지가 낮아 1단계 층으로는
날씨·조명보다 낫지 않다. 쓸 수 있는 형태는 두 가지뿐이다.
- `parking lot / intersection / construction / 기타` 4분류(§5: 정지 η² 0.18, box 수 η² 0.14 — 후보 중 유일하게
  두 축을 조금씩 통제). val에도 33/45/19/53으로 비슷하게 존재.
- 장면 복잡도 proxy(construction·parking lot·crosswalk → box 많음, nature·narrow·speed → 적음)를 cell 내부 정렬 축으로.

## 3. Q2. 데이터셋에서 그대로 쓸 수 있는 요소

| 테이블/파일 | 필드 | 단위 | 논문 사용 | cold-start 사용 가능 | 비고 |
|---|---|---|---|---|---|
| `scene.json` | `description` | scene | ○ (night/rain만) | ○ | 자유 텍스트, annotator 작성. 구 671종 |
| `scene.json` | `nbr_samples` | scene | × | ○ | 39–41, 정보량 없음 |
| `log.json` | `location` | log | × | **○** | BOS 390 / ON 148 / QT 92 / HV 70. 날씨와 V 0.49 |
| `log.json` | `vehicle` | log | × | △ | n008 = Boston, n015 = Singapore. location과 동일 정보 |
| `log.json` | `date_captured`, `logfile` | log | × | △ | 2018-05~11. 계절·시간대. 시각은 logfile명(현지시각)에 있음 |
| `sample.json` | `timestamp` | 2 Hz | ○ (간접) | ○ | |
| `ego_pose.json` | `translation`, `rotation` | 20 Hz(LiDAR) / 12 Hz(cam) | ○ (궤적·command) | ○ | 사람 라벨 아님(localization 출력) |
| `calibrated_sensor`, `sensor` | 외부/내부 파라미터 | 고정 | × | ○ | 피처 가치 없음 |
| CAN `pose` | `pos, orientation, vel, accel, rotation_rate` | 50 Hz | ○ (`v0`, `can_bus` 18-dim) | ○ | converter가 읽는 유일한 속도원 |
| CAN `steeranglefeedback` | `value` | 100 Hz | ○ (`lcf[8]` 곡률) | ○ | |
| CAN `vehicle_monitor` | `left_signal, right_signal, brake, throttle, steering, yaw_rate, vehicle_speed, gear_position` | 2 Hz | × | **○** | 방향지시등 = 운전자 의도의 유일한 직접 기록 |
| CAN `zoe_veh_info` | 4륜 속도, `longitudinal_accel, transversal_accel, regen, odom` | 100 Hz | × | ○ | 가감속 직접 측정 |
| CAN `zoesensors` | brake/steering/throttle 센서 raw | ~950 Hz | × | △ | vehicle_monitor와 중복 |
| CAN `ms_imu` | `linear_accel, q, rotation_rate` | 100 Hz | × | △ | pose와 중복 |
| CAN `route` | 계획 경로 polyline | scene | × | ○ | 목적지 경로. 교차로·회전 예정 여부 추출 가능(미계산) |
| CAN `meta` | 채널별 통계(min/max/mean/std) | scene | × | ○ | **scene 통계가 이미 계산돼 있음** (예: `POSE.vel.max`) |
| `maps_expansion/*.json` | lane, road_segment(`is_intersection`), ped_crossing, stop_line … | 고정 자산 | × (학습 GT로만) | ○ | 도시 4곳 전체를 덮는 고정 지도. scene별 라벨 비용 0 |
| `samples/CAM_*`, `LIDAR_TOP`, `RADAR_*` | 원본 센서 | 2 Hz keyframe | × | ○ | §4.3 |
| `sample_annotation`, `instance`, `category`, `attribute`, `visibility` | 3D box, 속성 | ann | × | **×** | 사람 라벨. pkl의 `gt_boxes/gt_names/gt_velocity/gt_agent_*/num_lidar_pts/num_radar_pts/valid_flag`도 여기서 나옴 |

CAN은 15개 train scene(nuScenes blacklist)에 없고 1개 scene은 `vehicle_monitor`가 비어 있다(16/700 결측). 결측
scene은 pose 기반 속도로 대체된다(`scene_attrs.frame_speed`).

## 4. Q3. 전처리로 만들 수 있는 요소

### 4.1 ego kinematics (pkl의 ego 필드만 사용: `gt_ego_lcf_feat`, `ego2global_*`, `timestamp`, `gt_ego_fut_trajs`)

| 피처 | 수식 | 분포 (p10 / 중앙값 / p90) | 통제하는 것 |
|---|---|---|---|
| `v_mean` | CAN `v0` keyframe 평균(결측 시 pose 속도) | 0.40 / 5.24 / 8.65 m/s | 논문의 cell 내 정렬 축 |
| `v_max`, `v_std` | max, std | 2.4 / 7.7 / 10.5 ; 0.3 / 1.4 / 2.8 | |
| `stop_frac` | mean(v < 0.5 m/s) | 0 / 0 / 0.81 (완전 정지 47, ≥50 % 125, 0 % 392) | 정지 scene 비중 |
| `dist_m` | Σ v·Δt | 6 / 103 / 169 m | v_mean과 ρ 1.00 (중복) |
| `acc_absmean`, `acc_min`, `acc_max` | Δv/Δt | 0.12 / 0.42 / 0.69 ; −1.84 / −0.91 / −0.10 m/s² | 급제동 |
| `yaw_total` | \|yaw_end − yaw_start\| (unwrap) | 0.1° / 8.4° / 89° | command cell η² 0.65; L 중앙값 70°, R 81°, S 1.9°, O 23° |
| `yaw_abssum` | Σ\|Δyaw\| | 2° / 27° / 103° | 구불구불함(S-curve·lane change 포함) |
| `yawrate_absmax` | max \|Δyaw/Δt\| | 0.5 / 4.7 / 24.6 °/s | |
| `curv_absmean` | mean \|lcf[8]\| (조향각 기반) | 0.03 / 0.21 / 0.88 | yaw와 ρ 0.7–0.9 |
| `cmd_S/R/L`, `clip_cmd` | 논문 규칙 | S 423 / R 132 / L 112 / O 33 | |
| `fut_lat_absmax`, `fut_lat_absmean` | 3 s 미래점 횡오프셋 \|x\| | 0.03 / 0.51 / 2.23 m | command의 연속형 원천(±2 m 임계) |

### 4.2 CAN 추가 채널 (converter 미사용)

| 피처 | 출처 | 분포 (p10 / 중앙값 / p90) | 관찰 |
|---|---|---|---|
| `left_signal_frac`, `right_signal_frac` | `vehicle_monitor` 메시지 중 on 비율 | 0 / 0 / 0.30 ; 0 / 0 / 0.38 | L cell 0.19 / R cell 0.17이지만 **S cell(CAN 있는 415개) 중 144개가 메시지의 10 % 이상 지시등 사용**(좌 55, 우 75, 양쪽 14) → 차선 변경·갓길 정차 등 궤적 임계(2 m)에 안 걸리는 조작 |
| `brake_frac`, `brake_mean` | `brake > 0` 비율, 평균 | 0 / 0.08 / 0.83 | stop_frac과 ρ 0.87 |
| `throttle_mean` | | 0.6 / 79 / 159 | |
| `steering_absmax`, `yawrate_absmax` | | 5.5 / 52 / 250 ; 0.6 / 4.9 / 24.7 | pose 기반 yaw rate와 ρ 0.99 (중복) |
| `lat_acc_absmax`, `lon_acc_min/max` | `zoe_veh_info` | 0.52 / 0.60 / 0.74 (단위 미상) ; −2.4 / −1.35 / −0.4 m/s² | 급제동 (`acc_min`과 ρ 0.83) |
| `regen_mean` | 회생제동 | 0 / 10 / 24 | 내리막·감속 |
| `gear_set` | | 거의 전부 7(D) | 정보 없음 |

### 4.3 센서 원본에서 (keyframe당 CAM_FRONT 1장, LIDAR_TOP 1 sweep)

| 피처 | 수식 | 결과 |
|---|---|---|
| `lum_mean` | CAM_FRONT를 200×113 grayscale로 축소한 뒤 평균, scene 평균 | night 단어 scene 중앙값 43.1(p90 75.9), day 107.0(p10 100.8). **AUC 0.992**, night 비율 분위 임계 92.8로 자르면 700개 중 694개 일치 |
| `lum_dark_frac` | 화소 < 40 비율 | night 0.62 / day 0.04 |
| `n_pts`, `n_pts_30m` | LiDAR 점 수(전체 / 반경 30 m) | 전체는 34,720으로 일정(센서 고정). 30 m 내도 rain η² 0.08로 약함 |
| `n_pts_ground_30m` | 반경 30 m ∧ z < −1.3 m (노면) | rain 12,794 vs 14,819; Boston만 AUC 0.81 |
| `int_mean_30m` | 반경 30 m 평균 intensity | rain 11 vs 19 (η² 0.28); **Boston만 AUC 0.905**(rain 10.3 vs 14.7). 단 지역 효과가 더 큼: ON 23, QT 26.5, HV 24.6 vs BOS 14.7 → 지역별 정규화 필요 |

불일치 6건은 모두 설명 가능하다. 어둡지만 night 단어가 없는 3건(scene-0177 "Parking toll gate", 0697, 0854)은
황혼·그늘, night 단어가 있지만 밝은 3건(scene-0996, 1108 "wet front camera", 1109 "lightning")은 조명·번개·젖은
렌즈다. 지역별 night 휘도도 다르다(QT 34.5, HV 61.1 — HV는 주택가 가로등). 즉 `night`는 연속 휘도의 이진화이고,
센서값은 "얼마나 어두운가"까지 준다.

### 4.4 HD map에서 (후보, 이번에 미계산 — devkit이 Docker에만 있음)

`NuScenesMap`으로 keyframe ego 위치를 질의하면 scene별로 다음을 라벨 비용 0으로 얻는다: 교차로(`road_segment.
is_intersection`) 체류 비율, `lane` 밖(주차장·광장) 비율, 20 m 내 `ped_crossing`·`stop_line`·`traffic_light` 수,
현재 lane의 곡률·차선 수. description의 intersection(238)·parking lot(157)·crosswalk(98) 단어를 **객관적 지도 질의로
대체**할 수 있고, One North처럼 단어가 적은 지역에서도 동일 기준이 적용된다. CAN `route`와 결합하면 "교차로에서
좌회전 예정" 같은 의도 라벨도 가능하다.

### 4.5 중복성 (Spearman \|ρ\| > 0.8 쌍)

`v_mean ≈ dist_m ≈ CAN vehicle_speed`(1.00), `yawrate(pose) ≈ yawrate(CAN)`(0.99), `stop_frac ≈ brake_frac`(0.87),
`yaw_total ≈ yaw_abssum ≈ fut_lat_absmax`(0.87–0.95), `acc_min ≈ CAN lon_acc_min`(0.83). 독립적으로 남는 축은
**속도/정지, 선회량, 급제동, 휘도, LiDAR intensity, 지역, 지시등** 정도다. pose 기반 값과 CAN 값이 거의 같으므로
CAN이 없는 15 scene도 pose만으로 동일 피처를 만들 수 있다.

## 5. 층화 후보가 실제로 무엇을 통제하는가

각 후보를 1단계 층으로 썼을 때 cell 크기, γ = 0.5·n0 = 70 배분, 그리고 다른 속성의 분산 중 층이 설명하는 비율(η²;
1이면 완전 통제, 0이면 무관). `nbox`는 참고용(annotation).

| 층 | cell(pool / 배분) | 속도 | 정지 | yaw | 급제동 | box 수 | 휘도 | LiDAR pts | night | rain |
|---|---|---|---|---|---|---|---|---|---|---|
| **weather(논문)** | DS 491/34, DR 125/17, NS 71/13, NR 13/6 | 0.04 | 0.00 | 0.00 | 0.01 | 0.08 | 0.84 | 0.14 | 1 | 1 |
| location | BOS 390/28, ON 148/17, QT 92/13, HV 70/12 | 0.05 | 0.02 | 0.02 | 0.00 | **0.26** | 0.41 | 0.39 | 0.53 | 0.14 |
| location × lighting | BOS-D 390/24, ON-D 148/15, QT-D 59/10, HV-N 51/9, QT-N 33/7, HV-D 19/5 | 0.05 | 0.03 | 0.02 | 0.01 | 0.26 | 0.84 | 0.40 | 1 | 0.15 |
| scenario(단어) | other 261/22, intersection 190/18, parking 157/17, construction 92/13 | 0.11 | 0.18 | 0.03 | 0.01 | 0.14 | 0.11 | 0.01 | 0.09 | 0.02 |
| pedestrian 단어 | ped 439/40, noped 261/30 | 0.02 | 0.01 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0 | 0.02 |
| command(논문 2단계) | S 423/30, R 132/17, L 112/15, O 33/8 | 0.06 | 0.12 | **0.65** | 0.01 | 0.03 | 0.01 | 0.02 | 0.01 | 0 |
| stop_bin | never 392/28, <50 % 183/19, ≥50 % 78/13, all 47/10 | 0.63 | **0.93** | 0.09 | 0.15 | 0.01 | 0.02 | 0.00 | 0.01 | 0 |
| yaw_bin | <10° 367/30, >45° 207/22, 10–45° 126/18 | 0.04 | 0.11 | **0.85** | 0.01 | 0.04 | 0.03 | 0.03 | 0.02 | 0.01 |
| speed_bin | 5–8 268/22, 2–5 188/18, <2 142/16, >8 102/14 | **0.89** | 0.80 | 0.09 | 0.08 | 0.04 | 0.01 | 0.01 | 0.01 | 0 |
| nbox_bin(참고, annotation) | 15–30 232/20, 30–50 195/19, >50 152/16, <15 121/15 | 0.06 | 0.02 | 0.01 | 0.00 | 0.82 | 0.12 | 0.08 | 0.12 | 0.02 |

후보 간 Cramér V: weather–location 0.49, command–yaw_bin 0.59, stop_bin–speed_bin 0.60, 나머지 ≤ 0.26. 즉
(날씨 | 지역), (command | yaw), (정지 | 속도)는 각각 같은 정보의 두 표현이고, 세 묶음은 서로 거의 독립이다.
논문의 ED는 묶음 1·2를 층으로, 묶음 3을 cell 내 정렬로 쓴다.

현재 ED10 split(`tools/active/splits/ed10_gamma0.5.json`)이 실제로 덮는 것(선택 70 / pool 비율):
- 지역: BOS 37(pool 56 %), HV 13(10 %), QT 11(13 %), ON **9**(21 %) → γ = 0.5가 NS/NR을 키우면서 Singapore 두
  지역이 과대, One North가 과소 표집된다. 날씨 층의 부작용.
- 정지: never 43 / <50 % 16 / ≥50 % 8 / all 3 (pool 56/26/11/7 %) → 속도 정렬 덕에 대체로 비례하지만 완전 정지 scene은
  4 %로 pool(7 %)보다 적다. 평균 속도 5.50 vs pool 4.98.
- 선회: <10° 32 / >45° 22 / 10–45° 16 (pool 52/30/18 %) → command 층이 선회 scene을 의도대로 늘림.
- scenario 단어: other 30 / intersection 20 / parking 12 / construction 8 (pool 37/27/22/13 %).

## 6. 다음에 시도할 대안 (pool / γ = 0.5 배분, n0 = 70)

### (a) 1단계 location × lighting, 2단계 command — 지역 편중을 명시적으로 통제

| | L | O | R | S | 합 |
|---|---|---|---|---|---|
| BOS-D | 44/5 | 11/2 | 68/6 | 267/11 | 390/24 |
| ON-D | 40/4 | 9/2 | 24/3 | 75/6 | 148/15 |
| QT-D | 9/2 | 3/1 | 18/3 | 29/4 | 59/10 |
| HV-N | 7/2 | 3/1 | 13/2 | 28/4 | 51/9 |
| QT-N | 9/2 | 4/1 | 5/1 | 15/3 | 33/7 |
| HV-D | 3/1 | 3/1 | 4/1 | 9/2 | 19/5 |

rain은 Boston 안에서만 cell 내 정렬 축(LiDAR intensity 또는 단어)으로 쓴다. 논문 split 대비 ON 9 → 15, HV 13 → 14,
QT 11 → 17, BOS 37 → 24. 비교 실험의 가설: "ED 이득의 일부는 Singapore 과대표집에서 온다".

### (b) 2단계를 τc command 대신 yaw 구간으로 — 규칙 임계값(τc, ±2 m) 제거

| | <10° | 10–45° | >45° | 합 |
|---|---|---|---|---|
| BOS-D | 245/12 | 45/5 | 100/7 | 390/24 |
| ON-D | 62/6 | 28/4 | 58/5 | 148/15 |
| QT-D | 17/3 | 22/4 | 20/3 | 59/10 |
| HV-N | 26/4 | 13/3 | 12/2 | 51/9 |
| QT-N | 7/2 | 15/3 | 11/2 | 33/7 |
| HV-D | 10/2 | 3/1 | 6/2 | 19/5 |

`yaw_total`은 command cell의 분산 65 %를 설명하면서(§5) O처럼 정의가 모호한 cell이 없다. Overtake를 따로 보고 싶으면
`yaw_total < 20° ∧ fut_lat_absmax > 1.5 m`로 세 번째 축을 더한다.

### (c) 1단계 weather(논문 그대로), 2단계 stop_bin — 속도 정렬 대신 정지를 층으로

| | never | <50 % | ≥50 % | all | 합 |
|---|---|---|---|---|---|
| DS | 273/14 | 135/9 | 50/6 | 33/5 | 491/34 |
| DR | 63/6 | 35/5 | 16/3 | 11/3 | 125/17 |
| NS | 50/6 | 8/2 | 10/3 | 3/2 | 71/13 |
| NR | 6/3 | 5/2 | 2/1 | 0/0 | 13/6 |

완전 정지 scene이 3 → 10으로 늘어난다. val의 collision 지표가 정지 scene(scene-0344/0345)에서 결정되는 점
(`docs/activead_ed10.md` §7)을 생각하면 정지 scene 표집량이 collision에 직접 영향을 줄 가능성이 있다.

### (d) 단어 기반 scenario × command (참고)

| | L | O | R | S | 합 |
|---|---|---|---|---|---|
| other | 47/5 | 22/3 | 54/5 | 138/9 | 261/22 |
| intersection | 21/4 | 1/1 | 26/4 | 142/9 | 190/18 |
| parking | 24/4 | 6/2 | 37/4 | 90/7 | 157/17 |
| construction | 20/3 | 4/2 | 15/3 | 53/5 | 92/13 |

지역 confound(§2.2)가 그대로 들어오므로 (a)보다 권하지 않는다. HD map 질의(§4.4)로 같은 분류를 객관화한 뒤 쓰는 것이 낫다.

구현은 `ego_diversity_select.py`의 `allocate`/`regular_interval_indices`를 그대로 쓰고 1·2단계 키 함수만 바꾸면 된다
(이번 작업 범위 밖, 코드 변경 없음).

## 부록 A. 스크립트와 산출물 (scratchpad, 세션 ee2cb0b5)

- `/tmp/claude-1000/-home-junspring-VAD/ee2cb0b5-dc25-43d7-8baa-1ed8079d7663/scratchpad/cheap_features.py`
  — scene별 피처 표 생성. `--split train|val --workers 6 [--no-sensor]`. 출력 `scene_features_{train,val}.csv`(87열),
  센서 pass 캐시 `sens_{split}.pkl`.
- 같은 폴더의 `cheap_report.py` — §2·4·5의 표 출력(`report.txt`), `scene_features_train_aug.csv`(bin 열 추가).
- 열 접두사: `raw_`(원본 그대로) · `kw_`(description 키워드) · `d_`(pkl ego 필드로 계산) · `can_`(vehicle_monitor,
  zoe_veh_info) · `sens_`(CAM_FRONT, LIDAR_TOP) · `ref_`(**annotation, cheap 아님**, 참고용).

## 부록 B. 키워드 그룹 정규식

```
intersection: intersection                       turn_word: \bturn
overtake_lanechange: overtak|lane change|pass(ing)? (truck|car|bus)
parking_lot: parking lot                         construction: construction|cones?\b|barrier|crane|excavator
pedestrian: \bped|pedestrian|jaywalk|person|wheelchair|stroller      jaywalker: jaywalk
crosswalk: crosswalk|crossing                    bus: \bbus             truck_trailer: truck|trailer
two_wheeler: bicycl|bike|scooter|motorcycl|cyclist
traffic_density: dense|congestion|heavy traffic|busy|low traffic|high traffic
speed_word: high speed|low speed|slow            stop_wait: \bwait|stopped|stop\b|red light
bump_slope_bridge: bump|slope|bridge             narrow_residential: narrow|residential|big street|wide
nature: nature|trees?\b|park\b                   wet_reflection_dark: reflection|wet|dark|glare
animal: animal|dog                               roundabout: roundabout
```
