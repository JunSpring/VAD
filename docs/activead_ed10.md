# ActiveAD Ego-Diversity 10% cold-start 재현 — VAD-Tiny (Docker, RTX 4090 1장)

*ActiveAD: Planning-Oriented Active Learning for End-to-End Autonomous Driving* (CVPR 2024,
`docs/ActiveAD.pdf`)의 초기 선택(initial selection) 실험을 재현한다. nuScenes train 700 scene 중
Ego-Diversity로 70 scene을 고르고 VAD-Tiny를 학습한다.

## 논문에서 가져온 설정

| 항목 | 값 | 출처 |
|---|---|---|
| 모델 | VAD-Tiny, VAD 기본 하이퍼파라미터, ImageNet R50, 매번 from scratch | Sec. 5.1 |
| 최적화 | AdamW, lr 2e-4, wd 0.01, CosineAnnealing, **20 epoch** | Sec. 5.1 |
| 선택 단위 / 예산 | scene 단위; 10% = **70 / 700** train scene | Table 4 |
| 날씨·조명 | `scene.json` description에 `night` → Night, `rain` → Rainy | Sec. 4.1 |
| 주행 명령 | keyframe별 VAD `gt_ego_fut_cmd`, clip 규칙 **τc = 4** | Sec. 4.1 / 5.1 |
| 비율 | Eq. 1 / Eq. 2, **γ = 0.5** | Sec. 5.1 |
| 셀 내 선택 | scene 평균 속도로 정렬 후 등간격 | Alg. 1 |
| 평가 | 전체 val(150 scene), 1·2·3 s L2 / collision | Table 1 |

참조 수치(VAD-Tiny, 10% ED): L2 avg **0.83 m**, collision avg **0.41–0.43 %**
(Random 0.86 / 0.67, 전체 데이터 0.70 / 0.25).

논문이 명시하지 않아 우리가 정한 것(모두 결정론적이며 split json에 기록됨):
1 GPU 배치를 gradient accumulation ×8로 모사(lr 2e-4 유지, warmup 500 step = 4000 iter),
단계별 largest-remainder 반올림, 중심 층화 등간격 인덱스, CAN `v0` 속도(없으면 pose 기반 대체),
seed 0 + `--deterministic`.

## 0. 1회성 준비 (호스트)

1. **CAN bus**: <https://www.nuscenes.org/download>(로그인)에서 `can_bus.zip`을 받아
   `/media/junspring/JunBoom/dataset/nuScenes/can_bus/scene-0001_meta.json`이 생기도록 압축 해제.
2. Map expansion v1.3은 `/media/junspring/JunBoom/dataset/nuScenes/maps_expansion/`에 있음.
3. `tools/active/setup_data_links.sh`가 심볼릭 링크로 `data/`를 구성하고 누락 항목을 보고함.

## 1. Docker

```bash
docker/build.sh                 # -> vad:cu118-torch2.0.1
docker/run.sh                   # 대화형 셸, 레포와 SSD를 호스트와 같은 경로에 마운트 (포트 공개는 TB_PORT=6006)
docker/run.sh python -c "import torch, mmcv, mmdet, mmdet3d, shapely, numpy; print(torch.cuda.is_available(), shapely.__version__, numpy.__version__)"
```

## 2. Info pkl 생성 (컨테이너 안, cwd = 레포 루트)

```bash
# v1.0-mini 스모크 테스트 (수 분)
python tools/data_converter/vad_nuscenes_converter.py nuscenes --root-path ./data/nuscenes_mini \
  --out-dir /media/junspring/JunBoom/dataset/nuScenes/vad_infos_mini --extra-tag vad_nuscenes --version v1.0-mini --canbus ./data
# 전체 trainval (약 25 분) -> 700 / 150 scene, 28130 / 6019 sample
python tools/data_converter/vad_nuscenes_converter.py nuscenes --root-path ./data/nuscenes \
  --out-dir /media/junspring/JunBoom/dataset/nuScenes/vad_infos --extra-tag vad_nuscenes --version v1.0-trainval \
  --canbus ./data --require-canbus
```

`--require-canbus`는 `data/can_bus/*_pose.json`이 없으면 중단한다. 이 옵션이 없으면 컨버터가 `can_bus`
(모델의 ego 상태 입력)에 조용히 0을 채워 논문 설정과 달라진다. mini 스모크는 CAN 없이도 된다
(ego 속도는 pose 기반으로 대체).

## 3. Ego-Diversity split

```bash
python tools/active/ego_diversity_select.py check  --train-pkl data/nuscenes/vad_nuscenes_infos_temporal_train.pkl
#   -> DS/DR/NS/NR = 491/125/71/13, S/R/L/O = 423/132/112/33 이 출력되어야 함 (논문 Table 4 "Complete")
python tools/active/ego_diversity_select.py select --train-pkl data/nuscenes/vad_nuscenes_infos_temporal_train.pkl \
  --gamma 0.5 --budget 70 --tau-c 4 --out tools/active/splits/ed10_gamma0.5.json
python tools/active/ego_diversity_select.py filter --train-pkl data/nuscenes/vad_nuscenes_infos_temporal_train.pkl \
  --scenes tools/active/splits/ed10_gamma0.5.json \
  --out /media/junspring/JunBoom/dataset/nuScenes/vad_infos/vad_nuscenes_infos_temporal_train_ed10.pkl
```

이 머신에서의 결과(2026-10-30): pool DS 303/89/76/23, DR 77/25/20/3, NS 38/14/13/6, NR 5/4/3/1
(S/R/L/O 순); 선택 DS 15/8/7/4, DR 7/4/4/2, NS 5/3/3/2, NR 2/2/1/1 = 70 scene / 2,816 frame; 명령 합계
S29/R17/L15/O9(논문 27/17/16/10). train scene 15개는 CAN 데이터가 없음(nuScenes 블랙리스트).

## 4. 학습

```bash
# 1 epoch dry run (속도 / 메모리 확인)
python -m torch.distributed.run --nproc_per_node=1 --master_port=29511 tools/train.py \
  projects/configs/VAD/VAD_tiny_e2e_activead_ed10.py --launcher pytorch --deterministic --seed 0 --no-validate \
  --work-dir work_dirs/vad_tiny_ed10_dry \
  --cfg-options total_epochs=1 runner.max_epochs=1 model.pts_bbox_head.tot_epoch=1 log_config.interval=10
# 본 학습 (20 epoch)
python -m torch.distributed.run --nproc_per_node=1 --master_port=29511 tools/train.py \
  projects/configs/VAD/VAD_tiny_e2e_activead_ed10.py --launcher pytorch --deterministic --seed 0 --no-validate \
  --work-dir work_dirs/vad_tiny_ed10_g0.5_seed0
# 재개: --resume-from work_dirs/vad_tiny_ed10_g0.5_seed0/latest.pth 추가
```

RTX 4090 측정값(mini dry run, 2026-10-30): 약 0.20 s/iter, GPU 메모리 4.3 GB, lr는 6.67e-5(= 2e-4/3
warmup)에서 시작 → 70 scene run(약 2,800 frame × 20 epoch = 56k iter)은 3–4시간. `tools/train.py`는
start method를 `fork`로 고정한다(mmcv의 `init_dist`가 `spawn`으로 바꾸면 데이터셋 pickle에 실패).

## 5. 평가 (1 GPU, 비분산 — docs/train_eval.md 요구사항)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/VAD/VAD_tiny_e2e_activead_ed10.py \
  work_dirs/vad_tiny_ed10_g0.5_seed0/epoch_20.pth --launcher none --eval bbox \
  --json_dir work_dirs/vad_tiny_ed10_g0.5_seed0/eval_ep20
python tools/active/eval_breakdown.py --metric-json work_dirs/vad_tiny_ed10_g0.5_seed0/eval_ep20/metric_record.json
```

`tools/test.py`는 이제 planning 지표(`planning/plan_L2_avg`, `planning/plan_obj_box_col_avg`, …)를
반환하고 샘플별 `metric_record.json`을 기록하므로, `eval_breakdown.py`가 논문 Table 3 형식의
Day/Night/Sunny/Rainy/명령별 표를 출력할 수 있다.

## 6. 결과 (run `vad_tiny_ed10_g0.5_seed0`, 2026-10-30, RTX 4090 1장, 3시간 18분)

전체 val(150 scene, 6,019 frame 중 `fut_valid_flag`인 5,119 frame), `epoch_20.pth`:

| 그룹 | scene | L2 1s / 2s / 3s / avg (m) | Col 1s / 2s / 3s / avg (%) |
|---|---|---|---|
| **All** | 150 | 0.470 / 0.776 / 1.158 / **0.801** | 0.58 / 0.64 / 0.76 / **0.66** |
| Day | 135 | 0.458 / 0.761 / 1.136 / 0.785 | 0.62 / 0.66 / 0.73 / 0.67 |
| Night | 15 | 0.571 / 0.918 / 1.359 / 0.949 | 0.20 / 0.54 / 1.01 / 0.58 |
| Sunny | 123 | 0.480 / 0.790 / 1.172 / 0.814 | 0.69 / 0.73 / 0.79 / 0.74 |
| Rainy | 27 | 0.423 / 0.714 / 1.094 / 0.743 | 0.05 / 0.27 / 0.58 / 0.30 |
| Go Straight | 93 | 0.451 / 0.733 / 1.076 / 0.753 | 0.77 / 0.80 / 0.81 / 0.80 |
| Turn Left | 24 | 0.488 / 0.824 / 1.250 / 0.854 | 0.12 / 0.37 / 0.59 / 0.36 |
| Turn Right | 24 | 0.502 / 0.855 / 1.316 / 0.891 | 0.49 / 0.46 / 0.69 / 0.55 |
| Overtake | 9 | 0.527 / 0.890 / 1.341 / 0.919 | 0.00 / 0.24 / 0.76 / 0.33 |

논문(VAD-Tiny, 10% Ego-Diversity): L2 avg 0.83 / Col avg 0.41–0.43; Random 0.86 / 0.67; 100% 데이터 0.70 / 0.25.
L2는 논문보다 약간 좋고, collision은 논문의 Random 수준이다. 보조 헤드: detection NDS 0.176 / mAP 0.157,
map mAP 0.078(divider 0.070, ped_crossing 0.043, boundary 0.121), motion EPA car 0.223 / ped 0.095,
minADE car 1.47 / ped 1.11.
파일: `work_dirs/vad_tiny_ed10_g0.5_seed0/{epoch_20.pth, eval_ep20/{eval_results,metric_record,breakdown}.json, ed10_gamma0.5.json}`;
로그 `work_dirs/train_ed10.log`, `work_dirs/eval_ed10.log`.
학습 손실(epoch 평균 total loss): 20.5(ep1) → 14.6(ep2) → 8.4(ep10) → 6.46(ep20), NaN 없음.

## 7. 충돌률이 논문과 다른 이유 (분석, 2026-10-30)

관측: L2 avg 0.801(논문 ED 0.83, 더 좋음), collision avg 0.66 %(논문 0.41–0.43, Random 0.67 수준).

1. **충돌 차이는 전부 연속된 val scene 두 개에서 나온다.** 1 s 충돌로 표시된 41 frame 중 32개가
   `scene-0344` / `scene-0345`("Wait at intersection, construction, bus")에 있다. 이 67 frame(val의 1.3 %)을
   제외하면 L2 avg 0.805(변화 없음), **collision avg 0.27 %**(0.10 / 0.25 / 0.45)로 논문 100% 데이터 수준이다.
2. **그 frame들에서 플래너는 맞고 지표가 틀렸다.** ego는 정지 상태(GT 1 s 이동 0.00 m), 예측 L2도 0.00 m이며,
   차 한 대가 ego 옆 0.58 m에 주차돼 있다. 1 s 충돌 frame의 78 %가 L2 < 0.2 m, 85 %가 GT 이동 < 0.3 m이다.
3. **근본 원인은 `metric_stp3.evaluate_single_coll`의 래스터화.** ego 박스를 0.5 m BEV 격자에 그리고 궤적
   오프셋을 `astype(np.int32)`(절삭)로 변환한다. 박스 모서리가 셀 경계에 정확히 놓여 **이웃 차 방향으로 1 mm만
   이동해도 플래그가 뒤집힌다**(`tools/active/collision_metric_probe.py`: GT 0.000 m → 충돌 없음; −0.001 m →
   6 step 모두 충돌; +0.05 m → 없음). GT 충돌 필터는 GT가 경계 반대편에 있어 소용없다. 정지한 ego의 mm 단위
   예측이 어느 쪽에 떨어지는지는 seed 노이즈이므로, 10% 모델의 충돌률은 장면 하나로 ±0.3 pt의 불확실성을 가진다.
   논문 자체의 Random–ED 차이(0.67 vs 0.43)와 Table 3의 Random Night 1.25 %도 같은 크기다.
4. **학습 설정은 충실하므로 원인이 아니다**: 백본 BN이 고정(`norm_eval=True`, `requires_grad=False`)이고
   temporal BEV 큐는 샘플 내부에서 만들어지므로 gradient accumulation ×8은 8 GPU 배치와 같다. lr 2e-4
   (autoscale 없음), wd 0.01, cosine, 20 epoch, warmup 500 optimizer step, epoch당 352 step으로 논문과 같고,
   손실은 단조 감소, NaN 없음.
5. **제거할 수 없는 잔여 차이**: 논문의 70 scene 목록은 비공개(명령 합계가 ±2 scene 다르고 셀 내 등간격 선택은
   그들의 속도 소스에 의존), 논문 각주는 VAD 베이스라인을 "업데이트"했다고 함(VAD-Tiny 100 %: 0.70 / 0.25 vs
   VAD README 0.78 / 0.38) → 평가 코드가 이 레포의 2023-08 릴리스와 다를 수 있음. Appendix D.2는 PDF에 없음.

결론: L2는 재현됨(논문보다 약간 좋음). 충돌 차이는 단일 장면의 지표 래스터화 아티팩트이며 학습 결함이 아니다.
비교는 L2로 하고, 충돌률은 `scene-0344/0345` 포함/제외를 함께 보고하거나 seed 평균을 쓴다.

### 7.1 체크포인트 간 변동 (같은 run, 전체 val, 제외 없음)

| 체크포인트 | L2 1s / 2s / 3s / avg | Col 1s / 2s / 3s / avg |
|---|---|---|
| epoch_18 | 0.473 / 0.777 / 1.153 / 0.801 | 0.10 / 0.26 / 0.48 / **0.28** |
| epoch_19 | 0.463 / 0.764 / 1.143 / 0.790 | 0.44 / 0.48 / 0.63 / **0.51** |
| epoch_20 | 0.470 / 0.776 / 1.158 / 0.801 | 0.58 / 0.64 / 0.76 / **0.66** |

L2는 같고 collision은 0.38 pt 다르다. epoch 18에서는 scene-0344/0345의 정지 frame들이 래스터 경계 반대편에
떨어졌기 때문이다. scene 단위 부트스트랩(epoch_20, val 150 scene을 5,000회 재추출): L2 avg 95 % CI
[0.75, 0.85]; Col avg 95 % CI [0.21, 1.45]; Col avg의 49 %가 scene 하나, 72 %가 scene 5개에서 나온다.
논문의 0.83 / 0.43은 두 구간 안에 있으므로, 단일 run으로는 collision ±0.3 pt / L2 ±0.05 m보다 정밀하게
확인도 반박도 할 수 없다. scene 제외는 논문과의 정당한 비교가 *아니다*(논문 수치도 그 모델이 겪은 모든
장면을 포함). 대신 seed 여러 개를 쓴다.
체크포인트별 scene-0344/0345 안의 1 s 충돌 frame: epoch_18 9개 중 0, epoch_19 39개 중 30, epoch_20 41개 중 32.

## 8. seed와 Random 베이스라인 (2026-10-30 09:29 KST 큐 등록)

```bash
# Random 10 % split (논문 "Random" 행), scene 단위 균등 추출, seed = 학습 seed
python tools/active/ego_diversity_select.py random --train-pkl data/nuscenes/vad_nuscenes_infos_temporal_train.pkl \
  --budget 70 --seed 1 --out tools/active/splits/rand10_s1.json
python tools/active/ego_diversity_select.py filter --train-pkl data/nuscenes/vad_nuscenes_infos_temporal_train.pkl \
  --scenes tools/active/splits/rand10_s1.json \
  --out /media/junspring/JunBoom/dataset/nuScenes/vad_infos/vad_nuscenes_infos_temporal_train_rand10_s1.pkl
tools/active/setup_data_links.sh            # 모든 train_*.pkl 을 data/nuscenes/ 에 링크
# 학습 -> 평가 -> breakdown 순차 큐 (컨테이너 1개, 항목당 약 3.6 h)
NAME=vad_queue docker/run.sh tools/active/run_queue.sh ed10:1 rand10_s1:1 ed10:2 rand10_s2:2
```
run 디렉터리: `work_dirs/vad_tiny_<split>_seed<seed>/{train.log,eval.log,queue.log,eval_ep20/}`. ED config를
`--cfg-options data.train.ann_file=...`로 재사용한다. 두 번째 큐(`rand10_s0:0`)를 첫 큐 뒤에 연결해 두
방법 모두 seed 0/1/2(3 대 3, 학습 seed로 쌍 비교)가 되게 했다.

Random = train 700 scene 중 70개 균등 추출(scene 경계 유지, 층화 없음):

| split | DS/DR/NS/NR | S/R/L/O | frame | ED와 겹침 |
|---|---|---|---|---|
| ED γ=0.5 | 34/17/13/6 | 29/17/15/9 | 2,816 | - |
| Random seed 0 | 46/17/6/1 | 49/8/9/4 | 2,816 | 9 scene |
| Random seed 1 | 50/12/7/1 | 44/16/8/2 | 2,816 | 5 scene |
| Random seed 2 | 48/14/6/2 | 41/21/8/0 | 2,817 | 7 scene |

### 8.1 3-비교군 설계 (사용자 요청, 2026-10-30)

논문에는 혼동하기 쉬운 두 베이스라인이 있다.
* **Random**(Table 1/2 "RA"/3): Alg. 1의 `Randomly select n0 samples K ⊂ [N]`, N = scene 수 → scene 단위
  균등 추출. 10 %: 0.86 / 0.67. = 우리의 `rand10_s{0,1,2}`.
* **γ = 1**(Table 4): pool 비례 층화 할당(P_x = n_x / Σn)에 결정론적 속도 정렬 등간격 선택. 본문은 이를
  "random selection과 비교해 속도 기반의 균일한 선택"이라 부른다. 10 %: 0.90 / 0.46, 분포 49/12/7/2,
  40/13/11/6. 우리 것(`ed10_g1`): 49/13/7/1, 43/13/11/3(반올림 차이, 논문 반올림 방식은 미명시), 2,811 frame,
  γ = 0.5와 2 scene 겹침.

최종 구성(각 3 seed, 학습 seed로 쌍 비교, 큐 순서): ED γ=0.5 s0(완료), s1, Random s1, ED s2, Random s2,
Random s0, 이어서 γ=1 s0/s1/s2(`work_dirs/vad_tiny_ed10_g1_seed*`, `tools/active/chain_gamma1.sh (copy of the one-off work_dirs/chain_gamma1.sh)`로 연결).
GPU 1장에서 약 9 × 3.6 h.

### 8.2 네 번째 비교군: frame 단위 Random (scene 경계 무시) + context frame

답하려는 질문: 선택 방법 간 차이 중 얼마가 단순한 *scene 커버리지*인가? train keyframe 28,130개의 10 % =
2,813 frame이 70개 scene 전체 대신 700개 중 약 690개 scene에 4개씩 흩어진다. 이것은 **같은 frame 수** 비교이며
**같은 라벨링 비용** 비교가 아니다(논문은 clip 단위 라벨링이 현실적 비용 모델이라고 주장). 따라서 논문이 정의한
베이스라인이 아닌 커버리지 상한으로 보고한다.

단순 frame subset이 불공정한 이유: VAD는 `queue_length=3`으로 학습한다. 즉 직전 3 keyframe 중 2개(매 iteration
무작위)를 **gradient 없이** 백본에 통과시켜(`VAD.obtain_history_bev`) temporal BEV를 만들고, 손실은 마지막
frame에만 걸린다. 이전 frame이 없으면 데이터셋이 현재 frame을 복제하므로, frame subset은 "움직임 없는"
history로 학습하고 실제 0.5 s history로 평가받게 된다.

해결(구현 완료): `ego_diversity_select.py random-frames --frac 0.1 --context 3 --seed N`이 target을 추출하고
각 target의 직전 3 keyframe을 기록한다. `filter`가 둘을 모두 pkl에 쓰되 이전 frame에는 `context_only=True`를
표시한다. `VADCustomNuScenesDataset`은 `context_only` info를 `__len__`/target(`target_idx`)에서 제외하지만
`data_infos`에는 남겨 큐가 평소처럼 인덱싱하게 한다. 따라서 epoch 길이는 2,813 iteration이고 temporal 입력은
전체 scene 학습과 정확히 같다. 검증(`tools/active/validate_frame_subset.py`): ed10 pkl은 변화 없음(target 2,816,
target 3의 큐 frame_idx [1,2,3]); frand10_s0: info 9,348 = target 2,813 + context 6,535, 길이 2,813,
target 7의 큐 [5,6,7].

split: `tools/active/splits/frand10_s{0,1,2}.json`(커버 scene 690 / 695 / 690); run
`work_dirs/vad_tiny_frand10_s*_seed*`, γ=1 큐 뒤에 `tools/active/chain_frand.sh (copy of the one-off work_dirs/chain_frand.sh)`로 연결(컨테이너 `vad_queue4`).
전체 큐: seed 0 ED 이후 11 run, 각 약 3.6 h → 2026-11-01 01:30 KST 전후 종료.

## 9. 날씨·조명 밖의 cheap label 검토

description의 다른 단어, 데이터셋에서 그대로 쓸 수 있는 필드(location, CAN 7채널, HD map, 센서 원본), 전처리로
만드는 ego kinematics·휘도·LiDAR intensity 피처를 비교하고 대안 층화 3개의 cell 표를 정리했다:
`docs/activead_cheap_labels.md` (2026-10-30). 요점: 날씨·조명은 사실상 지역 분할이고(night = Singapore, rain = Boston),
`night`/`rain` 단어는 센서값으로 재현되며(AUC 0.99 / 0.91), 주행 행동(정지·선회)은 현재 층이 통제하지 못한다.
