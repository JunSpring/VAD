# ActiveAD 10% cold-start (Ego-Diversity) 재현 — 중간 요약 / Interim summary

> 한글 버전 → 아래 영어 버전. Korean first, English below.

---

## 🇰🇷 한글

### 1. Ego-Diversity(ED)를 어떻게 구현했는지

- **선택 단위**: nuScenes train 700 scene 중 70 scene (10%). 논문과 같이 scene(clip)이 라벨링 단위.
- **1단계 — 날씨·조명**: nuScenes `scene.json`의 description 문자열에 `night` 포함 → Night, `rain` 포함 → Rainy, 나머지는 Day/Sunny. 결과 DS/DR/NS/NR = 491/125/71/13 — 논문 Table 4 "Complete"와 정확히 일치.
- **2단계 — 주행 명령**: VAD 컨버터가 keyframe마다 만드는 `gt_ego_fut_cmd`(3 s 후 ego의 횡방향 오프셋이 +2 m 이상 → Right, −2 m 이하 → Left, 그 외 Straight)를 scene 단위로 집계. 논문 규칙(τc = 4)을 그대로 적용: L ≥ 4이고 R ≥ 4 → Overtake, L > 4만 → Left, R > 4만 → Right, 그 외 Straight. 결과 S/R/L/O = 423/132/112/33 — 논문과 정확히 일치.
- **할당(γ = 0.5)**: Eq.1 P_x = n_x^γ / Σ n_z^γ, Eq.2 P_xy = P_x · n_xy^γ / Σ n_xz^γ, n0 = 70. 반올림은 단계별 largest-remainder(논문 미명시) → 1단계 34/17/13/6(논문과 동일), 명령 합계 S29/R17/L15/O9(논문 27/17/16/10, 셀 단위 반올림으로는 줄일 수 없는 ±2 차이).
- **셀 내 선택**: scene 평균 속도(CAN bus `v0`, 블랙리스트 scene은 pose 기반 속도) 오름차순 정렬 후 등간격 `floor((k+0.5)·n/K)`로 선택. 결정론적, seed 무관.
- **최종**: 70 scene / 2,816 frame. 선택 결과·파라미터·논문 대조값은 `tools/active/splits/ed10_gamma0.5.json`에 기록. 코드: `tools/active/scene_attrs.py`, `tools/active/ego_diversity_select.py` (`check / select / filter / random / random-frames`).

### 2. 학습 세부사항

- **하드웨어**: RTX 4090 1장 (24 GB), Docker 이미지 `vad:cu118-torch2.0.1` (CUDA 11.8, torch 2.0.1, mmcv-full 1.7.1, mmdet 2.28.2, mmdet3d 0.17.1, shapely 1.8.5).
- **데이터**: nuScenes v1.0-trainval + CAN bus + map expansion v1.3. VAD 컨버터로 전체 info pkl 생성(train 28,130 / val 6,019 keyframe) 후 70 scene만 필터한 pkl로 학습. 평가는 전체 val 150 scene.
- **모델/하이퍼파라미터**: VAD-Tiny(`VAD_tiny_e2e.py` 기반), ImageNet R50에서 from scratch, AdamW lr 2e-4, wd 0.01, backbone lr ×0.1, CosineAnnealing(min_lr_ratio 1e-3), linear warmup 500 optimizer step, grad clip 35, **20 epoch**.
- **배치**: 논문은 8 GPU × 1 = 배치 8. 우리는 1 GPU × 1에 **gradient accumulation ×8**로 유효 배치 8을 맞춤(mmcv `GradientCumulativeOptimizerHook`). 백본 BN이 고정(norm_eval)이고 temporal queue가 샘플 내부에서 만들어지므로 수학적으로 8 GPU 배치와 동일. epoch당 2,816 iter = 352 optimizer step, 20 epoch = 56k iter ≈ 7,040 step.
- **소요 시간**: 0.21 s/iter, 학습 3시간 18분, GPU 메모리 약 6 GB. seed 0, `--deterministic`.
- **평가**: `tools/test.py` 비분산 1 GPU, 전체 val, VAD 지표(L2 1/2/3 s, collision 1/2/3 s). 샘플별 지표를 json으로 저장하도록 수정해 Day/Night/Sunny/Rainy/명령별 breakdown(논문 Table 3 형식)을 뽑음.

### 3. 결과 (seed 0, epoch 20, 전체 val 5,119 valid frame)

| 설정 | L2 1s | L2 2s | L2 3s | **L2 avg (m)** | Col 1s | Col 2s | Col 3s | **Col avg (%)** |
|---|---|---|---|---|---|---|---|---|
| **우리 ED 10% (seed 0, ep20)** | 0.470 | 0.776 | 1.158 | **0.801** | 0.58 | 0.64 | 0.76 | **0.66** |
| 논문 ED 10% | 0.47 | 0.80 | 1.21 | 0.83 | 0.13 | 0.35 | 0.80 | 0.43 |
| 논문 Random 10% | 0.51 | 0.83 | 1.23 | 0.86 | 0.40 | 0.62 | 0.98 | 0.67 |
| 논문 100% 데이터 | 0.38 | 0.68 | 1.04 | 0.70 | 0.15 | 0.22 | 0.39 | 0.25 |

- L2는 논문보다 약간 좋고(0.80 vs 0.83), 1 s L2는 정확히 일치. 충돌률은 논문 ED보다 높고 논문 Random 수준.

**같은 run의 마지막 체크포인트 3개 (전체 val, 제외 없음)** — L2는 그대로인데 충돌률만 크게 흔들림:

| 체크포인트 | L2 1s | L2 2s | L2 3s | **L2 avg** | Col 1s | Col 2s | Col 3s | **Col avg** | scene-0344/0345의 1 s 충돌 frame |
|---|---|---|---|---|---|---|---|---|---|
| epoch 18 | 0.473 | 0.777 | 1.153 | **0.801** | 0.10 | 0.26 | 0.48 | **0.28** | 0 / 9 |
| epoch 19 | 0.463 | 0.764 | 1.143 | **0.790** | 0.44 | 0.48 | 0.63 | **0.51** | 30 / 39 |
| epoch 20 | 0.470 | 0.776 | 1.158 | **0.801** | 0.58 | 0.64 | 0.76 | **0.66** | 32 / 41 |

- 논문 ED의 0.43은 epoch 18(0.28)과 19(0.51) 사이에 있음. 마지막 열이 보여주듯 변동은 거의 전부 scene 두 개에서 나옴(4절 참조).
- 시나리오별: Night L2 0.95 / Col 0.58, Rainy 0.74 / 0.30, Turn Left 0.85 / 0.36, Turn Right 0.89 / 0.55, Overtake 0.92 / 0.33.
- 보조 헤드: detection NDS 0.176, map mAP 0.078. 학습 손실은 20 epoch 동안 단조 감소(20.5 → 6.46), NaN 없음.

### 4. 왜 논문과 다른가 (분석 결과)

- **충돌률 차이는 val scene 두 개에서 전부 나옴.** 1 s 충돌 41 frame 중 32개가 scene-0344/0345(교차로 대기, 옆에 버스·공사). ego는 완전 정지(GT 이동 0.00 m)이고 예측 L2도 0.00 m, 즉 궤적은 완벽한데 "충돌"로 집계.
- **원인은 VAD 충돌 지표의 래스터화.** ego 박스를 0.5 m BEV 격자에 그리고 좌표를 `int` 절삭하는데, 박스 모서리가 셀 경계에 걸려 있어 **1 mm만 옆차 쪽으로 움직여도 플래그가 뒤집힘**(실제 프레임으로 재현: GT 0.000 m → 충돌 0, −0.001 m → 6 step 모두 충돌). 옆차와 실제 간격은 0.58 m.
- **노이즈 크기 측정.** 같은 run의 epoch 18/19/20이 L2는 0.80/0.79/0.80으로 같은데 collision은 0.28/0.51/0.66. scene 단위 부트스트랩 95% 구간: L2 [0.75, 0.85], Col [0.21, 1.45]. 논문의 0.83/0.43은 둘 다 구간 안 → 단일 run으로는 논문과 "다르다"도 "더 좋다"도 말할 수 없음. 논문의 Random–ED 충돌 차이(0.24 pt)도 우리 체크포인트 간 변동(0.38 pt)보다 작음.
- **학습 설정은 원인이 아님.** grad accumulation은 8 GPU 배치와 동치, lr/wd/스케줄/epoch/step 수 모두 논문과 동일, 손실 정상 수렴.
- **제거 불가한 잔여 차이.** 논문의 70 scene 목록 비공개(셀 내 선택이 속도 소스에 따라 달라짐, 명령 합계 ±2), 논문이 VAD 베이스라인을 "업데이트"했다고 각주에 명시(VAD-Tiny 100%: 논문 0.70/0.25 vs VAD README 0.78/0.38) → 평가 코드 버전이 다를 가능성.
- **주의**: scene을 제외하면 Col 0.27이 되지만 이는 논문과의 정당한 비교가 아님(논문 수치도 그 모델이 겪은 장면을 모두 포함). 결론은 "seed 여러 개 없이는 충돌률 비교 불가".

### 5. 앞으로 할 실험 (큐 등록 완료, GPU 1장 순차 실행, 11/1 새벽 종료 예정)

4개 비교군 × 3 seed(학습 seed로 쌍 비교), 각 run 약 3.6 h:

1. **ED γ=0.5** (논문 방법) — seed 0 완료, seed 1·2 진행 중
2. **Random (scene 단위 균등, 논문 Table 1의 Random)** — seed 0·1·2
3. **ED γ=1** (논문 Table 4: pool 비례 층화 + 속도 등간격, 결정론적) — seed 0·1·2. Random → γ=1 → γ=0.5로 "층화·속도 등간격 자체의 효과"와 "소수 범주 가중의 효과"를 분리
4. **frame 단위 Random** (scene 경계 무시, 2,813 frame이 약 690 scene에 분산) — seed 0·1·2. 커버리지 상한 역할. VAD temporal queue가 직전 3 keyframe을 쓰므로 각 target의 직전 3 frame을 context 전용(손실 없음, epoch 길이 미포함)으로 함께 넣어 공정하게 비교

보고 방식: 비교군별 평균 ± 표준편차, 같은 seed끼리 쌍 비교, 충돌률은 epoch 18–20 범위 함께 제시, Night/Rainy/Turn 시나리오별 breakdown.

상세 문서: `docs/activead_ed10.md`

---

## 🇺🇸 English

### 1. How Ego-Diversity (ED) was implemented

- **Selection unit**: 70 of the 700 nuScenes train scenes (10%). As in the paper, the scene (clip) is the annotation unit.
- **Level 1 — weather/lighting**: from the free-text description in nuScenes `scene.json`: contains `night` → Night, contains `rain` → Rainy, otherwise Day/Sunny. Result DS/DR/NS/NR = 491/125/71/13, an exact match to the paper's Table 4 "Complete" row.
- **Level 2 — driving command**: VAD's per-keyframe `gt_ego_fut_cmd` (lateral offset of the ego pose 3 s ahead: ≥ +2 m → Right, ≤ −2 m → Left, else Straight) aggregated per scene, then the paper's rule with τc = 4: L ≥ 4 and R ≥ 4 → Overtake; only L > 4 → Left; only R > 4 → Right; else Straight. Result S/R/L/O = 423/132/112/33, exact match.
- **Allocation (γ = 0.5)**: Eq.1 P_x = n_x^γ / Σ n_z^γ, Eq.2 P_xy = P_x · n_xy^γ / Σ n_xz^γ, n0 = 70. Rounding (unspecified in the paper) is largest-remainder per level → level 1 = 34/17/13/6 (same as paper); command marginals S29/R17/L15/O9 vs paper 27/17/16/10 (±2, irreducible by any per-cell rounding).
- **In-cell pick**: sort scenes by mean speed (CAN bus `v0`; pose-based speed for the CAN-blacklisted scenes) and take regular intervals `floor((k+0.5)·n/K)`. Deterministic, seed-independent.
- **Result**: 70 scenes / 2,816 frames. Parameters, selected scenes and the paper's reference counts are logged in `tools/active/splits/ed10_gamma0.5.json`. Code: `tools/active/scene_attrs.py`, `tools/active/ego_diversity_select.py` (`check / select / filter / random / random-frames`).

### 2. Training details

- **Hardware**: one RTX 4090 (24 GB), Docker image `vad:cu118-torch2.0.1` (CUDA 11.8, torch 2.0.1, mmcv-full 1.7.1, mmdet 2.28.2, mmdet3d 0.17.1, shapely 1.8.5).
- **Data**: nuScenes v1.0-trainval + CAN bus + map expansion v1.3. Full info pkls generated with the VAD converter (28,130 train / 6,019 val keyframes), then filtered to the 70 selected scenes. Evaluation on the full val split (150 scenes).
- **Model / hyper-parameters**: VAD-Tiny (`VAD_tiny_e2e.py` based), trained from scratch from ImageNet R50; AdamW lr 2e-4, wd 0.01, backbone lr ×0.1, CosineAnnealing (min_lr_ratio 1e-3), linear warmup of 500 optimizer steps, grad clip 35, **20 epochs**.
- **Batch**: the paper uses 8 GPUs × 1 = batch 8. We use 1 GPU × 1 with **gradient accumulation ×8** (mmcv `GradientCumulativeOptimizerHook`) for an effective batch of 8. Backbone BN is frozen (norm_eval) and the temporal queue is built inside each sample, so this is mathematically equivalent to the 8-GPU batch. 2,816 iters/epoch = 352 optimizer steps; 20 epochs = 56k iters ≈ 7,040 steps.
- **Cost**: 0.21 s/iter, 3 h 18 min of training, ~6 GB GPU memory. Seed 0, `--deterministic`.
- **Evaluation**: `tools/test.py`, single GPU non-distributed, full val, VAD metrics (L2 and collision at 1/2/3 s). Patched to dump per-sample metrics to json so that the Day/Night/Sunny/Rainy/command breakdown (paper Table 3 format) can be computed.

### 3. Results (seed 0, epoch 20, full val, 5,119 valid frames)

| Setting | L2 1s | L2 2s | L2 3s | **L2 avg (m)** | Col 1s | Col 2s | Col 3s | **Col avg (%)** |
|---|---|---|---|---|---|---|---|---|
| **Ours, ED 10% (seed 0, ep20)** | 0.470 | 0.776 | 1.158 | **0.801** | 0.58 | 0.64 | 0.76 | **0.66** |
| Paper ED 10% | 0.47 | 0.80 | 1.21 | 0.83 | 0.13 | 0.35 | 0.80 | 0.43 |
| Paper Random 10% | 0.51 | 0.83 | 1.23 | 0.86 | 0.40 | 0.62 | 0.98 | 0.67 |
| Paper 100% data | 0.38 | 0.68 | 1.04 | 0.70 | 0.15 | 0.22 | 0.39 | 0.25 |

- L2 is slightly better than the paper (0.80 vs 0.83) and the 1 s L2 matches exactly. Collision is above the paper's ED and at its Random level.

**Last three checkpoints of the same run (full val, nothing excluded)** — L2 is stable while collision swings widely:

| Checkpoint | L2 1s | L2 2s | L2 3s | **L2 avg** | Col 1s | Col 2s | Col 3s | **Col avg** | 1 s collision frames in scene-0344/0345 |
|---|---|---|---|---|---|---|---|---|---|
| epoch 18 | 0.473 | 0.777 | 1.153 | **0.801** | 0.10 | 0.26 | 0.48 | **0.28** | 0 / 9 |
| epoch 19 | 0.463 | 0.764 | 1.143 | **0.790** | 0.44 | 0.48 | 0.63 | **0.51** | 30 / 39 |
| epoch 20 | 0.470 | 0.776 | 1.158 | **0.801** | 0.58 | 0.64 | 0.76 | **0.66** | 32 / 41 |

- The paper's ED value of 0.43 lies between epoch 18 (0.28) and epoch 19 (0.51). As the last column shows, almost all of the swing comes from two scenes (see section 4).
- Per scenario: Night L2 0.95 / Col 0.58, Rainy 0.74 / 0.30, Turn Left 0.85 / 0.36, Turn Right 0.89 / 0.55, Overtake 0.92 / 0.33.
- Auxiliary heads: detection NDS 0.176, map mAP 0.078. Training loss decreased monotonically over 20 epochs (20.5 → 6.46), no NaN.

### 4. Why the numbers differ from the paper (analysis)

- **The entire collision gap comes from two val scenes.** 32 of the 41 frames flagged as colliding at 1 s are in scene-0344/0345 (waiting at an intersection next to a bus and construction). The ego is fully stopped (GT displacement 0.00 m) and the predicted L2 is 0.00 m, i.e. the trajectory is perfect yet counted as a collision.
- **Root cause: rasterisation in VAD's collision metric.** The ego footprint is drawn on a 0.5 m BEV grid and coordinates are truncated to `int`; the footprint edges sit exactly on cell boundaries, so **a 1 mm shift towards the neighbour flips the flag** (reproduced on the real frame: GT 0.000 m → no collision, −0.001 m → collision at all 6 steps). The physical gap to the neighbouring car is 0.58 m.
- **Noise level measured.** Epochs 18/19/20 of the same run have L2 0.80/0.79/0.80 but collision 0.28/0.51/0.66. Scene-level bootstrap 95% CIs: L2 [0.75, 0.85], Col [0.21, 1.45]. The paper's 0.83/0.43 lies inside both, so a single run can claim neither "different" nor "better". The paper's own Random–ED collision gap (0.24 pt) is smaller than our checkpoint-to-checkpoint swing (0.38 pt).
- **Training setup is not the cause.** Gradient accumulation is equivalent to the 8-GPU batch; lr/wd/schedule/epochs/step counts match the paper; the loss converged normally.
- **Irreducible differences.** The paper's 70 scenes are unpublished (in-cell picks depend on their speed source; command marginals ±2), and a footnote says the VAD baselines were "updated" (VAD-Tiny 100%: paper 0.70/0.25 vs VAD README 0.78/0.38), so the evaluation code version may differ.
- **Caveat**: excluding the two scenes gives Col 0.27, but that is not a fair comparison with the paper (its number also includes whatever its model hit). The conclusion is: collision cannot be compared without multiple seeds.

### 5. Planned experiments (already queued, sequential on one GPU, finishing ~Nov 1 early morning)

Four arms × 3 seeds (paired by training seed), ~3.6 h per run:

1. **ED γ=0.5** (the paper's method) — seed 0 done, seeds 1·2 running
2. **Random (scene-level uniform, the paper's Table 1 "Random")** — seeds 0·1·2
3. **ED γ=1** (paper Table 4: pool-proportional stratification + speed intervals, deterministic) — seeds 0·1·2. Random → γ=1 → γ=0.5 separates the effect of stratification/speed-interval picking from the effect of up-weighting minority categories.
4. **Frame-level Random** (scene boundaries ignored; 2,813 frames spread over ~690 scenes) — seeds 0·1·2, as a coverage upper bound. Because VAD's temporal queue uses the 3 preceding keyframes, each target's 3 predecessors are included as context-only frames (no loss, not counted in the epoch) so the comparison is fair.

Reporting: mean ± std per arm, paired comparison by seed, collision reported with its epoch 18–20 range, and Night/Rainy/Turn scenario breakdowns.

Full write-up: `docs/activead_ed10.md`
