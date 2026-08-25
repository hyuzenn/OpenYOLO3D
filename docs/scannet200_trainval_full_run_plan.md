# ScanNet200 Train(1201)+Val(312) 전체 M22 Ablation 실행 계획

> 조사·계획 문서. **실제 다운로드/대규모 생성은 아직 실행하지 않음.**
> 코드 근거: `evaluate/__init__.py`(scene 목록 주입), `run_evaluation.py`,
> `method_scannet/streaming/tools/generate_mask3d_cache.py`,
> `models/Mask3D/mask3d/datasets/preprocessing/scannet_preprocessing.py`,
> `scripts/download_scannet200_filtered.sh`. 측정: `du`/`ls`/`comm`/`df` (2026-06-04).

---

## 1. 현재 디스크 상태 (data/scannet200 기준)

| 항목 | 값 | 근거 |
|---|---|---|
| 존재하는 scene 디렉토리 | **312개** | `ls -1d data/scannet200/scene*_*` |
| 그중 TRAIN split | **0개** | `comm -12 present train.txt` |
| 그중 VAL split | **312개 (전부)** | `comm -12 present val.txt` |
| **TRAIN 누락 scene** | **1201개 (전부 없음)** | `comm -13 present train.txt` |
| split 밖 scene | 0개 | `comm -23` |
| scene 데이터 총량(val) | **~118 GB** (GT 제외 ~117.8 GB) | `du -sh data/scannet200` |
| proposal cache `.pt` | **312개 / 9.5 GB** | `output/scannet200/scannet200_masks` |
| ground_truth `.txt` | **312개 / 239 MB** | `data/scannet200/ground_truth` |
| 가용 디스크 | **1703 GB** (FS 98% 사용) | `df -BG .` |

**scene 디렉토리 구성**(예: `scene0011_00`, 추출된 runtime 형태 — `.sens` 미보관):
```
0011_00.npy(11M)  color/(2374장,198M)  depth/(2374장,537M)
poses/(2374,9.4M)  intrinsics.txt  scene0011_00_vh_clean_2.ply(9.3M)
```
- 평균 ≈ **386 MB/scene** (depth가 용량 대부분). `.sens`는 전 scene에 0개 → val은 이미 추출 완료 상태로만 저장됨.

**결론(1):** train split은 디스크에 **하나도 없음**. val 312만 완비(데이터+`.pt`+GT).

---

## 2. train 1201을 파이프라인이 처리하기 위한 최소 자산 (scene당)

`run_evaluation.py` + `utils.WORLD_2_CAM` + `generate_mask3d_cache.py`가 실제로 여는 경로 기준 (전부 split-agnostic, scene 이름만 다름):

| 자산 | 경로 | 생성 주체 | 비고 |
|---|---|---|---|
| scene 디렉토리 | `data/scannet200/<scene>/` | SensReader 추출 + 복사 | 아래 5개 필수 |
| └ `color/<i>.jpg` | 〃 | SensReader (`--export_color_images`) | RGB 프레임 |
| └ `depth/<i>.png` | 〃 | SensReader (`--export_depth_images`) | **용량 대부분** |
| └ `poses/<i>.txt` | 〃 | SensReader (`--export_poses`) | per-frame extrinsic |
| └ `intrinsics.txt` | 〃 | SensReader (`--export_intrinsics` → rename) | |
| └ `<scene>_vh_clean_2.ply` | 〃 | 다운로드 후 복사 | mesh (glob `*.ply`) |
| └ `<id>.npy` | 〃 | Mask3D preprocessing | 좌표+RGB feature |
| proposal cache | `<cache_dir>/<scene>.pt` | `generate_mask3d_cache.py` | **재계산 필수**(다운로드 불가) |
| ground_truth | `data/scannet200/ground_truth/<scene>.txt` | Mask3D preprocessing | eval 정답 |

`generate_mask3d_cache.py` 기본값: `--cache-dir results/2026-05-13_mask3d_cache`,
`--config pretrained/config_scannet200.yaml`, scene 목록은 `SCENE_NAMES_SCANNET200`
(= 이번 패치로 `SCANNET200_SCENES_FILE`로 train 주입 가능).

**현재 누락된 변환 도구 (BLOCKER — 변환 전 반드시 해결):**
| 도구 | 상태 | 해결 |
|---|---|---|
| SensReader (`reader.py`/`SensorData.py`) | ❌ `third_party/ScanNet` 비어있음 | ScanNet repo의 `SensReader/python` 확보 |
| `scannetv2-labels.combined.tsv` | ❌ 없음 | `download-scannet.py --label_map` (1회) |
| Mask3D preprocessing | ✅ 존재 | `.../preprocessing/scannet_preprocessing.py` |
| `download-scannet.py` (TUM) | ✅ 존재 (13409 B) | repo 루트 |

---

## 3. 디스크 타당성 (가용 1703 GB)

per-scene 평균(현재 val 실측): 데이터 386 MB · `.pt` 31 MB · GT 0.78 MB.

| 시나리오 | train 1201 추정 | train+val 총 footprint | 1703 GB 내 가능? |
|---|---|---|---|
| **추출 형태만 보관**(`.sens` 미보관) | 데이터 ~463 GB + `.pt` ~37 GB + GT ~1 GB ≈ **~501 GB** | val 현재(~128 GB) + train(~501 GB) ≈ **~629 GB** | ✅ 여유 ~1 TB |
| **`.sens` 전량 보관** | `.sens` ~1–2 GB/scene × 1201 ≈ **~1.2–2.4 TB** (+추출분 ~0.5 TB) | **~1.7–2.9 TB** | ❌ **초과** (sens만으로도 위험) |

**결론(3) — 권장 다운로드 전략: `.sens` 스트리밍 후 즉시 삭제.**
- `.sens`를 전부 동시에 보관하면 1703 GB를 넘김(특히 train .sens 단독 ~1.8 TB).
- scene별로 **다운로드 → SensReader 추출 → `.sens` 삭제**를 배치(20~50 scene)로 반복 → 피크 증가분은 `배치크기 × ~1.5 GB`뿐.
- 최종 보관은 추출 형태만(~501 GB). train+val 합쳐 **~629 GB**로 가용 내 충분.
- (선택) 정리: val `.pt`가 `output/scannet200/scannet200_masks`와 `results/2026-05-13_mask3d_cache` **두 곳에 중복(~19 GB)** — 한쪽 제거로 9.5 GB 회수 가능(소유자/사용처 확인 후).

---

## 4. 다운로드 완료 후 실행 순서 (문서화 — 미실행)

> 모든 단계는 `SCANNET200_SCENES_FILE`로 scene 목록을 주입한다(코드 무수정).
> train+val 통합 평가를 위해 통합 split 파일을 만든다:
> `cat splits/scannetv2_train.txt splits/scannetv2_val.txt > splits/scannetv2_trainval.txt` (1513줄)

### Phase 0 — 사전 도구 (1회)
1. SensReader 확보: ScanNet repo의 `SensReader/python/{reader.py,SensorData.py}` → `third_party/ScanNet/`.
2. label map: `yes | python download-scannet.py -o data/raw/scannet --label_map`
   → `scannetv2-labels.combined.tsv`.

### Phase 1 — train 원본 다운로드 (스트리밍, `.sens` 일시적)
```bash
DOWNLOAD_SCRIPT=./download-scannet.py SPLIT=train INCLUDE_SENS=1 \
  bash scripts/download_scannet200_filtered.sh        # 배치/LIMIT로 분할 권장
```
- 받는 타입: `_vh_clean_2.ply`, `_vh_clean_2.labels.ply`, `.aggregation.json`,
  `_vh_clean_2.0.010000.segs.json`, `.txt`, `.sens`.

### Phase 2 — 데이터 변환 (scene별: 추출 → `.sens` 삭제)
1. SensReader 추출 → `data/scannet200/<scene>/{color,depth,poses}` + `intrinsics.txt`,
   `intrinsic/intrinsic_depth.txt → intrinsics.txt` rename, `_vh_clean_2.ply` 복사.
2. **추출 직후 해당 `.sens` 삭제**(디스크 피크 억제).
3. Mask3D preprocessing → `<id>.npy` + `ground_truth/<scene>.txt`:
   ```bash
   cd models/Mask3D && python -m datasets.preprocessing.scannet_preprocessing \
     preprocess --data_dir <raw> --save_dir <save> --scannet200 true
   ```

### Phase 3 — proposal cache 생성 (GPU, A100 권장)
```bash
SCANNET200_SCENES_FILE=data/scannet200/splits/scannetv2_train.txt \
python -m method_scannet.streaming.tools.generate_mask3d_cache \
  --cache-dir output/scannet200/scannet200_masks \
  --config pretrained/config_scannet200.yaml
```
- 기존 val `.pt`와 **같은 디렉토리**에 train `.pt`를 채움(생성기는 존재 파일 skip → 통합 캐시 완성).
- 추정 +37 GB.

### Phase 4 — inference + evaluation (train+val 1513 통합)
```bash
SCANNET200_SCENES_FILE=data/scannet200/splits/scannetv2_trainval.txt \
python -m method_scannet.eval_method_22_conf_skip \
  --dataset_name scannet200 \
  --path_to_3d_masks ./output/scannet200/scannet200_masks \
  <M22 ablation 플래그>
```
- `data/`, `ground_truth/`, `.pt` 모두 scene 이름 기반 → 통합 목록이면 그대로 동작.

---

## ⚠️ 방법론적 주의 (코드 아님 — 결과 보고 시 명시 필수)
`pretrained/checkpoints/scannet200_val.ckpt`는 **train split으로 학습된** Mask3D 체크포인트다.
이를 train scene에 추론하면 **in-sample(학습 데이터) proposal**이라 train 구간 성능이
낙관적으로 부풀 수 있다. Train+Val 합산 mAP 보고 시 train/val 분리 수치를 함께 제시할 것.

---

## 최종 결론
- **현재:** val 312만 완비, **train 1201 전량 부재**. 코드(scene 목록)는 이미 train 대응 완료.
- **디스크:** `.sens` 미보관 전략이면 train+val ~629 GB로 1703 GB 내 충분. `.sens` 전량 보관은 불가.
- **선행 BLOCKER:** SensReader + label map tsv 확보(둘 다 현재 없음).
- 위 Phase 0→4 순서로 진행하면 Train+Val 전체에서 M22 ablation 실행 가능.
