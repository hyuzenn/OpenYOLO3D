CLAUDE.md

🛠 Quick Start (Commands)
  Environment: conda activate openyolo3d (Python 3.10 based on pycache)
  GPU 할당: coss_agpu -g 1

📂 실험 로그 저장 규칙 (results/)
  포맷: results/<YYYY-MM-DD>_<experiment>_v<NN>/
    - YYYY-MM-DD : 실행 날짜 (예: 2026-05-05)
    - experiment : 실험 종류 (snake_case)
        · scannet_demo       : 단일 씬 데모
        · scannet_eval       : ScanNet200 전체 평가
        · replica_eval       : Replica 평가
        · ablation_<name>    : ablation 실험 (예: ablation_topk)
    - NN         : 같은 날·같은 실험 내 버전 (01부터 시작, zero-padded)

  예시:
    results/2026-05-05_scannet_eval_v01/
    results/2026-05-05_scannet_eval_v02/
    results/2026-05-05_ablation_topk_v01/

  디렉토리 내부 표준 파일:
    run.log         : stdout/stderr 전체 로그
    config.yaml     : 실험에 사용한 config 스냅샷 (재현성)
    metrics.json    : 최종 평가 지표 (mAP, mIoU 등)
    outputs/        : 산출물 (ply, npy, 시각화 등)
    notes.md        : 실험 메모 (선택)

  주의:
    - results/ 는 .gitignore 됨. 단 results/experiment_tracker.md 는 추적됨.
    - 같은 실험을 재실행할 때는 v 번호를 올리고, 기존 디렉토리는 덮어쓰지 말 것.

  실험 스크립트 헤더 (pbs/sh 공통):
    DATE=$(date +%F)
    EXP=scannet_eval                  # ← 실험 종류만 바꿔서 재사용
    CONFIG=pretrained/config_scannet200.yaml
    N=$(printf '%02d' $(($(ls -d results/${DATE}_${EXP}_v* 2>/dev/null | wc -l) + 1)))
    RUN_DIR=results/${DATE}_${EXP}_v${N}
    mkdir -p "$RUN_DIR/outputs"
    cp "$CONFIG" "$RUN_DIR/config.yaml"
    # 이후 출력은 $RUN_DIR/outputs/ 로, stdout/stderr 는 $RUN_DIR/run.log 로 리다이렉트
    # 예: python run_evaluation.py --dataset scannet200 ... 2>&1 | tee "$RUN_DIR/run.log"
