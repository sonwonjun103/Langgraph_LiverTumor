# Langgraph_LiverTumor

이 문서는 코드 정리를 시작하기 전에 프로젝트의 전체 흐름과 정리할 항목을 한곳에 모아두기 위한 README입니다.

## 1. Data

- Data1: 47 cases
- Data2: 87 cases
- 전체 기준: 2,000 tumor size 이상
- Split
  - Train: 100
  - Test: 34

## 2. Registration

### Phase Registration

- A, P, D phase registration 정리
- Affine transformation
- Deformable registration
- Ablation study 대상

### Learning-Based Registration

- VoxelMorph
- TransMorph
- Learning-based registration 실험 정리

## 3. Registration Quality Metrics

- TotalSegmentator를 이용한 3-phase liver Dice 평가
- 목표 기준
  - Dice > 0.95
- 0.95가 안될시 다시 registration 시도

## 4. Model

### Candidate Models

- nnUNet
- U-Net
- 3D SAM adapter
- Transformer-based model

### Development

- 자체 모델 개발 여부 검토
- 기존 모델 baseline 정리
- 모델별 input/output 형식 통일

### Evaluation

- Segmentation
  - Dice
  - IoU
- Detection
  - Confusion matrix

## 5. LangGraph

- 전체 pipeline orchestration 후보
- Data preprocessing, registration, model inference, evaluation 단계를 graph node로 분리할 수 있는지 검토

## TODO

- [ ] 데이터 폴더 구조 정리 -> 이건 단지 train 을위한 경로 지정
- [ ] train/test split 파일 생성
- [ ] Resampling 코드 생성
- [ ] registration 코드 분리
- [ ] affine/deformable registration ablation 정리
- [ ] VoxelMorph/TransMorph 실험 코드 정리 -> 해도 되고 안해도 되고
- [ ] TotalSegmentator 기반 Dice 평가 코드 정리
- [ ] baseline model 목록 확정
    - nnUNetv2, 3D UNet, 3D SAM adapter, transformer-based model
- [ ] segmentation/detection evaluation script 정리
- [ ] LangGraph 적용 여부 결정
