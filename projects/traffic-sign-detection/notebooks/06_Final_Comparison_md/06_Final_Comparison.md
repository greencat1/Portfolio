# Full Model Comparison Table

| Model | Dataset | mAP50 | mAP50-95 | Precision | Recall |
|-------|---------|-------|----------|-----------|--------|
| **Baseline YOLO** | Train | 81.32% | 70.77% | 85.72% | 76.54% |
| **Baseline YOLO** | Test | 81.36% | 71.07% | 84.48% | 73.28% |
| | | | | | |
| **YOLO (preprocessed)** | Train | 84.27% | 73.51% | 90.25% | 82.12% |
| **YOLO (preprocessed)** | Test | 84.30% | 73.67% | 82.50% | 79.63% |
| | | | | | |
| **Hybrid (YOLO + 5 ResNet)** | Train | **97.92%** | **89.45%** | **97.85%** | **97.30%** |
| **Hybrid (YOLO + 5 ResNet)** | Test | **94.07%** | **85.60%** | **90.33%** | **91.71%** |

## Improvement Analysis

### Baseline → Preprocessed YOLO (Test)

| Metric | Baseline | Preprocessed YOLO | Change |
|--------|----------|-------------------|--------|
| mAP50 | 81.36% | 84.30% | **+2.94%** |
| mAP50-95 | 71.07% | 73.67% | **+2.60%** |
| Precision | 84.48% | 82.50% | **-1.98%** |
| Recall | 73.28% | 79.63% | **+6.35%** |

### Preprocessed YOLO → Hybrid (Test)

| Metric | Preprocessed YOLO | Hybrid | Change |
|--------|-------------------|--------|--------|
| mAP50 | 84.30% | 94.07% | **+9.77%** |
| mAP50-95 | 73.67% | 85.60% | **+11.93%** |
| Precision | 82.50% | 90.33% | **+7.83%** |
| Recall | 79.63% | 91.71% | **+12.08%** |

### Baseline → Hybrid (Test)

| Metric | Baseline | Hybrid | Total Improvement |
|--------|----------|--------|-------------------|
| mAP50 | 81.36% | 94.07% | **+12.71%** |
| mAP50-95 | 71.07% | 85.60% | **+14.53%** |
| Precision | 84.48% | 90.33% | **+5.85%** |
| Recall | 73.28% | 91.71% | **+18.43%** |

## Key Observations

### Baseline YOLO
- Lowest performance across all metrics
- Recall (73.28%) is particularly weak
- Precision (84.48%) is moderate

### Preprocessed YOLO
- Preprocessing improved recall by +6.35%
- mAP50 improved by +2.94%
- Small trade-off: precision decreased by -1.98%
- Test performance close to train (good generalization)

### Hybrid System
- **Best performance across ALL metrics**
- mAP50: 94.07% (best)
- mAP50-95: 85.60% (best - requires precise bounding boxes)
- Precision: 90.33% (best - fewest false positives)
- Recall: 91.71% (best - finds most signs)

### Train-Test Gap (Hybrid)

| Metric | Train | Test | Gap |
|--------|-------|------|-----|
| mAP50 | 97.92% | 94.07% | -3.85% |
| mAP50-95 | 89.45% | 85.60% | -3.85% |
| Precision | 97.85% | 90.33% | -7.52% |
| Recall | 97.30% | 91.71% | -5.59% |

- Mild overfitting (acceptable for production)

## Conclusions

1. **Hybrid system outperforms pure YOLO in EVERY metric**:
   - mAP50: +9.77%
   - mAP50-95: +11.93%
   - Precision: +7.83%
   - Recall: +12.08%

2. **Total improvement from baseline to hybrid**:
   - mAP50: +12.71%
   - mAP50-95: +14.53%
   - Precision: +5.85%
   - Recall: +18.43%

3. **Hybrid system is production-ready** with 94% mAP50 and 91.7% recall on test set


```python

```
