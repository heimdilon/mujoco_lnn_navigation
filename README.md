# MuJoCo LNN/CfC Mobil Robot Navigasyonu

Bu proje, diferansiyel sürüşlü bir mobil robotun MuJoCo ortamında 32 ışınlı
LiDAR gözlemiyle statik ve dinamik engelli haritalarda hedefe ulaşmasını
inceler. Ana yöntem saf LNN/CfC politikasıdır. Değerlendirme sırasında safety
filter, otomatik waypoint veya kural tabanlı kontrol kullanılmaz; aksiyon
doğrudan öğrenilmiş politikadan gelir.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heimdilon/mujoco_lnn_navigation/blob/master/notebooks/colab_custom22_training.ipynb)

## Güncel Durum

Son deneyde `cfc_deep192_custom22_dynamic_dagger2` modeli Colab üzerinde
sıfırdan eğitildi ve 22 custom harita + 2 dinamik harita üzerinde test edildi.

| Model | Mimari | Harita | Başarı | Çarpışma | Timeout |
| --- | --- | ---: | ---: | ---: | ---: |
| `cfc_deep192_custom22_dynamic_dagger2` | `Linear+Tanh -> CfC -> CfC` | 24 | 21/24 | 3/24 | 0/24 |

Kırılım:

| Küme | Harita | Başarı | Not |
| --- | ---: | ---: | --- |
| Custom train | 18 | 17/18 | `custom_map_16` çarpıştı |
| Custom holdout | 4 | 3/4 | `custom_map_03` çarpıştı |
| Dynamic train | 1 | 1/1 | `dynamic_open_single` başarılı |
| Dynamic holdout | 1 | 0/1 | `dynamic_crossing` çarpıştı |

Güncel haftalık rapor:

- [11 Mayıs 2026 haftalık raporu PDF](report/weekly_2026_05_11/build/weekly_report.pdf)
- [LaTeX kaynağı](report/weekly_2026_05_11/weekly_report.tex)
- [Haftalık metrik CSV](report/weekly_2026_05_11/weekly_metrics.csv)

Ana proje raporu:

- [Proje raporu PDF](report/build/main.pdf)

## Saf Deep CfC/LNN Mimarisi

Güncel model `cfc_deep` adıyla kayıtlıdır. Yapı:

```text
obs(38)
  -> Linear(38, 192)
  -> Tanh
  -> CfC(hidden=192, dt=0.08)
  -> CfC(hidden=192, dt=0.08)
  -> actor: [linear, angular]
  -> critic: value
```

Gözlem ve aksiyon sözleşmesi:

| Boyut | Açıklama |
| ---: | --- |
| 38 | Hedef mesafesi, hedef açısı, hız (2), önceki aksiyon (2), 32 LiDAR ışını |
| 2 | Aksiyon: `[linear, angular]` ve her bileşen `[-1, 1]` aralığında |

Bu mimari saf LNN/CfC çizgisini korur. Eğitimde BC/DAgger için öğretmen
trajektori üretimi yapılabilir, fakat değerlendirme anında politika kendi
aksiyonunu üretir.

## Dinamik Haritalar

Ortam artık `map.dynamic_obstacles` alanıyla hareketli engelleri destekler.
Mevcut gözlem boyutu değişmedi; hareketli engeller LiDAR ışınlarına ve çarpışma
kontrolüne dahil edilir.

Eklenen dinamik haritalar:

| Harita | Split | Açıklama |
| --- | --- | --- |
| `configs/maps/dynamic_open_single.yaml` | train | Açık alanda tek hareketli silindir |
| `configs/maps/dynamic_crossing.yaml` | holdout | Koridor benzeri statik engeller + iki hareketli engel |

Rollout PNG/GIF çıktıları hareketli engel izlerini de gösterir.

## Colab ile Sıfırdan Eğitim

Notebook:

- [`notebooks/colab_custom22_training.ipynb`](notebooks/colab_custom22_training.ipynb)

Colab üzerinde:

1. `Runtime > Change runtime type > GPU` seç.
2. Notebook'u üstten alta çalıştır.
3. Sıfırdan eğitim için `RESUME_CHECKPOINT = ""` boş kalmalı.

Varsayılan güncel ayarlar:

```python
RUN_NAME = "cfc_deep192_custom22_dynamic_dagger2"
SPLIT_CONFIG = "configs/splits/custom22_dynamic_seed25462877008.yaml"
TRAIN_CONFIG = "configs/train/bc_cfc_deep192_dynamic_maps.yaml"
BC_EPOCHS_FROM_SCRATCH = 600
DAGGER_ITERATIONS = 2
```

Eğitim sonunda beklenen checkpoint:

```text
results/cfc_deep192_custom22_dynamic_dagger2/latest.pt
```

Not: Büyük checkpoint dosyaları ve `results/` klasörü Git'e eklenmez.

## Yerel Kurulum

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

Son doğrulamada testler:

```text
25 passed
```

## Güncel Modeli Değerlendirme

Colab'da eğitilen checkpoint'i `results/cfc_deep192_custom22_dynamic_dagger2/latest.pt`
konumuna koyduktan sonra tüm 24 haritada saf politika değerlendirmesi:

```powershell
.\.venv\Scripts\python.exe scripts\batch_evaluate.py ^
  --map-configs ^
    configs\maps\custom_map_01.yaml configs\maps\custom_map_02.yaml ^
    configs\maps\custom_map_04.yaml configs\maps\custom_map_05.yaml ^
    configs\maps\custom_map_06.yaml configs\maps\custom_map_07.yaml ^
    configs\maps\custom_map_09.yaml configs\maps\custom_map_10.yaml ^
    configs\maps\custom_map_11.yaml configs\maps\custom_map_12.yaml ^
    configs\maps\custom_map_13.yaml configs\maps\custom_map_14.yaml ^
    configs\maps\custom_map_16.yaml configs\maps\custom_map_18.yaml ^
    configs\maps\custom_map_19.yaml configs\maps\custom_map_20.yaml ^
    configs\maps\custom_map_21.yaml configs\maps\custom_map_22.yaml ^
    configs\maps\dynamic_open_single.yaml ^
    configs\maps\custom_map_03.yaml configs\maps\custom_map_08.yaml ^
    configs\maps\custom_map_15.yaml configs\maps\custom_map_17.yaml ^
    configs\maps\dynamic_crossing.yaml ^
  --checkpoint results\cfc_deep192_custom22_dynamic_dagger2\latest.pt ^
  --episodes 4 ^
  --run-name cfc_deep192_custom22_dynamic_dagger2_eval_all24_visual ^
  --max-steps 900 ^
  --goal-observation-max 10.0 ^
  --device cpu
```

Çıktılar:

```text
results/cfc_deep192_custom22_dynamic_dagger2_eval_all24_visual/summary.csv
results/cfc_deep192_custom22_dynamic_dagger2_eval_all24_visual/*/rollout.png
results/cfc_deep192_custom22_dynamic_dagger2_eval_all24_visual/*/rollout.gif
```

## Eğitim Komutları

BC eğitimi:

```powershell
.\.venv\Scripts\python.exe scripts\train_bc.py ^
  --split-config configs\splits\custom22_dynamic_seed25462877008.yaml ^
  --train-config configs\train\bc_cfc_deep192_dynamic_maps.yaml ^
  --run-name cfc_deep192_custom22_dynamic_dagger2 ^
  --epochs 600 ^
  --save-interval 50 ^
  --no-final-eval ^
  --device cuda
```

DAgger devam eğitimi:

```powershell
.\.venv\Scripts\python.exe scripts\train_bc.py ^
  --split-config configs\splits\custom22_dynamic_seed25462877008.yaml ^
  --train-config configs\train\bc_cfc_deep192_dynamic_maps.yaml ^
  --run-name cfc_deep192_custom22_dynamic_dagger2 ^
  --resume results\cfc_deep192_custom22_dynamic_dagger2\latest.pt ^
  --epochs 0 ^
  --dagger-iterations 2 ^
  --dagger-rollouts-per-map 4 ^
  --dagger-epochs 80 ^
  --save-interval 50 ^
  --no-final-eval ^
  --device cuda
```

## Eski Deneylerden Kısa Tarihçe

Önceki tek katmanlı CfC modeli `cfc_radius010_custom22_dagger2`, 24 haritalık
testte 10/24 başarı üretmişti. Yeni iki katmanlı `cfc_deep192` yapısı aynı test
yüzeyinde 21/24 başarıya çıktı. Kalan hata modu timeout değil, belirli
haritalarda çarpışmadır.

Başarısız güncel haritalar:

- `custom_map_16`
- `custom_map_03`
- `dynamic_crossing`

Bu nedenle sonraki çalışma, saf LNN yapısını bozmadan çarpışma davranışını
azaltmaya odaklanmalıdır.

## Faydalı Araçlar

Harita editörü:

```powershell
.\.venv\Scripts\python.exe scripts\map_editor.py --port 8765
```

Politika karşılaştırması:

```powershell
.\.venv\Scripts\python.exe scripts\compare_policies.py ^
  --map-configs configs\maps\custom_map_01.yaml configs\maps\custom_map_02.yaml ^
  --policies mlp cfc cfc_deep gru lstm ^
  --epochs 60 ^
  --episodes 4 ^
  --max-steps 900
```

Haftalık raporu Tectonic ile yeniden derleme:

```powershell
tectonic --outdir report\weekly_2026_05_11\build report\weekly_2026_05_11\weekly_report.tex
```

## Klasör Yapısı

```text
configs/                 Eğitim, split ve harita YAML dosyaları
notebooks/               Colab eğitim notebook'u
scripts/                 Eğitim, değerlendirme, harita editörü ve analiz scriptleri
source/mujoco_lnn_nav/   Ortam, model, eğitim ve yardımcı modüller
tests/                   Birim ve smoke testler
report/                  Ana rapor, haftalık raporlar ve PDF çıktıları
```

## Notlar

- `results/` ve büyük `.pt` checkpoint dosyaları `.gitignore` ile dışarıda tutulur.
- GitHub'daki Colab rozeti `master` branch'ini açar.
- Güncel notebook sıfırdan `cfc_deep192_custom22_dynamic_dagger2` deneyini çalıştıracak şekilde ayarlanmıştır.
