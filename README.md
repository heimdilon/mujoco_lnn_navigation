# MuJoCo LNN Tabanlı Öğrenmeli Robot Navigasyonu

Bu repo, MuJoCo ortamında diferansiyel sürüşlü bir robot için Liquid Neural Network (LNN/CfC) odaklı öğrenmeli navigasyon denemeleri yapmak üzere hazırlanmıştır. LiDAR burada temel yöntem değil, robotun çevreyi algılamak için kullandığı sensör modelidir.

Projenin ana amacı, aynı gözlem ve aksiyon sözleşmesi altında LNN/CfC tabanlı politikaları MLP, GRU ve LSTM gibi klasik ağlarla karşılaştırmaktır. Vize teslimindeki MVP kapsamında, bu hattı çalışır ve ölçülebilir hale getirmek için GRU tabanlı ilk yöntem raporlanmıştır.

Vize teslimindeki mevcut deney hattı:

- Robot çevreyi 32 ışınlı 2D LiDAR/range sensörü ile algılar.
- Politika girdisi 38 boyutludur.
- Politika çıktısı `[linear, angular]` aksiyonudur.
- İlk çalışan temel yöntem GRU tabanlı politikadır.
- LNN/CfC politikası `ncps.torch.CfC` ile gerçek recurrent hidden-state kullanan bir modeldir.
- LNN/CfC karşılaştırması projenin ana hedefidir.
- Eğitimde Behavioral Cloning ve DAgger kullanılır.
- A* yalnızca eğitimde öğretmen etiketi üretmek için kullanılır; değerlendirme sırasında kapalıdır.

## Proje Sözleşmesi

Gözlem boyutu: `38`

Gözlem içeriği:

- normalize hedef mesafesi
- normalize hedef açısı
- normalize doğrusal hız
- normalize açısal hız
- önceki aksiyon, iki değer
- 32 normalize LiDAR/range ışını

Aksiyon boyutu: `2`

Aksiyon içeriği:

- `[linear, angular]`
- değer aralığı `[-1, 1]`

## Kurulum

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest
```

## Testleri Çalıştırma

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Beklenen mevcut sonuç:

```text
19 passed
```

## Vize Sonucunu Yeniden Üretme

Vize raporundaki iki manuel harita sonucunu yeniden üretmek için:

```powershell
.\.venv\Scripts\python.exe scripts\batch_evaluate.py --map-configs configs\maps\custom_map_01.yaml configs\maps\custom_map_02.yaml --checkpoint results\custom_maps_gru_bc_dagger\latest.pt --episodes 4 --run-name batch_manual_maps_pure_gru_dagger --max-steps 900 --goal-observation-max 10
```

Bu komutta `--auto-waypoints` kullanılmamalıdır. Böylece değerlendirme A* destekli değil, saf policy değerlendirmesi olur.

Beklenen özet:

| Harita | Başarı | Çarpışma | Timeout | Ortalama adım |
| --- | ---: | ---: | ---: | ---: |
| `custom_map_01` | `1.000` | `0.000` | `0.000` | `459.0` |
| `custom_map_02` | `1.000` | `0.000` | `0.000` | `521.0` |

Komut çıktıları `results/<run-name>` altında `summary.csv`, `summary.json`, `eval.csv`, `eval.json`, `rollout.png` ve `rollout.gif` olarak oluşur.

## Harita Editörü

Manuel harita, başlangıç noktası, hedef noktası ve çizgi/duvar engelleri üretmek için:

```powershell
.\.venv\Scripts\python.exe scripts\map_editor.py --port 8765
```

Editör, sabit harita konfigürasyonlarını `configs/maps` altında YAML dosyası olarak üretir. Bu dosyalar doğrudan eğitim veya değerlendirmede kullanılabilir.

## Procedural Harita Üretimi

Çok sayıda doğrulanmış harita üretmek için:

```powershell
.\.venv\Scripts\python.exe scripts\generate_maps.py --type all --difficulty medium --count 12 --seed 1000 --out configs\maps\generated --gallery results\map_gallery\generated_maps.png
```

Desteklenen harita tipleri:

- `easy_open`: az engelli açık alan
- `clutter`: rastgele kutu/silindir engelli alan
- `wall_gap`: tek geçitli duvar
- `zigzag`: ardışık geçitli zikzak engeller
- `corridor`: dar koridor ve kapı benzeri iç engeller
- `u_trap`: U şekilli tuzak/kaçış senaryosu

Her harita üretilirken start/goal noktalarının serbest olduğu ve A* ile en az bir geçerli yol bulunduğu kontrol edilir. A* burada yalnızca harita kalite kontrolü içindir; saf policy değerlendirmesinde kullanılmamalıdır.

## Manuel Harita Çoğaltma ve Holdout Testi

Sekiz manuel haritalık deneyde iki harita eğitim dışında tutulur:

- eğitim: `custom_map_01`, `custom_map_02`, `custom_map_04`, `custom_map_05`, `custom_map_06`, `custom_map_07`
- holdout test: `custom_map_03`, `custom_map_08`

Split kaydı:

```text
configs/splits/custom8_seed25462877008.yaml
```

Eğitim verisini artırmak için yalnızca eğitim haritalarından doğrulanmış varyasyonlar üretilebilir:

```powershell
.\.venv\Scripts\python.exe scripts\augment_maps.py --split-config configs\splits\custom8_seed25462877008.yaml --variants-per-map 6 --out configs\maps\augmented\custom8_seed25462877008_aug_v6 --split-out configs\splits\custom8_seed25462877008_aug_v6.yaml --gallery results\map_gallery\custom8_seed25462877008_aug_v6.png
```

Bu komut 6 orijinal eğitim haritasını korur, her biri için 6 varyasyon üretir ve toplam 42 eğitim haritalık yeni split oluşturur. Holdout haritaları çoğaltılmaz ve eğitime eklenmez.

## Eğitim

Kısa bir PPO smoke eğitimi:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --task-config configs\task\sparse_goal.yaml --train-config configs\train\ppo_mlp.yaml --run-name sparse_goal_smoke --steps 4096 --num-envs 8 --eval-episodes 8
```

Manuel haritalar üzerinde BC/DAgger hattı için ilgili eğitim konfigürasyonları `configs/train` altında bulunur.

Augmented split üzerinde CfC/LNN eğitimi için:

```powershell
.\.venv\Scripts\python.exe scripts\train_bc.py --split-config configs\splits\custom8_seed25462877008_aug_v6.yaml --train-config configs\train\bc_cfc_augmented_maps.yaml --run-name cfc_custom8_aug_v6 --policy cfc --dagger-iterations 2 --dagger-rollouts-per-map 1 --dagger-epochs 30 --dagger-noise 0.05 --dagger-expert-mix 0.25 --device cpu --no-final-eval
```

`cfc` ve `lnn` policy adları aynı gerçek `ncps.torch.CfC` modeline gider. Bu model önceki feed-forward prototip değildir; evaluation sırasında hidden state taşır ve BC eğitiminde `forward_sequence` ile tam sekans üzerinde optimize edilir.

Sonrasında görülmemiş iki holdout haritasında saf policy değerlendirmesi:

```powershell
.\.venv\Scripts\python.exe scripts\batch_evaluate.py --map-configs configs\maps\custom_map_03.yaml configs\maps\custom_map_08.yaml --checkpoint results\cfc_custom8_aug_v6\latest.pt --episodes 4 --run-name cfc_custom8_aug_v6_holdout_eval --max-steps 900 --goal-observation-max 10 --device cpu
```

## Değerlendirme

Tek checkpoint değerlendirme:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --task-config configs\task\sparse_goal.yaml --checkpoint results\sparse_goal_smoke\latest.pt --episodes 32 --run-name sparse_goal_eval
```

Birden fazla manuel haritada aynı checkpoint değerlendirme:

```powershell
.\.venv\Scripts\python.exe scripts\batch_evaluate.py --map-configs configs\maps\custom_map_01.yaml configs\maps\custom_map_02.yaml --checkpoint results\custom_maps_gru_bc_dagger\latest.pt --episodes 4 --run-name batch_manual_maps_pure_gru_dagger --max-steps 900 --goal-observation-max 10
```

## Model Karşılaştırması

MLP, CfC/LNN, GRU ve LSTM politikalarını aynı BC/DAgger hattında hızlı karşılaştırmak için:

```powershell
.\.venv\Scripts\python.exe scripts\compare_policies.py --map-configs configs\maps\custom_map_01.yaml configs\maps\custom_map_02.yaml --run-name policy_comparison_quick_manual --policies mlp cfc gru lstm --epochs 60 --dagger-iterations 1 --dagger-rollouts-per-map 2 --dagger-epochs 15 --episodes 4 --max-steps 900 --goal-observation-max 10 --no-gif --no-png
```

Bu karşılaştırma vize tesliminin ana sonucu değildir; raporda sonraki adım olarak ele alınmıştır.

## MuJoCo Viewer

Canlı MuJoCo viewer ile checkpoint izlemek için:

```powershell
.\.venv\Scripts\python.exe scripts\watch_mujoco.py --task-config configs\task\sparse_goal.yaml --checkpoint results\sparse_goal_smoke\latest.pt
```

## Raporu Derleme

Vize raporu LaTeX kaynağı:

```text
report\main.tex
```

PDF üretmek için:

```powershell
xelatex -interaction=nonstopmode -halt-on-error -output-directory=report\build report\main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=report\build report\main.tex
```

PDF çıktısı:

```text
report\build\main.pdf
```

## Klasör Yapısı

```text
configs/                 görev, eğitim ve manuel harita YAML dosyaları
scripts/                 eğitim, değerlendirme, harita editörü ve analiz scriptleri
source/mujoco_lnn_nav/   MuJoCo ortamı, modeller ve yardımcı modüller
tests/                   birim testleri ve smoke testler
tools/map_editor/        tarayıcı tabanlı harita çizim arayüzü
report/                  LaTeX rapor, görseller, tablolar ve PDF çıktısı
```

## Teslim Notları

- Büyük checkpoint dosyaları GitHub'a eklenmez.
- `results/` klasörü `.gitignore` ile dışarıda bırakılır.
- Rapor için gerekli küçük PNG/GIF görselleri `report/figures` altında tutulur.
- Mevcut MVP, LNN/CfC çalışmasına zemin hazırlayan GRU + BC/DAgger sonucudur.
- MLP, GRU, LSTM ve CfC/LNN karşılaştırması sonraki çalışma olarak planlanmıştır.
