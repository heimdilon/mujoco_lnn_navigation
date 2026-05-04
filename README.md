# MuJoCo LiDAR Tabanlı Öğrenmeli Navigasyon

Bu repo, MuJoCo ortamında diferansiyel sürüşlü bir robotun LiDAR tabanlı öğrenmeli navigasyonunu denemek için hazırlanmıştır. Proje önceki Isaac/IsaacLab çalışmalarından bağımsızdır; kod içinde Isaac, IsaacLab veya `kit.exe` bağımlılığı yoktur.

Vize teslimindeki ana yöntem:

- Robot çevreyi 32 ışınlı 2D LiDAR/range sensörü ile algılar.
- Politika girdisi 38 boyutludur.
- Politika çıktısı `[linear, angular]` aksiyonudur.
- İlk temel yöntem GRU tabanlı politikadır.
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
11 passed
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

## Eğitim

Kısa bir PPO smoke eğitimi:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --task-config configs\task\sparse_goal.yaml --train-config configs\train\ppo_mlp.yaml --run-name sparse_goal_smoke --steps 4096 --num-envs 8 --eval-episodes 8
```

Manuel haritalar üzerinde BC/DAgger hattı için ilgili eğitim konfigürasyonları `configs/train` altında bulunur.

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
- Mevcut MVP kapsamı GRU + BC/DAgger sonucudur.
- MLP, LSTM ve CfC/LNN karşılaştırması sonraki çalışma olarak planlanmıştır.
