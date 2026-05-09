# MuJoCo LNN/CfC Öğrenmeli Robot Navigasyonu

**📄 [Proje Raporu (PDF)](report/build/main.pdf)**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heimdilon/mujoco_lnn_navigation/blob/master/notebooks/colab_custom22_training.ipynb)

MuJoCo ortamında diferansiyel sürüşlü bir robotun 32 ışınlı LiDAR gözlemiyle
engel içeren haritalarda hedefe ulaşması araştırılmıştır.
Projenin hedefi LNN/CfC tabanlı politikayı GRU ve LSTM gibi klasik yinelemeli
ağlarla aynı koşullar altında karşılaştırmaktır.

## Colab ile Eğitim

Custom22 CfC eğitimini Google Colab üzerinde çalıştırmak için
[`notebooks/colab_custom22_training.ipynb`](notebooks/colab_custom22_training.ipynb)
dosyasını aç veya yukarıdaki **Open in Colab** rozetini kullan.

Colab'de `Runtime > Change runtime type > GPU` seçildikten sonra notebook sırasıyla
repo kurulumunu, 22 haritalı BC eğitimini, 2 DAgger iterasyonunu ve train+holdout
haritalarda saf politika değerlendirmesini çalıştırır. Lokal makinedeki `latest.pt`
checkpoint'inden devam etmek istersen dosyayı Google Drive'a koyup notebook içindeki
`RESUME_CHECKPOINT` alanına Drive yolunu yazman yeterli.

Not: Colab rozeti GitHub'daki `master` branch'i açar; notebook ve custom22
konfigürasyonlarının Colab'de görünmesi için değişikliklerin GitHub'a push edilmiş
olması gerekir.

## Sonuçlar

### Vize — GRU + BC/DAgger (2 harita)

| Harita | Başarı | Çarpışma | Ort. Adım |
| --- | ---: | ---: | ---: |
| `custom_map_01` | 1.00 | 0.00 | 459 |
| `custom_map_02` | 1.00 | 0.00 | 521 |

### Final — CfC düzeltilmiş + DAgger (6 harita, max-steps = 900)

| Harita | GRU BC | CfC kırık | CfC düzeltilmiş |
| --- | ---: | ---: | ---: |
| `custom_map_01` | çarpışma | çarpışma | **%100 başarı** |
| `custom_map_02` | çarpışma | çarpışma | timeout |
| `custom_map_04` | çarpışma | çarpışma | timeout |
| `custom_map_05` | çarpışma | çarpışma | timeout |
| `custom_map_06` | çarpışma | çarpışma | timeout |
| `custom_map_07` | çarpışma | timeout | **%100 başarı** |

Timeout: çarpışma olmadan 900 adım — robot engelleri atlıyor, hedef takibi henüz tam değil.

### CfC Başarısızlığının Kök Nedenleri

1. **Giriş kodlayıcısı eksikliği** — GRU'da `Linear(38,128)+Tanh` var, CfC'de yoktu.
   Ham 38-d gözlem (farklı ölçekler) doğrudan ODE hücresine giriyordu.
2. **Asimetrik eğitim bütçesi** — CfC 220 dönem / 12 demo, GRU 500 dönem / 24 demo.
3. **Yanlış ODE zaman adımı** — `ncps` varsayılanı Δt = 1.0 s, gerçek kontrol adımı 0.08 s.

Üç düzeltme + 2 DAgger iterasyonu: **%100 collision → 2/6 harita başarısı**.

---

## Kurulum

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

Beklenen: `21 passed`

---

## Yeniden Üretme

### Vize baseline (GRU)

```powershell
.\.venv\Scripts\python.exe scripts\batch_evaluate.py ^
  --map-configs configs\maps\custom_map_01.yaml configs\maps\custom_map_02.yaml ^
  --checkpoint results\custom_maps_gru_bc_dagger\latest.pt ^
  --episodes 4 --run-name vize_gru_eval --max-steps 900 --goal-observation-max 10
```

### CfC düzeltilmiş eğitimi

```powershell
.\.venv\Scripts\python.exe scripts\train_bc.py ^
  --train-config configs\train\bc_cfc_augmented_maps.yaml ^
  --split-config configs\splits\custom8_seed25462877008.yaml ^
  --run-name cfc_fixed_bc --save-interval 50 --no-final-eval --device cuda
```

Ardından 2 DAgger iterasyonu:

```powershell
.\.venv\Scripts\python.exe scripts\train_bc.py ^
  --train-config configs\train\bc_cfc_augmented_maps.yaml ^
  --split-config configs\splits\custom8_seed25462877008.yaml ^
  --run-name cfc_fixed_dagger ^
  --resume results\cfc_fixed_bc\latest.pt ^
  --dagger-iterations 2 --dagger-rollouts-per-map 4 ^
  --dagger-epochs 80 --epochs 0 --no-final-eval --device cuda
```

### Değerlendirme

```powershell
.\.venv\Scripts\python.exe scripts\batch_evaluate.py ^
  --map-configs configs\maps\custom_map_01.yaml configs\maps\custom_map_02.yaml ^
    configs\maps\custom_map_04.yaml configs\maps\custom_map_05.yaml ^
    configs\maps\custom_map_06.yaml configs\maps\custom_map_07.yaml ^
  --checkpoint results\cfc_fixed_dagger\latest.pt ^
  --episodes 4 --run-name cfc_final_eval --max-steps 900 --device cuda
```

---

## Proje Sözleşmesi

| Boyut | Açıklama |
| ---: | --- |
| 38 | Gözlem: hedef mesafesi, hedef açısı, hız (2), önceki aksiyon (2), 32 LiDAR ışını |
| 2 | Aksiyon: `[linear, angular]` ∈ [-1, 1] |

---

## Klasör Yapısı

```
configs/                 görev, eğitim ve harita YAML dosyaları
scripts/                 eğitim, değerlendirme ve analiz scriptleri
source/mujoco_lnn_nav/   MuJoCo ortamı, modeller, yardımcı modüller
tests/                   birim ve smoke testler
report/                  LaTeX rapor ve PDF
```

## Araçlar

**Harita editörü:**
```powershell
.\.venv\Scripts\python.exe scripts\map_editor.py --port 8765
```

**Model karşılaştırması:**
```powershell
.\.venv\Scripts\python.exe scripts\compare_policies.py ^
  --map-configs configs\maps\custom_map_01.yaml configs\maps\custom_map_02.yaml ^
  --policies mlp cfc gru lstm --epochs 60 --episodes 4 --max-steps 900
```

**Raporu yeniden derlemek için:**
```powershell
pdflatex -interaction=nonstopmode -output-directory=report\build report\main.tex
pdflatex -interaction=nonstopmode -output-directory=report\build report\main.tex
```

---

> Büyük checkpoint dosyaları (`.pt`) ve `results/` klasörü `.gitignore` ile dışarıda bırakılmıştır.
