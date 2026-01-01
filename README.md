# Pekiştirmeli Öğrenme Projesi

## Proje Hakkında
Bu proje, pekiştirmeli öğrenme (Reinforcement Learning) yöntemlerini kullanarak geliştirilmiş bir otonom kontrol sistemidir. PPO (Proximal Policy Optimization) algoritması kullanılarak eğitilen ajanlar, farklı atmosferik koşullarda (rüzgar, türbülans, vb.) görevleri yerine getirmek üzere tasarlanmıştır.

## Kurulum

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

## Kullanım

Eğitim başlatmak için:
```bash
python train.py
```

Test etmek için:
```bash
python evaluate.py
```

## Dosya Yapısı
- `train.py`: Model eğitimi
- `evaluate.py`: Model değerlendirmesi
- `models/`: Eğitilmiş modeller (Git üzerinde takip edilmez)
- `logs/`: Eğitim logları (Git üzerinde takip edilmez)
