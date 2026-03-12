# Katkıda Bulunma

RAG Master Class'a katkıda bulunmak istediğiniz için teşekkürler!

## Nasıl Katkıda Bulunulur

1. Repoyu fork'layın
2. Feature branch oluşturun: `git checkout -b feature/ozelliginiz`
3. Değişikliklerinizi commit'leyin: `git commit -m "Özellik ekle"`
4. Branch'e push'layın: `git push origin feature/ozelliginiz`
5. Pull Request açın

## Geliştirme Ortamı Kurulumu

```bash
# Repoyu klonlayın
git clone https://github.com/YOUR_USERNAME/rag-master-class.git
cd rag-master-class

# Sanal ortam oluşturun
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Bağımlılıkları kurun
pip install -r Classical-RAG/requirements.txt
pip install -r Agentic-RAG/requirements.txt

# Ortam değişkenlerini kopyalayın
cp .env.example .env
# .env dosyasını API anahtarlarınızla düzenleyin
```

## Kurallar

- Açık ve anlaşılır commit mesajları yazın
- Yeni fonksiyon ve sınıflara docstring ekleyin
- Mevcut kod stiliyle tutarlı olun
- PR göndermeden önce değişikliklerinizi test edin
- Yeni özellik eklerseniz README'yi güncelleyin

## Sorun Bildirme

- Hata raporları ve özellik istekleri için GitHub Issues kullanın
- Hatalar için yeniden üretme adımlarını ekleyin
- Python sürümünüzü ve işletim sisteminizi belirtin

## Kod Stili

- Python kodu için PEP 8 kurallarına uyun
- Mümkün olduğunca type hint kullanın
- Fonksiyonları odaklı ve küçük tutun

## Lisans

Katkıda bulunarak, katkılarınızın MIT Lisansı altında lisanslanacağını kabul etmiş olursunuz.
