FROM node:16
WORKDIR /app

# ÖNCE sadece paket manifest dosyalarını kopyala
COPY package.json package-lock.json ./

# package-lock.json'a göre GÜVENİLİR kurulum için npm ci kullan
RUN npm ci

# Uygulama kodunun geri kalanını kopyala
COPY . .

# İzinleri ayarla ('node' kullanıcısı için)
RUN chown -R node:node /app

CMD ["npm", "start"]
