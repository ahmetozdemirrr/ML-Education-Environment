FROM node:18
WORKDIR /app

# ÖNCE sadece paket manifest dosyalarını kopyala
COPY package.json package-lock.json ./

# npm ci yerine npm install kullan (lock file uyumsuzlukları için)
RUN npm install

# Uygulama kodunun geri kalanını kopyala
COPY . .

# İzinleri ayarla ('node' kullanıcısı için)
RUN chown -R node:node /app

CMD ["npm", "start"]
