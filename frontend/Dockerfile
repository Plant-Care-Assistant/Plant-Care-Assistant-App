FROM node:22-alpine

WORKDIR /app

# Kopiujemy pliki zależności
COPY package*.json ./
RUN npm ci

# Kopiujemy resztę kodu
COPY . .

# Next.js domyślnie działa na porcie 3000
EXPOSE 3000

# Uruchamiamy w trybie deweloperskim
CMD ["npm", "run", "dev"]
