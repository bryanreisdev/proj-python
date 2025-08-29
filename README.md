# 🔗 AI Tools - Plataforma de Inteligência Artificial

[![React](https://img.shields.io/badge/React-19.1.1-blue.svg)](https://reactjs.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-purple.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-yellow.svg)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%20Lite-2.14-red.svg)](https://tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-✓-teal.svg)](https://docker.com/)
[![Redis](https://img.shields.io/badge/Redis-7-alpine-orange.svg)](https://redis.io/)

Uma plataforma moderna e completa de ferramentas de IA que combina geração de QR Codes e análise demográfica avançada com tecnologia de ponta.

## ✨ Funcionalidades

### 📱 Geração de QR Codes
- **Geração Instantânea**: Crie QR codes a partir de qualquer URL
- **Download em PNG**: Baixe seus QR codes em alta qualidade
- **Impressão Otimizada**: Interface dedicada para impressão
- **Cache Inteligente**: Resultados salvos para acesso rápido

### 👤 Análise Demográfica Avançada
- **Detecção de Faces**: Identifica múltiplas pessoas na imagem
- **Estimativa de Idade**: Análise precisa com faixas etárias
- **Classificação de Gênero**: Predição com níveis de confiança
- **Características Faciais**: Formato do rosto, tipo de pele, olhos, etc.
- **Análise de Etnia**: Detecção de características étnicas
- **Expressões Faciais**: Identificação de emoções e sorrisos
- **Estatísticas Globais**: Resumos detalhados da análise

## 🛠️ Stack Tecnológico

### Frontend
- **React.js 19.1.1** - Interface moderna e responsiva
- **Tailwind CSS 3.4** - Estilização utilitária e design system
- **PostCSS** - Processamento CSS avançado

### Backend
- **Python 3.8+** - Linguagem principal
- **Flask 2.3.3** - Framework web leve e flexível
- **OpenCV 4.8.1** - Processamento de imagem e visão computacional
- **TensorFlow Lite 2.14** - Modelos de IA otimizados
- **NumPy 1.24** - Computação numérica

### Infraestrutura
- **Docker** - Containerização completa
- **Redis 7** - Cache e sessões
- **Gunicorn** - Servidor WSGI para produção
- **Nginx** - Proxy reverso e servidor web

### Bibliotecas Especializadas
- **qrcode[pil]** - Geração de QR codes
- **flask-cors** - Cross-origin resource sharing
- **geoip2** - Geolocalização
- **user-agents** - Detecção de dispositivos

## 🚀 Instalação e Execução

### Pré-requisitos
- Docker e Docker Compose
- Git

### Execução Rápida com Docker

```bash
# Clone o repositório
git clone https://github.com/bryanreisdev/ai-tools.git
cd ai-tools

# Execute com Docker Compose
docker-compose up -d

# Acesse a aplicação
# Frontend: http://localhost:3000
# Backend API: http://localhost:5000
# Redis: localhost:6379
```

### Desenvolvimento Local

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

pip install -r requirements.txt
python app.py
```

#### Frontend
```bash
cd frontend
npm install
npm start
```

## 📁 Estrutura do Projeto

```
ai-tools/
├── frontend/                 # Aplicação React
│   ├── src/
│   │   ├── App.js           # Componente principal
│   │   ├── FaceAnalyzerOverlay.jsx
│   │   └── ...
│   ├── public/
│   ├── Dockerfile
│   └── package.json
├── backend/                  # API Flask
│   ├── app.py               # Aplicação principal
│   ├── analytics_manager.py # Gerenciador de analytics
│   ├── cache_manager.py     # Sistema de cache
│   ├── demographics_detector.py # Detector demográfico
│   ├── emotion_detector.py  # Detector de emoções
│   ├── url_services.py      # Serviços de QR code
│   ├── utils.py             # Utilitários
│   ├── config.py            # Configurações
│   ├── requirements.txt
│   └── Dockerfile
├── models/                   # Modelos de IA
│   ├── age_regression.tflite
│   └── gender.tflite
├── docker-compose.yml        # Orquestração Docker
├── docker-compose.backend.yml
└── README.md
```

## 🔧 Configuração

### Variáveis de Ambiente

#### Frontend (.env)
```env
REACT_APP_API_BASE=http://localhost:5000
```

#### Backend (.env)
```env
FLASK_ENV=development
FLASK_DEBUG=false
REDIS_URL=redis://redis:6379
```

### Modelos de IA

Os modelos TensorFlow Lite estão incluídos no diretório `models/`:
- `age_regression.tflite` - Modelo para estimativa de idade
- `gender.tflite` - Modelo para classificação de gênero

## 📊 API Endpoints

### QR Code
- `POST /api/qrcode` - Gerar QR code
- `POST /api/qrcode/download` - Download do QR code
- `POST /api/qrcode/base64` - QR code em base64

### Demografia
- `POST /api/demographics` - Análise demográfica completa

### Analytics
- `GET /api/analytics` - Estatísticas de uso
- `POST /api/analytics/event` - Registrar evento

## 🎯 Características Técnicas

### Performance
- **Cache Redis**: Resultados em cache para melhor performance
- **Processamento Assíncrono**: Análises não bloqueiam a interface
- **Otimização de Modelos**: TensorFlow Lite para inferência rápida
- **Compressão de Imagens**: Processamento eficiente de arquivos grandes

### Segurança
- **Validação de Entrada**: Sanitização de dados
- **Limite de Tamanho**: Máximo 50MB por upload
- **CORS Configurado**: Controle de acesso cross-origin
- **Error Handling**: Tratamento robusto de erros

### Escalabilidade
- **Arquitetura Containerizada**: Fácil deploy e escalonamento
- **Microserviços**: Separação clara entre frontend e backend
- **Cache Distribuído**: Redis para sessões e cache
- **Load Balancing Ready**: Preparado para balanceamento de carga

## 🎨 Interface do Usuário

### Design System
- **Tema Futurístico**: Interface moderna com glass morphism
- **Responsivo**: Otimizado para mobile e desktop
- **Animações Suaves**: Transições e efeitos visuais
- **Acessibilidade**: Suporte a navegação por teclado

### Componentes
- **Tabs Dinâmicos**: Navegação entre funcionalidades
- **Upload Drag & Drop**: Interface intuitiva para imagens
- **Progress Indicators**: Feedback visual durante processamento
- **Resultados Detalhados**: Visualizações ricas dos dados

## 📈 Analytics e Monitoramento

### Métricas Coletadas
- **Uso de Funcionalidades**: QR codes gerados, análises realizadas
- **Performance**: Tempo de resposta, taxa de erro
- **Dispositivos**: Informações sobre navegadores e sistemas
- **Geolocalização**: Dados de localização dos usuários

### Cache Analytics
- **Hit Rate**: Eficiência do sistema de cache
- **Storage Usage**: Uso do Redis
- **Performance Metrics**: Tempo de resposta do cache

## 🚀 Deploy

### Produção com Docker
```bash
# Build das imagens
docker-compose -f docker-compose.yml build

# Deploy
docker-compose -f docker-compose.yml up -d
```

### Cloud Deployment
O projeto está preparado para deploy em:
- **AWS**: ECS, EC2, S3 + CloudFront
- **Google Cloud**: GKE, Cloud Run
- **Azure**: AKS, App Service
- **Heroku**: Container deployment

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👨‍💻 Desenvolvedor

**Bryan Reis**

- [LinkedIn](https://linkedin.com/in/bryan-reis)
- [GitHub](https://github.com/bryanreisdev)

## 🙏 Agradecimentos

- **OpenCV** pela biblioteca de visão computacional
- **TensorFlow** pelos modelos de IA
- **React** pela framework frontend
- **Flask** pela simplicidade e flexibilidade
- **Docker** pela containerização

---

⭐ **Se este projeto te ajudou, considere dar uma estrela no repositório!**
