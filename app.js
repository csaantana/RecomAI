'use strict';
const express    = require('express');
const http       = require('http');
const { Server } = require('socket.io');
const path       = require('path');
const routes     = require('./routes');

// ─────────────────────────────────────────────────────────────────────────────
// SERVIDOR — ponto de entrada da aplicação
//
// Responsabilidades:
//   1. Configura o Express (HTTP) e o Socket.io (WebSocket) no mesmo servidor
//   2. Registra as rotas da API
//   3. Ao subir, dispara o treino automático após 2s (deixa o browser conectar)
//   4. Sincroniza novos clientes Socket.io com o estado atual do treino
// ─────────────────────────────────────────────────────────────────────────────

const expressApp = express();
const httpServer = http.createServer(expressApp);
const socketIo   = new Server(httpServer, { cors: { origin: '*' } });
const PORT       = 3000;

expressApp.use(express.json());
expressApp.use(express.static(path.join(__dirname, 'public')));
expressApp.set('io', socketIo);

expressApp.use('/api', routes);
expressApp.get('/', (_req, res) => res.sendFile(path.join(__dirname, 'views', 'index.html')));

// ── Sincronização de clientes Socket.io ──────────────────────────────────────
// Quando um novo browser conecta pode chegar em 3 momentos:
//   A) Antes de qualquer época → não envia nada; training:start chegará logo
//   B) Durante o treino        → replica épocas já gravadas + recebe as restantes ao vivo
//   C) Após o treino concluir  → replica todo o histórico + training:done
socketIo.on('connection', clientSocket => {
  const state = require('./models/stateModel');

  if (state.epochHistory.length) {
    // Casos B e C: envia histórico para popular os gráficos
    clientSocket.emit('training:history', state.epochHistory);
    // Não manda training:start — ele chama resetCharts() e apagaria o histórico
  } else if (state.isTraining) {
    // Caso A: treino iniciou mas ainda não há épocas gravadas
    clientSocket.emit('training:start', { totalEpochs: 60 });
  }

  if (state.trainingComplete) {
    clientSocket.emit('training:done', state.lastMetrics);
  }

  console.log('🔌 Browser conectado');
  clientSocket.on('disconnect', () => console.log('🔌 Browser desconectado'));
});

// ── Inicialização ─────────────────────────────────────────────────────────────
httpServer.listen(PORT, () => {
  console.log(`\n✅  App  → http://localhost:${PORT}`);

  // BrowserSync: abre o browser na porta 3001 (proxy reverso com auto-reload)
  try {
    const browserSync = require('browser-sync').create();
    browserSync.init({
      proxy   : `localhost:${PORT}`,
      port    : PORT + 1,
      files   : ['views/**/*.html', 'public/**/*'],
      open    : true,
      notify  : false,
      logLevel: 'silent',
    });
    console.log(`🔄  Sync → http://localhost:${PORT + 1}  (auto-reload ativo)`);
  } catch (err) {
    console.warn('browser-sync não disponível:', err.message);
  }

  // Aguarda 2s antes de iniciar o treino.
  // Motivo: o BrowserSync precisa abrir o browser e o browser precisa estabelecer
  // a conexão socket.io. Sem este delay o treino começa antes do cliente conectar
  // e os primeiros eventos de progresso são perdidos → gráficos ficam em branco.
  const { triggerTraining } = require('./controllers/modelController');
  setTimeout(() => {
    console.log('\n🧠  Iniciando treino automático do modelo...');
    triggerTraining(socketIo).then(() => {
      if (require('./models/stateModel').trainingComplete) {
        console.log('🎯  Modelo pronto — recomendações disponíveis sem novo treino.\n');
      }
    });
  }, 2000);
});
