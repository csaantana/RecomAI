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
//   3. Ao subir, dispara o treino automático do modelo
//   4. Sincroniza novos clientes Socket.io com o estado atual do treino
// ─────────────────────────────────────────────────────────────────────────────

const expressApp = express();
const httpServer = http.createServer(expressApp);
const socketIo   = new Server(httpServer, { cors: { origin: '*' } });
const PORT       = 3000;

// Middleware: lê JSON no body e serve arquivos estáticos da pasta /public
expressApp.use(express.json());
expressApp.use(express.static(path.join(__dirname, 'public')));

// Disponibiliza a instância do Socket.io para os controllers via req.app.get('io')
expressApp.set('io', socketIo);

expressApp.use('/api', routes);
expressApp.get('/', (_req, res) => res.sendFile(path.join(__dirname, 'views', 'index.html')));

// ── Sincronização de clientes Socket.io ──────────────────────────────────────
// Quando um novo browser conecta (ou reconecta após refresh), pode chegar em 3 momentos:
//   A) Antes do treino começar  → não faz nada, o evento 'training:start' chegará logo
//   B) Durante o treino         → reproduz épocas já registradas + recebe o restante ao vivo
//   C) Após o treino concluir   → reproduz todo o histórico + recebe 'training:done'
socketIo.on('connection', clientSocket => {
  const state = require('./models/stateModel');

  if (state.epochHistory.length) {
    // Caso B ou C — cliente chegou durante ou após o treino.
    // Manda o histórico de épocas primeiro para popular os gráficos.
    clientSocket.emit('training:history', state.epochHistory);
    // NÃO mandamos training:start aqui: o handler no browser chama
    // resetCharts() ao receber training:start, o que apagaria o histórico.
    // Os eventos training:progress seguintes chegam naturalmente via io.emit().
  } else if (state.isTraining) {
    // Caso A — treino começou mas ainda não há épocas registradas.
    clientSocket.emit('training:start', { totalEpochs: 60 });
  }

  // Treino já concluído → informa direto para habilitar o botão de recomendação
  if (state.trainingComplete) {
    clientSocket.emit('training:done', state.lastMetrics);
  }

  console.log('🔌 Browser conectado');
  clientSocket.on('disconnect', () => console.log('🔌 Browser desconectado'));
});

// ── Inicialização ─────────────────────────────────────────────────────────────
httpServer.listen(PORT, () => {
  console.log(`\n✅  App  → http://localhost:${PORT}`);

  // Treina o modelo uma única vez ao subir o servidor.
  // Após o treino, qualquer seleção de usuário / mudança de carrinho usa apenas
  // inferência (scoreProducts) — rápida, sem precisar retreinar.
  const { triggerTraining } = require('./controllers/modelController');
  console.log('🧠  Iniciando treino automático do modelo...');
  triggerTraining(socketIo).then(() => {
    if (require('./models/stateModel').trainingComplete) {
      console.log('🎯  Modelo pronto — recomendações disponíveis sem novo treino.\n');
    }
  });

  // BrowserSync: proxy reverso que monitora arquivos e recarrega o browser automaticamente
  try {
    const browserSync = require('browser-sync').create();
    browserSync.init({
      proxy   : `localhost:${PORT}`, // encaminha requests para o Express
      port    : PORT + 1,            // browser-sync escuta na porta 3001
      files   : ['views/**/*.html', 'public/**/*'], // arquivos monitorados
      open    : true,
      notify  : false,
      logLevel: 'silent',
    });
    console.log(`🔄  Sync → http://localhost:${PORT + 1}  (auto-reload ativo)\n`);
  } catch (err) {
    console.warn('browser-sync não disponível:', err.message);
  }
});
