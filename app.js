'use strict';
const express    = require('express');
const http       = require('http');
const { Server } = require('socket.io');
const path       = require('path');
const routes     = require('./routes');

const app    = express();
const server = http.createServer(app);
const io     = new Server(server, { cors: { origin: '*' } });
const PORT   = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
app.set('io', io);

app.use('/api', routes);
app.get('/', (_req, res) => res.sendFile(path.join(__dirname, 'views', 'index.html')));

// ── Socket.io ─────────────────────────────────────────────────────────────────
// When a client connects mid-training or after training, catch it up immediately.
io.on('connection', socket => {
  const state = require('./models/stateModel');

  if (state.epochHistory.length) {
    // Replay all epochs recorded so far (covers mid-training joins and post-training page refreshes)
    socket.emit('training:history', state.epochHistory);
  }

  if (state.trainingComplete) {
    socket.emit('training:done', state.lastMetrics);
  } else if (state.isTraining) {
    socket.emit('training:start', { totalEpochs: 60 });
  }

  console.log('🔌 Client connected');
  socket.on('disconnect', () => console.log('🔌 Client disconnected'));
});

// ── Start ─────────────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`\n✅  App  → http://localhost:${PORT}`);

  // Auto-train once on startup — model stays in memory for the lifetime of the process
  const { triggerTraining } = require('./controllers/modelController');
  console.log('🧠  Auto-training recommendation model...');
  triggerTraining(io).then(() => {
    if (require('./models/stateModel').trainingComplete) {
      console.log('🎯  Model ready — recommendations available without retraining.\n');
    }
  });

  // BrowserSync proxy (auto-reload on view/asset changes)
  try {
    const bs = require('browser-sync').create();
    bs.init({
      proxy   : `localhost:${PORT}`,
      port    : PORT + 1,
      files   : ['views/**/*.html', 'public/**/*'],
      open    : true,
      notify  : false,
      logLevel: 'silent',
    });
    console.log(`🔄  Sync → http://localhost:${PORT + 1}  (auto-reload ativo)\n`);
  } catch (e) {
    console.warn('browser-sync não disponível:', e.message);
  }
});
