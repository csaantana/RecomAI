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

io.on('connection', socket => {
  console.log('🔌 Client connected');
  socket.on('disconnect', () => console.log('🔌 Client disconnected'));
});

server.listen(PORT, () => {
  console.log(`\n✅  App  → http://localhost:${PORT}`);

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
