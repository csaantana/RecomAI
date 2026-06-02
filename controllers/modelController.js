'use strict';
const usersData = require('../data/users.json');
const products  = require('../data/products.json');
const state     = require('../models/stateModel');
const { trainModel, scoreProducts } = require('../models/recommendationModel');

// ── Shared training logic (used by auto-train on startup & manual retrain) ──

async function triggerTraining(io) {
  if (state.isTraining) return;

  state.isTraining       = true;
  state.trainingComplete = false;
  state.epochHistory     = [];

  io.emit('training:start', { totalEpochs: 60 });

  try {
    const { model, history, sampleCount } = await trainModel(
      usersData,
      products,
      (epoch, logs) => {
        const acc    = logs.acc     ?? logs.accuracy     ?? 0;
        const valAcc = logs.val_acc ?? logs.val_accuracy ?? 0;
        const data   = {
          epoch      : epoch + 1,
          totalEpochs: 60,
          loss       : +logs.loss.toFixed(4),
          accuracy   : +acc.toFixed(4),
          valLoss    : +(logs.val_loss ?? 0).toFixed(4),
          valAccuracy: +valAcc.toFixed(4),
        };
        state.epochHistory.push(data);
        io.emit('training:progress', data);
      }
    );

    state.model           = model;
    state.trainingComplete = true;

    const histAcc    = history.history.acc ?? history.history.accuracy ?? [];
    state.lastMetrics = {
      sampleCount,
      finalLoss    : +history.history.loss.at(-1).toFixed(4),
      finalAccuracy: histAcc.length ? +histAcc.at(-1).toFixed(4) : 0,
    };

    io.emit('training:done', state.lastMetrics);
  } catch (err) {
    console.error('[Trainer]', err);
    io.emit('training:error', { message: err.message });
  } finally {
    state.isTraining = false;
  }
}

// ── Controller handlers ───────────────────────────────────────────────────────

exports.triggerTraining = triggerTraining;

exports.train = async (req, res) => {
  if (state.isTraining) {
    return res.status(409).json({ error: 'Treinamento já em andamento' });
  }
  const io = req.app.get('io');
  res.json({ message: 'Retreinamento iniciado' });
  await triggerTraining(io);
};

exports.recommend = (req, res) => {
  if (!state.model)        return res.status(400).json({ error: 'Modelo não treinado ainda' });
  if (!state.cart.length)  return res.status(400).json({ error: 'Carrinho está vazio' });

  const sorted         = scoreProducts(state.model, state.cart, products);
  const recommendation = sorted.find(p => !p.inCart) ?? null;

  res.json({ sortedProducts: sorted, recommendation });
};
