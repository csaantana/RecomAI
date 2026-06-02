'use strict';
const usersData = require('../data/users.json');
const products  = require('../data/products.json');
const state     = require('../models/stateModel');
const { trainModel, scoreProducts } = require('../models/recommendationModel');

exports.train = async (req, res) => {
  if (state.isTraining) {
    return res.status(409).json({ error: 'Treinamento já em andamento' });
  }

  const io = req.app.get('io');
  state.isTraining = true;
  res.json({ message: 'Treinamento iniciado' });

  try {
    const { model, history, sampleCount } = await trainModel(
      usersData,
      products,
      (epoch, logs) => {
        const acc    = logs.acc     ?? logs.accuracy     ?? 0;
        const valAcc = logs.val_acc ?? logs.val_accuracy ?? 0;
        io.emit('training:progress', {
          epoch      : epoch + 1,
          totalEpochs: 60,
          loss       : +logs.loss.toFixed(4),
          accuracy   : +acc.toFixed(4),
          valLoss    : +(logs.val_loss ?? 0).toFixed(4),
          valAccuracy: +valAcc.toFixed(4),
        });
      }
    );

    state.model = model;

    const histAcc = history.history.acc ?? history.history.accuracy ?? [];
    io.emit('training:done', {
      sampleCount,
      finalLoss    : +history.history.loss.at(-1).toFixed(4),
      finalAccuracy: histAcc.length ? +histAcc.at(-1).toFixed(4) : 0,
    });
  } catch (err) {
    console.error('[ModelController] train error:', err);
    io.emit('training:error', { message: err.message });
  } finally {
    state.isTraining = false;
  }
};

exports.recommend = (req, res) => {
  if (!state.model)        return res.status(400).json({ error: 'Modelo não treinado ainda' });
  if (!state.cart.length)  return res.status(400).json({ error: 'Carrinho está vazio' });

  const sorted         = scoreProducts(state.model, state.cart, products);
  const recommendation = sorted.find(p => !p.inCart) ?? null;

  res.json({ sortedProducts: sorted, recommendation });
};
