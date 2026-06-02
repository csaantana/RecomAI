'use strict';
const allUsers    = require('../data/users.json');    // histórico de compras dos 10 usuários
const allProducts = require('../data/products.json'); // catálogo completo (30 produtos)
const state       = require('../models/stateModel');
const { trainModel, scoreProducts } = require('../models/recommendationModel');

// ─────────────────────────────────────────────────────────────────────────────
// triggerTraining
//
// Função central de treinamento — chamada automaticamente no startup do servidor
// e opcionalmente pelo endpoint POST /api/model/train (retreino manual).
//
// Fluxo:
//   1. Gera ~3000 amostras a partir do histórico dos usuários
//   2. Treina a rede por 60 épocas
//   3. Emite eventos Socket.io a cada época → browser atualiza os gráficos em tempo real
//   4. Ao terminar, persiste o modelo treinado no state (em memória)
//
// O modelo treinado fica disponível para todos os requests subsequentes
// sem precisar de novo treino — apenas a inferência (scoreProducts) é chamada.
// ─────────────────────────────────────────────────────────────────────────────
async function triggerTraining(io) {
  if (state.isTraining) return; // evita treinos simultâneos

  // Reinicia o estado de treino
  state.isTraining       = true;
  state.trainingComplete = false;
  state.epochHistory     = []; // histórico de épocas para clientes que conectam tarde

  io.emit('training:start', { totalEpochs: 60 });

  try {
    const { model, history, sampleCount } = await trainModel(
      allUsers,
      allProducts,

      // Callback chamado ao fim de cada época pelo TF.js
      (epochIndex, metrics) => {
        // TF.js pode retornar 'acc' ou 'accuracy' dependendo da versão
        const trainAccuracy = metrics.acc      ?? metrics.accuracy      ?? 0;
        const valAccuracy   = metrics.val_acc  ?? metrics.val_accuracy  ?? 0;

        const epochSnapshot = {
          epoch      : epochIndex + 1,  // começa em 1 para exibição
          totalEpochs: 60,
          loss       : +metrics.loss.toFixed(4),
          accuracy   : +trainAccuracy.toFixed(4),
          valLoss    : +(metrics.val_loss ?? 0).toFixed(4),
          valAccuracy: +valAccuracy.toFixed(4),
        };

        // Salva no histórico para reproduzir ao browser que reconectar após o treino
        state.epochHistory.push(epochSnapshot);
        io.emit('training:progress', epochSnapshot);
      }
    );

    // Persiste o modelo treinado — sobrevive a qualquer número de requests
    state.model           = model;
    state.trainingComplete = true;

    // Extrai métricas finais do histórico retornado pelo TF.js
    const accuracyHistory = history.history.acc ?? history.history.accuracy ?? [];
    state.lastMetrics = {
      sampleCount,
      finalLoss    : +history.history.loss.at(-1).toFixed(4),
      finalAccuracy: accuracyHistory.length ? +accuracyHistory.at(-1).toFixed(4) : 0,
    };

    io.emit('training:done', state.lastMetrics);

  } catch (trainingError) {
    console.error('[ModelController] Erro durante o treino:', trainingError);
    io.emit('training:error', { message: trainingError.message });

  } finally {
    state.isTraining = false;
  }
}

// ── Endpoints HTTP ────────────────────────────────────────────────────────────

exports.triggerTraining = triggerTraining;

// POST /api/model/train — retreino manual (opcional, modelo já treina no startup)
exports.train = async (req, res) => {
  if (state.isTraining) {
    return res.status(409).json({ error: 'Treinamento já em andamento' });
  }

  const io = req.app.get('io'); // acessa a instância Socket.io registrada no app
  res.json({ message: 'Retreinamento iniciado' }); // responde imediatamente (treino é assíncrono)
  await triggerTraining(io);
};

// POST /api/model/recommend — pontua e ordena o catálogo com base no carrinho atual
exports.recommend = (req, res) => {
  if (!state.model)        return res.status(400).json({ error: 'Modelo não treinado ainda' });
  if (!state.cart.length)  return res.status(400).json({ error: 'Carrinho está vazio' });

  // Pontua todos os produtos usando o modelo treinado e o vetor do carrinho atual
  const rankedProducts = scoreProducts(state.model, state.cart, allProducts);

  // A recomendação principal é o produto com maior score que ainda não está no carrinho
  const topRecommendation = rankedProducts.find(product => !product.inCart) ?? null;

  res.json({ sortedProducts: rankedProducts, recommendation: topRecommendation });
};
