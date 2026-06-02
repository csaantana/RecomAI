'use strict';
const tf = require('@tensorflow/tfjs');

const allUsers    = require('../data/users.json');
const allProducts = require('../data/products.json');
const state       = require('../models/stateModel');

const {
  trainModel,
  scoreProducts,
  createEmbeddingExtractor,
  buildCartContextVector,
  productToFeatureVector,
  FEATURE_DIM,
  CART_FEATURE_DIM,
} = require('../models/recommendationModel');

const {
  isAvailable      : isQdrantAvailable,
  indexProducts    : indexProductsInQdrant,
  searchSimilarProducts,
} = require('../models/vectorStore');

// ─────────────────────────────────────────────────────────────────────────────
// triggerTraining — chamada no startup e opcionalmente pelo endpoint de retreino
//
// Etapas:
//   1. Gera ~4000 amostras de treino (usuários × subconjuntos de carrinho)
//   2. Treina a rede neural por 60 épocas (emite progresso via Socket.io)
//   3. Persiste o modelo treinado no state
//   4. Se Qdrant estiver disponível:
//        → cria o extrator de embeddings (sub-modelo até a camada Dense(32))
//        → indexa todos os 100 produtos no Qdrant com seus vetores de 32 dims
// ─────────────────────────────────────────────────────────────────────────────
async function triggerTraining(io) {
  if (state.isTraining) return;

  state.isTraining       = true;
  state.trainingComplete = false;
  state.vectorIndexReady = false;
  state.epochHistory     = [];

  io.emit('training:start', { totalEpochs: 60 });

  try {
    // ── Passo 1 & 2: Treinar a rede neural ──────────────────────────────────
    const { model, history, sampleCount } = await trainModel(
      allUsers,
      allProducts,
      async (epochIndex, metrics) => {
        const trainAccuracy = metrics.acc      ?? metrics.accuracy      ?? 0;
        const valAccuracy   = metrics.val_acc  ?? metrics.val_accuracy  ?? 0;

        const epochSnapshot = {
          epoch      : epochIndex + 1,
          totalEpochs: 60,
          loss       : +metrics.loss.toFixed(4),
          accuracy   : +trainAccuracy.toFixed(4),
          valLoss    : +(metrics.val_loss ?? 0).toFixed(4),
          valAccuracy: +valAccuracy.toFixed(4),
        };

        state.epochHistory.push(epochSnapshot); // salvo para clientes que reconectam
        io.emit('training:progress', epochSnapshot);

        // O backend puro JS do TF.js bloqueia o event loop durante cada época.
        // setImmediate garante que o Node.js passe pela fase de I/O (poll) antes
        // de continuar — é onde o socket.io realmente envia os frames WebSocket.
        // setTimeout(0) dispara na fase de timers, ANTES de I/O, então não serve.
        await new Promise(resolve => setImmediate(resolve));
      }
    );

    // ── Passo 3: Persiste o modelo treinado ──────────────────────────────────
    state.model           = model;
    state.trainingComplete = true;

    const accuracyHistory = history.history.acc ?? history.history.accuracy ?? [];
    state.lastMetrics = {
      sampleCount,
      finalLoss    : +history.history.loss.at(-1).toFixed(4),
      finalAccuracy: accuracyHistory.length ? +accuracyHistory.at(-1).toFixed(4) : 0,
    };

    io.emit('training:done', state.lastMetrics);

    // ── Passo 4: Indexar produtos no Qdrant (se disponível) ──────────────────
    const qdrantOnline = await isQdrantAvailable();

    if (qdrantOnline) {
      io.emit('vector:indexing', { total: allProducts.length });

      // Cria sub-modelo que retorna a saída da camada 'embedding_layer' (32 dims)
      state.embeddingExtractor = createEmbeddingExtractor(model);

      await indexProductsInQdrant(
        allProducts,
        state.embeddingExtractor,
        productToFeatureVector,
        CART_FEATURE_DIM
      );

      state.vectorIndexReady = true;
      io.emit('vector:indexed', { count: allProducts.length });
      console.log('[ModelController] Qdrant indexado com sucesso.');
    } else {
      io.emit('vector:unavailable');
      console.warn('[ModelController] Qdrant offline — usando scoring direto como fallback.');
    }

  } catch (trainingError) {
    console.error('[ModelController] Erro durante o treino:', trainingError);
    io.emit('training:error', { message: trainingError.message });
  } finally {
    state.isTraining = false;
  }
}

// ── Endpoints HTTP ────────────────────────────────────────────────────────────

exports.triggerTraining = triggerTraining;

// POST /api/model/train — retreino manual
exports.train = async (req, res) => {
  if (state.isTraining) {
    return res.status(409).json({ error: 'Treinamento já em andamento' });
  }
  const io = req.app.get('io');
  res.json({ message: 'Retreinamento iniciado' });
  await triggerTraining(io);
};

// POST /api/model/recommend — recomendação em dois estágios (Qdrant + re-rank)
exports.recommend = async (req, res) => {
  if (!state.model)       return res.status(400).json({ error: 'Modelo não treinado ainda' });
  if (!state.cart.length) return res.status(400).json({ error: 'Carrinho está vazio' });

  const userAge    = state.sessionUser.age;
  const cartIds    = new Set(state.cart.map(p => p.id));

  // ── Fluxo A: Qdrant disponível → Retrieval + Re-rank ─────────────────────
  if (state.vectorIndexReady && state.embeddingExtractor) {
    try {
      // Estágio 1 — Retrieval: computa embedding do carrinho e busca ANN no Qdrant
      // O input do extrator: [vetorCarrinho(19), zeros(18)] → mesma forma do treino
      const cartContextVector = buildCartContextVector(state.cart, userAge);
      const queryModelInput   = [...cartContextVector, ...new Array(FEATURE_DIM).fill(0.0)];
      const queryEmbedding    = Array.from(
        state.embeddingExtractor.predict(tf.tensor2d([queryModelInput])).dataSync()
      );

      // Qdrant retorna top-50 produtos mais similares ao vetor do carrinho
      const candidates = await searchSimilarProducts(queryEmbedding, 50, cartIds);

      // Estágio 2 — Re-rank: pontua os 50 candidatos com o modelo neural completo
      // (mais preciso que a similaridade cosseno do Qdrant sozinha)
      const rerankedCandidates = scoreProducts(state.model, state.cart, candidates, userAge);

      // Adiciona itens do carrinho ao final (já selecionados)
      const cartItemsWithFlag = state.cart.map(p => ({ ...p, score: -1, inCart: true }));
      const finalRankedList   = [...rerankedCandidates, ...cartItemsWithFlag];
      const topRecommendation = finalRankedList.find(p => !p.inCart) ?? null;

      return res.json({
        sortedProducts  : finalRankedList,
        recommendation  : topRecommendation,
        usedVectorSearch: true,
        candidatesFound : candidates.length,
      });

    } catch (qdrantError) {
      // Qdrant caiu em runtime → cai no fallback abaixo
      console.warn('[Recommend] Qdrant indisponível em runtime, usando fallback:', qdrantError.message);
    }
  }

  // ── Fluxo B: Fallback direto (sem Qdrant) → pontua todos os 100 produtos ─
  const rankedProducts    = scoreProducts(state.model, state.cart, allProducts, userAge);
  const topRecommendation = rankedProducts.find(p => !p.inCart) ?? null;

  res.json({
    sortedProducts  : rankedProducts,
    recommendation  : topRecommendation,
    usedVectorSearch: false,
    candidatesFound : allProducts.length,
  });
};
