'use strict';
const tf = require('@tensorflow/tfjs');

// ─────────────────────────────────────────────────────────────────────────────
// RECOMMENDATION MODEL
//
// Abordagem: modelo pairwise + Two-Stage Retrieval
//   Treino  : concat( vetor do carrinho com idade | vetor do produto candidato )
//   Saída   : probabilidade [0–1] de que o candidato combine com o carrinho
//
// Com Qdrant:
//   1. Pós-treino → extrai embeddings de 32 dims de cada produto → indexa no Qdrant
//   2. Inferência → computa embedding do carrinho → busca ANN no Qdrant (top-50)
//                → re-rank dos 50 candidatos com o modelo neural completo
//
// Sem Qdrant (fallback):
//   → pontua todos os 100 produtos diretamente com o modelo
// ─────────────────────────────────────────────────────────────────────────────

// ── 1. Espaço de features ─────────────────────────────────────────────────────
// Produto → 18 dims: 6 (categoria) + 1 (preço) + 11 (cor)
// Carrinho → 19 dims: 18 (média dos produtos) + 1 (idade normalizada do usuário)
// Input do modelo → 37 dims: concat(carrinho[19], produto[18])
const PRODUCT_CATEGORIES = ['Electronics', 'Clothing', 'Sports', 'Home', 'Beauty', 'Books'];
const PRODUCT_COLORS     = ['Silver', 'Black', 'Gray', 'White', 'Blue', 'Red', 'Brown', 'Gold', 'Purple', 'Green', 'Orange'];
const MAX_PRICE_BRL      = 5000;
const MAX_AGE            = 100;
const FEATURE_DIM        = PRODUCT_CATEGORIES.length + 1 + PRODUCT_COLORS.length; // 18
const CART_FEATURE_DIM   = FEATURE_DIM + 1;  // 18 + 1 (idade) = 19
const INPUT_DIM          = CART_FEATURE_DIM + FEATURE_DIM; // 19 + 18 = 37

// ── 2. Encoding de produto → vetor numérico (18 dims) ────────────────────────
// One-hot: somente a posição correspondente recebe 1.0.
// Preço dividido pelo máximo para ficar na mesma escala [0,1] que o restante.
function productToFeatureVector(product) {
  const categoryOneHot  = PRODUCT_CATEGORIES.map(cat   => (cat   === product.category ? 1.0 : 0.0));
  const normalizedPrice = [Math.min(product.price / MAX_PRICE_BRL, 1.0)];
  const colorOneHot     = PRODUCT_COLORS.map(color => (color === product.color    ? 1.0 : 0.0));

  return [...categoryOneHot, ...normalizedPrice, ...colorOneHot]; // 18 dims
}

// Média dos vetores de um conjunto de produtos.
// Representa o "perfil geral" do carrinho independente do número de itens.
function averageFeatureVector(productVectors) {
  if (!productVectors.length) return new Array(FEATURE_DIM).fill(0);

  const sumVector = new Array(FEATURE_DIM).fill(0);
  for (const vector of productVectors) {
    vector.forEach((value, i) => { sumVector[i] += value; });
  }
  return sumVector.map(sum => sum / productVectors.length);
}

// ── 3. Vetor de contexto do carrinho (19 dims) ────────────────────────────────
// A idade do usuário é a última dimensão — permite ao modelo capturar padrões
// de preferência por faixa etária (ex: jovens → eletrônicos, adultos → culinária).
function buildCartContextVector(cartProducts, userAge) {
  const avgProductFeatures = averageFeatureVector(cartProducts.map(productToFeatureVector));
  const normalizedAge      = (userAge ?? 28) / MAX_AGE;

  return [...avgProductFeatures, normalizedAge]; // 19 dims
}

// ── 4. Geração de combinações ─────────────────────────────────────────────────
// Retorna todos os subconjuntos de tamanho k.
// Ex: getCombinations([A,B,C], 2) → [[A,B],[A,C],[B,C]]
function getCombinations(array, subsetSize) {
  if (subsetSize === 0) return [[]];
  if (!array.length || array.length < subsetSize) return [];

  const [first, ...rest] = array;
  const withFirst    = getCombinations(rest, subsetSize - 1).map(combo => [first, ...combo]);
  const withoutFirst = getCombinations(rest, subsetSize);
  return [...withFirst, ...withoutFirst];
}

function shuffleInPlace(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

// ── 5. Geração dos dados de treinamento ───────────────────────────────────────
//
// Para cada usuário com N compras, simula todos os estados possíveis de carrinho
// (subconjuntos de tamanho k = 1..N-1) e gera:
//   - Pares positivos : produto que o usuário comprou dado aquele carrinho parcial → label 1
//   - Pares negativos : produto aleatório não comprado → label 0  (3× por positivo)
//
// A idade do usuário é incluída no vetor de contexto do carrinho.
// Resultado: ~4000 amostras com 100 produtos e usuários com 12 compras médias.
function generateTrainingData(users, products) {
  const productById   = {};
  products.forEach(product => { productById[product.id] = product; });

  const allProductIds = products.map(product => product.id);
  const inputVectors  = []; // X — shape [N, 37]
  const outputLabels  = []; // y — shape [N]

  for (const user of users) {
    const userAge           = user.age ?? 28;
    const purchasedProducts = user.purchaseHistory
      .map(id => productById[id])
      .filter(Boolean);

    if (purchasedProducts.length < 2) continue;

    const purchasedIds = new Set(purchasedProducts.map(p => p.id));
    const negativePool = allProductIds.filter(id => !purchasedIds.has(id));

    for (let cartSize = 1; cartSize < purchasedProducts.length; cartSize++) {
      const cartSubsets = getCombinations(purchasedProducts, cartSize);

      for (const cartSubset of cartSubsets) {
        const notYetBought      = purchasedProducts.filter(p => !cartSubset.includes(p));
        const cartContextVector = buildCartContextVector(cartSubset, userAge);

        // Pares positivos: produtos que o usuário comprou além do carrinho atual
        for (const boughtProduct of notYetBought) {
          inputVectors.push([...cartContextVector, ...productToFeatureVector(boughtProduct)]);
          outputLabels.push(1.0);
        }

        // Pares negativos: 3 produtos aleatórios não comprados pelo usuário
        const negatives = shuffleInPlace([...negativePool]).slice(0, notYetBought.length * 3);
        for (const negativeId of negatives) {
          inputVectors.push([...cartContextVector, ...productToFeatureVector(productById[negativeId])]);
          outputLabels.push(0.0);
        }
      }
    }
  }

  // Embaralha inputs e labels juntos mantendo o alinhamento
  const shuffledIndices = shuffleInPlace(Array.from({ length: outputLabels.length }, (_, i) => i));
  return {
    inputVectors: shuffledIndices.map(i => inputVectors[i]),
    outputLabels: shuffledIndices.map(i => outputLabels[i]),
    totalSamples: outputLabels.length,
  };
}

// ── 6. Arquitetura da rede neural (Input: 37 dims) ────────────────────────────
//
// [carrinho+idade (19)] + [produto candidato (18)] → Dense(128) → BN → Drop(0.3)
//                                                  → Dense(64)  → Drop(0.2)
//                                                  → Dense(32)  ← embedding
//                                                  → Dense(1, sigmoid) → probabilidade
function buildModel() {
  const model = tf.sequential({ name: 'RecommendationNet' });

  model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [INPUT_DIM] }));
  model.add(tf.layers.batchNormalization()); // normaliza ativações → treino mais estável
  model.add(tf.layers.dropout({ rate: 0.3 })); // desativa 30% aleatoriamente → evita overfitting

  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));

  // Camada de embedding — saída de 32 dims usada pelo Qdrant
  model.add(tf.layers.dense({ units: 32, activation: 'relu', name: 'embedding_layer' }));

  // Sigmoid: converte para probabilidade binária [0, 1]
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(0.001), // Adam ajusta lr por parâmetro automaticamente
    loss     : 'binaryCrossentropy', // loss padrão para classificação binária
    metrics  : ['accuracy'],
  });

  return model;
}

// ── 7. Treinamento ────────────────────────────────────────────────────────────
// Converte os dados em tensores, treina por 60 épocas e chama onEpochEnd a cada
// época (o controlador usa este callback para emitir progresso via Socket.io).
async function trainModel(users, products, onEpochEnd) {
  const { inputVectors, outputLabels, totalSamples } = generateTrainingData(users, products);
  console.log(`[RecomModel] Amostras de treino: ${totalSamples}`);

  const inputTensor = tf.tensor2d(inputVectors); // shape: [totalSamples, 37]
  const labelTensor = tf.tensor1d(outputLabels); // shape: [totalSamples]

  const model = buildModel();

  const trainingHistory = await model.fit(inputTensor, labelTensor, {
    epochs         : 60,
    batchSize      : 32,   // atualiza pesos a cada 32 amostras
    validationSplit: 0.2,  // 20% reservado para detectar overfitting
    shuffle        : true,
    callbacks: {
      onEpochEnd: async (epochIndex, metrics) => {
        if (onEpochEnd) await onEpochEnd(epochIndex, metrics);
      },
    },
  });

  inputTensor.dispose(); // libera memória dos tensores de entrada
  labelTensor.dispose();

  return { model, history: trainingHistory, sampleCount: totalSamples };
}

// ── 8. Extrator de Embedding ──────────────────────────────────────────────────
// Cria um sub-modelo que usa os mesmos pesos do modelo treinado mas retorna
// a saída da camada 'embedding_layer' (32 dims) em vez da probabilidade final.
// Esse vetor de 32 dims é o que fica armazenado no Qdrant para cada produto.
function createEmbeddingExtractor(trainedModel) {
  const embeddingLayer = trainedModel.getLayer('embedding_layer');
  return tf.model({
    inputs : trainedModel.inputs,
    outputs: embeddingLayer.output,
  });
}

// ── 9. Inferência — Scoring direto (fallback sem Qdrant) ─────────────────────
// Pontua todos os produtos do catálogo contra o carrinho atual.
// Retorna lista ordenada: não-carrinho por score desc, carrinho ao final.
function scoreProducts(trainedModel, cartProducts, allProducts, userAge = 28) {
  if (!trainedModel || !cartProducts.length) {
    return allProducts.map(product => ({ ...product, score: 0, inCart: false }));
  }

  const cartContextVector = buildCartContextVector(cartProducts, userAge);
  const cartProductIds    = new Set(cartProducts.map(p => p.id));

  return tf.tidy(() => {
    const scoringInputs = allProducts.map(product => [
      ...cartContextVector,
      ...productToFeatureVector(product),
    ]);

    const predictionScores = Array.from(
      trainedModel.predict(tf.tensor2d(scoringInputs)).dataSync()
    );

    return allProducts
      .map((product, i) => ({
        ...product,
        score : +predictionScores[i].toFixed(4),
        inCart: cartProductIds.has(product.id),
      }))
      .sort((a, b) => {
        if (a.inCart !== b.inCart) return a.inCart ? 1 : -1;
        return b.score - a.score;
      });
  });
}

module.exports = {
  trainModel,
  scoreProducts,
  createEmbeddingExtractor,
  buildCartContextVector,
  productToFeatureVector,
  FEATURE_DIM,
  CART_FEATURE_DIM,
  INPUT_DIM,
};
