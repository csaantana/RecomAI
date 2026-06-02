'use strict';
const tf = require('@tensorflow/tfjs');

const CATEGORIES  = ['Electronics', 'Clothing', 'Sports', 'Home', 'Beauty', 'Books'];
const COLORS      = ['Silver', 'Black', 'Gray', 'White', 'Blue', 'Red', 'Brown', 'Gold', 'Purple', 'Green', 'Orange'];
const MAX_PRICE   = 5000;
const FEATURE_DIM = CATEGORIES.length + 1 + COLORS.length; // 6 + 1 + 11 = 18

// ── Feature encoding ─────────────────────────────────────────────────────────

function productToVector(p) {
  const cat   = CATEGORIES.map(c => (c === p.category ? 1.0 : 0.0));
  const price = [Math.min(p.price / MAX_PRICE, 1.0)];
  const color = COLORS.map(c => (c === p.color ? 1.0 : 0.0));
  return [...cat, ...price, ...color];
}

function meanVector(vecs) {
  if (!vecs.length) return new Array(FEATURE_DIM).fill(0);
  const sum = new Array(FEATURE_DIM).fill(0);
  for (const v of vecs) v.forEach((x, i) => { sum[i] += x; });
  return sum.map(x => x / vecs.length);
}

// ── Combinatorics ─────────────────────────────────────────────────────────────

function getCombinations(arr, k) {
  if (k === 0) return [[]];
  if (!arr.length || arr.length < k) return [];
  const [head, ...tail] = arr;
  return [
    ...getCombinations(tail, k - 1).map(c => [head, ...c]),
    ...getCombinations(tail, k),
  ];
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ── Training data generation ──────────────────────────────────────────────────
//
// Strategy: for each user with N purchases, generate all subsets of size k
// (k = 1..N-1) as "cart context". Each subset produces:
//   - positive samples: remaining purchased products (label = 1)
//   - negative samples: 3 × |remaining| random un-purchased products (label = 0)
//
// ~300 samples/user × 10 users ≈ 3000 training samples total.

function generateTrainingData(users, products) {
  const pMap   = {};
  products.forEach(p => { pMap[p.id] = p; });
  const allIds = products.map(p => p.id);

  const inputs = [];
  const labels = [];

  for (const user of users) {
    const bought = user.purchaseHistory.map(id => pMap[id]).filter(Boolean);
    if (bought.length < 2) continue;

    const boughtIds = new Set(bought.map(p => p.id));
    const negPool   = allIds.filter(id => !boughtIds.has(id));

    for (let k = 1; k < bought.length; k++) {
      for (const subset of getCombinations(bought, k)) {
        const remaining = bought.filter(p => !subset.includes(p));
        const ctxVec    = meanVector(subset.map(productToVector));

        // Positive samples
        for (const pos of remaining) {
          inputs.push([...ctxVec, ...productToVector(pos)]);
          labels.push(1.0);
        }

        // 3 negatives per positive
        const negs = shuffle([...negPool]).slice(0, remaining.length * 3);
        for (const negId of negs) {
          inputs.push([...ctxVec, ...productToVector(pMap[negId])]);
          labels.push(0.0);
        }
      }
    }
  }

  // Final shuffle (zip inputs+labels together to keep alignment)
  const idx = shuffle(Array.from({ length: labels.length }, (_, i) => i));
  return {
    inputs: idx.map(i => inputs[i]),
    labels: idx.map(i => labels[i]),
    count : labels.length,
  };
}

// ── Model architecture ────────────────────────────────────────────────────────

function buildModel() {
  const model = tf.sequential({ name: 'RecommendationNet' });

  model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [FEATURE_DIM * 2] }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1,  activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss     : 'binaryCrossentropy',
    metrics  : ['accuracy'],
  });

  return model;
}

// ── Public API ────────────────────────────────────────────────────────────────

async function trainModel(users, products, onEpochEnd) {
  const { inputs, labels, count } = generateTrainingData(users, products);
  console.log(`[RecomModel] Training samples generated: ${count}`);

  const X       = tf.tensor2d(inputs);
  const y       = tf.tensor1d(labels);
  const model   = buildModel();

  const history = await model.fit(X, y, {
    epochs         : 60,
    batchSize      : 32,
    validationSplit: 0.2,
    shuffle        : true,
    callbacks      : {
      onEpochEnd: async (epoch, logs) => {
        if (onEpochEnd) await onEpochEnd(epoch, logs);
      },
    },
  });

  X.dispose();
  y.dispose();

  return { model, history, sampleCount: count };
}

function scoreProducts(model, cartProducts, allProducts) {
  if (!model || !cartProducts.length) {
    return allProducts.map(p => ({ ...p, score: 0, inCart: false }));
  }

  const ctxVec  = meanVector(cartProducts.map(productToVector));
  const cartIds = new Set(cartProducts.map(p => p.id));

  return tf.tidy(() => {
    const raw    = allProducts.map(p => [...ctxVec, ...productToVector(p)]);
    const tensor = tf.tensor2d(raw);
    const scores = Array.from(model.predict(tensor).dataSync());

    return allProducts
      .map((p, i) => ({ ...p, score: +scores[i].toFixed(4), inCart: cartIds.has(p.id) }))
      .sort((a, b) => {
        if (a.inCart !== b.inCart) return a.inCart ? 1 : -1;
        return b.score - a.score;
      });
  });
}

module.exports = { trainModel, scoreProducts, FEATURE_DIM };
