'use strict';
const tf = require('@tensorflow/tfjs');

// ─────────────────────────────────────────────────────────────────────────────
// RECOMMENDATION MODEL
//
// Abordagem: modelo pairwise ("par de produtos")
//   Entrada : concat( vetor do carrinho atual | vetor do produto candidato )
//   Saída   : probabilidade [0–1] de que o produto candidato combine com o carrinho
//
// O modelo aprende padrões de co-compra a partir do histórico dos 10 usuários.
// Uma vez treinado, basta passar qualquer carrinho que ele rankeia o catálogo.
// ─────────────────────────────────────────────────────────────────────────────

// ── 1. Espaço de features ─────────────────────────────────────────────────────
// Cada produto vira um vetor de 18 números:
//   6 (categoria, one-hot) + 1 (preço normalizado) + 11 (cor, one-hot) = 18
const PRODUCT_CATEGORIES = ['Electronics', 'Clothing', 'Sports', 'Home', 'Beauty', 'Books'];
const PRODUCT_COLORS     = ['Silver', 'Black', 'Gray', 'White', 'Blue', 'Red', 'Brown', 'Gold', 'Purple', 'Green', 'Orange'];
const MAX_PRICE_BRL      = 5000;
const FEATURE_DIM        = PRODUCT_CATEGORIES.length + 1 + PRODUCT_COLORS.length; // 18

// ── 2. Encoding de produto → vetor numérico ───────────────────────────────────
// One-hot: a posição correspondente recebe 1.0, o restante fica 0.0.
// O preço é dividido pelo máximo esperado para ficar na mesma escala [0, 1]
// (sem isso, features com valores grandes dominam as pequenas numericamente).
function productToFeatureVector(product) {
  const categoryOneHot = PRODUCT_CATEGORIES.map(cat => (cat === product.category ? 1.0 : 0.0));
  const normalizedPrice = [Math.min(product.price / MAX_PRICE_BRL, 1.0)];
  const colorOneHot    = PRODUCT_COLORS.map(color => (color === product.color ? 1.0 : 0.0));

  return [...categoryOneHot, ...normalizedPrice, ...colorOneHot]; // 18 dimensões
}

// Representa um conjunto de produtos (ex: carrinho) como a média dos seus vetores.
// A média captura o "perfil geral" do carrinho sem depender do número de itens.
function averageFeatureVector(productFeatureVectors) {
  if (!productFeatureVectors.length) return new Array(FEATURE_DIM).fill(0);

  const sumVector = new Array(FEATURE_DIM).fill(0);
  for (const featureVector of productFeatureVectors) {
    featureVector.forEach((value, dimension) => { sumVector[dimension] += value; });
  }
  return sumVector.map(sum => sum / productFeatureVectors.length);
}

// ── 3. Geração de combinações (helper para dados de treino) ───────────────────
// Retorna todos os subconjuntos de tamanho k de um array.
// Ex: getCombinations([A,B,C], 2) → [[A,B], [A,C], [B,C]]
function getCombinations(array, subsetSize) {
  if (subsetSize === 0) return [[]];
  if (!array.length || array.length < subsetSize) return [];

  const [firstItem, ...remainingItems] = array;
  const subsetsWithFirst    = getCombinations(remainingItems, subsetSize - 1).map(combo => [firstItem, ...combo]);
  const subsetsWithoutFirst = getCombinations(remainingItems, subsetSize);
  return [...subsetsWithFirst, ...subsetsWithoutFirst];
}

function shuffleInPlace(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const randomIndex = Math.floor(Math.random() * (i + 1));
    [array[i], array[randomIndex]] = [array[randomIndex], array[i]];
  }
  return array;
}

// ── 4. Geração dos dados de treinamento ───────────────────────────────────────
//
// Estratégia: simula todos os estados possíveis de carrinho de cada usuário.
//
// Para cada usuário com N compras e para cada tamanho k de "carrinho parcial"
// (k = 1 até N-1):
//   → gera todos os subconjuntos de tamanho k como contexto de carrinho
//   → produtos restantes comprados pelo usuário  = label 1 (par positivo)
//   → produtos aleatórios não comprados          = label 0 (par negativo, 3x)
//
// Resultado: ~300 amostras/usuário × 10 usuários ≈ 3000 amostras balanceadas,
// cobrindo muitos cenários de compra sem precisar de dataset externo.
function generateTrainingData(users, products) {
  // Índice rápido para buscar produto por id
  const productById = {};
  products.forEach(product => { productById[product.id] = product; });

  const allProductIds = products.map(product => product.id);

  // X = vetores de entrada (36 dims cada), y = labels (0 ou 1)
  const inputVectors = [];
  const outputLabels = [];

  for (const user of users) {
    const purchasedProducts = user.purchaseHistory
      .map(productId => productById[productId])
      .filter(Boolean); // descarta ids inválidos

    if (purchasedProducts.length < 2) continue; // precisa de ao menos 2 itens para formar par

    const purchasedProductIds = new Set(purchasedProducts.map(p => p.id));
    const notPurchasedIds     = allProductIds.filter(id => !purchasedProductIds.has(id));

    for (let cartSize = 1; cartSize < purchasedProducts.length; cartSize++) {
      const cartSubsets = getCombinations(purchasedProducts, cartSize);

      for (const cartSubset of cartSubsets) {
        // Produtos que o usuário comprou mas ainda não estão neste carrinho parcial
        const productsNotYetInCart = purchasedProducts.filter(p => !cartSubset.includes(p));

        // Vetor do carrinho = média dos vetores dos itens do subconjunto atual
        const cartContextVector = averageFeatureVector(cartSubset.map(productToFeatureVector));

        // Pares positivos: carrinho → produto que o usuário realmente comprou
        for (const boughtProduct of productsNotYetInCart) {
          inputVectors.push([...cartContextVector, ...productToFeatureVector(boughtProduct)]);
          outputLabels.push(1.0);
        }

        // Pares negativos: carrinho → produto aleatório que o usuário NÃO comprou
        // 3 negativos por positivo mantém o dataset balanceado
        const negativeProductIds = shuffleInPlace([...notPurchasedIds])
          .slice(0, productsNotYetInCart.length * 3);

        for (const negativeId of negativeProductIds) {
          inputVectors.push([...cartContextVector, ...productToFeatureVector(productById[negativeId])]);
          outputLabels.push(0.0);
        }
      }
    }
  }

  // Embaralha inputs e labels juntos (mantendo alinhamento) para evitar viés de ordem
  const shuffledIndices = shuffleInPlace(Array.from({ length: outputLabels.length }, (_, i) => i));
  return {
    inputVectors : shuffledIndices.map(i => inputVectors[i]),
    outputLabels : shuffledIndices.map(i => outputLabels[i]),
    totalSamples : outputLabels.length,
  };
}

// ── 5. Arquitetura da rede neural ─────────────────────────────────────────────
//
// Entrada (36): [ vetor do carrinho (18) | vetor do produto candidato (18) ]
// Saída   (1) : probabilidade sigmoid → o quão bem o produto combina com o carrinho
//
// As camadas Dense comprimem progressivamente a representação.
// BatchNormalization estabiliza os gradientes → treino mais rápido e consistente.
// Dropout "desliga" neurônios aleatórios durante o treino → evita memorizar os dados.
function buildModel() {
  const model = tf.sequential({ name: 'RecommendationNet' });

  model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [FEATURE_DIM * 2] }));
  model.add(tf.layers.batchNormalization()); // normaliza ativações entre camadas
  model.add(tf.layers.dropout({ rate: 0.3 })); // desativa 30% dos neurônios aleatoriamente

  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));

  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));

  // Sigmoid: converte a saída em probabilidade entre 0 e 1
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(0.001),  // Adam ajusta a taxa de aprendizado por parâmetro automaticamente
    loss     : 'binaryCrossentropy', // loss padrão para classificação binária (comprou / não comprou)
    metrics  : ['accuracy'],
  });

  return model;
}

// ── 6. Treinamento ────────────────────────────────────────────────────────────
// Converte os dados em tensores (formato numérico do TF), treina a rede por 60
// épocas e chama onEpochEnd a cada época (usado para transmitir métricas ao browser).
async function trainModel(users, products, onEpochEnd) {
  const { inputVectors, outputLabels, totalSamples } = generateTrainingData(users, products);
  console.log(`[RecomModel] Amostras de treino geradas: ${totalSamples}`);

  // Tensores: matrizes numéricas que o TF.js processa (suporte a GPU/WASM)
  const inputTensor = tf.tensor2d(inputVectors);  // shape: [totalSamples, 36]
  const labelTensor = tf.tensor1d(outputLabels);  // shape: [totalSamples]

  const model = buildModel();

  const trainingHistory = await model.fit(inputTensor, labelTensor, {
    epochs         : 60,
    batchSize      : 32,   // atualiza os pesos a cada 32 amostras (não uma por vez)
    validationSplit: 0.2,  // 20% dos dados reservados para medir overfitting
    shuffle        : true,
    callbacks: {
      // Chamado ao fim de cada época → controlador emite evento Socket.io ao browser
      onEpochEnd: async (epochIndex, metrics) => {
        if (onEpochEnd) await onEpochEnd(epochIndex, metrics);
      },
    },
  });

  // Libera a memória dos tensores de entrada (o modelo retém apenas seus pesos)
  inputTensor.dispose();
  labelTensor.dispose();

  return { model, history: trainingHistory, sampleCount: totalSamples };
}

// ── 7. Inferência (Recomendação) ──────────────────────────────────────────────
// Dado o carrinho atual, atribui um score a cada produto do catálogo e ordena.
// Produtos já no carrinho vão para o final da lista (já foram escolhidos).
function scoreProducts(trainedModel, cartProducts, allProducts) {
  // Sem modelo ou carrinho vazio: retorna catálogo sem ordenação
  if (!trainedModel || !cartProducts.length) {
    return allProducts.map(product => ({ ...product, score: 0, inCart: false }));
  }

  // Representa o carrinho como um único vetor (média dos itens)
  const cartContextVector = averageFeatureVector(cartProducts.map(productToFeatureVector));
  const cartProductIds    = new Set(cartProducts.map(p => p.id));

  // tf.tidy() descarta automaticamente tensores intermediários após o bloco → sem memory leak
  return tf.tidy(() => {
    // Monta uma linha de input para cada produto: [contextoCarrinho | vetorProduto]
    const scoringInputMatrix = allProducts.map(product => [
      ...cartContextVector,
      ...productToFeatureVector(product),
    ]);

    // Roda o modelo em batch para todos os produtos de uma vez (eficiente)
    const predictionScores = Array.from(
      trainedModel.predict(tf.tensor2d(scoringInputMatrix)).dataSync()
    );

    return allProducts
      .map((product, index) => ({
        ...product,
        score : +predictionScores[index].toFixed(4), // probabilidade arredondada
        inCart: cartProductIds.has(product.id),
      }))
      .sort((productA, productB) => {
        // Itens do carrinho sempre vão para o fim (já selecionados)
        if (productA.inCart !== productB.inCart) return productA.inCart ? 1 : -1;
        // Demais produtos: maior score primeiro
        return productB.score - productA.score;
      });
  });
}

module.exports = { trainModel, scoreProducts, FEATURE_DIM };
