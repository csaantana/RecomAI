'use strict';
const tf             = require('@tensorflow/tfjs');
const { QdrantClient } = require('@qdrant/js-client-rest');

// ─────────────────────────────────────────────────────────────────────────────
// VECTOR STORE — camada de acesso ao Qdrant
//
// Qdrant é um banco de dados vetorial que armazena embeddings (vetores numéricos)
// e realiza busca por similaridade via ANN (Approximate Nearest Neighbor).
//
// Fluxo:
//   1. indexProducts() → extrai embedding de 32 dims de cada produto via modelo TF.js
//                      → upserta todos os vetores no Qdrant
//   2. searchSimilarProducts() → recebe vetor do carrinho → Qdrant retorna top-K similares
//
// Por que é mais rápido que scoring direto?
//   Scoring direto: O(n) forward passes na rede neural
//   Qdrant ANN:     O(log n) busca no índice HNSW — escalável para milhões de vetores
// ─────────────────────────────────────────────────────────────────────────────

const COLLECTION_NAME = 'product_embeddings';
const EMBEDDING_DIM   = 32;                                  // dimensão da camada 'embedding_layer'
const QDRANT_URL      = process.env.QDRANT_URL ?? 'http://localhost:6333';

const qdrantClient = new QdrantClient({ url: QDRANT_URL });

// Verifica se o Qdrant está disponível (Docker rodando) — sem lançar exceção
async function isAvailable() {
  try {
    await qdrantClient.getCollections();
    return true;
  } catch {
    return false;
  }
}

// Cria (ou recria) a collection no Qdrant com métrica de similaridade cosseno
async function createCollection() {
  const { collections } = await qdrantClient.getCollections();

  // Recria do zero a cada treino para garantir que os embeddings estejam atualizados
  if (collections.some(c => c.name === COLLECTION_NAME)) {
    await qdrantClient.deleteCollection(COLLECTION_NAME);
  }

  await qdrantClient.createCollection(COLLECTION_NAME, {
    vectors: {
      size    : EMBEDDING_DIM,  // 32 dims
      distance: 'Cosine',       // similaridade cosseno: ignora magnitude, foca na direção do vetor
    },
  });
}

// Indexa todos os produtos no Qdrant
// Para cada produto, extrai o embedding via extrator e upserta com o payload completo
async function indexProducts(products, embeddingExtractor, productToFeatureVector, cartFeatureDim) {
  await createCollection();

  const points = [];
  for (let index = 0; index < products.length; index++) {
    const product = products[index];

    // Input do extrator: carrinho vazio (zeros) + features do produto
    // O carrinho vazio (0s) faz com que o embedding reflita apenas o produto em si
    const emptyCartContext = new Array(cartFeatureDim).fill(0.0);
    const productFeatures  = productToFeatureVector(product);
    const modelInput       = [...emptyCartContext, ...productFeatures];

    // Extrai embedding de 32 dims passando pelo modelo até a camada 'embedding_layer'
    const embeddingTensor = embeddingExtractor.predict(tf.tensor2d([modelInput]));
    const embeddingVector = Array.from(embeddingTensor.dataSync());
    embeddingTensor.dispose(); // libera memória imediatamente

    points.push({
      id     : index + 1,        // Qdrant exige id numérico inteiro ou UUID string
      vector : embeddingVector,  // vetor de 32 dims
      payload: { ...product },   // metadados do produto — recuperados junto com o resultado da busca
    });
  }

  // Envia todos os pontos de uma vez (batch upsert)
  await qdrantClient.upsert(COLLECTION_NAME, { wait: true, points });
  console.log(`[VectorStore] ${points.length} produtos indexados (${EMBEDDING_DIM} dims, Cosine).`);
}

// Busca os K produtos mais similares ao vetor de consulta do carrinho
// Exclui os produtos que já estão no carrinho (já selecionados)
async function searchSimilarProducts(queryVector, topK, excludeProductIds) {
  const results = await qdrantClient.search(COLLECTION_NAME, {
    vector      : queryVector,
    limit       : topK + excludeProductIds.size, // busca extra para compensar as exclusões
    with_payload: true,
  });

  return results
    .filter(result => !excludeProductIds.has(result.payload.id)) // remove itens do carrinho
    .slice(0, topK)
    .map(result => ({
      ...result.payload,
      qdrantScore: +result.score.toFixed(4), // score de similaridade cosseno [0, 1]
    }));
}

module.exports = { isAvailable, indexProducts, searchSimilarProducts, EMBEDDING_DIM };
