'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// STATE — memória compartilhada da aplicação
//
// Node.js carrega cada módulo uma única vez e mantém a referência em cache.
// Todo arquivo que faz require('./stateModel') recebe o MESMO objeto,
// tornando este módulo o repositório central de estado em memória.
// ─────────────────────────────────────────────────────────────────────────────
module.exports = {
  // Carrinho atual — array de objetos produto { id, name, category, price, color }
  cart: [],

  // Modelo TF.js treinado — null até o treino inicial concluir
  model: null,

  // Sub-modelo que extrai o embedding de 32 dims da penúltima camada da rede.
  // Usado para indexar produtos no Qdrant e para gerar o vetor de consulta do carrinho.
  embeddingExtractor: null,

  // Impede treinos simultâneos (ex: dois requests POST /model/train ao mesmo tempo)
  isTraining: false,

  // Torna-se true ao fim do treino — habilita o botão "Run Recommendation" no frontend
  trainingComplete: false,

  // true quando os embeddings de todos os produtos estão indexados no Qdrant
  vectorIndexReady: false,

  // Métricas do treino mais recente — enviadas ao browser via socket 'training:done'
  lastMetrics: null,

  // Histórico de cada época — reproduzido para clientes que conectam durante ou após o treino
  // Garante que um page refresh sempre exibe os gráficos completos sem retreinar
  epochHistory: [],

  // Usuário ativo na sessão (pode ser um dos 10 usuários fixos ou o usuário vazio)
  // A idade é a última dimensão do vetor de contexto do carrinho
  sessionUser: { age: 28 }, // padrão: usuário vazio com 28 anos
};
