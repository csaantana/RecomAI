'use strict';
const allProducts = require('../data/products.json');
const state       = require('../models/stateModel');
const { scoreProducts } = require('../models/recommendationModel');

// GET /api/products — retorna o catálogo completo
// Se o modelo já estiver treinado e houver itens no carrinho, retorna os produtos
// ordenados por score de recomendação. Caso contrário, retorna na ordem original.
exports.getAll = (_req, res) => {
  if (state.model && state.cart.length) {
    return res.json(scoreProducts(state.model, state.cart, allProducts));
  }

  // Modelo não treinado ou carrinho vazio: retorna sem score
  res.json(allProducts.map(product => ({ ...product, score: 0, inCart: false })));
};
