'use strict';
const allProducts = require('../data/products.json');
const state       = require('../models/stateModel');
const { scoreProducts } = require('../models/recommendationModel');

// GET /api/products — retorna o catálogo completo
// Se modelo treinado + carrinho não vazio: retorna ordenado por score de recomendação
// Caso contrário: retorna na ordem original sem score
exports.getAll = (_req, res) => {
  if (state.model && state.cart.length) {
    const userAge = state.sessionUser.age; // usa idade do usuário ativo na sessão
    return res.json(scoreProducts(state.model, state.cart, allProducts, userAge));
  }

  res.json(allProducts.map(product => ({ ...product, score: 0, inCart: false })));
};
