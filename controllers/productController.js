'use strict';
const products = require('../data/products.json');
const state    = require('../models/stateModel');
const { scoreProducts } = require('../models/recommendationModel');

exports.getAll = (_req, res) => {
  if (state.model && state.cart.length) {
    return res.json(scoreProducts(state.model, state.cart, products));
  }
  res.json(products.map(p => ({ ...p, score: 0, inCart: false })));
};
