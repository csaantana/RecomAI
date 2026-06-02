'use strict';
const allUsers    = require('../data/users.json');
const allProducts = require('../data/products.json');
const state       = require('../models/stateModel');

// Índice de produtos por id → busca O(1) em vez de O(n) a cada request
const productById = {};
allProducts.forEach(product => { productById[product.id] = product; });

// GET /api/users — lista os 10 usuários com seus históricos de compra
exports.getAll = (_req, res) => res.json(allUsers);

// GET /api/cart — retorna o carrinho em memória do usuário atual
exports.getCart = (_req, res) => res.json(state.cart);

// POST /api/cart/add { productId } — adiciona um produto ao carrinho
exports.addToCart = (req, res) => {
  const { productId } = req.body;
  const product = productById[productId];

  if (!product) {
    return res.status(404).json({ error: 'Produto não encontrado' });
  }
  if (state.cart.some(cartItem => cartItem.id === productId)) {
    return res.status(409).json({ error: 'Produto já está no carrinho' });
  }

  state.cart.push(product);
  res.json(state.cart);
};

// DELETE /api/cart/:productId — remove um produto do carrinho pelo id
exports.removeFromCart = (req, res) => {
  state.cart = state.cart.filter(cartItem => cartItem.id !== req.params.productId);
  res.json(state.cart);
};

// POST /api/cart/clear — esvazia o carrinho
exports.clearCart = (_req, res) => {
  state.cart = [];
  res.json(state.cart);
};

// POST /api/cart/load/:userId — carrega o histórico de compras de um usuário
// como carrinho inicial (permite testar recomendações com perfis reais)
exports.loadUserCart = (req, res) => {
  const user = allUsers.find(u => u.id === req.params.userId);
  if (!user) return res.status(404).json({ error: 'Usuário não encontrado' });

  // Converte lista de ids em objetos de produto completos
  state.cart = user.purchaseHistory
    .map(productId => productById[productId])
    .filter(Boolean); // descarta ids que não existam no catálogo

  res.json(state.cart);
};
