'use strict';
const usersData = require('../data/users.json');
const products  = require('../data/products.json');
const state     = require('../models/stateModel');

const pMap = {};
products.forEach(p => { pMap[p.id] = p; });

exports.getAll = (_req, res) => res.json(usersData);

exports.getCart = (_req, res) => res.json(state.cart);

exports.addToCart = (req, res) => {
  const { productId } = req.body;
  const product = pMap[productId];
  if (!product)                          return res.status(404).json({ error: 'Produto não encontrado' });
  if (state.cart.some(p => p.id === productId)) return res.status(409).json({ error: 'Já no carrinho' });
  state.cart.push(product);
  res.json(state.cart);
};

exports.removeFromCart = (req, res) => {
  state.cart = state.cart.filter(p => p.id !== req.params.productId);
  res.json(state.cart);
};

exports.clearCart = (_req, res) => {
  state.cart = [];
  res.json(state.cart);
};

exports.loadUserCart = (req, res) => {
  const user = usersData.find(u => u.id === req.params.userId);
  if (!user) return res.status(404).json({ error: 'Usuário não encontrado' });
  state.cart = user.purchaseHistory.map(id => pMap[id]).filter(Boolean);
  res.json(state.cart);
};
