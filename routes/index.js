'use strict';
const express = require('express');
const router  = express.Router();

const productController = require('../controllers/productController');
const userController    = require('../controllers/userController');
const modelController   = require('../controllers/modelController');

// ── Produtos ──────────────────────────────────────────────────────────────────
router.get('/products', productController.getAll);          // catálogo (ordenado por score se modelo ativo)

// ── Usuários e carrinho ───────────────────────────────────────────────────────
router.get('/users',                 userController.getAll);         // lista os 10 usuários
router.get('/cart',                  userController.getCart);        // carrinho atual
router.post('/cart/add',             userController.addToCart);      // adiciona produto { productId }
router.delete('/cart/:productId',    userController.removeFromCart); // remove produto por id
router.post('/cart/clear',           userController.clearCart);      // esvazia o carrinho
router.post('/cart/load/:userId',    userController.loadUserCart);   // carrega histórico do usuário

// ── Modelo de recomendação ────────────────────────────────────────────────────
router.post('/model/train',          modelController.train);         // retreino manual (opcional)
router.post('/model/recommend',      modelController.recommend);     // pontua e ordena o catálogo

module.exports = router;
