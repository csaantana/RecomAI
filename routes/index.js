'use strict';
const express = require('express');
const router  = express.Router();

const product = require('../controllers/productController');
const user    = require('../controllers/userController');
const model   = require('../controllers/modelController');

router.get('/products',           product.getAll);
router.get('/users',              user.getAll);
router.get('/cart',               user.getCart);
router.post('/cart/add',          user.addToCart);
router.delete('/cart/:productId', user.removeFromCart);
router.post('/cart/clear',        user.clearCart);
router.post('/cart/load/:userId', user.loadUserCart);
router.post('/model/train',       model.train);
router.post('/model/recommend',   model.recommend);

module.exports = router;
