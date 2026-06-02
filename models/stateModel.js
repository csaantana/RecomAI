'use strict';

module.exports = {
  cart            : [],
  model           : null,
  isTraining      : false,
  trainingComplete: false,
  lastMetrics     : null,
  epochHistory    : [],   // replayed to clients that connect mid/post-training
};
