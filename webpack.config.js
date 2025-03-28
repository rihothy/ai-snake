const path = require('path');

module.exports = {
  entry: {
    'index': './src/js/index.js',
    'train.worker': './src/js/train.worker.js'
  },
  
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].js',
    clean: true
  },
  
  mode: 'development',
  
  devtool: 'inline-source-map'
}; 