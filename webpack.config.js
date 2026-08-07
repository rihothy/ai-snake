const TerserPlugin = require("terser-webpack-plugin");
const path = require('path');

module.exports = (env, argv) => ({
  entry: {
    'index': './src/js/index.js',
  },

  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].js',
    clean: true
  },

  optimization: {
    minimizer: [
      new TerserPlugin({
        extractComments: false,
      }),
    ],
  },

  mode: argv.mode || 'development',

  // Inline source maps massively inflate the production bundle (~18MB).
  // Keep them for development, omit them in the release build.
  devtool: argv.mode === 'production' ? false : 'inline-source-map'
}); 