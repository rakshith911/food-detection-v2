module.exports = function(api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      // Must run before any class-feature plugins so that TypeScript `declare` fields
      // (used by expo-file-system's new NativeModule architecture) are stripped first.
      ['@babel/plugin-transform-typescript', { allowDeclareFields: true }],
      '@babel/plugin-transform-class-static-block',
      '@babel/plugin-transform-private-methods',
      ['@babel/plugin-transform-class-properties', { loose: false }],
      '@babel/plugin-transform-private-property-in-object',
    ],
  };
};

