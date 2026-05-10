// Learn more https://docs.expo.dev/guides/customizing-metro
// TEMPORARILY DISABLED Sentry Metro config to restore console.log
// const { getSentryExpoConfig } = require('@sentry/react-native/metro');
const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Configure server for emulator/localhost access
config.server = {
  ...config.server,
  enhanceMiddleware: (middleware) => {
    return middleware;
  },
  // For emulator, bind to localhost
  port: 8081,
};

// Add support for AWS Amplify ESM modules (.mjs files) and CSS
config.resolver.sourceExts.push('mjs', 'cjs', 'css');

// Configure SVG transformer - must be done before other transformer configs
const { transformer, resolver } = config;

if (!config.resolver.assetExts.includes('onnx')) {
  config.resolver.assetExts.push('onnx');
}

config.transformer = {
  ...transformer,
  babelTransformerPath: require.resolve('react-native-svg-transformer'),
  getTransformOptions: transformer.getTransformOptions,
};

config.resolver = {
  ...resolver,
  assetExts: resolver.assetExts.filter((ext) => ext !== 'svg'),
  sourceExts: [...resolver.sourceExts, 'svg'],
};

// Ensure proper resolution of node_modules
config.resolver.nodeModulesPaths = [
  path.resolve(__dirname, 'node_modules'),
];

// Enable package exports for better module resolution
config.resolver.unstable_enablePackageExports = true;

// Custom resolver to help with AWS Amplify module resolution
const originalResolveRequest = config.resolver.resolveRequest;
config.resolver.resolveRequest = (context, moduleName, platform) => {
  // Handle @aws-amplify/core resolution
  if (moduleName === '@aws-amplify/core' || moduleName.startsWith('@aws-amplify/')) {
    try {
      const resolvedPath = require.resolve(moduleName, {
        paths: [path.resolve(__dirname, 'node_modules')],
      });
      return {
        type: 'sourceFile',
        filePath: resolvedPath,
      };
    } catch (e) {
      // Fall through to default resolver
    }
  }
  
  // Use default resolver for everything else
  if (originalResolveRequest) {
    return originalResolveRequest(context, moduleName, platform);
  }
  return context.resolveRequest(context, moduleName, platform);
};

module.exports = config;
