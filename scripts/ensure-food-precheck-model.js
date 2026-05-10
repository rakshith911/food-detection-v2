const fs = require('fs');
const https = require('https');
const path = require('path');

const modelPath = path.resolve(__dirname, '../assets/models/food_precheck.onnx');
const modelUrl = process.env.FOOD_PRECHECK_MODEL_URL;

const download = (url, destination, redirectCount = 0) =>
  new Promise((resolve, reject) => {
    if (redirectCount > 5) {
      reject(new Error('Too many redirects while downloading food precheck model'));
      return;
    }

    https
      .get(url, (response) => {
        const statusCode = response.statusCode || 0;
        const location = response.headers.location;

        if (statusCode >= 300 && statusCode < 400 && location) {
          response.resume();
          const nextUrl = new URL(location, url).toString();
          download(nextUrl, destination, redirectCount + 1).then(resolve, reject);
          return;
        }

        if (statusCode !== 200) {
          response.resume();
          reject(new Error(`Model download failed with HTTP ${statusCode}`));
          return;
        }

        const tempPath = `${destination}.tmp`;
        const file = fs.createWriteStream(tempPath);
        response.pipe(file);

        file.on('finish', () => {
          file.close(() => {
            fs.renameSync(tempPath, destination);
            resolve();
          });
        });

        file.on('error', (error) => {
          fs.rmSync(tempPath, { force: true });
          reject(error);
        });
      })
      .on('error', reject);
  });

const main = async () => {
  if (fs.existsSync(modelPath)) {
    const stats = fs.statSync(modelPath);
    if (stats.size > 0) {
      console.log(`[FoodPrecheck] Model already exists at ${modelPath}`);
      return;
    }
  }

  if (!modelUrl) {
    throw new Error(
      'Missing assets/models/food_precheck.onnx. Set FOOD_PRECHECK_MODEL_URL in EAS secrets so the build can download the model.'
    );
  }

  fs.mkdirSync(path.dirname(modelPath), { recursive: true });
  console.log('[FoodPrecheck] Downloading model for EAS build...');
  await download(modelUrl, modelPath);
  console.log(`[FoodPrecheck] Model downloaded to ${modelPath}`);
};

main().catch((error) => {
  console.error(`[FoodPrecheck] ${error.message}`);
  process.exit(1);
});
