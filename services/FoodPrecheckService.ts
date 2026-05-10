import { Asset } from 'expo-asset';
import * as ImageManipulator from 'expo-image-manipulator';
import jpeg from 'jpeg-js';
import type * as Ort from 'onnxruntime-react-native';
import foodPrecheckModel from '../assets/models/food_precheck.onnx';

export type FoodPrecheckVerdict = 'allow' | 'reject';

export interface FoodPrecheckResult {
  verdict: FoodPrecheckVerdict;
  isFood: boolean;
  confidence: number;
  label: 'food' | 'not_food' | 'unknown';
  reason: string;
}

const MODEL_INPUT_SIZE = 384;
const FOOD_CONFIDENCE_THRESHOLD = 0.65;
const FOOD_CLASS_INDEX = 0;
const NOT_FOOD_CLASS_INDEX = 1;
const CHANNEL_MEAN = [0.485, 0.456, 0.406];
const CHANNEL_STD = [0.229, 0.224, 0.225];

let ortModulePromise: Promise<typeof Ort> | null = null;
let sessionPromise: Promise<Ort.InferenceSession> | null = null;

const loadOrt = async () => {
  if (!ortModulePromise) {
    ortModulePromise = import('onnxruntime-react-native').then((module) => {
      const maybeDefault = (module as unknown as { default?: typeof Ort }).default;
      return maybeDefault || module;
    });
  }
  return ortModulePromise;
};

const softmax = (values: number[]) => {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
};

const loadSession = async () => {
  if (sessionPromise) {
    return sessionPromise;
  }

  sessionPromise = (async () => {
    const ort = await loadOrt();
    if (!ort.InferenceSession?.create) {
      throw new Error('ONNX Runtime native module is unavailable. Rebuild the iOS/Android dev client.');
    }

    // NextViT food/not-food classifier:
    // classes: [food_or_drink, not_food_or_drink], input: RGB float32 NCHW 1x3x384x384.
    const asset = Asset.fromModule(foodPrecheckModel);
    await asset.downloadAsync();
    const modelUri = asset.localUri || asset.uri;
    if (!modelUri) {
      throw new Error('Food verifier model could not be loaded from assets');
    }
    return ort.InferenceSession.create(modelUri);
  })();

  return sessionPromise;
};

const resizeBilinearRgb = (
  source: Uint8Array,
  sourceWidth: number,
  sourceHeight: number,
  targetSize: number
) => {
  const resized = new Uint8Array(targetSize * targetSize * 3);
  const xRatio = sourceWidth / targetSize;
  const yRatio = sourceHeight / targetSize;

  for (let y = 0; y < targetSize; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor((y + 0.5) * yRatio));
    for (let x = 0; x < targetSize; x += 1) {
      const sourceX = Math.min(sourceWidth - 1, Math.floor((x + 0.5) * xRatio));
      const sourceIndex = (sourceY * sourceWidth + sourceX) * 4;
      const targetIndex = (y * targetSize + x) * 3;
      resized[targetIndex] = source[sourceIndex];
      resized[targetIndex + 1] = source[sourceIndex + 1];
      resized[targetIndex + 2] = source[sourceIndex + 2];
    }
  }

  return resized;
};

const imageUriToTensor = async (imageUri: string) => {
  const ort = await loadOrt();
  const normalizedImage = await ImageManipulator.manipulateAsync(
    imageUri,
    [{ resize: { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE } }],
    {
      compress: 1,
      format: ImageManipulator.SaveFormat.JPEG,
    }
  );
  const imageResponse = await fetch(normalizedImage.uri);
  const imageBytes = new Uint8Array(await imageResponse.arrayBuffer());
  const decoded = jpeg.decode(imageBytes, { useTArray: true });
  const resizedRgb = resizeBilinearRgb(
    decoded.data,
    decoded.width,
    decoded.height,
    MODEL_INPUT_SIZE
  );
  const planeSize = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
  const input = new Float32Array(3 * planeSize);

  for (let pixelIndex = 0; pixelIndex < planeSize; pixelIndex += 1) {
    const rgbIndex = pixelIndex * 3;
    input[pixelIndex] = (resizedRgb[rgbIndex] / 255 - CHANNEL_MEAN[0]) / CHANNEL_STD[0];
    input[planeSize + pixelIndex] = (resizedRgb[rgbIndex + 1] / 255 - CHANNEL_MEAN[1]) / CHANNEL_STD[1];
    input[planeSize * 2 + pixelIndex] = (resizedRgb[rgbIndex + 2] / 255 - CHANNEL_MEAN[2]) / CHANNEL_STD[2];
  }

  return new ort.Tensor('float32', input, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
};

const readClassifierScores = (output: Ort.Tensor) => {
  const rawValues = Array.from(output.data as Float32Array | number[]);
  const values = rawValues.length > 2 ? rawValues.slice(0, 2) : rawValues;
  if (values.length < 2) {
    throw new Error('Food verifier model must return two class scores: food and not_food');
  }

  const looksLikeProbabilities =
    values.every((value) => value >= 0 && value <= 1) &&
    Math.abs(values.reduce((total, value) => total + value, 0) - 1) < 0.05;

  return looksLikeProbabilities ? values : softmax(values);
};

export async function verifyFoodImageOnDevice(imageUri: string): Promise<FoodPrecheckResult> {
  try {
    const session = await loadSession();
    const inputTensor = await imageUriToTensor(imageUri);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const outputs = await session.run({ [inputName]: inputTensor });
    const scores = readClassifierScores(outputs[outputName]);
    const foodConfidence = scores[FOOD_CLASS_INDEX] || 0;
    const notFoodConfidence = scores[NOT_FOOD_CLASS_INDEX] || 0;
    const isFood = foodConfidence >= FOOD_CONFIDENCE_THRESHOLD && foodConfidence > notFoodConfidence;

    return {
      verdict: isFood ? 'allow' : 'reject',
      isFood,
      confidence: isFood ? foodConfidence : Math.max(notFoodConfidence, 1 - foodConfidence),
      label: isFood ? 'food' : 'not_food',
      reason: isFood
        ? 'Food detected locally on device'
        : 'The selected image does not look like food',
    };
  } catch (error) {
    console.warn('[FoodPrecheck] Verification failed:', error);
    return {
      verdict: 'reject',
      isFood: false,
      confidence: 0,
      label: 'unknown',
      reason: error instanceof Error ? error.message : 'Food verifier failed',
    };
  }
}
