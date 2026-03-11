import React from 'react';
import { Image, ActivityIndicator, StyleSheet, View, ImageProps } from 'react-native';
import { Image as ExpoImage } from 'expo-image';

type OptimizedImageProps = ImageProps & {
  cachePolicy?: 'none' | 'memory' | 'disk' | 'memory-disk';
  priority?: 'low' | 'normal' | 'high';
  cacheKey?: string;
  showLoader?: boolean;
  onImageLoad?: () => void;
  onImageError?: () => void;
};

export default function OptimizedImage({
  source,
  resizeMode = 'cover',
  cachePolicy = 'memory-disk',
  priority = 'normal',
  cacheKey,
  onImageLoad,
  onImageError,
  showLoader = false,
  style,
  onLoad: _onLoad,
  onError: _onError,
  ...props
}: OptimizedImageProps) {
  const [isLoading, setIsLoading] = React.useState(true);
  const [hasError, setHasError] = React.useState(false);

  const contentFit = resizeMode === 'cover' ? 'cover' : resizeMode === 'contain' ? 'contain' : resizeMode;

  const handleLoad = () => {
    setIsLoading(false);
    setHasError(false);
    onImageLoad?.();
  };

  const handleError = () => {
    setIsLoading(false);
    setHasError(true);
    onImageError?.();
  };

  if (typeof source === 'number') {
    return (
      <View style={[styles.container, style]}>
        <Image
          source={source}
          style={[styles.image, style]}
          resizeMode={resizeMode}
          onLoad={handleLoad}
          onError={handleError}
          {...props}
        />
        {showLoader && isLoading && (
          <View style={styles.loaderContainer}>
            <ActivityIndicator size="small" color="#7BA21B" />
          </View>
        )}
      </View>
    );
  }

  return (
    <View style={[styles.container, style]}>
      <ExpoImage
        source={source}
        style={StyleSheet.absoluteFill}
        contentFit={contentFit as any}
        cachePolicy={cachePolicy}
        priority={priority}
        {...(cacheKey ? { cacheKey } : {})}
        transition={200}
        onLoad={handleLoad}
        onError={handleError}
      />
      {showLoader && isLoading && (
        <View style={styles.loaderContainer}>
          <ActivityIndicator size="small" color="#7BA21B" />
        </View>
      )}
      {hasError && (
        <View style={styles.skeletonContainer} />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    overflow: 'hidden',
    backgroundColor: '#E5E7EB',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  loaderContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
  skeletonContainer: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#E5E7EB',
  },
});

