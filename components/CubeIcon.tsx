import React from 'react';
import Svg, { Polygon } from 'react-native-svg';

interface CubeIconProps {
  size?: number;
  color?: string;
}

export default function CubeIcon({ size = 15, color = '#7BA21B' }: CubeIconProps) {
  const isLight = color === '#FFFFFF';
  const strokeColor = isLight ? 'rgba(255,255,255,0.6)' : '#4d6b10';

  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      {/* Top face */}
      <Polygon
        points="12,2 22,7 12,12 2,7"
        fill={color}
        stroke={strokeColor}
        strokeWidth="1.2"
        strokeLinejoin="round"
      />
      {/* Left face */}
      <Polygon
        points="2,7 12,12 12,22 2,17"
        fill={color}
        stroke={strokeColor}
        strokeWidth="1.2"
        strokeLinejoin="round"
        opacity={0.75}
      />
      {/* Right face */}
      <Polygon
        points="22,7 22,17 12,22 12,12"
        fill={color}
        stroke={strokeColor}
        strokeWidth="1.2"
        strokeLinejoin="round"
        opacity={0.55}
      />
    </Svg>
  );
}
